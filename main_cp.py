#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Hard No-Box SSL Training Script (Fixed with KNN Validation)

import os
import argparse
import random
import shutil
import time
import warnings
import math
import datetime
import numpy as np

try:
    import distutils.version
except Exception:
    import sys, types, setuptools

    distutils = types.ModuleType("distutils")
    distutils.version = setuptools._distutils.version
    sys.modules["distutils"] = distutils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "backend:cudaMallocAsync,"
    "max_split_size_mb:128,"
    "garbage_collection_threshold:0.8,"
    "expandable_segments:True"
)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

# === 引用原作者的 Brain ===
import moco.builder_inter

# === 引用你的 Body ===
from dataset import get_pretraining_set_inter
from dataset import get_finetune_validation_set
from dataset import get_finetune_training_set  # 新增：用于KNN底库
from options import options_pretraining as options

# --------------------
# 命令行参数设置
# --------------------
parser = argparse.ArgumentParser(description='Hard No-Box MoCo Training')

# 通用参数
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate')  # 建议0.005适配BS=64
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('--checkpoint-path', default='./checkpoints_attack', type=str, help='checkpoint dir')

parser.add_argument('--epochs', default=350, type=int, help='number of total epochs')
parser.add_argument('--frame', default=1000, type=int, help='number of frames')
parser.add_argument('--num-workers', default=8, type=int, help='num of workers')
parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size')
parser.add_argument('--schedule', default=[160, 260], nargs='*', type=int, help='lr schedule')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

# MoCo 核心参数
parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--moco-k', default=1024, type=int, help='queue size')  # 建议1024或512
parser.add_argument('--moco-m', default=0.999, type=float, help='moco momentum')
parser.add_argument('--moco-t', default=0.07, type=float, help='softmax temperature')
parser.add_argument('--mlp', action='store_true', help='use mlp head')  # 建议开启，加上 --mlp
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')  # 建议开启，加上 --cos

# 增强开关
parser.add_argument('--use-augmentation', default=True, type=bool, help='use augmentation')


def main():
    print(f"开始时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    start_time = datetime.datetime.now()

    args = parser.parse_args()

    # 显存优化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

    end_time = datetime.datetime.now()
    print(f"结束时间: {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"训练耗时: {end_time - start_time}")


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # === 路径管理 ===
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(args.checkpoint_path, f"run_hardnobox_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # === 获取配置 ===
    if hasattr(options, 'opts_yhx_pretrain'):
        opts = options.opts_yhx_pretrain()
    else:
        print("Error: opts_yhx_pretrain not found in options_pretraining.py")
        return

    # 强制同步参数
    opts.train_feeder_args['input_size'] = 64
    if 'l_ratio' not in opts.train_feeder_args:
        opts.train_feeder_args['l_ratio'] = [0.1, 1]

    # === 模型构建 ===
    print(f"[Info] Creating Hard No-Box MoCo Model (AGCN + GRU)...")
    model = moco.builder_inter.MoCo(
        skeleton_representation='seq-based_and_graph-based',
        args_bi_gru=opts.bi_gru_model_args,
        args_agcn=opts.agcn_model_args,
        args_hcn=None,
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        mlp=args.mlp
    ).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Resume
    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")

    cudnn.benchmark = True

    # === 数据加载 ===
    # 1. 训练集 (MoCo Update): 双视图，无标签返回
    train_dataset = get_pretraining_set_inter(opts)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, persistent_workers=True)

    # 2. 内存集 (KNN Gallery): 训练集，无增强，带标签
    # 我们使用 get_finetune_training_set (它调用 feeder_downstream，不带双视图增强)
    # 注意：需要把 options 里的 train_feeder_args 复制给 memory loader 使用
    # 但 feeder_downstream 需要 label_path，ensure it is in opts
    memory_dataset = get_finetune_training_set(opts)
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, persistent_workers=True)

    # 3. 验证集 (KNN Query): 验证集，无增强，带标签
    # 配置验证集使用 graph-based (因为我们只验证 AGCN 分支)
    opts.val_feeder_args['input_representation'] = 'graph-based'
    val_dataset = get_finetune_validation_set(opts)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, persistent_workers=True)

    writer = SummaryWriter(log_dir=tb_dir)

    # --------------------
    # 训练循环
    # --------------------
    best_acc = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # === 训练 ===
        loss, acc_seq, acc_graph = train(
            train_loader, model, criterion, optimizer, epoch, args, device
        )

        writer.add_scalar('Train/Loss', loss, epoch)
        writer.add_scalar('Train/InstDisc_Seq_Acc', acc_seq, epoch)
        writer.add_scalar('Train/InstDisc_Graph_Acc', acc_graph, epoch)

        print(f"Epoch [{epoch:03d}] Loss: {loss:.4f} | Seq Acc: {acc_seq:.2f}% | Graph Acc: {acc_graph:.2f}%")

        # === KNN 验证 (Identity Recognition) ===
        # 每 10 轮验证一次，或者最后几轮
        if (epoch % args.print_freq == 0) or (epoch >= args.epochs - 5):
            val_acc = evaluate_knn(model, memory_loader, val_loader, device)
            print(f"[Validation] KNN Top-1 Accuracy: {val_acc:.2f}%")
            writer.add_scalar('Val/KNN_Acc', val_acc, epoch)

            # 保存最优模型
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, is_best=True, filename=os.path.join(ckpt_dir, f'checkpoint_{epoch:04d}.pth.tar'))

        # 常规保存
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(ckpt_dir, f'checkpoint_{epoch:04d}.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch, args, device):
    """
    Hard No-Box MoCo Training Loop
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_seq = AverageMeter('Acc@Seq', ':6.2f')
    top1_graph = AverageMeter('Acc@Graph', ':6.2f')

    model.train()
    end = time.time()

    for i, (input_s1_v1, input_s2_v1, input_s1_v2, input_s2_v2) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # 搬运数据到 GPU
        input_s1_v1 = input_s1_v1.float().to(device, non_blocking=True)
        input_s2_v1 = input_s2_v1.float().to(device, non_blocking=True)
        input_s1_v2 = input_s1_v2.float().to(device, non_blocking=True)
        input_s2_v2 = input_s2_v2.float().to(device, non_blocking=True)

        # Forward
        (logits_seq, logits_graph), (labels_seq, labels_graph) = model(
            input_s1_v1, input_s2_v1, input_s1_v2, input_s2_v2
        )

        # Loss
        loss_seq = criterion(logits_seq, labels_seq)
        loss_graph = criterion(logits_graph, labels_graph)
        loss = loss_seq + loss_graph

        # Metrics (Instance Discrimination Acc)
        acc_s = accuracy(logits_seq, labels_seq, topk=(1,))[0]
        acc_g = accuracy(logits_graph, labels_graph, topk=(1,))[0]

        losses.update(loss.item(), input_s1_v1.size(0))
        top1_seq.update(acc_s.item(), input_s1_v1.size(0))
        top1_graph.update(acc_g.item(), input_s1_v1.size(0))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f"  Iter [{i}/{len(train_loader)}] Loss {losses.val:.4f} ({losses.avg:.4f})")

    return losses.avg, top1_seq.avg, top1_graph.avg


@torch.no_grad()
def evaluate_knn(model, memory_loader, val_loader, device, k=1):
    """
    使用 KNN 验证 AGCN 分支的特征质量
    """
    model.eval()
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []

    # 1. 构建 Memory Bank (底库)
    # 使用 feeder_downstream，它返回 (data, label, ...)
    for batch in memory_loader:
        if len(batch) == 3:
            data, label, _ = batch
        else:
            data, label = batch
        data = data.float().to(device)

        # 只用 encoder_r (AGCN) 提取特征
        # 注意: AGCN forward 需要支持 knn_eval 参数返回特征，或者默认返回
        # 原作者代码 AGCN.forward 有 knn_eval 参数
        feat = model.encoder_r(data, knn_eval=True)
        feat = torch.nn.functional.normalize(feat, dim=1)

        train_features.append(feat.cpu())
        train_labels.append(label)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # 2. 提取验证集特征 (Query)
    for batch in val_loader:
        if len(batch) == 3:
            data, label, _ = batch
        else:
            data, label = batch
        data = data.float().to(device)

        feat = model.encoder_r(data, knn_eval=True)
        feat = torch.nn.functional.normalize(feat, dim=1)

        val_features.append(feat.cpu())
        val_labels.append(label)

    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    # 3. 计算 KNN (K=1)
    # 余弦相似度 = 归一化后的点积
    sim_mat = torch.mm(val_features, train_features.t())
    _, topk_idx = sim_mat.topk(k, dim=1, largest=True)

    # 预测
    topk_labels = train_labels[topk_idx]  # (N_val, K)
    pred_labels = topk_labels[:, 0]  # K=1

    correct = (pred_labels == val_labels).sum().item()
    acc = correct / val_labels.size(0) * 100.0

    return acc


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0;
        self.avg = 0;
        self.sum = 0;
        self.count = 0

    def update(self, val, n=1):
        self.val = val;
        self.sum += val * n;
        self.count += n;
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            if epoch >= milestone: lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'model_best.pth.tar'))


if __name__ == '__main__':
    main()