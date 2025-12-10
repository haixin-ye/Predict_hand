#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Hard No-Box SSL Training Script (User's Body + Author's Brain)

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
from options import options_pretraining as options

# --------------------
# 命令行参数设置
# --------------------
parser = argparse.ArgumentParser(description='Hard No-Box MoCo Training')

# 通用参数 (Your Body)
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('--checkpoint-path', default='./checkpoints_attack', type=str, help='checkpoint dir')

parser.add_argument('--epochs', default=350, type=int, help='number of total epochs')
parser.add_argument('--frame', default=1000, type=int, help='number of frames')
parser.add_argument('--num-workers', default=8, type=int, help='num of workers')
parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size')
parser.add_argument('--schedule', default=[160, 260], nargs='*', type=int, help='lr schedule')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

# MoCo 核心参数 (Author's Brain)
parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--moco-k', default=16384, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco-m', default=0.999, type=float, help='moco momentum')
parser.add_argument('--moco-t', default=0.07, type=float, help='softmax temperature')
parser.add_argument('--mlp', action='store_true', help='use mlp head')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

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
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting.')

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

    # === 路径管理 (Your Body Style) ===
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(args.checkpoint_path, f"run_hardnobox_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # === 获取配置 ===
    # 优先加载你的手部数据配置 opts_yhx_pretrain
    if hasattr(options, 'opts_yhx_pretrain'):
        opts = options.opts_yhx_pretrain()
    else:
        opts = options.opts_ntu_60_cross_view()  # Fallback

    # 强制同步命令行参数
    opts.train_feeder_args['uniform_frame_length'] = args.frame
    opts.val_feeder_args['uniform_frame_length'] = args.frame
    opts.train_feeder_args['input_size'] = 64  # 固定输入给 GRU/AGCN 的长度，推荐 64

    # 增强配置：Hard No-Box 的 feeder 需要 l_ratio 参数，我们在这里确保它存在
    if 'l_ratio' not in opts.train_feeder_args:
        opts.train_feeder_args['l_ratio'] = [0.1, 1]

    # === 模型构建 (Author's Brain) ===
    print(f"[Info] Creating Hard No-Box MoCo Model (AGCN + GRU)...")
    model = moco.builder_inter.MoCo(
        skeleton_representation='seq-based_and_graph-based',  # 激活双流
        args_bi_gru=opts.bi_gru_model_args,  # 你的 GRU 参数
        args_agcn=opts.agcn_model_args,  # 你的 AGCN 参数
        args_hcn=None,
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        mlp=args.mlp
    ).to(device)

    # Loss & Optimizer
    # 原作者使用 CrossEntropyLoss 配合 builder_inter 返回的 (logits, labels)
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

    # === 数据加载 (Your Body) ===
    # 注意：这里使用的是 get_pretraining_set_inter，它调用 feeder_pretraining_inter.py
    # 该 feeder 现在返回 4 个 Tensor: (seq_v1, graph_v1, seq_v2, graph_v2)
    train_dataset = get_pretraining_set_inter(opts)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, persistent_workers=True)

    # 验证集 (用于监控)
    # 我们配置它只看 graph-based，用于 evaluate_closedset_knn
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

        # === Train Step (Author's Logic inside Your Loop) ===
        loss, acc_seq, acc_graph = train(
            train_loader, model, criterion, optimizer, epoch, args, device
        )

        # 日志记录 (Your Style)
        writer.add_scalar('Train/Loss', loss, epoch)
        writer.add_scalar('Train/Acc_Seq', acc_seq, epoch)
        writer.add_scalar('Train/Acc_Graph', acc_graph, epoch)

        # 打印信息
        print(f"Epoch [{epoch:03d}] Loss: {loss:.4f} | Seq Acc: {acc_seq:.2f}% | Graph Acc: {acc_graph:.2f}%")

        # === Validation Step (Optional but recommended) ===
        # 每隔几轮验证一次 AGCN 分支的 KNN 准确率
        if (epoch % args.print_freq == 0) or (epoch == args.epochs - 1):
            # 注意：如果你的 feeder_pretraining 不返回 label，这个 KNN 可能无法运行
            # 这里假设你已经按我之前的建议修改了 feeder 或者是无监督 loss 监控
            # 如果不想跑验证，可以注释掉下面两行
            # val_acc = evaluate_closedset_knn_agcn(model, train_loader, val_loader, device)
            # print(f"[Val] KNN Top1 (AGCN Branch): {val_acc:.4f}")
            pass

            # 保存 Checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(ckpt_dir, f'checkpoint_{epoch:04d}.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch, args, device):
    """
    Hard No-Box 风格的训练循环：四路输入 -> 双流模型 -> 互对比 Loss
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_seq = AverageMeter('Acc@Seq', ':6.2f')
    top1_graph = AverageMeter('Acc@Graph', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1_seq, top1_graph],
        prefix="Epoch: [{}] LR: {:.4f}".format(epoch, optimizer.param_groups[0]['lr']))

    model.train()
    end = time.time()

    # feeder 吐出 4 份数据 (如果你的 feeder 修改正确的话)
    # 格式: (Seq_Q, Graph_Q, Seq_K, Graph_K)
    for i, (input_s1_v1, input_s2_v1, input_s1_v2, input_s2_v2) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # 搬运数据
        input_s1_v1 = input_s1_v1.float().to(device, non_blocking=True)  # Seq View 1
        input_s2_v1 = input_s2_v1.float().to(device, non_blocking=True)  # Graph View 1
        input_s1_v2 = input_s1_v2.float().to(device, non_blocking=True)  # Seq View 2
        input_s2_v2 = input_s2_v2.float().to(device, non_blocking=True)  # Graph View 2

        # === Author's Brain: Forward ===
        # MoCo Forward 内部计算 query, key, queue, 和 logits
        (logits_seq, logits_graph), (labels_seq, labels_graph) = model(
            input_s1_v1, input_s2_v1, input_s1_v2, input_s2_v2
        )

        # === Author's Brain: Loss ===
        loss_seq = criterion(logits_seq, labels_seq)
        loss_graph = criterion(logits_graph, labels_graph)
        loss = loss_seq + loss_graph

        # 统计精度 (Instance Discrimination Accuracy)
        acc_s, _ = accuracy(logits_seq, labels_seq, topk=(1,))
        acc_g, _ = accuracy(logits_graph, labels_graph, topk=(1,))

        # 更新仪表盘
        losses.update(loss.item(), input_s1_v1.size(0))
        top1_seq.update(acc_s[0].item(), input_s1_v1.size(0))
        top1_graph.update(acc_g[0].item(), input_s1_v1.size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1_seq.avg, top1_graph.avg


# --------------------
# 辅助函数 (Your Utilities)
# --------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""

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
        return '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'


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