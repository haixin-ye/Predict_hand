import json
import re
from collections import defaultdict
from lightgbm import LGBMClassifier
import numpy as np
from fastdtw import fastdtw
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # 导入 tqdm 库
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, classification_report

from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import os

from torch.nn.utils.rnn import pack_padded_sequence

from typing import Dict, Any, Union

from torch.nn.utils.rnn import pack_padded_sequence


class MotionPredictorHybrid(nn.Module):
    """
    Hybrid 折中方案（下一帧 9 维）：
      输入: (B, L, 3) 历史序列（支持 padding + lengths）
      输出: (B, 9) = [pos_next(3), vel_next(3), acc_next(3)]
      位置 = 融合( 直接回归的 pos_direct, 物理一致的 pos_phys )

    设计要点：
      - LSTM 编码序列 → shared bottleneck
      - head_va 输出 v_{t+1}, a_{t+1}（6 维）
      - head_pos 直接输出 pos_direct（3 维）
      - gate_alpha ∈ (0,1) 学习融合权重：pos = (1-α)*pos_direct + α*pos_phys
    """

    def __init__(self,
                 input_dim=3,
                 hidden_dim=128,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.0,
                 learn_alpha=True,  # 学习融合权重 α
                 init_alpha=0.3):  # α 初值（靠近物理或数据可调）
        super().__init__()
        self.num_layers = num_layers
        self.directions = 2 if bidirectional else 1
        self.learn_alpha = learn_alpha

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        feat_dim = hidden_dim * self.directions

        # shared bottleneck
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
        )

        # 速度+加速度头：6 维
        self.head_va = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # [vx,vy,vz, ax,ay,az]
        )

        # 直接位置头：3 维（数据驱动）
        self.head_pos = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # 融合权重 α：标量或逐样本标量（Sigmoid 输出 0~1）
        if learn_alpha:
            self.head_alpha = nn.Sequential(
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            # 用一个可训练/固定的全局参数也可以；这里做成 buffer 固定
            self.register_buffer("alpha_const", torch.tensor(float(init_alpha)))

    @staticmethod
    def _gather_last_true_pos(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        从 pad 后序列中按 lengths 取出每条序列最后一个真实位置 x_t
        x: (B,L,3), lengths: (B,)
        """
        B, L, _ = x.shape
        lengths = torch.clamp(lengths, 1, L)
        idx = (lengths - 1).to(x.device)
        b = torch.arange(B, device=x.device)
        return x[b, idx, :]  # (B,3)

    def forward(self, x, lengths=None, dt: Union[float, torch.Tensor] = 0.01):
        """
        x: (B, L, 3)
        lengths: (B,)
        dt: 标量或 (B,) tensor
        return: (B, 9) = [pos_next, vel_next, acc_next]
        """
        # 1) LSTM 编码
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.detach().cpu(),
                                          batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        B = x.size(0)
        hidden_dim = self.lstm.hidden_size
        h_n = h_n.view(self.num_layers, self.directions, B, hidden_dim)
        last = h_n[-1].transpose(0, 1).reshape(B, -1)  # (B, hidden*dirs)

        z = self.backbone(last)  # (B,128)

        # 2) 预测 v_{t+1}, a_{t+1}
        va = self.head_va(z)  # (B,6)
        vel_next = va[:, :3]
        acc_next = va[:, 3:]

        # 3) x_t & dt 处理
        if lengths is None:
            x_t = x[:, -1, :]
        else:
            x_t = self._gather_last_true_pos(x, lengths)  # (B,3)

        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, dtype=x.dtype, device=x.device)
        dt_vec = dt.view(-1, 1) if dt.dim() == 1 else dt.view(1, 1).expand(B, 1)

        # 4) 两路位置：物理一致 & 直接回归
        pos_phys = x_t + vel_next * dt_vec + 0.5 * acc_next * (dt_vec ** 2)  # (B,3)
        pos_direct = self.head_pos(z)  # (B,3)

        # 5) 学习/使用融合权重 α
        if self.learn_alpha:
            alpha = self.head_alpha(z)  # (B,1), 0~1
        else:
            alpha = self.alpha_const.view(1, 1).expand(B, 1)  # 固定 α

        pos_next = (1.0 - alpha) * pos_direct + alpha * pos_phys  # (B,3)

        out = torch.cat([pos_next, vel_next, acc_next], dim=-1)  # (B,9)
        return out, {"pos_direct": pos_direct, "pos_phys": pos_phys, "alpha": alpha}


def train_model(
        train_loader,
        model,
        epochs,
        lr,
        device,
        grad_clip=None,
        dt=0.01,  # ← 帧间隔（标量或 (B,) tensor 也行，直接传给模型）
        w_pos=1.0,  # ← 位置损失权重
        w_vel=0,  # ← 速度损失权重
        w_acc=0,  # ← 加速度损失权重
        w_consist=0.05,  # ← 一致性正则（pos_direct ≈ pos_phys）
        use_amp=False  # ← 可选：混合精度
):
    """
    期望 train_loader 每个 batch 返回: (X, lengths, y)
      - X: (B, L, 3) 已 pad 的历史窗口
      - lengths: (B,) 真实步长
      - y: (B, 9) 下一帧标签 [pos, vel, acc]
    模型:
      - 若是 Hybrid，forward(X, lengths=..., dt=...) -> (pred, aux)
      - 若是直接 9 维，forward(X, lengths=..., dt=...) -> pred
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_cnt = 0

        for batch in train_loader:
            # 支持 dict 或 tuple
            if isinstance(batch, dict):
                X = batch["X"]
                lengths = batch["lengths"]
                y = batch["y"]
            else:
                X, lengths, y = batch

            X = X.to(device)  # (B, L, 3)
            y = y.to(device)  # (B, 9)
            lengths = lengths.to(device)  # (B,)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # ---- 前向：兼容 Hybrid / 非 Hybrid ----
                out = model(X, lengths=lengths, dt=dt)
                if isinstance(out, tuple):
                    pred, aux = out  # pred: (B,9)
                else:
                    pred, aux = out, {}

                # ---- 损失：多任务 + 轻一致性 ----
                # 位置/速度/加速度
                loss_pos = mse(pred[:, :3], y[:, :3])
                loss_vel = mse(pred[:, 3:6], y[:, 3:6])
                loss_acc = mse(pred[:, 6:9], y[:, 6:9])

                loss = w_pos * loss_pos + w_vel * loss_vel + w_acc * loss_acc

                # 若有辅助量，加入一致性正则（Hybrid）
                if "pos_direct" in aux and "pos_phys" in aux and w_consist > 0.0:
                    loss_cons = mse(aux["pos_direct"], aux["pos_phys"])
                    loss = loss + w_consist * loss_cons
                else:
                    loss_cons = None

            # ---- 反传与优化 ----
            scaler.scale(loss).backward()
            if grad_clip is not None:
                # 注意：grad clip 要在 unscale_ 之后（混合精度）
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = X.size(0)
            total_loss += float(loss.item()) * bs
            total_cnt += bs

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / max(1, total_cnt):.9f}")

    return model


def read_coordinates(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()[1:-1]  # 去掉括号
            x, y, z = map(float, line.split(','))
            coordinates.append([x, y, z])

    coordinates = np.array(coordinates)

    return coordinates


def proce(names_list, ttip, pad_value=0.0, pad_to=None):
    """
    读取 Left / Right wrist 轨迹，合并为批次，并进行“按最大长度 padding + 显式掩码”。
    返回:
        all_results = [
            left_data,   # (N_left,  L_left, 3)
            left_mask,   # (N_left,  L_left)
            left_lengths,# (N_left,)
            right_data,  # (N_right, L_right,3)
            right_mask,  # (N_right, L_right)
            right_lengths# (N_right,)
        ]
    参数:
        pad_value: 填充值（默认 0.0；配合 mask 使用即可）
        pad_to:    手动指定目标长度（不指定则取该侧批次的最大长度）
    说明:
        - mask 中真实步长为 1.0，padding 为 0.0；
        - lengths 记录每条序列原始长度；
        - 若后续用到 RNN，可结合 pack_padded_sequence；用 Transformer/CNN 则在 loss/attention 里使用 mask。
    """
    all_results = []

    all_data_left = []
    all_data_right = []

    # 你已有的全局变量/函数：sides_list, data_file_name, read_coordinates
    # 假设 sides_list = ["Left", "Right"]

    for name in names_list:

        # 每个 tip 可能产出多个样本（每秒 1 个）
        for tip_num in ttip:
            side_points = {side: [] for side in sides_list}  # 'Left'/'Right' -> list of (T,3)

            for side in sides_list:

                wrist_file = fr"/home/fx2/project/Predict_hand/data/All_Data/{name}/{side}/Index/{tip_num}/{side}_Wrist.txt"
                # print(wrist_file)

                if os.path.exists(wrist_file):
                    arr = read_coordinates(wrist_file)
                    if isinstance(arr, np.ndarray) and arr.size > 0:
                        if arr.ndim == 1:
                            print("原始数据为 1D，尝试 reshape 为 (T,3)")
                            if arr.size % 3 == 0:
                                arr = arr.reshape(-1, 3)
                            else:
                                arr = None
                        if arr is not None and arr.ndim == 2 and arr.shape[1] == 3:
                            side_points[side].append(arr)
                else:
                    print("文件不存在")

            # 汇总每侧 wrist 点（把该 tip 下的多段拼接）
            if side_points.get('Left'):
                left_all = np.concatenate(side_points['Left'], axis=0)
                all_data_left.append(left_all)
            else:
                print(f"[Warning] {name} tip={tip_num} Left wrist data is missing (skip if no paired chunks).")

            if side_points.get('Right'):
                right_all = np.concatenate(side_points['Right'], axis=0)
                all_data_right.append(right_all)
            else:
                print(f"[Warning] {name} tip={tip_num} Right wrist data is missing (skip if no paired chunks).")

    def pad_with_mask(arr_list, pad_value=0.0, pad_to=None):
        """
        对列表中 (X,3) 数组按最大长度 padding，并返回 data/mask/lengths。
        data: (N, L, 3) ；mask: (N, L)，真实=1，填充=0；lengths: (N,)
        pad_to: 可手动指定目标长度；默认取该批次的最大 X。
        """
        if not arr_list:
            return None, None, None

        lengths = np.array([a.shape[0] for a in arr_list], dtype=int)
        L = int(pad_to if pad_to is not None else lengths.max())
        N = len(arr_list)

        data = np.full((N, L, 3), pad_value, dtype=float)
        mask = np.zeros((N, L), dtype=float)

        for i, a in enumerate(arr_list):
            n = a.shape[0]
            take = min(n, L)  # 以防 pad_to 比某些序列还短
            data[i, :take] = a[:take]
            mask[i, :take] = 1.0
        return data, mask, lengths

    # 分别对 Left / Right 做 padding + mask
    left_data, left_mask, left_lengths = pad_with_mask(all_data_left, pad_value=pad_value, pad_to=pad_to)
    right_data, right_mask, right_lengths = pad_with_mask(all_data_right, pad_value=pad_value, pad_to=pad_to)

    # 打印形状（为空时做保护）
    if left_data is not None:
        print("Left  data shape:", left_data.shape, "mask shape:", left_mask.shape)
    else:
        print("Left  data is empty.")

    if right_data is not None:
        print("Right data shape:", right_data.shape, "mask shape:", right_mask.shape)
    else:
        print("Right data is empty.")

    all_results.append(left_data)
    all_results.append(left_mask)
    all_results.append(left_lengths)
    all_results.append(right_data)
    all_results.append(right_mask)
    all_results.append(right_lengths)

    return all_results


# =============== 1) 变长滑窗 Dataset（保留每条轨迹边界） ===============
class SlidingWindowDataset(Dataset):
    def __init__(self, coords, seq_len, stride, lengths, dt,
                 require_acc_history: bool = False):
        """
        只用于“下一帧”监督：
          y = [pos_{t+1}, vel_{t+1}, acc_{t+1}]
        - coords: list[(Ti,3)] 或 np.ndarray (N,L,3)
        - lengths: 每条轨迹真实长度（只在真实区间内切窗）
        - seq_len/stride/dt: 窗口长度/步长/帧间隔
        - require_acc_history:
            True  -> 仅保留可计算加速度的窗口（至少 3 帧历史）
            False -> 历史不足 3 帧时，加速度用 0 填充
        """
        super().__init__()
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.dt = float(max(1e-12, dt))  # 防止除零
        self.require_acc_history = bool(require_acc_history)

        if isinstance(coords, np.ndarray) and coords.ndim == 3:
            self.trajs = [coords[i] for i in range(coords.shape[0])]
        elif isinstance(coords, list):
            self.trajs = coords
        else:
            raise ValueError("coords 必须是 (N,L,3) 的 ndarray 或 list[(Ti,3)]")

        if lengths is not None:
            self.eff_lengths = [int(l) for l in list(lengths)]
        else:
            self.eff_lengths = [t.shape[0] for t in self.trajs]

        # —— 索引构建：强制“有下一帧” —— #
        self.index = []  # (traj_id, start_idx, length)
        for tid, arr in enumerate(self.trajs):
            T_eff = int(self.eff_lengths[tid])
            T_eff = max(0, min(T_eff, arr.shape[0]))

            # 需要窗口 [s, ..., s+seq_len-1]，目标是下一帧 n = s+seq_len
            # 要求 n <= T_eff-1  ->  s <= T_eff - seq_len - 1
            upper = T_eff - self.seq_len - 1
            if upper < 0:
                continue

            # 若要求能算加速度，需要至少 3 帧历史：seq_len >= 3
            if self.require_acc_history and self.seq_len < 3:
                # 在索引阶段直接禁用这类配置
                raise ValueError("require_acc_history=True 时，seq_len 必须 >= 3")

            for s in range(0, upper + 1, self.stride):
                self.index.append((tid, s, self.seq_len))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        tid, s, length = self.index[idx]
        arr = np.asarray(self.trajs[tid], dtype=float)

        # 窗口与下一帧
        seq_np = np.asarray(arr[s:s + length], dtype=float)  # (L,3)
        n = s + length  # 下一帧索引（保证存在）

        dt = self.dt
        pos_t = arr[n - 1]  # x_t
        pos_next = arr[n]  # x_{t+1}
        vel_next = (pos_next - pos_t) / dt  # v_{t+1}

        # 加速度：需要 x_{t-1} 才能算 v_t
        if self.seq_len >= 2:
            v_t = (pos_t - arr[n - 2]) / dt
            acc_next = (vel_next - v_t) / dt
        else:
            # seq_len==1 时无法计算 a；仅当 require_acc_history=False 才允许 0 填充
            if self.require_acc_history:
                raise RuntimeError("索引构造已避免该情况；请检查 seq_len 与配置。")
            acc_next = np.zeros(3, float)

        y = np.concatenate([pos_next, vel_next, acc_next], axis=0).astype(np.float32)  # (9,)

        return (
            torch.from_numpy(seq_np).float(),  # (L,3)
            torch.tensor(length, dtype=torch.long),  # ()
            torch.from_numpy(y).float()  # (9,)
        )


# =============== 2) collate：pad 到同批最大长度，并返回 lengths ===============
def collate_pad_to_max(batch, pad_value=0.0):
    """
    batch: list of (seq:(Ti,3) tensor, length_i: long, y:(9,))
    returns:
      X: (B, L, 3)  padded 到本 batch 最大长度 L
      lengths: (B,)
      y: (B, 9)
    """
    seqs, lens, ys = zip(*batch)
    lens = torch.stack(lens, dim=0)  # (B,)
    L = int(lens.max().item())
    B = len(seqs)
    X = torch.full((B, L, 3), pad_value, dtype=torch.float32)
    for i, s in enumerate(seqs):
        n = s.size(0)
        X[i, :n] = s
    y = torch.stack(ys, dim=0)  # (B, 9)
    return X, lens, y


# =============== 3) cross_validate：使用变长 Dataset + collate + lengths ===============
def cross_validate(coords, seq_len, k_folds, batch_size, epochs, lr, device, lengthsss, dt,
                   stride=1, pad_value=0.0, grad_clip=None,
                   w_pos=1.0, w_vel=0.2, w_acc=0.1, w_consist=0.05):
    """
    说明：
      - 假定 SlidingWindowDataset 产出的 y 是“下一帧”的 9 维 [pos, vel, acc]
      - 兼容两类模型：
          * Hybrid: forward(X, lengths, dt) -> (pred(B,9), aux dict)
          * 直接9维: forward(X, lengths, dt) -> pred(B,9)  (dt 可被忽略)
      - 使用多任务损失 + 轻度一致性正则（若 aux 可用）
    """
    # 统一成列表以做 KFold（按“轨迹”为单位划分）
    if isinstance(coords, np.ndarray) and coords.ndim == 3:
        traj_list = [coords[i] for i in range(coords.shape[0])]
    elif isinstance(coords, list):
        traj_list = coords
    else:
        raise ValueError("coords 必须是 (N,X,3) 的 ndarray 或 list[(Ti,3)]")

    num_samples = len(traj_list)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_losses = []
    best_model = None
    best_val_loss = float("inf")

    mse = nn.MSELoss()

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(num_samples))):
        print(f"\n===== Fold {fold + 1}/{k_folds} =====")
        train_coords = [traj_list[i] for i in train_idx]
        val_coords = [traj_list[i] for i in val_idx]
        train_lengths = [lengthsss[i] for i in train_idx]
        val_lengths = [lengthsss[i] for i in val_idx]

        train_dataset = SlidingWindowDataset(train_coords, seq_len=seq_len, stride=stride,
                                             lengths=train_lengths, dt=dt)
        val_dataset = SlidingWindowDataset(val_coords, seq_len=seq_len, stride=stride,
                                           lengths=val_lengths, dt=dt)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda b: collate_pad_to_max(b, pad_value=pad_value))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=lambda b: collate_pad_to_max(b, pad_value=pad_value))

        # 每折新建模型（Hybrid 或你的直接9维模型都可以）
        model = MotionPredictorHybrid(
            input_dim=3,
            hidden_dim=128,
            num_layers=2,  # 双层
            bidirectional=True,  # 双向
            dropout=0.1,  # LSTM 内部dropout（仅num_layers>1时生效）
            learn_alpha=True,
            init_alpha=0.4
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # 我们希望验证损失越小越好
            factor=0.7,  # 没有改进就把 LR 乘以 0.5
            patience=20,  # 连续 5 个 epoch 没提升才降 LR
            verbose=True  # 控制台打印 LR 变化
            # 你也可以加 min_lr=1e-6 做下限
        )
        # 训练
        for epoch in range(epochs):
            model.train()
            total_loss, total_cnt = 0.0, 0
            # pbar = tqdm(train_loader, desc=f"Fold {fold + 1} Epoch {epoch + 1}/{epochs} [Train]")
            for X, lengths, y in train_loader:
                X = X.to(device)  # (B, L, 3)
                y = y.to(device)  # (B, 9)
                lengths = lengths.to(device)  # (B,)

                # —— forward：兼容 (pred, aux) 或 pred —— #
                out = model(X, lengths=lengths, dt=dt)
                if isinstance(out, tuple):
                    pred, aux = out
                else:
                    pred, aux = out, {}

                # —— 多任务损失 —— #
                loss_pos = mse(pred[:, :3], y[:, :3])
                loss_vel = mse(pred[:, 3:6], y[:, 3:6])
                loss_acc = mse(pred[:, 6:9], y[:, 6:9])
                loss = w_pos * loss_pos + w_vel * loss_vel + w_acc * loss_acc

                # —— 轻一致性（Hybrid 可用）—— #
                if "pos_direct" in aux and "pos_phys" in aux and w_consist > 0.0:
                    if epoch > 175:
                        loss_cons = mse(aux["pos_direct"], aux["pos_phys"])
                        loss = loss + w_consist * loss_cons

                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                bs = X.size(0)
                total_loss += loss.item() * bs
                total_cnt += bs
            #     pbar.set_postfix({
            #         "loss": f"{loss.item():.6f}",
            #         "pos": f"{loss_pos.item():.6f}",
            #         "vel": f"{loss_vel.item():.6f}",
            #         "acc": f"{loss_acc.item():.6f}"
            #     })

            avg_train_loss = total_loss / max(1, total_cnt)

            # 验证
            model.eval()
            val_loss, val_cnt = 0.0, 0
            # pbar_val = tqdm(val_loader, desc=f"Fold {fold + 1} Epoch {epoch + 1}/{epochs} [Val]")
            with torch.no_grad():
                for X, lengths, y in val_loader:
                    X = X.to(device)
                    y = y.to(device)
                    lengths = lengths.to(device)

                    out = model(X, lengths=lengths, dt=dt)
                    if isinstance(out, tuple):
                        pred, aux = out
                    else:
                        pred, aux = out, {}

                    lp = mse(pred[:, :3], y[:, :3])
                    lv = mse(pred[:, 3:6], y[:, 3:6])
                    la = mse(pred[:, 6:9], y[:, 6:9])
                    l = w_pos * lp + w_vel * lv + w_acc * la
                    # 验证阶段一般不加一致性也行；若想看其数值可同样加入
                    if "pos_direct" in aux and "pos_phys" in aux and w_consist > 0.0:
                        lc = mse(aux["pos_direct"], aux["pos_phys"])
                        l = l + w_consist * lc

                    bs = X.size(0)
                    val_loss += l.item() * bs
                    val_cnt += bs
            #         pbar_val.set_postfix({"loss": f"{l.item():.6f}"})

            avg_val_loss = val_loss / max(1, val_cnt)
            # print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.9f}, Val Loss: {avg_val_loss:.9f}")

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.9f}, Val Loss: {avg_val_loss:.9f}, LR: {current_lr:.6g}")
            scheduler.step(avg_val_loss)

        fold_losses.append(avg_val_loss)

        # 记录最佳
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model
            print(f"==> 更新最佳模型 (Fold {fold + 1}, Val Loss {best_val_loss:.10f})")

    print("\n===== Cross-validation results =====")
    for i, loss in enumerate(fold_losses):
        print(f"Fold {i + 1}: Val Loss = {loss:.9f}")
    print(f"Mean Val Loss: {np.mean(fold_losses):.9f}")

    return best_model, fold_losses


# def load_model(model_path, device="cpu"):
#     model = MotionPredictorHybrid().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     return model


# def predict_next(model, input_seq, device="cpu", dt=0.01):
#     """
#     input_seq: numpy 数组 (seq_len, 3)，最近几帧坐标
#     输出: 下一帧预测值 [x,y,z] (长度为 3 的数组)
#     """
#     x = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 3)
#     lengths = torch.tensor([x.size(1)], dtype=torch.long).to(device)  # (1,) 长度信息
#
#     with torch.no_grad():
#         out = model(x, lengths=lengths, dt=dt)
#         if isinstance(out, tuple):
#             pred, aux = out
#         else:
#             pred = out
#         pred = pred.cpu().numpy()  # (1,9)
#
#     pos_next = pred[0, :3]  # 取前三维作为下一帧位置
#     return pos_next


@torch.no_grad()
def evaluate_model_on_windows(
        model: torch.nn.Module,
        test_coords,  # list[(Ti,3)] 或 np.ndarray(N, L, 3)
        test_lengths,  # list[int] or np.ndarray(N,)
        dt: float,
        seq_len: int,
        batch_size: int,
        device: str = "cpu",
        stride: int = 1,
        pad_value: float = 0.0,
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    在测试集上评估 (只评估“下一帧”的位置三维误差)。
    依赖:
      - SlidingWindowDataset（应产出下一帧 9 维标签）
      - collate_pad_to_max
      - 模型 forward 支持 (x, lengths, dt)；Hybrid 返回 (pred, aux)
    """
    # 1) 构建 Dataset / DataLoader
    test_dataset = SlidingWindowDataset(
        coords=test_coords, seq_len=seq_len, stride=stride, lengths=test_lengths, dt=dt
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_pad_to_max(b, pad_value=pad_value)
    )

    model = model.to(device)
    model.eval()

    # 2) 累积误差（仅位置）
    abs_errs_all = []  # (M,3)
    l2_errs_all = []  # (M,)
    preds_all = []  # (M,3)
    trues_all = []  # (M,3)

    pbar = tqdm(test_loader, desc="[Test] evaluating", disable=not verbose)
    for X, lengths, y in pbar:
        X = X.to(device)  # (B, L, 3)
        y = y.to(device)  # (B, 9)  目标是下一帧 [pos, vel, acc]
        lengths = lengths.to(device)

        # ---- 兼容 Hybrid / 非 Hybrid，并把 dt 传入 ----
        out = model(X, lengths=lengths, dt=dt)
        if isinstance(out, tuple):
            pred, _aux = out  # pred: (B, 9)
        else:
            pred = out  # (B, 9)

        pred_pos = pred[:, :3]  # 只评估位置
        true_pos = y[:, :3]

        err_vec = pred_pos - true_pos  # (B, 3)
        abs_err = torch.abs(err_vec)  # (B, 3)
        l2_err = torch.linalg.norm(err_vec, dim=1)  # (B,)

        abs_errs_all.append(abs_err.cpu().numpy())
        l2_errs_all.append(l2_err.cpu().numpy())
        preds_all.append(pred_pos.detach().cpu().numpy())
        trues_all.append(true_pos.detach().cpu().numpy())

    if len(abs_errs_all) == 0:
        return {"message": "No test samples were generated. Check lengths/seq_len/stride."}

    abs_errs_all = np.concatenate(abs_errs_all, axis=0)  # (M,3)
    l2_errs_all = np.concatenate(l2_errs_all, axis=0)  # (M,)
    preds_all = np.concatenate(preds_all, axis=0)  # (M,3)
    trues_all = np.concatenate(trues_all, axis=0)  # (M,3)

    # 3) 统计指标
    def _safe_rmse(x):
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")

    mae_xyz = abs_errs_all.mean(axis=0)  # (3,)
    rmse_xyz = np.sqrt((abs_errs_all ** 2).mean(axis=0))  # (3,)

    mae_l2 = float(np.mean(l2_errs_all))
    rmse_l2 = _safe_rmse(l2_errs_all)
    max_l2 = float(np.max(l2_errs_all))
    p90_l2 = float(np.percentile(l2_errs_all, 90))
    p95_l2 = float(np.percentile(l2_errs_all, 95))
    median_l2 = float(np.median(l2_errs_all))

    summary = {
        "count": int(l2_errs_all.size),
        "per_axis": {
            "MAE": {"x": float(mae_xyz[0]), "y": float(mae_xyz[1]), "z": float(mae_xyz[2])},
            "RMSE": {"x": float(rmse_xyz[0]), "y": float(rmse_xyz[1]), "z": float(rmse_xyz[2])},
        },
        "vector_error": {
            "MAE_L2": mae_l2,
            "RMSE_L2": rmse_l2,
            "median_L2": median_l2,
            "P90_L2": p90_l2,
            "P95_L2": p95_l2,
            "max_L2": max_l2,
        },
        "pred_samples": preds_all[:10].tolist(),
        "true_samples": trues_all[:10].tolist(),
    }

    if verbose:
        print("\n=== Test Summary (next-frame position only) ===")
        print(f"Samples: {summary['count']}")
        print(f"Per-axis MAE:  x={summary['per_axis']['MAE']['x']:.6f}, "
              f"y={summary['per_axis']['MAE']['y']:.6f}, z={summary['per_axis']['MAE']['z']:.6f}")
        print(f"Per-axis RMSE: x={summary['per_axis']['RMSE']['x']:.6f}, "
              f"y={summary['per_axis']['RMSE']['y']:.6f}, z={summary['per_axis']['RMSE']['z']:.6f}")
        print(f"L2  MAE:   {summary['vector_error']['MAE_L2']:.6f}")
        print(f"L2  RMSE:  {summary['vector_error']['RMSE_L2']:.6f}")
        print(f"L2  median:{summary['vector_error']['median_L2']:.6f} | "
              f"P90:{summary['vector_error']['P90_L2']:.6f} | "
              f"P95:{summary['vector_error']['P95_L2']:.6f} | "
              f"max:{summary['vector_error']['max_L2']:.6f}")

    return summary


# TODO 参数设置
names_1 = []
name_all_1 = []
sample_rate = -1

# setting = "offline_OO"
setting = "offline_O"
# setting = "offline_H"
# setting = "offline_V"
# setting = "offline_ALL"

# 加噪前的数据
# data_file_name = "AllData"
data_file_name = "All_Data"

if setting == "offline_OO":
    sample_rate = 60.0

    names_1 = [
        "CYS", "FWQ", "FXZ", "HHY", "JJB",
        "LDY", "LGH", "LJL", "MKQ", "WJY",
        "WJZ", "WXY", "XZC", "YHP", "YHX",
        "ZH", "ZJW"
    ]

    name_all_1 = [
        "OO_CYS1", "OO_CYS2", "OO_CYS3", "OO_CYS4", "OO_CYS5", "OO_CYS6", "OO_CYS7", "OO_CYS8", "OO_CYS9", "OO_CYS10",
        "OO_FWQ1", "OO_FWQ2", "OO_FWQ3", "OO_FWQ4", "OO_FWQ5", "OO_FWQ6", "OO_FWQ7", "OO_FWQ8", "OO_FWQ9", "OO_FWQ10",
        "OO_FXZ1", "OO_FXZ2", "OO_FXZ3", "OO_FXZ4", "OO_FXZ5", "OO_FXZ6", "OO_FXZ7", "OO_FXZ8", "OO_FXZ9", "OO_FXZ10",
        "OO_HHY1", "OO_HHY2", "OO_HHY3", "OO_HHY4", "OO_HHY5", "OO_HHY6", "OO_HHY7", "OO_HHY8", "OO_HHY9", "OO_HHY10",
        "OO_JJB1", "OO_JJB2", "OO_JJB3", "OO_JJB4", "OO_JJB5", "OO_JJB6", "OO_JJB7", "OO_JJB8", "OO_JJB9", "OO_JJB10",
        "OO_LDY1", "OO_LDY2", "OO_LDY3", "OO_LDY4", "OO_LDY5", "OO_LDY6", "OO_LDY7", "OO_LDY8", "OO_LDY9", "OO_LDY10",
        "OO_LGH1", "OO_LGH2", "OO_LGH3", "OO_LGH4", "OO_LGH5", "OO_LGH6", "OO_LGH7", "OO_LGH8", "OO_LGH9", "OO_LGH10",
        "OO_LJL1", "OO_LJL2", "OO_LJL3", "OO_LJL4", "OO_LJL5", "OO_LJL6", "OO_LJL7", "OO_LJL8", "OO_LJL9", "OO_LJL10",
        "OO_MKQ1", "OO_MKQ2", "OO_MKQ3", "OO_MKQ4", "OO_MKQ5", "OO_MKQ6", "OO_MKQ7", "OO_MKQ8", "OO_MKQ9", "OO_MKQ10",
        "OO_WJY1", "OO_WJY2", "OO_WJY3", "OO_WJY4", "OO_WJY5", "OO_WJY6", "OO_WJY7", "OO_WJY8", "OO_WJY9", "OO_WJY10",
        "OO_WJZ1", "OO_WJZ2", "OO_WJZ3", "OO_WJZ4", "OO_WJZ5", "OO_WJZ6", "OO_WJZ7", "OO_WJZ8", "OO_WJZ9", "OO_WJZ10",
        "OO_WXY1", "OO_WXY2", "OO_WXY3", "OO_WXY4", "OO_WXY5", "OO_WXY6", "OO_WXY7", "OO_WXY8", "OO_WXY9", "OO_WXY10",
        "OO_XZC1", "OO_XZC2", "OO_XZC3", "OO_XZC4", "OO_XZC5", "OO_XZC6", "OO_XZC7", "OO_XZC8", "OO_XZC9", "OO_XZC10",
        "OO_YHP1", "OO_YHP2", "OO_YHP3", "OO_YHP4", "OO_YHP5", "OO_YHP6", "OO_YHP7", "OO_YHP8", "OO_YHP9", "OO_YHP10",
        "OO_YHX1", "OO_YHX2", "OO_YHX3", "OO_YHX4", "OO_YHX5", "OO_YHX6", "OO_YHX7", "OO_YHX8", "OO_YHX9", "OO_YHX10",
        "OO_ZH1", "OO_ZH2", "OO_ZH3", "OO_ZH4", "OO_ZH5", "OO_ZH6", "OO_ZH7", "OO_ZH8", "OO_ZH9", "OO_ZH10",
        "OO_ZJW1", "OO_ZJW2", "OO_ZJW3", "OO_ZJW4", "OO_ZJW5", "OO_ZJW6", "OO_ZJW7", "OO_ZJW8", "OO_ZJW9", "OO_ZJW10"
    ]

if setting == "offline_H":
    sample_rate = 60.0

    names_1 = [
        "FR", "GHY", "JJB", "LDE", "MKQ",
        "MYL", "NBW", "WJY", "YHP", "YRY",
        "ZH", "ZZH", "ZZR"
    ]

    name_all_1 = [
        "H_FR1", "H_FR2", "H_FR3",
        "H_GHY1", "H_GHY2", "H_GHY3",
        "H_JJB1", "H_JJB2", "H_JJB3",
        "H_LDE1", "H_LDE2", "H_LDE3",
        "H_MKQ1", "H_MKQ2", "H_MKQ3",
        "H_MYL1", "H_MYL2", "H_MYL3",
        "H_NBW1", "H_NBW2", "H_NBW3",
        "H_WJY1", "H_WJY2", "H_WJY3",
        "H_YHP1", "H_YHP2", "H_YHP3",
        "H_YRY1", "H_YRY2", "H_YRY3",
        "H_ZH1", "H_ZH2", "H_ZH3",
        "H_ZZH1", "H_ZZH2", "H_ZZH3",
        "H_ZZR1", "H_ZZR2", "H_ZZR3"
    ]

if setting == "offline_O":
    sample_rate = 60.0

    names_1 = [
        "CYS", "FWQ", "FXZ", "GDK", "HHY",
        "JJB", "LDY", "LGH", "LJL", "MKQ",
        "SWJ", "WJZ", "WXY", "XXY", "XZC",
        "YHP", "YHX", "YRY", "ZH", "ZJW",
        "ZZR"
    ]

    name_all_1 = [
        "O_CYS1", "O_CYS2", "O_CYS3", "O_CYS4", "O_CYS5", "O_CYS6", "O_CYS7", "O_CYS8", "O_CYS9", "O_CYS10",
        "O_FWQ1", "O_FWQ2", "O_FWQ3", "O_FWQ4", "O_FWQ5", "O_FWQ6", "O_FWQ7", "O_FWQ8", "O_FWQ9", "O_FWQ10",
        "O_FXZ1", "O_FXZ2", "O_FXZ3", "O_FXZ4", "O_FXZ5", "O_FXZ6", "O_FXZ7", "O_FXZ8", "O_FXZ9", "O_FXZ10",
        "O_FXZ11", "O_FXZ12", "O_FXZ13", "O_FXZ14", "O_FXZ15", "O_FXZ16", "O_FXZ17", "O_FXZ18", "O_FXZ19", "O_FXZ20",
        "O_FXZ21",
        "O_GDK1", "O_GDK2", "O_GDK3", "O_GDK4", "O_GDK5", "O_GDK6", "O_GDK7", "O_GDK8", "O_GDK9", "O_GDK10", "O_GDK11",
        "O_HHY1", "O_HHY2", "O_HHY3", "O_HHY4", "O_HHY5", "O_HHY6", "O_HHY7", "O_HHY8", "O_HHY9", "O_HHY10", "O_HHY11",
        "O_JJB1", "O_JJB2", "O_JJB3", "O_JJB4", "O_JJB5", "O_JJB6", "O_JJB7", "O_JJB8", "O_JJB9", "O_JJB10",
        "O_JJB11", "O_JJB12", "O_JJB13", "O_JJB14", "O_JJB15", "O_JJB16", "O_JJB17", "O_JJB18", "O_JJB19", "O_JJB20",
        "O_JJB21",
        "O_LDY1", "O_LDY2", "O_LDY3", "O_LDY4", "O_LDY5", "O_LDY6", "O_LDY7", "O_LDY8", "O_LDY9", "O_LDY10",
        "O_LDY11", "O_LDY12", "O_LDY13", "O_LDY14", "O_LDY15", "O_LDY16", "O_LDY17", "O_LDY18", "O_LDY19", "O_LDY20",
        "O_LGH1", "O_LGH2", "O_LGH3", "O_LGH4", "O_LGH5", "O_LGH6", "O_LGH7", "O_LGH8", "O_LGH9", "O_LGH10",
        "O_LGH11", "O_LGH12", "O_LGH13", "O_LGH14", "O_LGH15", "O_LGH16", "O_LGH17", "O_LGH18", "O_LGH19", "O_LGH20",
        "O_LGH21",
        "O_LJL1", "O_LJL2", "O_LJL3", "O_LJL4", "O_LJL5", "O_LJL6", "O_LJL7", "O_LJL8", "O_LJL9", "O_LJL10",
        "O_LJL11", "O_LJL12", "O_LJL13", "O_LJL14", "O_LJL15", "O_LJL16", "O_LJL17", "O_LJL18", "O_LJL19", "O_LJL20",
        "O_LJL21",
        "O_MKQ1", "O_MKQ2", "O_MKQ3", "O_MKQ4", "O_MKQ5", "O_MKQ6", "O_MKQ7", "O_MKQ8", "O_MKQ9", "O_MKQ10",
        "O_MKQ11", "O_MKQ12", "O_MKQ13", "O_MKQ14", "O_MKQ15", "O_MKQ16", "O_MKQ17", "O_MKQ18", "O_MKQ19", "O_MKQ20",
        "O_MKQ21",
        "O_SWJ1", "O_SWJ2", "O_SWJ3", "O_SWJ4", "O_SWJ5", "O_SWJ6", "O_SWJ7", "O_SWJ8", "O_SWJ9", "O_SWJ10",
        "O_SWJ11", "O_SWJ12", "O_SWJ13", "O_SWJ14", "O_SWJ15", "O_SWJ16", "O_SWJ17", "O_SWJ18", "O_SWJ19", "O_SWJ20",
        "O_SWJ21",
        "O_WJZ1", "O_WJZ2", "O_WJZ3", "O_WJZ4", "O_WJZ5", "O_WJZ6", "O_WJZ7", "O_WJZ8", "O_WJZ9", "O_WJZ10",
        "O_WJZ11", "O_WJZ12", "O_WJZ13", "O_WJZ14", "O_WJZ15", "O_WJZ16", "O_WJZ17", "O_WJZ18", "O_WJZ19", "O_WJZ20",
        "O_WXY1", "O_WXY2", "O_WXY3", "O_WXY4", "O_WXY5", "O_WXY6", "O_WXY7", "O_WXY8", "O_WXY9", "O_WXY10", "O_WXY11",
        "O_XXY1", "O_XXY2", "O_XXY3", "O_XXY4", "O_XXY5", "O_XXY6", "O_XXY7", "O_XXY8", "O_XXY9", "O_XXY10",
        "O_XZC1", "O_XZC2", "O_XZC3", "O_XZC4", "O_XZC5", "O_XZC6", "O_XZC7", "O_XZC8", "O_XZC9", "O_XZC10",
        "O_YHP1", "O_YHP2", "O_YHP3", "O_YHP4", "O_YHP5", "O_YHP6", "O_YHP7", "O_YHP8", "O_YHP9", "O_YHP10",
        "O_YHP11", "O_YHP12", "O_YHP13", "O_YHP14", "O_YHP15", "O_YHP16", "O_YHP17", "O_YHP18", "O_YHP19", "O_YHP20",
        "O_YHP21",
        "O_YHX1", "O_YHX2", "O_YHX3", "O_YHX4", "O_YHX5", "O_YHX6", "O_YHX7", "O_YHX8", "O_YHX9", "O_YHX10",
        "O_YRY1", "O_YRY2", "O_YRY3", "O_YRY4", "O_YRY5", "O_YRY6", "O_YRY7", "O_YRY8", "O_YRY9", "O_YRY10",
        "O_ZH1", "O_ZH2", "O_ZH3", "O_ZH4", "O_ZH5", "O_ZH6", "O_ZH7", "O_ZH8", "O_ZH9", "O_ZH10",
        "O_ZH11", "O_ZH12", "O_ZH13", "O_ZH14", "O_ZH15", "O_ZH16", "O_ZH17", "O_ZH18", "O_ZH19", "O_ZH20",
        "O_ZJW1", "O_ZJW2", "O_ZJW3", "O_ZJW4", "O_ZJW5", "O_ZJW6", "O_ZJW7", "O_ZJW8", "O_ZJW9", "O_ZJW10",
        "O_ZJW11", "O_ZJW12", "O_ZJW13", "O_ZJW14", "O_ZJW15", "O_ZJW16", "O_ZJW17", "O_ZJW18", "O_ZJW19",
        "O_ZZR1", "O_ZZR2", "O_ZZR3", "O_ZZR4", "O_ZZR5", "O_ZZR6", "O_ZZR7", "O_ZZR8", "O_ZZR9", "O_ZZR10"
    ]

if setting == "offline_V":
    sample_rate = 40.0

    names_1 = [
        "CYS", "FWQ", "HHY", "JJB", "LDY",
        "LGH", "LRQ", "LSY", "MKQ", "SWJ",
        "WA", "WJY", "WJZ", "WXY", "XZC",
        "YHP", "YHX", "YRY", "ZH", "ZJW"
    ]

    name_all_1 = [
        "V_CYS1", "V_CYS2", "V_CYS3", "V_CYS4", "V_CYS5", "V_CYS6", "V_CYS7", "V_CYS8", "V_CYS9", "V_CYS10",
        "V_FWQ1", "V_FWQ2", "V_FWQ3", "V_FWQ4", "V_FWQ5", "V_FWQ7", "V_FWQ8", "V_FWQ9", "V_FWQ10",
        "V_HHY1", "V_HHY2", "V_HHY3", "V_HHY4", "V_HHY5", "V_HHY6", "V_HHY7", "V_HHY8", "V_HHY9", "V_HHY10",
        "V_JJB1", "V_JJB2", "V_JJB3", "V_JJB4", "V_JJB5", "V_JJB6", "V_JJB7", "V_JJB8", "V_JJB9", "V_JJB10",
        "V_LDY1", "V_LDY2", "V_LDY3", "V_LDY4", "V_LDY5", "V_LDY6", "V_LDY7", "V_LDY8", "V_LDY9", "V_LDY10",
        "V_LGH1", "V_LGH2", "V_LGH3", "V_LGH4", "V_LGH5", "V_LGH6", "V_LGH7", "V_LGH8", "V_LGH9", "V_LGH10",
        "V_LRQ1", "V_LRQ2", "V_LRQ3", "V_LRQ4", "V_LRQ5", "V_LRQ6", "V_LRQ7", "V_LRQ8", "V_LRQ9", "V_LRQ10",
        "V_LSY1", "V_LSY2", "V_LSY3", "V_LSY4", "V_LSY5", "V_LSY6", "V_LSY7", "V_LSY8", "V_LSY9", "V_LSY10",
        "V_MKQ1", "V_MKQ2", "V_MKQ3", "V_MKQ4", "V_MKQ5", "V_MKQ6", "V_MKQ7", "V_MKQ8", "V_MKQ9", "V_MKQ10",
        "V_SWJ1", "V_SWJ2", "V_SWJ3", "V_SWJ4", "V_SWJ5", "V_SWJ6", "V_SWJ7", "V_SWJ8", "V_SWJ9", "V_SWJ10",
        "V_WA1", "V_WA2", "V_WA3", "V_WA4", "V_WA5", "V_WA6", "V_WA7", "V_WA8", "V_WA9", "V_WA10",
        "V_WJY1", "V_WJY2", "V_WJY3", "V_WJY4", "V_WJY5", "V_WJY6", "V_WJY7", "V_WJY8", "V_WJY9", "V_WJY10",
        "V_WJZ1", "V_WJZ2", "V_WJZ3", "V_WJZ4", "V_WJZ5", "V_WJZ6", "V_WJZ7", "V_WJZ8", "V_WJZ9", "V_WJZ10",
        "V_WXY1", "V_WXY2", "V_WXY3", "V_WXY4", "V_WXY5", "V_WXY6", "V_WXY7", "V_WXY8", "V_WXY9", "V_WXY10",
        "V_XZC1", "V_XZC2", "V_XZC3", "V_XZC4", "V_XZC5", "V_XZC6", "V_XZC7", "V_XZC8", "V_XZC9", "V_XZC10",
        "V_YHP1", "V_YHP2", "V_YHP3", "V_YHP4", "V_YHP5", "V_YHP6", "V_YHP7", "V_YHP8", "V_YHP9", "V_YHP10",
        "V_YHX1", "V_YHX2", "V_YHX3", "V_YHX4", "V_YHX5", "V_YHX6", "V_YHX7", "V_YHX8", "V_YHX9", "V_YHX10",
        "V_YRY1", "V_YRY2", "V_YRY3", "V_YRY4", "V_YRY5", "V_YRY6", "V_YRY7", "V_YRY8", "V_YRY9", "V_YRY10",
        "V_ZH1", "V_ZH2", "V_ZH3", "V_ZH4", "V_ZH5", "V_ZH6", "V_ZH7", "V_ZH8", "V_ZH9", "V_ZH10",
        "V_ZJW1", "V_ZJW2", "V_ZJW3", "V_ZJW4", "V_ZJW5", "V_ZJW6", "V_ZJW7", "V_ZJW8", "V_ZJW9", "V_ZJW10"
    ]

if setting == "offline_ALL":
    sample_rate = 60.0
    names_1 = [
        'CYS', 'FR', 'FWQ', 'FXZ', 'GDK', 'GHY', 'HHY',
        'JJB', 'LDE', 'LDY', 'LGH', 'LJL', 'LRQ', 'LSY',
        'MKQ', 'MYL', 'NBW', 'SWJ', 'WA', 'WJY', 'WJZ',
        'WXY', 'XXY', 'XZC', 'YHP', 'YHX', 'YRY', 'ZH',
        'ZJW', 'ZZH', 'ZZR'
    ]

    name_all_1 = [
        "H_FR1", "H_FR2", "H_FR3",
        "H_GHY1", "H_GHY2", "H_GHY3",
        "H_JJB1", "H_JJB2", "H_JJB3",
        "H_LDE1", "H_LDE2", "H_LDE3",
        "H_MKQ1", "H_MKQ2", "H_MKQ3",
        "H_MYL1", "H_MYL2", "H_MYL3",
        "H_NBW1", "H_NBW2", "H_NBW3",
        "H_WJY1", "H_WJY2", "H_WJY3",
        "H_YHP1", "H_YHP2", "H_YHP3",
        "H_YRY1", "H_YRY2", "H_YRY3",
        "H_ZH1", "H_ZH2", "H_ZH3",
        "H_ZZH1", "H_ZZH2", "H_ZZH3",
        "H_ZZR1", "H_ZZR2", "H_ZZR3",

        "OO_CYS1", "OO_CYS2", "OO_CYS3", "OO_CYS4", "OO_CYS5", "OO_CYS6", "OO_CYS7", "OO_CYS8",
        "OO_CYS9", "OO_CYS10",
        "OO_FWQ1", "OO_FWQ2", "OO_FWQ3", "OO_FWQ4", "OO_FWQ5", "OO_FWQ6", "OO_FWQ7", "OO_FWQ8", "OO_FWQ9", "OO_FWQ10",
        "OO_FXZ1", "OO_FXZ2", "OO_FXZ3", "OO_FXZ4", "OO_FXZ5", "OO_FXZ6", "OO_FXZ7", "OO_FXZ8", "OO_FXZ9", "OO_FXZ10",
        "OO_HHY1", "OO_HHY2", "OO_HHY3", "OO_HHY4", "OO_HHY5", "OO_HHY6", "OO_HHY7", "OO_HHY8", "OO_HHY9", "OO_HHY10",
        "OO_JJB1", "OO_JJB2", "OO_JJB3", "OO_JJB4", "OO_JJB5", "OO_JJB6", "OO_JJB7", "OO_JJB8", "OO_JJB9", "OO_JJB10",
        "OO_LDY1", "OO_LDY2", "OO_LDY3", "OO_LDY4", "OO_LDY5", "OO_LDY6", "OO_LDY7", "OO_LDY8", "OO_LDY9", "OO_LDY10",
        "OO_LGH1", "OO_LGH2", "OO_LGH3", "OO_LGH4", "OO_LGH5", "OO_LGH6", "OO_LGH7", "OO_LGH8", "OO_LGH9", "OO_LGH10",
        "OO_LJL1", "OO_LJL2", "OO_LJL3", "OO_LJL4", "OO_LJL5", "OO_LJL6", "OO_LJL7", "OO_LJL8", "OO_LJL9", "OO_LJL10",
        "OO_MKQ1", "OO_MKQ2", "OO_MKQ3", "OO_MKQ4", "OO_MKQ5", "OO_MKQ6", "OO_MKQ7", "OO_MKQ8", "OO_MKQ9", "OO_MKQ10",
        "OO_WJY1", "OO_WJY2", "OO_WJY3", "OO_WJY4", "OO_WJY5", "OO_WJY6", "OO_WJY7", "OO_WJY8", "OO_WJY9", "OO_WJY10",
        "OO_WJZ1", "OO_WJZ2", "OO_WJZ3", "OO_WJZ4", "OO_WJZ5", "OO_WJZ6", "OO_WJZ7", "OO_WJZ8", "OO_WJZ9", "OO_WJZ10",
        "OO_WXY1", "OO_WXY2", "OO_WXY3", "OO_WXY4", "OO_WXY5", "OO_WXY6", "OO_WXY7", "OO_WXY8", "OO_WXY9", "OO_WXY10",
        "OO_XZC1", "OO_XZC2", "OO_XZC3", "OO_XZC4", "OO_XZC5", "OO_XZC6", "OO_XZC7", "OO_XZC8", "OO_XZC9", "OO_XZC10",
        "OO_YHP1", "OO_YHP2", "OO_YHP3", "OO_YHP4", "OO_YHP5", "OO_YHP6", "OO_YHP7", "OO_YHP8", "OO_YHP9", "OO_YHP10",
        "OO_YHX1", "OO_YHX2", "OO_YHX3", "OO_YHX4", "OO_YHX5", "OO_YHX6", "OO_YHX7", "OO_YHX8", "OO_YHX9", "OO_YHX10",
        "OO_ZH1", "OO_ZH2", "OO_ZH3", "OO_ZH4", "OO_ZH5", "OO_ZH6", "OO_ZH7", "OO_ZH8", "OO_ZH9", "OO_ZH10",
        "OO_ZJW1", "OO_ZJW2", "OO_ZJW3", "OO_ZJW4", "OO_ZJW5", "OO_ZJW6", "OO_ZJW7", "OO_ZJW8", "OO_ZJW9", "OO_ZJW10",

        # "O_CYS1",
        # "O_CYS2", "O_CYS3", "O_CYS4", "O_CYS5", "O_CYS6", "O_CYS7", "O_CYS8", "O_CYS9", "O_CYS10",
        # "O_FWQ1", "O_FWQ2", "O_FWQ3", "O_FWQ4", "O_FWQ5", "O_FWQ6", "O_FWQ7", "O_FWQ8", "O_FWQ9", "O_FWQ10",
        # "O_FXZ1", "O_FXZ2", "O_FXZ3", "O_FXZ4", "O_FXZ5", "O_FXZ6", "O_FXZ7", "O_FXZ8", "O_FXZ9", "O_FXZ10",
        # "O_FXZ11", "O_FXZ12", "O_FXZ13", "O_FXZ14", "O_FXZ15", "O_FXZ16", "O_FXZ17", "O_FXZ18", "O_FXZ19", "O_FXZ20",
        # "O_FXZ21",
        # "O_GDK1", "O_GDK2", "O_GDK3", "O_GDK4", "O_GDK5", "O_GDK6", "O_GDK7", "O_GDK8", "O_GDK9", "O_GDK10", "O_GDK11",
        # "O_HHY1", "O_HHY2", "O_HHY3", "O_HHY4", "O_HHY5", "O_HHY6", "O_HHY7", "O_HHY8", "O_HHY9", "O_HHY10", "O_HHY11",
        # "O_JJB1", "O_JJB2", "O_JJB3", "O_JJB4", "O_JJB5", "O_JJB6", "O_JJB7", "O_JJB8", "O_JJB9", "O_JJB10",
        # "O_JJB11", "O_JJB12", "O_JJB13", "O_JJB14", "O_JJB15", "O_JJB16", "O_JJB17", "O_JJB18", "O_JJB19", "O_JJB20",
        # "O_JJB21",
        # "O_LDY1", "O_LDY2", "O_LDY3", "O_LDY4", "O_LDY5", "O_LDY6", "O_LDY7", "O_LDY8", "O_LDY9", "O_LDY10",
        # "O_LDY11", "O_LDY12", "O_LDY13", "O_LDY14", "O_LDY15", "O_LDY16", "O_LDY17", "O_LDY18", "O_LDY19", "O_LDY20",
        # "O_LGH1", "O_LGH2", "O_LGH3", "O_LGH4", "O_LGH5", "O_LGH6", "O_LGH7", "O_LGH8", "O_LGH9", "O_LGH10",
        # "O_LGH11", "O_LGH12", "O_LGH13", "O_LGH14", "O_LGH15", "O_LGH16", "O_LGH17", "O_LGH18", "O_LGH19", "O_LGH20",
        # "O_LGH21",
        # "O_LJL1", "O_LJL2", "O_LJL3", "O_LJL4", "O_LJL5", "O_LJL6", "O_LJL7", "O_LJL8", "O_LJL9", "O_LJL10",
        # "O_LJL11", "O_LJL12", "O_LJL13", "O_LJL14", "O_LJL15", "O_LJL16", "O_LJL17", "O_LJL18", "O_LJL19", "O_LJL20",
        # "O_LJL21",
        # "O_MKQ1", "O_MKQ2", "O_MKQ3", "O_MKQ4", "O_MKQ5", "O_MKQ6", "O_MKQ7", "O_MKQ8", "O_MKQ9", "O_MKQ10",
        # "O_MKQ11", "O_MKQ12", "O_MKQ13", "O_MKQ14", "O_MKQ15", "O_MKQ16", "O_MKQ17", "O_MKQ18", "O_MKQ19", "O_MKQ20",
        # "O_MKQ21",
        # "O_SWJ1", "O_SWJ2", "O_SWJ3", "O_SWJ4", "O_SWJ5", "O_SWJ6", "O_SWJ7", "O_SWJ8", "O_SWJ9", "O_SWJ10",
        # "O_SWJ11", "O_SWJ12", "O_SWJ13", "O_SWJ14", "O_SWJ15", "O_SWJ16", "O_SWJ17", "O_SWJ18", "O_SWJ19", "O_SWJ20",
        # "O_SWJ21",
        # "O_WJZ1", "O_WJZ2", "O_WJZ3", "O_WJZ4", "O_WJZ5", "O_WJZ6", "O_WJZ7", "O_WJZ8", "O_WJZ9", "O_WJZ10",
        # "O_WJZ11", "O_WJZ12", "O_WJZ13", "O_WJZ14", "O_WJZ15", "O_WJZ16", "O_WJZ17", "O_WJZ18", "O_WJZ19", "O_WJZ20",
        # "O_WXY1", "O_WXY2", "O_WXY3", "O_WXY4", "O_WXY5", "O_WXY6", "O_WXY7", "O_WXY8", "O_WXY9", "O_WXY10", "O_WXY11",
        # "O_XXY1", "O_XXY2", "O_XXY3", "O_XXY4", "O_XXY5", "O_XXY6", "O_XXY7", "O_XXY8", "O_XXY9", "O_XXY10",
        # "O_XZC1", "O_XZC2", "O_XZC3", "O_XZC4", "O_XZC5", "O_XZC6", "O_XZC7", "O_XZC8", "O_XZC9", "O_XZC10",
        # "O_YHP1", "O_YHP2", "O_YHP3", "O_YHP4", "O_YHP5", "O_YHP6", "O_YHP7", "O_YHP8", "O_YHP9", "O_YHP10",
        # "O_YHP11", "O_YHP12", "O_YHP13", "O_YHP14", "O_YHP15", "O_YHP16", "O_YHP17", "O_YHP18", "O_YHP19", "O_YHP20",
        # "O_YHP21",
        # "O_YHX1", "O_YHX2", "O_YHX3", "O_YHX4", "O_YHX5", "O_YHX6", "O_YHX7", "O_YHX8", "O_YHX9", "O_YHX10",
        # "O_YRY1", "O_YRY2", "O_YRY3", "O_YRY4", "O_YRY5", "O_YRY6", "O_YRY7", "O_YRY8", "O_YRY9", "O_YRY10",
        # "O_ZH1", "O_ZH2", "O_ZH3", "O_ZH4", "O_ZH5", "O_ZH6", "O_ZH7", "O_ZH8", "O_ZH9", "O_ZH10",
        # "O_ZH11", "O_ZH12", "O_ZH13", "O_ZH14", "O_ZH15", "O_ZH16", "O_ZH17", "O_ZH18", "O_ZH19", "O_ZH20",
        # "O_ZJW1", "O_ZJW2", "O_ZJW3", "O_ZJW4", "O_ZJW5", "O_ZJW6", "O_ZJW7", "O_ZJW8", "O_ZJW9", "O_ZJW10",
        # "O_ZJW11", "O_ZJW12", "O_ZJW13", "O_ZJW14", "O_ZJW15", "O_ZJW16", "O_ZJW17", "O_ZJW18", "O_ZJW19",
        # "O_ZZR1", "O_ZZR2", "O_ZZR3", "O_ZZR4", "O_ZZR5", "O_ZZR6", "O_ZZR7", "O_ZZR8", "O_ZZR9", "O_ZZR10"

        "V_CYS1", "V_CYS2",
        "V_CYS3", "V_CYS4", "V_CYS5", "V_CYS6", "V_CYS7", "V_CYS8", "V_CYS9", "V_CYS10",
        "V_FWQ1", "V_FWQ2", "V_FWQ3", "V_FWQ4", "V_FWQ5", "V_FWQ7", "V_FWQ8", "V_FWQ9", "V_FWQ10",
        "V_HHY1", "V_HHY2", "V_HHY3", "V_HHY4", "V_HHY5", "V_HHY6", "V_HHY7", "V_HHY8", "V_HHY9", "V_HHY10",
        "V_JJB1", "V_JJB2", "V_JJB3", "V_JJB4", "V_JJB5", "V_JJB6", "V_JJB7", "V_JJB8", "V_JJB9", "V_JJB10",
        "V_LDY1", "V_LDY2", "V_LDY3", "V_LDY4", "V_LDY5", "V_LDY6", "V_LDY7", "V_LDY8", "V_LDY9", "V_LDY10",
        "V_LGH1", "V_LGH2", "V_LGH3", "V_LGH4", "V_LGH5", "V_LGH6", "V_LGH7", "V_LGH8", "V_LGH9", "V_LGH10",
        "V_LRQ1", "V_LRQ2", "V_LRQ3", "V_LRQ4", "V_LRQ5", "V_LRQ6", "V_LRQ7", "V_LRQ8", "V_LRQ9", "V_LRQ10",
        "V_LSY1", "V_LSY2", "V_LSY3", "V_LSY4", "V_LSY5", "V_LSY6", "V_LSY7", "V_LSY8", "V_LSY9", "V_LSY10",
        "V_MKQ1", "V_MKQ2", "V_MKQ3", "V_MKQ4", "V_MKQ5", "V_MKQ6", "V_MKQ7", "V_MKQ8", "V_MKQ9", "V_MKQ10",
        "V_SWJ1", "V_SWJ2", "V_SWJ3", "V_SWJ4", "V_SWJ5", "V_SWJ6", "V_SWJ7", "V_SWJ8", "V_SWJ9", "V_SWJ10",
        "V_WA1", "V_WA2", "V_WA3", "V_WA4", "V_WA5", "V_WA6", "V_WA7", "V_WA8", "V_WA9", "V_WA10",
        "V_WJY1", "V_WJY2", "V_WJY3", "V_WJY4", "V_WJY5", "V_WJY6", "V_WJY7", "V_WJY8", "V_WJY9", "V_WJY10",
        "V_WJZ1", "V_WJZ2", "V_WJZ3", "V_WJZ4", "V_WJZ5", "V_WJZ6", "V_WJZ7", "V_WJZ8", "V_WJZ9", "V_WJZ10",
        "V_WXY1", "V_WXY2", "V_WXY3", "V_WXY4", "V_WXY5", "V_WXY6", "V_WXY7", "V_WXY8", "V_WXY9", "V_WXY10",
        "V_XZC1", "V_XZC2", "V_XZC3", "V_XZC4", "V_XZC5", "V_XZC6", "V_XZC7", "V_XZC8", "V_XZC9", "V_XZC10",
        "V_YHP1", "V_YHP2", "V_YHP3", "V_YHP4", "V_YHP5", "V_YHP6", "V_YHP7", "V_YHP8", "V_YHP9", "V_YHP10",
        "V_YHX1", "V_YHX2", "V_YHX3", "V_YHX4", "V_YHX5", "V_YHX6", "V_YHX7", "V_YHX8", "V_YHX9", "V_YHX10",
        "V_YRY1", "V_YRY2", "V_YRY3", "V_YRY4", "V_YRY5", "V_YRY6", "V_YRY7", "V_YRY8", "V_YRY9", "V_YRY10",
        "V_ZH1", "V_ZH2", "V_ZH3", "V_ZH4", "V_ZH5", "V_ZH6", "V_ZH7", "V_ZH8", "V_ZH9", "V_ZH10",
        "V_ZJW1", "V_ZJW2", "V_ZJW3", "V_ZJW4", "V_ZJW5", "V_ZJW6", "V_ZJW7", "V_ZJW8", "V_ZJW9", "V_ZJW10"
    ]

sides_list = [
    'Left',
    'Right'
]

tip_nums = [
    "1",
    "2",
    "3"
]


def main():
    # 名字与ID的映射
    name_id_map_1 = {name: idx for idx, name in enumerate(names_1)}

    # 输出结果
    print("name_id_map_1:", name_id_map_1)

    user_num = len(name_id_map_1)
    print("user_num:", user_num)

    average_test_result = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    all_results = proce(name_all_1, tip_nums)

    for i in [0, 3]:
        best_model, fold_losses = cross_validate(
            coords=all_results[i],
            seq_len=5,  # 输入 5 帧预测下一帧
            k_folds=3,  # 5 折交叉验证
            batch_size=1024,
            epochs=200,
            lr=0.0005,
            device=device,
            lengthsss=all_results[i + 2],
            dt=1.0 / sample_rate
        )

        print("\n最终交叉验证结果:")
        print(fold_losses)
        print(f"平均验证损失: {np.mean(fold_losses):.9f}")

        # 保存最佳模型
        torch.save(best_model.state_dict(), fr"E:\pythonProject\Predict_hand\model_save\final_model_{i}.pth")
        print(fr"最佳模型已保存 -> D:\dp_handjoint\mlp_dt\final_model_{i}.pth")


if __name__ == "__main__":
    main()

# # 构造一个测试序列 (5帧轨迹)
# test_seq = np.array([
#     [1.0, 0.0, 0.0],
#     [0.9, 0.1, 0.0],
#     [0.8, 0.2, 0.0],
#     [0.7, 0.3, 0.0],
#     [0.9, 0.9, 0.0]
# ], dtype=np.float32)
# # 加载模型
# model = load_model(fr"D:\dp_handjoint\mlp\final_model_0.pth", device=device)
# # 预测下一帧
# pred_next = predict_next(model, test_seq, device=device)
# print("Predicted next frame:", pred_next[:3])


# best_model = load_model(fr"D:\dp_handjoint\mlp\final_model_0.pth", device=device)
#
# for i in [0, 3]:
#     summary = evaluate_model_on_windows(
#         model=best_model,
#         test_coords=all_results[i],       # np.ndarray(N,L,3) 或 list[(Ti,3)]
#         test_lengths=all_results[i+2],     # list[int] / np.ndarray(N,)
#         dt=1.0 / sample_rate,                       # 举例：10ms
#         seq_len=5,                     # 评估用的输入窗口长度
#         batch_size=128,
#         device=device,
#         stride=1,
#         pad_value=0.0,
#         verbose=True
#     )
