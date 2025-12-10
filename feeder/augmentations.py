import torch.nn.functional as F
import torch
import random
import numpy as np
import math


def joint_courruption(input_data, std=0.01):
    """
    关节加噪：模拟传感器噪声
    """
    out = input_data.copy()
    C, T, V, M = input_data.shape  # 动态获取维度

    flip_prob = random.random()

    if flip_prob < 0.5:
        # 随机将一部分关节置为0 (模拟遮挡/丢失)
        # 动态根据 V 的数量来决定 mask 多少个
        num_drop = random.randint(int(V * 0.1), int(V * 0.3))
        joint_indicies = np.random.choice(V, num_drop, replace=False)
        out[:, :, joint_indicies, :] = 0
        return out

    else:
        # 对部分关节添加高斯噪声
        num_noise = random.randint(int(V * 0.1), int(V * 0.3))
        joint_indicies = np.random.choice(V, num_noise, replace=False)

        # 生成噪声
        noise = np.random.normal(0, std, (C, T, num_noise, M))
        out[:, :, joint_indicies, :] = out[:, :, joint_indicies, :] + noise
        return out


def random_rotate(input_data, max_angle=0.2):
    """
    [新增] 纯旋转增强 (Rigid Rotation)
    替代原来的 pose_augmentation (Shear)，因为剪切会改变骨长，破坏身份特征。
    旋转只改变视角，保留几何结构。
    """
    C, T, V, M = input_data.shape
    data_new = input_data.copy()

    # 随机生成旋转角度 (弧度制)
    # 对于手部数据，通常绕 Y 轴或 Z 轴旋转比较合理，这里实现通用的绕轴旋转
    theta = random.uniform(-max_angle, max_angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # 构建旋转矩阵 (假设数据通道顺序是 x, y, z)
    # 这里演示绕 Y 轴旋转 (通常 Y 轴是垂直轴)
    # x' = x cos - z sin
    # z' = x sin + z cos
    x = data_new[0, :, :, :]
    z = data_new[2, :, :, :]

    new_x = x * cos_t - z * sin_t
    new_z = x * sin_t + z * cos_t

    data_new[0, :, :, :] = new_x
    data_new[2, :, :, :] = new_z

    return data_new


def temporal_cropresize(input_data, num_of_frames, l_ratio, output_size):
    C, T, V, M = input_data.shape

    # Temporal crop
    min_crop_length = 64

    scale = np.random.rand(1) * (l_ratio[1] - l_ratio[0]) + l_ratio[0]
    temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames * scale)), min_crop_length), num_of_frames)

    # 确保 start 索引合法
    if num_of_frames - temporal_crop_length + 1 > 0:
        start = np.random.randint(0, num_of_frames - temporal_crop_length + 1)
    else:
        start = 0
        temporal_crop_length = num_of_frames  # 兜底

    temporal_context = input_data[:, start:start + temporal_crop_length, :, :]

    # interpolate
    temporal_context = torch.tensor(temporal_context, dtype=torch.float)
    # Reshape: (C*V*M, T)
    temporal_context = temporal_context.permute(0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
    temporal_context = temporal_context[None, :, :, None]  # (1, C*V*M, T, 1)

    # 双线性插值缩放到 output_size
    temporal_context = F.interpolate(temporal_context, size=(output_size, 1), mode='bilinear', align_corners=False)

    temporal_context = temporal_context.squeeze(dim=3).squeeze(dim=0)
    # Reshape back: (C, V, M, T_new) -> (C, T_new, V, M)
    temporal_context = temporal_context.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

    return temporal_context


def crop_subsequence(input_data, num_of_frames, l_ratio, output_size):
    # 这个函数保留原样即可，主要用于下游微调时的裁剪
    C, T, V, M = input_data.shape

    if l_ratio[0] == 0.1:
        min_crop_length = 64
        scale = np.random.rand(1) * (l_ratio[1] - l_ratio[0]) + l_ratio[0]
        temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames * scale)), min_crop_length),
                                          num_of_frames)

        if num_of_frames - temporal_crop_length + 1 > 0:
            start = np.random.randint(0, num_of_frames - temporal_crop_length + 1)
        else:
            start = 0
            temporal_crop_length = num_of_frames

        temporal_crop = input_data[:, start:start + temporal_crop_length, :, :]
        temporal_crop = torch.tensor(temporal_crop, dtype=torch.float)
        temporal_crop = temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
        temporal_crop = temporal_crop[None, :, :, None]
        temporal_crop = F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear', align_corners=False)
        temporal_crop = temporal_crop.squeeze(dim=3).squeeze(dim=0)
        temporal_crop = temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()
        return temporal_crop

    else:
        start = int((1 - l_ratio[0]) * num_of_frames / 2)
        data = input_data[:, start:num_of_frames - start, :, :]
        temporal_crop_length = data.shape[1]

        temporal_crop = torch.tensor(data, dtype=torch.float)
        temporal_crop = temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
        temporal_crop = temporal_crop[None, :, :, None]
        temporal_crop = F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear', align_corners=False)
        temporal_crop = temporal_crop.squeeze(dim=3).squeeze(dim=0)
        temporal_crop = temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()
        return temporal_crop