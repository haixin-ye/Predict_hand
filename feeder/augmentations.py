import torch.nn.functional as F
import torch
import random
import numpy as np
import math


def joint_courruption(input_data, std=0.01):
    """
    关节加噪：模拟传感器噪声 (保留)
    微小的抖动不会改变宏观骨长，有助于提高模型对噪点的鲁棒性。
    """
    out = input_data.copy()
    C, T, V, M = input_data.shape  # 动态获取维度

    flip_prob = random.random()

    if flip_prob < 0.5:
        # 模拟遮挡/丢帧：随机将一部分关节置为0
        # 动态根据 V 的数量来决定 mask 多少个 (例如遮挡 10%-30% 的点)
        num_drop = random.randint(int(V * 0.1), int(V * 0.3))
        # 确保至少丢1个，或者如果V太小就不丢
        num_drop = max(1, 3)

        joint_indicies = np.random.choice(V, num_drop, replace=False)
        out[:, :, joint_indicies, :] = 0
        return out

    else:
        # 模拟噪声：对部分关节添加高斯噪声
        num_noise = random.randint(int(V * 0.1), int(V * 0.3))
        num_noise = max(1, num_noise)

        joint_indicies = np.random.choice(V, num_noise, replace=False)

        # 生成噪声 (C, T, num_noise, M)
        noise = np.random.normal(0, std, (C, T, num_noise, M))
        out[:, :, joint_indicies, :] = out[:, :, joint_indicies, :] + noise
        return out


def random_rotate(input_data, max_angle=0.3):
    """
    [新增] 纯旋转增强 (Rigid Rotation)
    替代原来的 pose_augmentation (Shear)。
    核心目的：改变视角，但【绝对保持骨骼长度不变】。
    """
    # input_data: (C, T, V, M)
    data_new = input_data.copy()

    # 随机生成旋转角度 (弧度制)
    # 针对手部数据，全方位的3D旋转是安全的
    alpha = random.uniform(-max_angle, max_angle)  # 绕X轴
    beta = random.uniform(-max_angle, max_angle)  # 绕Y轴
    gamma = random.uniform(-max_angle, max_angle)  # 绕Z轴

    # 旋转矩阵 (Rotation Matrix)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])

    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])

    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])

    # 组合旋转矩阵 R = Rz * Ry * Rx
    R = np.dot(Rz, np.dot(Ry, Rx))

    # 执行旋转: Data * R.T
    # transpose(1,2,3,0) 把 C(3) 放到最后 -> (T, V, M, 3)
    # dot 之后 -> (T, V, M, 3)
    # transpose(3,0,1,2) 还原 -> (3, T, V, M)
    result = np.dot(data_new.transpose([1, 2, 3, 0]), R.transpose())
    output = result.transpose(3, 0, 1, 2)

    return output


def temporal_cropresize(input_data, num_of_frames, l_ratio, output_size):
    """
    时域裁剪：改变动作速度 (保留)
    这对身份识别是安全的，因为动作快慢不影响身份。
    """
    C, T, V, M = input_data.shape

    # 最小裁剪长度
    min_crop_length = 32  # 稍微改小一点以适应较短的输入

    # 随机计算裁剪长度
    scale = np.random.rand(1) * (l_ratio[1] - l_ratio[0]) + l_ratio[0]
    temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames * scale)), min_crop_length), num_of_frames)

    # 随机选择起始点
    if num_of_frames - temporal_crop_length + 1 > 0:
        start = np.random.randint(0, num_of_frames - temporal_crop_length + 1)
    else:
        start = 0
        temporal_crop_length = num_of_frames

    temporal_context = input_data[:, start:start + temporal_crop_length, :, :]

    # 双线性插值缩放回 output_size
    temporal_context = torch.tensor(temporal_context, dtype=torch.float)
    # Reshape: (C*V*M, T)
    temporal_context = temporal_context.permute(0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
    temporal_context = temporal_context[None, :, :, None]  # (1, C*V*M, T, 1)

    temporal_context = F.interpolate(temporal_context, size=(output_size, 1), mode='bilinear', align_corners=False)

    temporal_context = temporal_context.squeeze(dim=3).squeeze(dim=0)
    # Reshape back: (C, V, M, T_new) -> (C, T_new, V, M)
    temporal_context = temporal_context.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

    return temporal_context


# 下面的函数如果没用到可以不用管，但保留也没事
def crop_subsequence(input_data, num_of_frames, l_ratio, output_size):
    # ... (保持原样)
    return input_data  # 占位