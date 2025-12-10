import pickle
import torch
import numpy as np
import random

try:
    from feeder import augmentations
except:
    import augmentations


class Feeder(torch.utils.data.Dataset):
    """
    MoCo 双流数据加载器 (适配手部身份识别)
    返回: (Seq_V1, Graph_V1, Seq_V2, Graph_V2)
    """

    def __init__(self,
                 data_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 # 兼容性参数 (options里可能会传，这里接住但不一定用)
                 input_representations=None,
                 mmap=True,
                 label_path=None,
                 **kwargs):

        self.data_path = data_path
        self.num_frame_path = num_frame_path
        self.input_size = input_size
        self.l_ratio = l_ratio

        self.load_data(mmap)

        # 获取数据维度: N, C, T, V, M
        self.N, self.C, self.T, self.V, self.M = self.data.shape

        print(f"[Feeder Info] Data Shape: {self.data.shape}")
        print(f"[Feeder Info] Joints(V)={self.V}, Persons(M)={self.M} => GRU Input Dim={self.C * self.V * self.M}")

    def load_data(self, mmap):
        # data: N C T V M
        # (注：如果 Deal2.py 生成的是 N,3,T,40,1，这里读出来的就是这个形状)
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        self.number_of_frames = np.load(self.num_frame_path)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        # get raw input
        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        number_of_frames = self.number_of_frames[index]

        # =========================================
        # 生成 View 1 (Query)
        # =========================================
        # 1. 时域增强 (改变速度，不影响身份)
        data_v1 = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)

        # 2. 空域增强 (随机旋转 or 加噪)
        # 严禁使用 Shear/Scale (会改变骨长)
        if random.random() < 0.5:
            data_v1 = augmentations.random_rotate(data_v1)
        else:
            data_v1 = augmentations.joint_courruption(data_v1)

        # =========================================
        # 生成 View 2 (Key)
        # =========================================
        # 1. 时域增强
        data_v2 = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)

        # 2. 空域增强
        if random.random() < 0.5:
            data_v2 = augmentations.random_rotate(data_v2)
        else:
            data_v2 = augmentations.joint_courruption(data_v2)

        # =========================================
        # 格式转换：同时提供 Sequence 和 Graph 格式
        # =========================================

        # 1. Graph Input (C, T, V, M) -> 保持原样，给 AGCN
        graph_v1 = data_v1.astype('float32')
        graph_v2 = data_v2.astype('float32')

        # 2. Sequence Input (T, Input_Dim) -> 给 GRU
        # 动态计算维度，不再硬编码 150
        # transpose: (C, T, V, M) -> (T, V, M, C) -> reshape (T, V*M*C)
        seq_v1 = data_v1.transpose(1, 2, 3, 0).reshape(self.input_size, -1).astype('float32')
        seq_v2 = data_v2.transpose(1, 2, 3, 0).reshape(self.input_size, -1).astype('float32')

        return seq_v1, graph_v1, seq_v2, graph_v2