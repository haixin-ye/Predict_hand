import pickle
import torch
import numpy as np

np.set_printoptions(threshold=np.inf)
import random

try:
    from feeder import augmentations
except:
    import augmentations


class Feeder(torch.utils.data.Dataset):
    """
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 # input_representations, # 保持接口兼容，但在代码中逻辑已固定为双流
                 mmap=True,
                 # 兼容性参数，防止options传入报错
                 label_path=None,
                 **kwargs):

        self.data_path = data_path
        self.num_frame_path = num_frame_path
        self.input_size = input_size
        self.crop_resize = True
        self.l_ratio = l_ratio

        self.load_data(mmap)
        # 获取数据维度: N, C, T, V, M
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        print(f"[Feeder] Data Shape: {self.data.shape}, Samples: {len(self.number_of_frames)}")
        print(f"[Feeder] Joints(V): {self.V}, Channels(C): {self.C}, Persons(M): {self.M}")
        print(f"[Feeder] L_Ratio: {self.l_ratio}")

    def load_data(self, mmap):
        # data: N C T V M (注意：Deal2.py 生成的是 N, 3, T, 40, 1)
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        self.number_of_frames = np.load(self.num_frame_path)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get raw input
        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        number_of_frames = self.number_of_frames[index]

        # === View 1 Generation ===
        # 1. Temporal Crop-Resize
        data_numpy_v1_crop = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio,
                                                               self.input_size)

        # 2. Spatial Augmentation (Randomly select)
        if random.random() < 0.5:
            # 【关键修改】使用 random_rotate 替代 pose_augmentation(Shear)
            # 旋转不改变骨长，适合身份识别任务
            data_numpy_v1 = augmentations.random_rotate(data_numpy_v1_crop)
        else:
            # 关节加噪
            data_numpy_v1 = augmentations.joint_courruption(data_numpy_v1_crop)

        # === View 2 Generation ===
        # 1. Temporal Crop-Resize
        data_numpy_v2_crop = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio,
                                                               self.input_size)

        # 2. Spatial Augmentation
        if random.random() < 0.5:
            data_numpy_v2 = augmentations.random_rotate(data_numpy_v2_crop)
        else:
            data_numpy_v2 = augmentations.joint_courruption(data_numpy_v2_crop)

        # === Format Conversion for Model Inputs ===
        # 无论 options 怎么填，我们这里固定输出 Sequence 和 Graph 两种格式，供给 MoCo 训练

        # 1. Sequence-based input (GRU)
        # Shape change: (C, T, V, M) -> (T, V, M, C) -> (T, V*M*C)
        # 例如: (3, 64, 40, 1) -> (64, 40*1*3) = (64, 120)
        input_s1_v1 = data_numpy_v1.transpose(1, 2, 3, 0)
        input_s1_v1 = input_s1_v1.reshape(self.input_size, -1).astype('float32')

        input_s1_v2 = data_numpy_v2.transpose(1, 2, 3, 0)
        input_s1_v2 = input_s1_v2.reshape(self.input_size, -1).astype('float32')

        # 2. Graph-based input (AGCN)
        # Shape: (C, T, V, M) -> 保持不变 (3, 64, 40, 1)
        input_s2_v1 = data_numpy_v1.astype('float32')
        input_s2_v2 = data_numpy_v2.astype('float32')

        # 返回 4 个 Tensor: (Seq_V1, Graph_V1, Seq_V2, Graph_V2)
        return input_s1_v1, input_s2_v1, input_s1_v2, input_s2_v2