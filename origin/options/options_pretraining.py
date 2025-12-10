# options/options_pretraining.py

# -----------------------------
# 模型结构参数定义
# -----------------------------

# AGCN 图卷积编码器配置
agcn_model_arguments = {
    "num_class": 128,  # MoCo 输出特征维度
    "num_point": 40,  # 【你的数据】关节点数 40
    "num_person": 1,  # 【你的数据】人数 1
    "graph_args": {
        'labeling_mode': 'spatial'
    }
}

# BIGRU 序列编码器配置
bi_gru_model_arguments = {
    "en_input_size": 120,  # 【你的数据】输入维度 = 3(xyz) * 40关节 * 1人 = 120
    "en_hidden_size": 1024,  # 隐藏层维度，保持原作者设置
    "en_num_layers": 3,  # 层数
    "num_class": 128  # MoCo 输出特征维度
}


# -----------------------------
# 配置类：适配 Hard No-Box 的手部数据预训练
# -----------------------------
class opts_yhx_pretrain:
    def __init__(self):
        # 1. 绑定模型参数 (Brain)
        self.agcn_model_args = agcn_model_arguments

        # 【关键修正】必须传入 GRU 参数，否则双流架构无法初始化
        self.bi_gru_model_args = bi_gru_model_arguments

        self.hcn_model_args = None  # 未使用 HCN，留空即可

        # 2. 训练集 Feeder 配置 (Body - Input)
        # 适配 feeder_pretraining_inter.py 的参数要求
        self.train_feeder_args = {
            'data_path': '/home/fx2/project/MoCo_hand/data_yhx/8group/AGCN_bianma—AGCN_bianma/Identity verification/npy_8:2/train_data_joint.npy',
            'num_frame_path': '/home/fx2/project/MoCo_hand/data_yhx/8group/AGCN_bianma—AGCN_bianma/Identity verification/npy_8:2/train_num_frame.npy',

            # 【重要】原作者 feeder 需要的参数：
            # l_ratio: 时域裁剪的比例范围 [min, max]，用于模拟速度变化
            'l_ratio': [0.1, 1],

            # input_size: 模型输入的固定帧数。
            # Feeder 会把你的 1000 帧裁剪/缩放成这个长度喂给模型。
            # 建议设为 64 (原作者设置)，既能节省显存，又能捕捉局部运动特征。
            'input_size': 64,

            # 你的 label_path 在预训练阶段其实不需要（SSL是无监督的），
            # 但为了防止报错可以留着，或者 feeder 会忽略它
            'label_path': '/home/fx2/project/MoCo_hand/data_yhx/8group/AGCN_bianma—AGCN_bianma/Identity verification/npy_8:2/train_label.npy',
        }

        # 3. 验证集 Feeder 配置 (Validation)
        # 通常命名为 test_feeder_args 以匹配 dataset.py 的通用接口
        self.test_feeder_args = {
            'data_path': '/home/fx2/project/MoCo_hand/data_yhx/8group/AGCN_bianma—MLP_bianma/Adversarial_sample/0.01_test/npy_8:2_attacked/val_data_joint.npy',
            'num_frame_path': '/home/fx2/project/MoCo_hand/data_yhx/8group/AGCN_bianma—MLP_bianma/Adversarial_sample/0.01_test/npy_8:2_attacked/val_num_frame.npy',
            'label_path': '/home/fx2/project/MoCo_hand/data_yhx/8group/AGCN_bianma—MLP_bianma/Adversarial_sample/0.01_test/npy_8:2_attacked/val_label.npy',

            'l_ratio': [0.1, 1],
            'input_size': 64,

            # 验证时通常只看 Graph 分支的表现
            'input_representation': 'graph-based'
        }