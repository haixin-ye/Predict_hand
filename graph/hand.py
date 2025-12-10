import sys
import numpy as np

# 引用同目录下的 tools
try:
    from graph import tools
except ImportError:
    import tools

num_node = 40

self_link = [(i, i) for i in range(num_node)]

# 你的 40 关节手部拓扑 (0-based)
inward_ori_index = [
    # 左手五指
    (1, 0), (2, 1), (3, 2),        # Index
    (5, 4), (6, 5), (7, 6),        # Middle
    (9, 8), (10, 9), (11, 10),     # Ring
    (13, 12), (14, 13), (15, 14),  # Pinky
    (17, 16), (18, 17),            # Thumb

    # 左手指根 -> 左手 Wrist (19)
    (0, 19), (4, 19), (8, 19), (12, 19), (16, 19),

    # 右手五指
    (21, 20), (22, 21), (23, 22),   # Index
    (25, 24), (26, 25), (27, 26),   # Middle
    (29, 28), (30, 29), (31, 30),   # Ring
    (33, 32), (34, 33), (35, 34),   # Pinky
    (37, 36), (38, 37),             # Thumb

    # 右手指根 -> 右手 Wrist (39)
    (20, 39), (24, 39), (28, 39), (32, 39), (36, 39),

    # 跨手 Wrist 双向连接
    (19, 39), (39, 19)
]

# 你的数据已经是 0-based，直接使用
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A