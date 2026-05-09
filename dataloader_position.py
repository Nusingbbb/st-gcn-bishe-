import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_and_clean_position(file_path):
    # 1. 读数据
    if ',' in open(file_path).read():
        data = np.loadtxt(file_path, delimiter=',')
    else:
        data = np.loadtxt(file_path)

    # 2. 统一 73 帧
    if data.shape[0] < 73:
        data = np.pad(data, ((0, 73 - data.shape[0]), (0, 0)), mode='constant')
    else:
        data = data[:73, :]

    data = data.reshape(73, 22, 3)

    # ==========================================
    # 🌟 物理装甲：绝对坐标 -> 相对尺度
    # ==========================================
    # 平移到原点 (骨盆作为世界中心)
    root_pos = data[:, 0:1, :].copy()
    data = data - root_pos

    # 骨架尺度归一化 (消除受试者高矮胖瘦差异)
    pelvis = data[0, 0, :]
    chest = data[0, 2, :]
    torso_length = np.linalg.norm(chest - pelvis) + 1e-6
    data = data / torso_length

    return data


class UIPRMD_Dataset(Dataset):
    def __init__(self, current_folder, train_folder):
        self.current_folder = current_folder
        self.files = [f for f in os.listdir(current_folder) if f.endswith('.txt')]

        # ==========================================
        # 🌟 绝杀点 1：固定唯一靶心 (Fixed Anchor)
        # ==========================================
        self.standard_pool = {}
        # 必须使用 sorted，保证每次运行抽取到的“教科书”是唯一的
        for file_name in sorted(os.listdir(train_folder)):
            if file_name.endswith('.txt') and '_inc' not in file_name:
                action_idx = int(file_name.split('_')[0].replace('m', '')) - 1
                # 只有当该动作的席位空缺时才存入，拒绝随机抽卡
                if action_idx not in self.standard_pool:
                    self.standard_pool[action_idx] = os.path.join(train_folder, file_name)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        test_file_path = os.path.join(self.current_folder, file_name)
        action_idx = int(file_name.split('_')[0].replace('m', '')) - 1

        # 加载测试动作
        data_test = load_and_clean_position(test_file_path)

        # 永远与那个绝对固定的唯一“标尺”对比
        ref_file_path = self.standard_pool[action_idx]
        data_ref = load_and_clean_position(ref_file_path)

        # ==========================================
        # 🌟 绝杀点 2：回归绝对纯净的医学先验标签
        # ==========================================
        # 只要带 '_inc' 就是 0.0 (错误动作)；不带就是 1.0 (正确动作)
        target_y = 0.0 if '_inc' in file_name else 1.0

        # 张量维度转换: (73, 22, 3) -> (3, 73, 22, 1)
        data_test = np.transpose(data_test, (2, 0, 1))
        data_test = np.expand_dims(data_test, axis=-1)
        data_ref = np.transpose(data_ref, (2, 0, 1))
        data_ref = np.expand_dims(data_ref, axis=-1)

        return torch.tensor(data_test, dtype=torch.float32), \
               torch.tensor(data_ref, dtype=torch.float32), \
               torch.tensor(target_y, dtype=torch.float32), \
               torch.tensor(action_idx, dtype=torch.long)


def get_position_dataloaders(base_path, batch_size=16):
    train_dir = os.path.join(base_path, 'train', 'train_position')
    val_dir = os.path.join(base_path, 'val', 'val_position')
    test_dir = os.path.join(base_path, 'test', 'test_position')

    train_ds = UIPRMD_Dataset(train_dir, train_dir)
    val_ds = UIPRMD_Dataset(val_dir, train_dir)
    test_ds = UIPRMD_Dataset(test_dir, train_dir)

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(val_ds, batch_size=batch_size, shuffle=False), \
           DataLoader(test_ds, batch_size=batch_size, shuffle=False)