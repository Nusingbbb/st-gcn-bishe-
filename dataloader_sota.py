import os
import numpy as np
import torch
import torch.nn.functional as F
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
# ... [保留你上一轮最新的 load_and_clean_position 橡皮筋插值函数不变] ...

class UIPRMD_Dataset(Dataset):
    def __init__(self, current_folder, train_folder):
        self.current_folder = current_folder
        self.files = [f for f in os.listdir(current_folder) if f.endswith('.txt')]

        # 🌟 SOTA 升级：集结全量黄金教科书库
        self.standard_pool = {i: [] for i in range(10)}
        for file_name in os.listdir(train_folder):
            if file_name.endswith('.txt') and '_inc' not in file_name:
                action_idx = int(file_name.split('_')[0].replace('m', '')) - 1
                self.standard_pool[action_idx].append(os.path.join(train_folder, file_name))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        test_file_path = os.path.join(self.current_folder, file_name)
        action_idx = int(file_name.split('_')[0].replace('m', '')) - 1

        data_test = load_and_clean_position(test_file_path)
        data_test = np.transpose(data_test, (2, 0, 1))
        data_test = np.expand_dims(data_test, axis=-1)

        # 🌟 提取该动作对应的所有 10 个专家标尺，并堆叠在一起
        refs_data = []
        for ref_path in self.standard_pool[action_idx]:
            ref_d = load_and_clean_position(ref_path)
            ref_d = np.transpose(ref_d, (2, 0, 1))
            ref_d = np.expand_dims(ref_d, axis=-1)
            refs_data.append(ref_d)

        # 形状变为: (10, 3, 73, 22, 1)
        refs_tensor = torch.tensor(np.stack(refs_data), dtype=torch.float32)

        target_y = 0.0 if '_inc' in file_name else 1.0

        return torch.tensor(data_test, dtype=torch.float32), \
            refs_tensor, \
            torch.tensor(target_y, dtype=torch.float32), \
            torch.tensor(action_idx, dtype=torch.long)

# ... get_position_dataloaders 保持不变 ...

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