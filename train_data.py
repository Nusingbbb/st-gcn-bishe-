import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

from model_new import STGCN_Regressor
from dataloader_position import get_position_dataloaders


def train_model():
    # ==========================================
    # 1. 基础配置
    # ==========================================
    BASE_PATH = r"E:\PythonProject2\bishe\data\latest_data"
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 400
    MARGIN = 1.35  # 对比损失的安全推斥边界

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 当前使用的计算设备: {device}")

    print(" 正在加载动态孪生数据流水线...")
    train_loader, val_loader, test_loader = get_position_dataloaders(BASE_PATH, batch_size=BATCH_SIZE)

    print(" 正在构建 ST-GCN 拓扑特征提取器...")
    model = STGCN_Regressor(in_channels=3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 🌟 新增：余弦退火调度器，让学习率平滑下降，榨干最后一点性能
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    best_val_rho = -1.0

    print("\n 硬监督 + 软度量 训练正式开始！\n" + "=" * 60)

    for epoch in range(NUM_EPOCHS):
        # ---------------- 训练阶段 ----------------
        model.train()
        train_loss = 0.0
        train_preds, train_trues = [], []

        for inputs_test, inputs_ref, target_y, action_classes in train_loader:
            inputs_test = inputs_test.to(device)
            inputs_ref = inputs_ref.to(device)
            target_y = target_y.to(device)  # 纯粹的 0.0 或 1.0

            optimizer.zero_grad()

            emb_test = model(inputs_test)
            emb_ref = model(inputs_ref)

            cos_sim = F.cosine_similarity(emb_test, emb_ref, dim=1)
            dist = 1.0 - cos_sim

            # 经典对比损失：1.0全力拉近，0.0全力推开
            loss_pull = target_y * (dist ** 2)
            loss_push = (1.0 - target_y) * (F.relu(MARGIN - dist) ** 2)
            loss = 0.5 * (loss_pull + loss_push).mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs_test.size(0)

            # ==========================================
            # 🌟 绝杀点 3：拆除 Clamp，回归纯粹的物理距离排位
            # ==========================================
            with torch.no_grad():
                # 距离(dist)越小，动作越好，所以用负距离 (-dist) 来代表真实排名分数
                pred_rank_score = -dist.detach()

                train_preds.extend(pred_rank_score.cpu().numpy())
                train_trues.extend(target_y.cpu().numpy())

        train_loss = train_loss / len(train_loader.dataset)
        t_rho, _ = spearmanr(train_trues, train_preds)
        if np.isnan(t_rho): t_rho = 0.0

        # ---------------- 验证阶段 ----------------
        model.eval()
        val_loss = 0.0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for inputs_test, inputs_ref, target_y, action_classes in val_loader:
                inputs_test = inputs_test.to(device)
                inputs_ref = inputs_ref.to(device)
                target_y = target_y.to(device)

                emb_test = model(inputs_test)
                emb_ref = model(inputs_ref)

                cos_sim = F.cosine_similarity(emb_test, emb_ref, dim=1)
                dist = 1.0 - cos_sim

                loss_pull = target_y * (dist ** 2)
                loss_push = (1.0 - target_y) * (F.relu(MARGIN - dist) ** 2)
                loss = 0.5 * (loss_pull + loss_push).mean()

                val_loss += loss.item() * inputs_test.size(0)

                # 验证集同样使用无截断的纯粹距离进行排位
                pred_rank_score = -dist.detach()

                val_preds.extend(pred_rank_score.cpu().numpy())
                val_trues.extend(target_y.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        v_rho, _ = spearmanr(val_trues, val_preds)
        if np.isnan(v_rho): v_rho = 0.0

        # 学习率步进
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ---------------- 终端打印与保存模型 ----------------
        print(f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] LR: {current_lr:.6f} | Train ρ: {t_rho:.4f} | Val ρ: {v_rho:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_loss_stgcn.pth')

        if v_rho > best_val_rho:
            best_val_rho = v_rho
            torch.save(model.state_dict(), 'best_rho_stgcn.pth')
            print(f"    🏆 突破历史新高！已封存最高排序模型 (Spearman ρ = {v_rho:.4f})")

    print("=" * 60 + "\n🎯 炼丹结束！表现最优的 ST-GCN 隐空间度量网络已被成功保存。")


if __name__ == '__main__':
    train_model()