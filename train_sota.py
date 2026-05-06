import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from model import STGCN_Regressor
from dataloader_sota import get_position_dataloaders


def train_model():
    BASE_PATH = r"E:\PythonProject2\bishe\data\latest_data"

    # 🌟 显存绝对安全方案：物理2 + 累加16 = 逻辑32
    BATCH_SIZE = 2
    ACCUMULATION_STEPS = 16

    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 400
    MARGIN = 1.35

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 SOTA 多中心度量引擎启动 (设备: {device})")
    print(
        f"📦 物理 Batch: {BATCH_SIZE} | 梯度累加: {ACCUMULATION_STEPS} | 逻辑等效 Batch: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print("🛡️ 已关闭 AMP，使用绝对稳定的 FP32 精度防止梯度爆炸！\n" + "=" * 60)

    train_loader, val_loader, test_loader = get_position_dataloaders(BASE_PATH, batch_size=BATCH_SIZE)
    model = STGCN_Regressor(in_channels=3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_rho = -1.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_preds, train_trues = [], []

        optimizer.zero_grad()

        # 训练集进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] Train", leave=False, colour='green')

        for i, (inputs_test, inputs_refs, target_y, action_classes) in enumerate(train_pbar):
            inputs_test = inputs_test.to(device)
            inputs_refs = inputs_refs.to(device)
            target_y = target_y.to(device)

            # 🛡️ 移除 AMP，全部使用默认的 float32 进行高精度计算
            if inputs_test.dim() == 4:
                inputs_test = inputs_test.unsqueeze(-1)
            emb_test = model(inputs_test)

            B = inputs_refs.size(0)
            num_anchors = inputs_refs.size(1)
            refs_flat = inputs_refs.contiguous().view(B * num_anchors, *inputs_refs.shape[2:])

            if refs_flat.dim() == 4:
                refs_flat = refs_flat.unsqueeze(-1)

            emb_refs_flat = model(refs_flat)
            emb_refs = emb_refs_flat.view(B, num_anchors, -1)

            cos_sims = F.cosine_similarity(emb_test.unsqueeze(1), emb_refs, dim=2)
            distances = 1.0 - cos_sims
            min_dist, _ = torch.min(distances, dim=1)

            loss_pull = target_y * (min_dist ** 2)
            loss_push = (1.0 - target_y) * (F.relu(MARGIN - min_dist) ** 2)
            loss = 0.5 * (loss_pull + loss_push).mean()

            # 梯度平摊
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            # 达到累加步数，或者到了最后一个 batch，执行更新
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                # 🌟 修复警告：保证 optimizer.step() 必定执行
                optimizer.step()
                optimizer.zero_grad()

            current_real_loss = loss.item() * ACCUMULATION_STEPS
            train_loss += current_real_loss * inputs_test.size(0)

            # 实时显示 Loss
            train_pbar.set_postfix({'Loss': f"{current_real_loss:.4f}"})

            with torch.no_grad():
                pred_rank_score = -min_dist.detach()
                train_preds.extend(pred_rank_score.cpu().numpy())
                train_trues.extend(target_y.cpu().numpy())

        train_loss = train_loss / len(train_loader.dataset)
        t_rho, _ = spearmanr(train_trues, train_preds)
        if np.isnan(t_rho): t_rho = 0.0

        # ---------------- 验证阶段 ----------------
        model.eval()
        val_preds, val_trues = [], []

        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] Val  ", leave=False, colour='blue')

        with torch.no_grad():
            for inputs_test, inputs_refs, target_y, action_classes in val_pbar:
                inputs_test = inputs_test.to(device)
                inputs_refs = inputs_refs.to(device)
                target_y = target_y.to(device)

                if inputs_test.dim() == 4:
                    inputs_test = inputs_test.unsqueeze(-1)
                emb_test = model(inputs_test)

                B = inputs_refs.size(0)
                num_anchors = inputs_refs.size(1)
                refs_flat = inputs_refs.contiguous().view(B * num_anchors, *inputs_refs.shape[2:])

                if refs_flat.dim() == 4:
                    refs_flat = refs_flat.unsqueeze(-1)

                emb_refs = model(refs_flat).view(B, num_anchors, -1)

                cos_sims = F.cosine_similarity(emb_test.unsqueeze(1), emb_refs, dim=2)
                min_dist, _ = torch.min(1.0 - cos_sims, dim=1)

                pred_rank_score = -min_dist.detach()
                val_preds.extend(pred_rank_score.cpu().numpy())
                val_trues.extend(target_y.cpu().numpy())

        v_rho, _ = spearmanr(val_trues, val_preds)
        if np.isnan(v_rho): v_rho = 0.0

        # 现在更新 scheduler 是 100% 安全的
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} (ρ: {t_rho:.4f}) | Val ρ: {v_rho:.4f}")

        if v_rho > best_val_rho:
            best_val_rho = v_rho
            torch.save(model.state_dict(), 'best_rho_sota_stgcn.pth')
            print(f"    🏆 突破 SOTA！多中心极值模型已封存 (Spearman ρ = {v_rho:.4f})")

    print("\n" + "=" * 60 + "\n🎯 炼丹结束！")


if __name__ == '__main__':
    train_model()