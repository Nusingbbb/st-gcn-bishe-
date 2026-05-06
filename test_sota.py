import os
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from model import STGCN_Regressor
from dataloader_position import load_and_clean_position

def process_single_file(file_path):
    data = load_and_clean_position(file_path)
    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=-1)
    data = np.expand_dims(data, axis=0)
    return torch.tensor(data, dtype=torch.float32)

def evaluate_test_set():
    BASE_PATH = r"E:\PythonProject2\bishe\data\latest_data"
    TRAIN_FOLDER = os.path.join(BASE_PATH, "train", "train_position")
    TEST_FOLDER = os.path.join(BASE_PATH, "test", "test_position")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 启动 SOTA 级多锚点期末考试引擎 (全量盲测模式)...")

    # ==========================================
    # 🌟 建立全量专家委员会 (Multi-Anchor Pool)
    # ==========================================
    standard_pool = {i: [] for i in range(10)}
    print(" 正在集结全量黄金教科书库...")
    for file_name in os.listdir(TRAIN_FOLDER):
        if file_name.endswith('.txt') and '_inc' not in file_name:
            action_idx = int(file_name.split('_')[0].replace('m', '')) - 1
            file_path = os.path.join(TRAIN_FOLDER, file_name)
            standard_pool[action_idx].append(file_path)

    model = STGCN_Regressor(in_channels=3).to(device)
    weight_path = 'best_rho_sota_stgcn.pth'
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        print(f"✅ 成功加载巅峰权重: {weight_path}")
    else:
        print("❌ 找不到 best_rho_sota_stgcn.pth！")
        return

    model.eval()

    test_preds_score, test_preds_rank, test_trues = [], [], []

    print("\n" + "=" * 70)
    print(" 🎯 开始对 Test 集进行全量多专家盲测 🎯")
    print("=" * 70)

    # 获取所有测试文件
    all_test_files = [f for f in os.listdir(TEST_FOLDER) if f.endswith('.txt')]

    # 🌟 核心修改：取消随机抽样，使用 sorted 排序保证测试顺序绝对一致，避免任何玄学波动
    test_files = sorted(all_test_files)
    sample_size = len(test_files)

    with torch.no_grad():
        # 提前提取专家委员会的所有特征，加速计算
        standard_embs = {i: [] for i in range(10)}
        for idx, paths in standard_pool.items():
            for p in paths:
                standard_embs[idx].append(model(process_single_file(p).to(device)))

        # 遍历全量样本
        for i, file_name in enumerate(test_files):
            test_file_path = os.path.join(TEST_FOLDER, file_name)

            # 读取动作种类 (0-9)
            action_idx = int(file_name.split('_')[0].replace('m', '')) - 1

            # 临床先验标签
            is_incorrect = '_inc' in file_name
            true_label = 0.0 if is_incorrect else 1.0

            # 提取测试动作特征
            inputs_test = process_single_file(test_file_path).to(device)
            emb_test = model(inputs_test)

            # 多锚点 KNN 距离检索
            all_distances = []
            for emb_ref in standard_embs[action_idx]:
                cos_sim = F.cosine_similarity(emb_test, emb_ref, dim=1)
                all_distances.append(1.0 - cos_sim.item())

            # 🌟 逻辑修正：将排序和求平均值的代码移到 for 循环外部！
            # 将距离从小到大排序
            all_distances.sort()

            # 取最近的 3 个专家的距离，求平均值 (Top-3 KNN)
            k = min(3, len(all_distances))
            best_dist = sum(all_distances[:k]) / k

            # 物理偏差映射为 0-100 分
            score = 100.0 * np.exp(-1.5 * best_dist)

            test_preds_score.append(score)
            test_preds_rank.append(-best_dist)
            test_trues.append(true_label)

            # 打印审查
            mark = "❌ 代偿错误" if is_incorrect else "✅ 标准动作"
            print(f"样本 {i + 1:03d}/{sample_size} | 类别: 动作{action_idx + 1:02d} | {mark} | AI得分: {score:5.1f} 分 | 源文件: {file_name}")

    # 计算全量样本的 Spearman ρ
    test_rho, p_value = spearmanr(test_trues, test_preds_rank)
    if np.isnan(test_rho):
        test_rho = 0.0
        print("\n⚠️ 提示：发生异常，无法计算排位相关性。")

    print("\n" + "★" * 70)
    print(f" 🏆 全量测试集 ({sample_size} 样本) Spearman ρ: {test_rho:.4f}")
    print(f" 📊 统计学 P-value: {p_value:.4e} (越小说明结果越具备统计学意义)")
    print("★" * 70)

if __name__ == '__main__':
    evaluate_test_set()