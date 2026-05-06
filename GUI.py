import os
import torch
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

# 假设您的代码文件名为 evaluation_logic.py，或者直接引用您的类
# 请确保 model.py 和 dataloader_position.py 在同一目录下
from model import STGCN_Regressor
from dataloader_position import load_and_clean_position
import torch.nn.functional as F


class RehabApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ST-GCN 康复动作量化评估系统")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f2f5")

        # 核心逻辑变量
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.standard_embeddings = {}
        self.MU = 0.4

        self.setup_ui()

    def setup_ui(self):
        # 标题
        header = tk.Label(self.root, text="康复动作评估系统 (ST-GCN)", font=("微软雅黑", 20, "bold"), bg="#1890ff",
                          fg="white", pady=10)
        header.pack(fill=tk.X)

        # 控制面板
        ctrl_frame = tk.Frame(self.root, bg="#f0f2f5", pady=20)
        ctrl_frame.pack()

        self.btn_load_model = ttk.Button(ctrl_frame, text="1. 加载模型权重", command=self.load_model_thread)
        self.btn_load_model.grid(row=0, column=0, padx=10)

        self.btn_init_pool = ttk.Button(ctrl_frame, text="2. 初始化标准库", command=self.init_pool_thread,
                                        state=tk.DISABLED)
        self.btn_init_pool.grid(row=0, column=1, padx=10)

        self.btn_test = ttk.Button(ctrl_frame, text="3. 开始盲测", command=self.start_test_thread, state=tk.DISABLED)
        self.btn_test.grid(row=0, column=2, padx=10)

        # 状态显示
        self.status_var = tk.StringVar(value="系统就绪，请加载模型...")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 结果显示区
        display_frame = tk.Frame(self.root, bg="white", padx=20, pady=20)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.result_text = tk.Text(display_frame, font=("Consolas", 11), state=tk.DISABLED, bg="#fafafa")
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # 进度条
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=10)

    # --- 逻辑包装 ---
    def log(self, message):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)

    def process_file(self, file_path):
        data = load_and_clean_position(file_path)
        data = np.transpose(data, (2, 0, 1))
        data = np.expand_dims(data, axis=(0, -1))
        return torch.tensor(data, dtype=torch.float32).to(self.device)

    # --- 线程化任务 ---
    def load_model_thread(self):
        path = filedialog.askopenfilename(title="选择模型权重", filetypes=[("PTH files", "*.pth")])
        if not path: return

        def task():
            try:
                self.status_var.set("正在加载模型...")
                self.model = STGCN_Regressor(in_channels=3).to(self.device)
                self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
                self.model.eval()
                self.log(f"模型加载成功！设备: {self.device}")
                self.btn_init_pool.config(state=tk.NORMAL)
                self.status_var.set("模型已准备好")
            except Exception as e:
                messagebox.showerror("错误", f"加载失败: {e}")

        threading.Thread(target=task).start()

    def init_pool_thread(self):
        folder = filedialog.askdirectory(title="选择训练集(标准动作)目录")
        if not folder: return

        def task():
            self.btn_init_pool.config(state=tk.DISABLED)
            self.log(f"🚀 开始扫描模板库: {folder}")
            paths_dict = {i: [] for i in range(10)}
            for f in os.listdir(folder):
                if f.endswith('.txt') and '_inc' not in f:
                    idx = int(f.split('_')[0].replace('m', '')) - 1
                    paths_dict[idx].append(os.path.join(folder, f))

            with torch.no_grad():
                for i, (idx, paths) in enumerate(paths_dict.items()):
                    self.standard_embeddings[idx] = []
                    for p in paths:
                        inp = self.process_file(p)
                        self.standard_embeddings[idx].append(self.model(inp))
                    self.progress['value'] = (i + 1) * 10
                    self.status_var.set(f"正在存入特征库: 动作 {idx + 1}")

            self.log("内存库加载完毕！")
            self.btn_test.config(state=tk.NORMAL)
            self.status_var.set("准备盲测")

        threading.Thread(target=task).start()

    def start_test_thread(self):
        folder = filedialog.askdirectory(title="选择测试集目录")
        if not folder: return

        def task():
            test_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
            samples = test_files[:10]  # 取前10个
            self.log(f"\n--- 开始盲测 (抽取 {len(samples)} 个样本) ---")

            with torch.no_grad():
                for f_name in samples:
                    path = os.path.join(folder, f_name)
                    act_idx = int(f_name.split('_')[0].replace('m', '')) - 1
                    is_inc = '_inc' in f_name

                    emb_test = self.model(self.process_file(path))
                    dists = [(1.0 - F.cosine_similarity(emb_test, ref, dim=1).item()) for ref in
                             self.standard_embeddings[act_idx]]

                    best_dist = min(dists)
                    score = 100.0 * np.exp(-(best_dist ** 2) / (2 * (self.MU ** 2)))

                    self.log(f"文件: {f_name}")
                    res_str = "错误" if is_inc else "标准"
                    self.log(f"   [标签]: {res_str}  [距离]: {best_dist:.4f}  [AI得分]: {score:.1f}")
            self.log("--- 盲测结束 ---")

        threading.Thread(target=task).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = RehabApp(root)
    root.mainloop()