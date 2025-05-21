import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import pandas as pd
import numpy as np
from imblearn.metrics import specificity_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from esm2_extractor2 import ESM2FeatureExtractor
from onehot_encoder import OneHotEncoderProtein
from ANOVA import FeatureReducer
from classifierDNN import classify_model_with_preprocessed_data_DNN
from classifierCNN import classify_model_with_preprocessed_data_CNN
from classifierMLP import classify_model_with_preprocessed_data_MLP

from DNN_Attention import DNN
from CNN_Attention import CNN
from MLP_Attention import MLP

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("蛋白质分类预测器")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # 设置主题颜色
        self.primary_color = "#2563eb"
        self.secondary_color = "#f97316"
        self.tertiary_color = "#10b981"
        self.neutral_color = "#64748b"

        # 变量
        self.positive_file = ""
        self.negative_file = ""
        self.output_filename = tk.StringVar(value="预测结果.csv")
        self.results_df = None
        self.feature_importance = None

        # 创建UI
        self.create_widgets()

    def create_widgets(self):
        # 顶部导航栏
        navbar = tk.Frame(self.root, bg=self.primary_color, height=50)
        navbar.pack(fill="x")
        tk.Label(navbar, text="蛋白质分类预测系统", font=("Arial", 16, "bold"),
                 bg=self.primary_color, fg="white").pack(pady=10, padx=20, side="left")

        # 主内容区域
        content_frame = tk.Frame(self.root, bg="#f0f0f0")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # 左侧面板 - 输入和控制
        left_frame = tk.Frame(content_frame, bg="#ffffff", relief="solid", bd=1)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # 文件选择区域
        file_frame = tk.LabelFrame(left_frame, text="数据集选择", font=("Arial", 12, "bold"),
                                   padx=20, pady=15, bg="#ffffff")
        file_frame.pack(fill="x", pady=(10, 20), padx=15)

        # 正样本文件
        tk.Label(file_frame, text="正样本数据集:", font=("Arial", 10), bg="#ffffff").grid(row=0, column=0, sticky="w",
                                                                                          pady=5)
        self.positive_entry = tk.Entry(file_frame, width=40, font=("Arial", 10))
        self.positive_entry.grid(row=0, column=1, padx=5, pady=5)
        browse_btn1 = tk.Button(file_frame, text="浏览", command=self.browse_positive,
                                bg=self.primary_color, fg="white", font=("Arial", 10),
                                relief="flat", padx=10)
        browse_btn1.grid(row=0, column=2, padx=5, pady=5)

        # 负样本文件
        tk.Label(file_frame, text="负样本数据集:", font=("Arial", 10), bg="#ffffff").grid(row=1, column=0, sticky="w",
                                                                                          pady=5)
        self.negative_entry = tk.Entry(file_frame, width=40, font=("Arial", 10))
        self.negative_entry.grid(row=1, column=1, padx=5, pady=5)
        browse_btn2 = tk.Button(file_frame, text="浏览", command=self.browse_negative,
                                bg=self.primary_color, fg="white", font=("Arial", 10),
                                relief="flat", padx=10)
        browse_btn2.grid(row=1, column=2, padx=5, pady=5)

        # 输出设置区域
        output_frame = tk.LabelFrame(left_frame, text="输出设置", font=("Arial", 12, "bold"),
                                     padx=20, pady=15, bg="#ffffff")
        output_frame.pack(fill="x", pady=(0, 20), padx=15)

        tk.Label(output_frame, text="输出文件名:", font=("Arial", 10), bg="#ffffff").grid(row=0, column=0, sticky="w",
                                                                                          pady=5)
        tk.Entry(output_frame, textvariable=self.output_filename, width=40, font=("Arial", 10)).grid(row=0, column=1,
                                                                                                     padx=5, pady=5)

        # 进度条
        progress_frame = tk.LabelFrame(left_frame, text="处理进度", font=("Arial", 12, "bold"),
                                       padx=20, pady=15, bg="#ffffff")
        progress_frame.pack(fill="x", pady=(0, 20), padx=15)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=100, mode='determinate')
        self.progress_bar.pack(fill="x", pady=10)

        self.status_label = tk.Label(progress_frame, text="就绪", font=("Arial", 10), bg="#ffffff",
                                     fg=self.neutral_color)
        self.status_label.pack(anchor="w")

        # 处理按钮
        process_frame = tk.Frame(left_frame, bg="#ffffff")
        process_frame.pack(pady=10, padx=15)

        run_btn = tk.Button(process_frame, text="运行预测", command=self.run_prediction,
                            bg=self.tertiary_color, fg="white", font=("Arial", 12, "bold"),
                            relief="flat", padx=20, pady=8, cursor="hand2")
        run_btn.pack()

        # 右侧面板 - 结果展示
        right_frame = tk.Frame(content_frame, bg="#ffffff", relief="solid", bd=1)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # 创建标签页
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill="both", expand=True)

        # 结果文本标签页
        results_text_frame = tk.Frame(notebook, bg="#ffffff")
        notebook.add(results_text_frame, text="结果报告")

        self.results_text = tk.Text(results_text_frame, height=15, font=("Arial", 10), wrap=tk.WORD)
        self.results_text.pack(fill="both", expand=True, pady=15, padx=15)
        scrollbar = tk.Scrollbar(self.results_text, command=self.results_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.results_text.config(yscrollcommand=scrollbar.set)

        # 结果表格标签页
        results_table_frame = tk.Frame(notebook, bg="#ffffff")
        notebook.add(results_table_frame, text="预测详情")

        # 创建表格
        columns = ["样本ID", "Actual", "Final_Predicted","DNN_Predicted", "DNN_Predicted_Probability",
                   "CNN_Predicted", "CNN_Predicted_Probability",
                   "MLP_Predicted", "MLP_Predicted_Probability"]

        self.results_tree = ttk.Treeview(results_table_frame, columns=columns, show="headings", height=10)

        for col in columns:
            # 设置列标题
            if col == "Actual":
                self.results_tree.heading(col, text="实际类别")
            elif col.endswith("_Predicted"):
                self.results_tree.heading(col, text=f"{col.split('_')[0]}预测结果")
            elif col.endswith("_Probability"):
                self.results_tree.heading(col, text=f"{col.split('_')[0]}预测概率")
            else:
                self.results_tree.heading(col, text=col)

            # 设置列宽
            width = 120 if "Probability" in col else 100
            self.results_tree.column(col, width=width, anchor="center")

        self.results_tree.pack(fill="both", expand=True, pady=15, padx=15)

        # 添加滚动条
        tree_scroll_y = tk.Scrollbar(results_table_frame, orient="vertical", command=self.results_tree.yview)
        tree_scroll_y.pack(side="right", fill="y")
        self.results_tree.configure(yscrollcommand=tree_scroll_y.set)

        tree_scroll_x = tk.Scrollbar(results_table_frame, orient="horizontal", command=self.results_tree.xview)
        tree_scroll_x.pack(side="bottom", fill="x")
        self.results_tree.configure(xscrollcommand=tree_scroll_x.set)

        # 特征重要性标签页
        feature_frame = tk.Frame(notebook, bg="#ffffff")
        notebook.add(feature_frame, text="特征重要性")

        self.feature_fig = Figure(figsize=(6, 4), dpi=100)
        self.feature_canvas = FigureCanvasTkAgg(self.feature_fig, master=feature_frame)
        self.feature_canvas.get_tk_widget().pack(fill="both", expand=True, pady=15, padx=15)

        # 底部按钮区域
        bottom_frame = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        bottom_frame.pack(fill="x", side="bottom")

        save_btn = tk.Button(bottom_frame, text="保存结果", command=self.save_results,
                             bg=self.primary_color, fg="white", font=("Arial", 10),
                             relief="flat", padx=15, pady=5, cursor="hand2")
        save_btn.pack(side="right", padx=20)

    def browse_positive(self):
        filename = filedialog.askopenfilename(title="选择正样本数据集",
                                              filetypes=(
                                              ("TXT文件", "*.txt"), ("FASTA文件", "*.fasta"), ("所有文件", "*.*")))
        if filename:
            self.positive_file = filename
            self.positive_entry.delete(0, tk.END)
            self.positive_entry.insert(0, filename)

    def browse_negative(self):
        filename = filedialog.askopenfilename(title="选择负样本数据集",
                                              filetypes=(
                                              ("TXT文件", "*.txt"), ("FASTA文件", "*.fasta"), ("所有文件", "*.*")))
        if filename:
            self.negative_file = filename
            self.negative_entry.delete(0, tk.END)
            self.negative_entry.insert(0, filename)

    def update_progress(self, value, status):
        self.progress_var.set(value)
        self.status_label.config(text=status)
        self.root.update_idletasks()

    def run_prediction(self):
        if not self.positive_file or not self.negative_file:
            messagebox.showerror("错误", "请选择正负数据集")
            return

        # 清空之前的结果
        self.results_text.delete(1.0, tk.END)
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        try:
            # 加载数据集
            self.update_progress(10, "正在加载数据集...")
            self.results_text.insert(tk.END, "正在加载数据集...\n")

            # 模拟数据加载，实际使用时请替换为真实数据加载代码
            # pos_data = pd.read_csv(self.positive_file)
            # neg_data = pd.read_csv(self.negative_file)

            # ==================== 1. ESM2特征提取 ====================
            self.update_progress(20, "正在提取ESM2特征...")
            self.results_text.insert(tk.END, "正在提取ESM2特征...\n")

            esm_extractor = ESM2FeatureExtractor()
            pos_esm, pos_ids = esm_extractor.extract_features(self.positive_file)
            neg_esm, neg_ids = esm_extractor.extract_features(self.negative_file)

            # 保存ESM2特征到CSV
            pos_esm_df = pd.DataFrame(pos_esm)
            pos_esm_df.to_csv("GUI/pos_esm_features.csv", index=False)
            neg_esm_df = pd.DataFrame(neg_esm)
            neg_esm_df.to_csv("GUI/neg_esm_features.csv", index=False)

            # 合并正负样本用于降维
            X_esm = np.vstack([pos_esm, neg_esm])
            y = np.array([1] * len(pos_esm) + [0] * len(neg_esm))

            # ==================== 2. ESM2特征降维 ====================
            self.update_progress(35, "正在对ESM2特征降维...")
            self.results_text.insert(tk.END, "正在对ESM2特征降维...\n")

            pos_esm_reduced, neg_esm_reduced = FeatureReducer()  # 降维到300维


            pos_esm_reduced.to_csv("GUI/pos_esm_ANOVA_features2.csv", index=False)
            neg_esm_reduced.to_csv("GUI/neg_esm_ANOVA_features2.csv", index=False)

            # ==================== 3. One-Hot特征提取 ====================
            self.update_progress(50, "正在提取One-Hot特征...")
            self.results_text.insert(tk.END, "正在提取One-Hot特征...\n")

            onehot_encoder = OneHotEncoderProtein()
            pos_onehot, _ = onehot_encoder.process_file(self.positive_file)
            neg_onehot, _ = onehot_encoder.process_file(self.negative_file)

            pos_onehot_df = pd.DataFrame(pos_onehot)
            pos_onehot_df.to_csv("GUI/pos_onehot_features.csv", index=False)
            neg_onehot_df = pd.DataFrame(neg_onehot)
            neg_onehot_df.to_csv("GUI/neg_onehot_features.csv", index=False)



            # ==================== 4. 特征合并 ====================
            self.update_progress(60, "正在合并特征...")
            self.results_text.insert(tk.END, "正在合并特征...\n")

            # 确保样本数量一致
            assert len(pos_esm_reduced) == len(pos_onehot)
            assert len(neg_esm_reduced) == len(neg_onehot)

            positive_data_path = 'GUI/pos_esm_ANOVA_features2.csv'
            negative_data_path = 'GUI/neg_esm_ANOVA_features2.csv'

            # Read positive and negative datasets
            positive_data1 = pd.read_csv(positive_data_path)
            negative_data1 = pd.read_csv(negative_data_path)

            # 水平拼接降维后的ESM2和One-Hot特征
            pos_features = np.hstack([pos_onehot, positive_data1])
            neg_features = np.hstack([neg_onehot, negative_data1])

            pos_features_df = pd.DataFrame(pos_features)
            pos_features_df.to_csv("GUI/pos_combined_features.csv", index=False)
            neg_features_df = pd.DataFrame(neg_features)
            neg_features_df.to_csv("GUI/neg_combined_features.csv", index=False)

            # ==================== 5. 准备最终数据集 ====================
            X = np.vstack([pos_features, neg_features])
            y = np.array([1] * len(pos_features) + [0] * len(neg_features))

            # ==================== 6. 分类 ====================
            self.update_progress(85, "正在运行分类器...")
            self.results_text.insert(tk.END, "正在运行分类器...\n")

            # 调用分类器函数
            result_df1, metrics_df1 = classify_model_with_preprocessed_data_DNN()
            result_df2, metrics_df2 = classify_model_with_preprocessed_data_CNN()
            result_df3, metrics_df3 = classify_model_with_preprocessed_data_MLP()

            # ==================== 显示单个分类器指标 ====================
            self.results_text.insert(tk.END, "=" * 50 + "\n")
            self.results_text.insert(tk.END, "各分类器性能指标:\n\n")

            # 显示DNN分类器指标
            self.results_text.insert(tk.END, "DNN-Attention 模型结果:\n")
            self.display_single_classifier_metrics(metrics_df1)

            # 显示CNN分类器指标
            self.results_text.insert(tk.END, "\nCNN-Attention 模型结果:\n")
            self.display_single_classifier_metrics(metrics_df2)

            # 显示MLP分类器指标
            self.results_text.insert(tk.END, "\nMLP-Attention 模型结果:\n")
            self.display_single_classifier_metrics(metrics_df3)

            # ==================== 多数类投票 ====================

            # 提取每个模型的预测结果和预测概率
            all_predictions = [result_df1['Predicted'], result_df2['Predicted'], result_df3['Predicted']]
            all_probs = [result_df1['Predicted_Probability'], result_df2['Predicted_Probability'],
                         result_df3['Predicted_Probability']]

            # 执行多数投票
            vote_results = []
            for i in range(len(y)):
                # 统计正类预测的数量
                positive_votes = sum([p[i] for p in all_predictions])
                # 如果有两个或以上的分类器预测为正，则最终结果为正
                final_prediction = 1 if positive_votes >= 2 else 0
                vote_results.append(final_prediction)

            # 计算多数投票的性能指标
            from sklearn.metrics import recall_score, accuracy_score, matthews_corrcoef, roc_auc_score

            vote_sn = recall_score(y, vote_results)
            vote_sp = specificity_score(y, vote_results)  # 请确保已定义specificity_score
            vote_acc = accuracy_score(y, vote_results)
            vote_mcc = matthews_corrcoef(y, vote_results)
            vote_auc = roc_auc_score(y, vote_results)

            # 打印多数投票的性能指标
            self.results_text.insert(tk.END, "=" * 50 + "\n")
            self.results_text.insert(tk.END, "多数投票集成模型结果:\n")
            self.results_text.insert(tk.END, f"灵敏度 (Sn): {vote_sn:.4f}\n")
            self.results_text.insert(tk.END, f"特异度 (Sp): {vote_sp:.4f}\n")
            self.results_text.insert(tk.END, f"准确率 (ACC): {vote_acc:.4f}\n")
            self.results_text.insert(tk.END, f"马修斯相关系数 (MCC): {vote_mcc:.4f}\n")
            self.results_text.insert(tk.END, f"曲线下面积 (AUC): {vote_auc:.4f}\n\n")

            # ==================== 显示结果 ====================

            # 在表格中显示预测结果
            self.display_results_in_table(
                sample_ids=pos_ids + neg_ids,  # 添加样本ID
                Actual=y,
                Final_Predicted=vote_results,  # 添加最终的多数投票预测结果
                DNN_Predicted=all_predictions[0],
                DNN_Predicted_Probability=all_probs[0],
                CNN_Predicted=all_predictions[1],
                CNN_Predicted_Probability=all_probs[1],
                MLP_Predicted=all_predictions[2],
                MLP_Predicted_Probability=all_probs[2]
            )

            # 构建最终结果DataFrame
            self.results_df = pd.DataFrame({
                'Actual': y,
                'Final_Predicted': vote_results,
                'DNN_Predicted': all_predictions[0],
                'DNN_Predicted_Probability': all_probs[0],
                'CNN_Predicted': all_predictions[1],
                'CNN_Predicted_Probability': all_probs[1],
                'MLP_Predicted': all_predictions[2],
                'MLP_Predicted_Probability': all_probs[2]

            })

            # 保存最终的结果
            self.results_df.to_csv("final_predictions.csv", index=False)

            # 更新结果文本
            self.results_text.insert(tk.END, f"已生成{len(self.results_df)}条预测结果\n")
            self.results_text.insert(tk.END, "请点击底部的'保存结果'按钮保存数据\n\n")

            # 更新进度条
            self.update_progress(100, "预测完成")



        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {str(e)}")
            self.results_text.insert(tk.END, f"\n错误: {str(e)}\n")
            self.update_progress(0, "处理失败")


    def save_results(self):
        if not hasattr(self, 'results_df') or self.results_df is None:
            messagebox.showerror("错误", "没有结果可保存，请先运行预测")
            return

        filename = self.output_filename.get()
        if not filename.endswith('.csv'):
            filename += '.csv'

        try:
            save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                     initialfile=filename,
                                                     filetypes=(("CSV文件", "*.csv"), ("所有文件", "*.*")))
            if save_path:
                self.results_df.to_csv(save_path, index=False)
                messagebox.showinfo("成功", f"结果已保存到 {save_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

    def display_results_in_table(self,  sample_ids, Actual,
                                 Final_Predicted,DNN_Predicted, DNN_Predicted_Probability,
                                 CNN_Predicted, CNN_Predicted_Probability,
                                 MLP_Predicted, MLP_Predicted_Probability):
        # 清空表格
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # 插入数据
        for i in range(len(Actual)):
            self.results_tree.insert("", tk.END, values=(
                sample_ids[i],  # 添加样本ID
                Actual[i],
                Final_Predicted[i],
                DNN_Predicted[i],
                f"{DNN_Predicted_Probability[i]:.4f}",  # 格式化概率值
                CNN_Predicted[i],
                f"{CNN_Predicted_Probability[i]:.4f}",
                MLP_Predicted[i],
                f"{MLP_Predicted_Probability[i]:.4f}"
            ))

    def display_single_classifier_metrics(self, metrics_df):
        """显示单个分类器的性能指标"""
        # 从DataFrame中提取指标值
        sn = metrics_df['Sensitivity (Sn)'].values[0]
        sp = metrics_df['Specificity (Sp)'].values[0]
        acc = metrics_df['Accuracy (Acc)'].values[0]
        mcc = metrics_df['Matthews Correlation Coefficient (MCC)'].values[0]
        auc = metrics_df['Area Under the ROC Curve (AUC)'].values[0]

        # 显示指标
        self.results_text.insert(tk.END, f"灵敏度 (Sn): {sn:.4f}\n")
        self.results_text.insert(tk.END, f"特异度 (Sp): {sp:.4f}\n")
        self.results_text.insert(tk.END, f"准确率 (ACC): {acc:.4f}\n")
        self.results_text.insert(tk.END, f"马修斯相关系数 (MCC): {mcc:.4f}\n")
        self.results_text.insert(tk.END, f"曲线下面积 (AUC): {auc:.4f}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()