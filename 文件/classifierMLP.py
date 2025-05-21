import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, roc_auc_score
from MLP_Attention import MLP

def classify_model_with_preprocessed_data_MLP( batch_size=64):
    """
    使用给定的模型对已经处理好的特征数据进行分类，计算性能指标，并返回结果。

    参数:
    model_path: str, 训练好的模型路径
    X: numpy.ndarray, 特征数据
    y: numpy.ndarray, 标签数据
    batch_size: int, 批量大小，默认64

    返回:
    result_df: pandas DataFrame, 预测结果
    metrics_df: pandas DataFrame, 性能指标
    """
    # Device settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 直接加载整个模型

    model = MLP().to(device)
    model.load_state_dict(torch.load('MLP_ATT-esm+anova3.pth', map_location=device))
    model.eval()

    positive_data_path = 'GUI/pos_combined_features.csv'
    negative_data_path = 'GUI/neg_combined_features.csv'

    # Read positive and negative datasets
    positive_data = pd.read_csv(positive_data_path)
    negative_data = pd.read_csv(negative_data_path)

    # Convert to tensors
    X_positive = torch.tensor(positive_data.values, dtype=torch.float32)
    X_negative = torch.tensor(negative_data.values, dtype=torch.float32)

    # Create labels for positive and negative datasets
    y_positive = torch.ones(X_positive.size(0), dtype=torch.float32)
    y_negative = torch.zeros(X_negative.size(0), dtype=torch.float32)

    # Concatenate positive and negative data
    X_test = torch.cat((X_positive, X_negative), dim=0)
    y_test = torch.cat((y_positive, y_negative), dim=0)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    y_true = []
    y_pred = []
    y_pred_proba = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch).squeeze()
            preds = torch.round(outputs)  # Predicted labels
            y_true.extend(y_batch.tolist())
            y_pred.extend(preds.tolist())
            y_pred_proba.extend(outputs.tolist())

    # 保存预测结果和实际结果到 CSV 文件
    result_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Predicted_Probability': y_pred_proba
    })

    result_df.to_csv('GUI/0-MLP-ATT-esm+anova3结果.csv', index=False)

    # 计算性能指标
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tn_t, fp_t, fn_t, tp_t = torch.tensor(tn), torch.tensor(fp), torch.tensor(fn), torch.tensor(tp)
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    mcc = ((tp_t * tn_t) - (fp_t * fn_t)) / torch.sqrt((tp_t + fp_t) * (tp_t + fn_t) * (tn_t + fp_t) * (tn_t + fn_t))
    mcc = mcc.item() if mcc.numel() == 1 else mcc
    auc = roc_auc_score(y_true, y_pred_proba)

    # 创建包含指标名称和对应值的字典
    metrics_dict = {
        'Sensitivity (Sn)': [sn],
        'Specificity (Sp)': [sp],
        'Accuracy (Acc)': [acc],
        'Matthews Correlation Coefficient (MCC)': [mcc],
        'Area Under the ROC Curve (AUC)': [auc]
    }

    # 将字典转换为DataFrame
    metrics_df = pd.DataFrame(metrics_dict)

    # 将DataFrame保存为CSV文件
    metrics_df.to_csv('GUI/0-MLP-ATT-esm+anova3结果指标.csv', index=False)

    # 返回结果和性能指标
    return result_df, metrics_df
