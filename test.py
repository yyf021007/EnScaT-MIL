import sys
import os
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import json
import math
from transformers import BertTokenizer, BertModel  # type: ignore
from dataset2 import FenziDataset, collate_single_task
from utils import calculate_metrics, validate2, validate_single_task, FocalLoss
from nystrom_attention import NystromAttention

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# 单任务配置
TASK_CONFIG = {
    'name': 'p53',
    'type': 'multiclass',
    'label_column': 'p53',
    'num_classes': 2
}

# =========================
# 统计学辅助函数（与CLAM版一致）
# =========================
def _safe_div(n, d):
    return n / d if d != 0 else 0.0

def wilson_ci(p: float, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * math.sqrt((p*(1-p)/n) + (z**2)/(4*n**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))

def compute_binary_metrics_from_cm(tp, tn, fp, fn):
    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    ppv  = _safe_div(tp, tp + fp)
    npv  = _safe_div(tn, tn + fn)
    return {"sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv}

def compute_binary_metrics_ci(tp, tn, fp, fn, z: float = 1.96):
    sens_n = tp + fn
    spec_n = tn + fp
    ppv_n  = tp + fp
    npv_n  = tn + fn
    sens_ci = wilson_ci(_safe_div(tp, sens_n), sens_n, z) if sens_n > 0 else (0.0, 0.0)
    spec_ci = wilson_ci(_safe_div(tn, spec_n), spec_n, z) if spec_n > 0 else (0.0, 0.0)
    ppv_ci  = wilson_ci(_safe_div(tp, ppv_n),  ppv_n,  z) if ppv_n  > 0 else (0.0, 0.0)
    npv_ci  = wilson_ci(_safe_div(tn, npv_n),  npv_n,  z) if npv_n  > 0 else (0.0, 0.0)
    return {"sensitivity_ci": sens_ci, "specificity_ci": spec_ci, "ppv_ci": ppv_ci, "npv_ci": npv_ci}

def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 2000, seed: int = 12, alpha: float = 0.95):
    rng = np.random.default_rng(seed)
    aucs = []
    base_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]; ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    if len(aucs) == 0:
        return base_auc, float("nan"), float("nan")
    low = np.quantile(aucs, (1 - alpha)/2)
    high = np.quantile(aucs, 1 - (1 - alpha)/2)
    return base_auc, float(low), float(high)

def evaluate_predictions(y_true: np.ndarray, y_prob_pos: np.ndarray, y_pred: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary') if len(np.unique(y_true)) > 1 else 0.0
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4: tn, fp, fn, tp = cm.ravel()
    else:
        tn = cm[0, 0] if y_true.max() == 0 else 0; fp = fn = tp = 0
    basic = compute_binary_metrics_from_cm(tp, tn, fp, fn)
    cis = compute_binary_metrics_ci(tp, tn, fp, fn)
    auc_base, auc_low, auc_high = bootstrap_auc_ci(y_true, y_prob_pos)
    return {
        "accuracy": acc, "f1_score": f1, "precision": prec, "recall": rec,
        "confusion_matrix": cm.tolist(),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "sensitivity": basic["sensitivity"], "specificity": basic["specificity"],
        "ppv": basic["ppv"], "npv": basic["npv"],
        "sensitivity_ci": cis["sensitivity_ci"], "specificity_ci": cis["specificity_ci"],
        "ppv_ci": cis["ppv_ci"], "npv_ci": cis["npv_ci"],
        "auc": auc_base, "auc_ci": (auc_low, auc_high)
    }

# =========================
# TransMIL 模型实现
# =========================
class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim//8,
            heads=8,
            num_landmarks=dim//2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class CSVEncoder:
    def __init__(self, model_name='/mnt/gemlab_data_2/User_database/yangyefeng/bert/bert_uncase/', max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # 推理模式
        self.max_length = max_length
        self.dense = torch.nn.Linear(768, 512)
        self.relu = torch.nn.ReLU()

    @torch.no_grad()
    def encode_line(self, line):
        inputs = self.tokenizer(line, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        outputs = self.model(**inputs)  # 默认CPU即可
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected_embedding = self.relu(self.dense(cls_embedding))
        return projected_embedding.detach().numpy().reshape(1, -1)

    @torch.no_grad()
    def encode_csv(self, csv_file, line_number):
        df = pd.read_csv(csv_file)
        if line_number >= len(df):
            raise ValueError("Line number exceeds the number of lines in the CSV file.")
        line = df.iloc[line_number].astype(str).str.cat(sep=' ')
        return self.encode_line(line)

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.encoder = CSVEncoder()
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, x):
        h = x['features'].float()
        h = self._fc1(h)  # [B, n, 512]

        # pad 成近似方阵
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        # Trans + PPEG + Trans
        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)

        # 读取一行CSV文本嵌入拼接
        encoded_vector = self.encoder.encode_csv('/data/yyf/Net/script/prompt.csv', 0)
        encoded_vector = torch.tensor(encoded_vector, dtype=torch.float32, device=h.device)
        encoded_vector = encoded_vector.unsqueeze(1)  # [1,1,512] -> broadcast到B时需按B复制
        if encoded_vector.shape[0] != B:
            encoded_vector = encoded_vector.repeat(B, 1, 1)
        h = torch.cat((h, encoded_vector), dim=1)

        # 取cls_token
        h = self.norm(h)[:, 0]
        return h

# 单任务 Slide Encoder（使用 TransMIL）
class SingleTaskSlideEncoder(nn.Module):
    def __init__(self, input_feature_dim=512):
        super().__init__()
        self.feature_encoder = TransMIL(n_classes=2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_feature_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, TASK_CONFIG['num_classes'])
        )
    def forward(self, x):
        f = self.feature_encoder(x)
        logits = self.classifier(f)
        return logits

# =========================
# 集成评测（含CI）
# =========================
def test_ensemble_models():
    """测试五折交叉验证的集成模型（含 Sens/Spec/PPV/NPV + Wilson CI 与 AUC bootstrap CI）"""
    # model_paths = [
    #     '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_musk/best_p53_pfmil_model_fold1_0425.pth',
    #     '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_musk/best_p53_pfmil_model_fold2_0425.pth',
    #     '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_musk/best_p53_pfmil_model_fold3_0425.pth',
    #     '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_musk/best_p53_pfmil_model_fold4_0425.pth',
    #     '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_musk/best_p53_pfmil_model_fold5_0425.pth'
    # ]

    model_paths = [
        '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_resnet/best_p53_pfmil_model_fold1_0425.pth',
        '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_resnet/best_p53_pfmil_model_fold2_0425.pth',
        '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_resnet/best_p53_pfmil_model_fold3_0425.pth',
        '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_resnet/best_p53_pfmil_model_fold4_0425.pth',
        '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_resnet/best_p53_pfmil_model_fold5_0425.pth'
    ]

    out_dir = '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_musk'
    os.makedirs(out_dir, exist_ok=True)

    # 预加载模型（你的原实现如此；如显存紧张可改为逐模型加载）
    models = []
    for model_path in model_paths:
        model = SingleTaskSlideEncoder(input_feature_dim=512).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        print(f"Loaded model: {model_path}")

    test_df = pd.read_csv('/data/yyf/Net/dataset/P53/p53_710_musk.csv').assign(dataset='dataset')
    # feats_paths = {'dataset': '/mnt/gemlab_data_3/User_database/yangyefeng/medical_image/process/p53_test_710/'}
    feats_paths = {'dataset': '/mnt/gemlab_data_3/User_database/yangyefeng/P53_musk/test/merge'}


    # test_df = pd.read_excel('/data/yyf/Net/dataset/P53/test_waibu5.xlsx').assign(dataset='dataset')
    # # feats_paths = {'dataset': '/mnt/gemlab_data_3/User_database/yangyefeng/medical_image/process/p53_test_710/'}
    # feats_paths = {'dataset': '/mnt/gemlab_data_3/User_database/yangyefeng/P53_2026/waibu_val/merge'}
    # # 

    # 
    test_dataset = FenziDataset(feats_paths, test_df, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, pin_memory=True)

    all_probs, all_preds, all_labels, all_slide_ids = [], [], [], []

    with torch.no_grad():
        for idx, (features, labels) in enumerate(tqdm(test_loader, desc="Testing (Ensemble-PFMIL)")):
            features = {'features': features.float().to(device)}
            labels = labels.to(device)
            slide_id = test_df.iloc[idx]['slide_id']
            all_slide_ids.append(slide_id)

            batch_probs = []
            for model in models:
                outputs = model(features)
                probs = torch.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())

            ensemble_probs = np.mean(batch_probs, axis=0)
            ensemble_pred = int(np.argmax(ensemble_probs, axis=1)[0])

            all_probs.append(ensemble_probs[0])
            all_preds.append(ensemble_pred)
            all_labels.append(int(labels.cpu().numpy()[0]))

            del features, labels, outputs
            torch.cuda.empty_cache()

    all_probs = np.array(all_probs); all_preds = np.array(all_preds); all_labels = np.array(all_labels)

    # 统一评估（含 CI）
    eval_pack = evaluate_predictions(all_labels, all_probs[:, 1], all_preds)

    # 打印
    print("\n" + "="*50)
    print("五折交叉验证集成模型测试结果 (PFMIL)")
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)
    print(f"Loaded model: {model_path}")

    test_df = pd.read_csv('/data/yyf/Net/dataset/P53/p53_710_musk.csv').assign(dataset='dataset')
    # feats_paths = {'dataset': '/mnt/gemlab_data_3/User_database/yangyefeng/medical_image/process/p53_test_710/'}
    feats_paths = {'dataset': '/mnt/gemlab_data_3/User_database/yangyefeng/P53_musk/test/merge'}


    # test_df = pd.read_excel('/data/yyf/Net/dataset/P53/test_waibu5.xlsx').assign(dataset='dataset')
    # # feats_paths = {'dataset': '/mnt/gemlab_data_3/User_database/yangyefeng/medical_image/process/p53_test_710/'}
    # feats_paths = {'dataset': '/mnt/gemlab_data_3/User_database/yangyefeng/P53_2026/waibu_val/merge'}
    # 

    # 
    test_dataset = FenziDataset(feats_paths, test_df, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, pin_memory=True)

    all_probs, all_preds, all_labels, all_slide_ids = [], [], [], []

    with torch.no_grad():
        for idx, (features, labels) in enumerate(tqdm(test_loader, desc="Testing (Ensemble-PFMIL)")):
            features = {'features': features.float().to(device)}
            labels = labels.to(device)
            slide_id = test_df.iloc[idx]['slide_id']
            all_slide_ids.append(slide_id)

            batch_probs = []
            for model in models:
                outputs = model(features)
                probs = torch.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())

            ensemble_probs = np.mean(batch_probs, axis=0)
            ensemble_pred = int(np.argmax(ensemble_probs, axis=1)[0])

            all_probs.append(ensemble_probs[0])
            all_preds.append(ensemble_pred)
            all_labels.append(int(labels.cpu().numpy()[0]))

            del features, labels, outputs
            torch.cuda.empty_cache()

    all_probs = np.array(all_probs); all_preds = np.array(all_preds); all_labels = np.array(all_labels)

    # 统一评估（含 CI）
    eval_pack = evaluate_predictions(all_labels, all_probs[:, 1], all_preds)

    # 打印
    print("\n" + "="*50)
    print("五折交叉验证集成模型测试结果 (PFMIL)")
    print("="*50)
    print(f"Accuracy: {eval_pack['accuracy']:.4f}")
    print(f"AUC: {eval_pack['auc']:.4f}  (95% CI: {eval_pack['auc_ci'][0]:.3f}–{eval_pack['auc_ci'][1]:.3f})")
    print(f"F1-Score: {eval_pack['f1_score']:.4f}")
    print(f"Sensitivity: {eval_pack['sensitivity']:.3f}  (95% CI: {eval_pack['sensitivity_ci'][0]:.3f}–{eval_pack['sensitivity_ci'][1]:.3f})")
    print(f"Specificity: {eval_pack['specificity']:.3f}  (95% CI: {eval_pack['specificity_ci'][0]:.3f}–{eval_pack['specificity_ci'][1]:.3f})")
    print(f"PPV: {eval_pack['ppv']:.3f}  (95% CI: {eval_pack['ppv_ci'][0]:.3f}–{eval_pack['ppv_ci'][1]:.3f})")
    print(f"NPV: {eval_pack['npv']:.3f}  (95% CI: {eval_pack['npv_ci'][0]:.3f}–{eval_pack['npv_ci'][1]:.3f})")
    print(f"\nConfusion Matrix:\n{np.array(eval_pack['confusion_matrix'])}")

    # 详细CSV（逐样本 + 总体指标冗余）
    detailed_rows = []
    for i in range(len(all_slide_ids)):
        y = int(all_labels[i]); yhat = int(all_preds[i])
        prob0, prob1 = float(all_probs[i, 0]), float(all_probs[i, 1])
        TP = 1 if (y==1 and yhat==1) else 0
        TN = 1 if (y==0 and yhat==0) else 0
        FP = 1 if (y==0 and yhat==1) else 0
        FN = 1 if (y==1 and yhat==0) else 0
        detailed_rows.append({
            'slide_id': all_slide_ids[i],
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'predicted_label': yhat, 'true_label': y,
            'prediction_probability_class0': prob0,
            'prediction_probability_class1': prob1,
            'accuracy_overall': eval_pack['accuracy'],
            'auc_overall': eval_pack['auc'],
            'auc_ci_low': eval_pack['auc_ci'][0],
            'auc_ci_high': eval_pack['auc_ci'][1],
            'f1_overall': eval_pack['f1_score'],
            'precision_overall': eval_pack['precision'],
            'recall_overall': eval_pack['recall'],
            'sensitivity_overall': eval_pack['sensitivity'],
            'sensitivity_ci_low': eval_pack['sensitivity_ci'][0],
            'sensitivity_ci_high': eval_pack['sensitivity_ci'][1],
            'specificity_overall': eval_pack['specificity'],
            'specificity_ci_low': eval_pack['specificity_ci'][0],
            'specificity_ci_high': eval_pack['specificity_ci'][1],
            'ppv_overall': eval_pack['ppv'],
            'ppv_ci_low': eval_pack['ppv_ci'][0],
            'ppv_ci_high': eval_pack['ppv_ci'][1],
            'npv_overall': eval_pack['npv'],
            'npv_ci_low': eval_pack['npv_ci'][0],
            'npv_ci_high': eval_pack['npv_ci'][1],
        })
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_musk/detailed_test_results_ensemble.csv'
    os.makedirs(os.path.dirname(detailed_path), exist_ok=True)
    detailed_df.to_csv(detailed_path, index=False)
    print(f"\n详细测试结果已保存到: {detailed_path}")

    # 汇总 JSON（含 ROC 数据）
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
    summary = {
        'model_type': 'PFMIL',
        'total_samples': int(len(all_slide_ids)),
        **{k: v for k, v in eval_pack.items() if k != 'confusion_matrix'},
        'confusion_matrix': eval_pack['confusion_matrix'],
        'roc_data': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}
    }
    json_path = '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_musk/test_results_ensemble_pfmil.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=4)

    # ROC 图
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {eval_pack["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - PFMIL (Ensemble)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_path = '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_musk/roc_curve_ensemble_pfmil.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight'); plt.close()

    # 错误样本
    error_df = detailed_df[detailed_df['predicted_label'] != detailed_df['true_label']]
    err_path = '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_musk/error_samples_ensemble.csv'
    error_df.to_csv(err_path, index=False)

    print(f"\n结果文件已保存到:")
    print(f"- 详细测试结果: {detailed_path}")
    print(f"- 汇总评估指标: {json_path}")
    print(f"- ROC曲线: {roc_path}")
    print(f"- 预测错误样本: {err_path}")

# =========================
# 各折评测（含CI）
# =========================
def test_individual_folds():
    """分别测试每个fold的模型性能并保存详细结果（含CI）"""
    feats_paths = {'dataset': '/mnt/gemlab_data_3/User_database/yangyefeng/P53_musk/test/merge'}
    out_dir = '/data/yyf/Net/result/p53_2026_0425(2)/pfmil_musk'
    os.makedirs(out_dir, exist_ok=True)

    all_fold_results = {}

    for fold in range(1, 6):
        print(f"\n{'='*50}")
        print(f"Testing Fold {fold}")
        print('='*50)

        # model_path = f'/data/yyf/Net/result/p53_2026_0425(2)/pfmil_musk/best_p53_pfmil_model_fold{fold}_0425.pth'
        # model_path = f'/data/yyf/Net/result/p53/pfmil/best_p53_pfmil_model_fold{fold}_903.pth'
        model_path = f'/data/yyf/Net/result/p53_2026_0425(2)/pfmil_resnet/best_p53_pfmil_model_fold{fold}_0425.pth'
        model = SingleTaskSlideEncoder(input_feature_dim=512).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"  Loaded model from: {model_path}")
        model.eval()

        test_df = pd.read_csv('/data/yyf/Net/dataset/P53/p53_710_musk.csv').assign(dataset='dataset')
        # test_df = pd.read_excel('/data/yyf/Net/dataset/P53/test_waibu5.xlsx').assign(dataset='dataset')

        test_dataset = FenziDataset(feats_paths, test_df, 'test')
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, pin_memory=True)

        all_probs, all_preds, all_labels, all_slide_ids = [], [], [], []

        with torch.no_grad():
            for idx, (features, labels) in enumerate(tqdm(test_loader, desc=f"Testing Fold {fold}")):
                features = {'features': features.float().to(device)}
                labels = labels.to(device)
                slide_id = test_df.iloc[idx]['slide_id']
                all_slide_ids.append(slide_id)

                outputs = model(features)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_probs.append(probs.cpu().numpy()[0])
                all_preds.append(int(preds.cpu().numpy()[0]))
                all_labels.append(int(labels.cpu().numpy()[0]))

                del features, labels, outputs
                torch.cuda.empty_cache()

        all_probs = np.array(all_probs); all_preds = np.array(all_preds); all_labels = np.array(all_labels)

        # 统一评估（含 CI）
        eval_pack = evaluate_predictions(all_labels, all_probs[:, 1], all_preds)

        # 保存该fold的详细结果
        detailed_rows = []
        for i in range(len(all_slide_ids)):
            y = int(all_labels[i]); yhat = int(all_preds[i])
            prob0, prob1 = float(all_probs[i, 0]), float(all_probs[i, 1])
            TP = 1 if (y==1 and yhat==1) else 0
            TN = 1 if (y==0 and yhat==0) else 0
            FP = 1 if (y==0 and yhat==1) else 0
            FN = 1 if (y==1 and yhat==0) else 0
            detailed_rows.append({
                'slide_id': all_slide_ids[i],
                'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
                'predicted_label': yhat, 'true_label': y,
                'prediction_probability_class0': prob0,
                'prediction_probability_class1': prob1,
                'accuracy_overall': eval_pack['accuracy'],
                'auc_overall': eval_pack['auc'],
                'auc_ci_low': eval_pack['auc_ci'][0],
                'auc_ci_high': eval_pack['auc_ci'][1],
                'f1_overall': eval_pack['f1_score'],
                'precision_overall': eval_pack['precision'],
                'recall_overall': eval_pack['recall'],
                'sensitivity_overall': eval_pack['sensitivity'],
                'sensitivity_ci_low': eval_pack['sensitivity_ci'][0],
                'sensitivity_ci_high': eval_pack['sensitivity_ci'][1],
                'specificity_overall': eval_pack['specificity'],
                'specificity_ci_low': eval_pack['specificity_ci'][0],
                'specificity_ci_high': eval_pack['specificity_ci'][1],
                'ppv_overall': eval_pack['ppv'],
                'ppv_ci_low': eval_pack['ppv_ci'][0],
                'ppv_ci_high': eval_pack['ppv_ci'][1],
                'npv_overall': eval_pack['npv'],
                'npv_ci_low': eval_pack['npv_ci'][0],
                'npv_ci_high': eval_pack['npv_ci'][1],
            })
        pd.DataFrame(detailed_rows).to_csv(f'{out_dir}/detailed_test_results_fold{fold}.csv', index=False)

        # 打印
        print(f"Fold {fold} Results:")
        print(f"  Accuracy: {eval_pack['accuracy']:.4f}")
        print(f"  AUC: {eval_pack['auc']:.4f}  (95% CI: {eval_pack['auc_ci'][0]:.3f}–{eval_pack['auc_ci'][1]:.3f})")
        print(f"  F1-Score: {eval_pack['f1_score']:.4f}")
        print(f"  Sensitivity: {eval_pack['sensitivity']:.3f}  (95% CI: {eval_pack['sensitivity_ci'][0]:.3f}–{eval_pack['sensitivity_ci'][1]:.3f})")
        print(f"  Specificity: {eval_pack['specificity']:.3f}  (95% CI: {eval_pack['specificity_ci'][0]:.3f}–{eval_pack['specificity_ci'][1]:.3f})")
        print(f"  PPV: {eval_pack['ppv']:.3f}  (95% CI: {eval_pack['ppv_ci'][0]:.3f}–{eval_pack['ppv_ci'][1]:.3f})")
        print(f"  NPV: {eval_pack['npv']:.3f}  (95% CI: {eval_pack['npv_ci'][0]:.3f}–{eval_pack['npv_ci'][1]:.3f})")
        print(f"  Confusion Matrix:\n  {np.array(eval_pack['confusion_matrix'])[0]}\n  {np.array(eval_pack['confusion_matrix'])[1]}")

        # 汇总该fold
        all_fold_results[f'fold_{fold}'] = {
            'accuracy': float(eval_pack['accuracy']),
            'auc': float(eval_pack['auc']),
            'auc_ci_low': float(eval_pack['auc_ci'][0]),
            'auc_ci_high': float(eval_pack['auc_ci'][1]),
            'f1_score': float(eval_pack['f1_score']),
            'precision': float(eval_pack['precision']),
            'recall': float(eval_pack['recall']),
            'sensitivity': float(eval_pack['sensitivity']),
            'sensitivity_ci_low': float(eval_pack['sensitivity_ci'][0]),
            'sensitivity_ci_high': float(eval_pack['sensitivity_ci'][1]),
            'specificity': float(eval_pack['specificity']),
            'specificity_ci_low': float(eval_pack['specificity_ci'][0]),
            'specificity_ci_high': float(eval_pack['specificity_ci'][1]),
            'ppv': float(eval_pack['ppv']),
            'ppv_ci_low': float(eval_pack['ppv_ci'][0]),
            'ppv_ci_high': float(eval_pack['ppv_ci'][1]),
            'npv': float(eval_pack['npv']),
            'npv_ci_low': float(eval_pack['npv_ci'][0]),
            'npv_ci_high': float(eval_pack['npv_ci'][1]),
            'TP': int(eval_pack['TP']), 'TN': int(eval_pack['TN']),
            'FP': int(eval_pack['FP']), 'FN': int(eval_pack['FN']),
            'confusion_matrix': eval_pack['confusion_matrix']
        }

    # 折间均值±标准差
    metrics_for_avg = ['accuracy', 'auc', 'f1_score', 'precision', 'recall', 'sensitivity', 'specificity', 'ppv', 'npv']
    avg_results = {}
    print(f"\n{'='*50}")
    print("Overall Results (Mean ± Std)")
    print('='*50)
    for metric in metrics_for_avg:
        values = [all_fold_results[f'fold_{i}'][metric] for i in range(1, 6)]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        avg_results[f'{metric}_mean'] = mean_val
        avg_results[f'{metric}_std'] = std_val
        print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

    final_results = {
        'model_type': 'PFMIL',
        'individual_folds': all_fold_results,
        'average_results': avg_results
    }
    with open('/data/yyf/Net/result/p53_2026_0425(2)/pfmil_musk/test_results_all_folds_pfmil.json', 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"\n各fold详细结果已保存到:")
    print(f"- 汇总结果: /data/yyf/Net/result/P53_new_710/pfmil/test_results_all_folds_pfmil.json")
    print(f"- 各fold详细结果: /data/yyf/Net/result/P53_new_710/pfmil/detailed_test_results_fold[1-5].csv")

if __name__ == "__main__":
    print("开始测试PFMIL五折交叉验证模型...")
    os.makedirs('/data/yyf/Net/result/P53_new_710/pfmil2/', exist_ok=True)
    print("\n1. 测试集成模型（5个模型的平均）")
    test_ensemble_models()
    print("\n2. 分别测试每个fold的模型")
    test_individual_folds()
    print("\n测试完成！")
