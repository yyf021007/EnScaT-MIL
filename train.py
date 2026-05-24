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
import math
from transformers import BertTokenizer, BertModel # type: ignore
from dataset2 import FenziDataset, collate_single_task
from utils import calculate_metrics, validate2, validate_single_task, FocalLoss
from nystrom_attention import NystromAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 单任务配置
TASK_CONFIG = {
    'name': 'p53',
    'type': 'multiclass',
    'label_column': 'p53',
    'num_classes': 2
}

# TransMIL 模型实现
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


# class NystromAttention(nn.Module):
#     def __init__(self, dim, dim_head=64, heads=8, num_landmarks=256, pinv_iterations=6, residual=True, residual_conv_kernel=33, eps=1e-8, dropout=0.):
#         super().__init__()
#         self.eps = eps
#         inner_dim = dim_head * heads

#         self.num_landmarks = num_landmarks
#         self.pinv_iterations = pinv_iterations

#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         )

#         self.residual = residual
#         if residual:
#             kernel_size = residual_conv_kernel
#             padding = residual_conv_kernel // 2
#             self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

#     def forward(self, x, mask=None, return_attn=False):
#         b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

#         # pad so that sequence can be evenly divided into m landmarks
#         remainder = n % m
#         if remainder > 0:
#             padding = m - remainder
#             x = torch.nn.functional.pad(x, (0, 0, padding, 0), value=0)

#             if mask is not None:
#                 mask = torch.nn.functional.pad(mask, (padding, 0), value=False)

#         # derive query, keys, values
#         q, k, v = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), (q, k, v))

#         # set masked positions to 0 in queries, keys, values
#         if mask is not None:
#             mask = mask.unsqueeze(1).unsqueeze(-1)
#             q, k, v = map(lambda t: t * mask, (q, k, v))

#         q = q * self.scale

#         # select landmarks
#         l = math.ceil(n / m)
#         landmark_einops_eq = '... (n l) d -> ... n d'
#         q_landmarks = q.reshape(b, h, m, l, -1).sum(dim=-2)
#         k_landmarks = k.reshape(b, h, m, l, -1).sum(dim=-2)

#         # produce values
#         einops_eq = '... i d, ... j d -> ... i j'
#         sim1 = torch.einsum(einops_eq, q, k_landmarks)
#         sim2 = torch.einsum(einops_eq, q_landmarks, k_landmarks)
#         sim3 = torch.einsum(einops_eq, q_landmarks, k)

#         # calculate attention matrix
#         attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
#         attn2_inv = self.moore_penrose_iter_pinv(attn2, iters)

#         out = (attn1 @ attn2_inv) @ (attn3 @ v)

#         # add depth-wise conv residual of values
#         if self.residual:
#             out += self.res_conv(v)

#         # merge and combine heads
#         out = out.transpose(1, 2).reshape(b, n, -1)
#         out = self.to_out(out)
#         out = out[:, -n:]

#         if return_attn:
#             return out, attn1, attn2, attn3
#         return out

#     def moore_penrose_iter_pinv(self, x, iters=6):
#         device = x.device

#         abs_x = torch.abs(x)
#         col = abs_x.sum(dim=-1)
#         row = abs_x.sum(dim=-2)
#         z = x.transpose(-1, -2).contiguous()
#         z = z / (torch.max(col) * torch.max(row))

#         I = torch.eye(x.shape[-1], device=device)
#         I = I.unsqueeze(0).unsqueeze(0)

#         for _ in range(iters):
#             xz = x @ z
#             z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

#         return z

class CSVEncoder:
    def __init__(self, model_name='/mnt/gemlab_data_2/User_database/yangyefeng/bert/bert_uncase/', max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.max_length = max_length
        self.dense = torch.nn.Linear(768, 512)
        self.relu = torch.nn.ReLU()

    def encode_line(self, line):
        inputs = self.tokenizer(line, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected_embedding = self.relu(self.dense(cls_embedding))
        return projected_embedding.detach().numpy().reshape(1, -1)

    def encode_csv(self, csv_file, line_number):
        df = pd.read_csv(csv_file)
        if line_number >= len(df):
            raise ValueError("Line number exceeds the number of lines in the CSV file.")
        line = df.iloc[line_number].astype(str).str.cat(sep=' ')
        return self.encode_line(line)

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
        
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        # #---->PPEG
        # h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        # #---->Translayer x2
        # h = self.layer2(h) #[B, N, 512]

        # #---->cls_token
        # h = self.norm(h)[:,0]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        encoded_vector = self.encoder.encode_csv('/data/yyf/Net/script/prompt.csv',0)
        encoded_vector = torch.tensor(encoded_vector, dtype=torch.float32).to(h.device)  # Convert to tensor and move to the same device as features
        encoded_vector = encoded_vector.unsqueeze(1)
        h = torch.cat((h, encoded_vector), dim=1)  # Concatenate features and encoded vector
        # ---->cls_token
        h = self.norm(h)[:, 0]

        #---->predict
        # logits = self._fc2(h) #[B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return h


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
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


# 单任务 Slide Encoder（使用 TransMIL）
class SingleTaskSlideEncoder(nn.Module):
    def __init__(self, input_feature_dim=512):
        super().__init__()
        # 使用 TransMIL
        self.feature_encoder = TransMIL(n_classes=2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_feature_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, TASK_CONFIG['num_classes'])
        )
    
    def forward(self, x):
        # TransMIL 直接返回 logits
        f = self.feature_encoder(x)
        logits = self.classifier(f)
        return logits


# 五折交叉验证训练流程
def train_single_fold(fold_num):
    """训练单个fold"""
    print(f"\n========== Training Fold {fold_num} ==========")
    
    model = SingleTaskSlideEncoder(input_feature_dim=512).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-5,
        weight_decay=1e-4
    )
     
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 数据加载 - 使用对应的fold文件
    # csv_path = f'/data/yyf/Net/dataset/P53/p53_fold_{fold_num}.csv'
    csv_path = f'/data/yyf/Net/dataset/P53/fold_2026_0326_resnet/oversample/fold_{fold_num}.xlsx'
    
    # df = pd.read_csv(csv_path).assign(dataset='dataset')
    df = pd.read_excel(csv_path).assign(dataset='dataset')
    
    # feats_paths = {
    #     'dataset': '/mnt/gemlab_data_3/User_database/yangyefeng/medical_image/process/P53_merge/'
    # }

    feats_paths = {
        'dataset': '/mnt/gemlab_data_3/User_database/yangyefeng/P53_musk/merge'
    }

    

    # 使用修改后的单任务数据集
    train_dataset = FenziDataset(feats_paths, df, 'train')
    val_dataset = FenziDataset(feats_paths, df, 'val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
    )

    # 训练循环
    best_acc = 0.0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    # Early stopping
    patience = 10
    patience_counter = 0
    
    for epoch in range(30):
        model.train()
        epoch_loss = 0.0
        
        for features, labels in tqdm(train_loader, desc=f'Fold {fold_num} - Epoch {epoch+1}'):
            features = {'features': features.to(device)}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            
            # 过滤无效标签
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue
                
            loss = criterion(outputs[valid_mask], labels[valid_mask].long())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            epoch_loss += loss.item()

            # 显式释放内存
            del features, labels, outputs
            torch.cuda.empty_cache()

        # 验证
        val_loss, val_metrics = validate_single_task(model, val_loader, criterion, device, TASK_CONFIG)
        
        # 保存最佳模型
        current_acc = val_metrics['auc']
        scheduler.step(current_acc)
        
        if current_acc > best_acc:
            best_acc = current_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'/data/yyf/Net/result/p53_2026_0425(2)/pfmil_resnet/best_p53_pfmil_model_fold{fold_num}_0425.pth')
            print(f" --------- Fold {fold_num}: Save best model (AUC: {best_acc:.4f}) --------- ")
        else:
            patience_counter += 1
            
        torch.save(model.state_dict(), f'/data/yyf/Net/result/p53_2026_0425(2)/pfmil_resnet/last_p53_pfmil_model_fold{fold_num}_0425.pth')
        
        # 打印日志
        print(f"Fold {fold_num} - Epoch {epoch+1}:"
            f"  Train Loss: {epoch_loss/len(train_loader):.4f}"
            f"  Val Loss: {val_loss:.4f}"
            f"  Accuracy: {val_metrics['accuracy']:.4f}"
            f"  F1-Score: {val_metrics['f1']:.4f}"
            f"  AUC: {val_metrics['auc']:.4f}"
            f"  Best AUC: {best_acc:.4f}"
        )
        print(f"  Confusion Matrix:\n{val_metrics['confusion_matrix']}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered for Fold {fold_num}")
            break
    
    return best_acc


def train_all_folds():
    """训练所有5个fold"""
    results = {}
    
    # 创建结果目录
    os.makedirs('/data/yyf/Net/result/p53_2026_0425(2)/pfmil_resnet', exist_ok=True)
    
    # 训练每个fold
    for fold in range(1, 6):
        best_auc = train_single_fold(fold)
        results[f'fold_{fold}'] = best_auc
        
        # 保存当前结果
        with open('/data/yyf/Net/result/p53_2026_0425(2)/pfmil_resnet/pfmil_5fold_results.txt', 'a') as f:
            f.write(f"Fold {fold}: Best AUC = {best_auc:.4f}\n")
    
    # 计算平均性能
    avg_auc = np.mean(list(results.values()))
    std_auc = np.std(list(results.values()))
    
    print("\n========== 5-Fold Cross Validation Results ==========")
    for fold, auc in results.items():
        print(f"{fold}: AUC = {auc:.4f}")
    print(f"Average AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    
    # 保存最终结果
    with open('/data/yyf/Net/result/p53_2026_0425(2)/pfmil_resnet/pfmil_5fold_results.txt', 'a') as f:
        f.write("\n========== Final Results ==========\n")
        f.write(f"Average AUC: {avg_auc:.4f} ± {std_auc:.4f}\n")
        f.write(f"All folds: {results}\n")
    
    return results


if __name__ == "__main__":
    # 运行5折交叉验证
    train_all_folds()