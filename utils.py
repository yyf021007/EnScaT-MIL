import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import h5py
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F  # 这是标准导入方式
from torch.utils.data import Dataset, DataLoader
# from trident.slide_encoder_models import ABMILSlideEncoder, CHIEFSlideEncoder
from model.TransMIL import TransMIL
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict

# 在utils.py中添加以下内容
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 自动处理多分类情况
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            # 自动匹配类别权重
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            alpha = self.alpha[targets]
            focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_metrics(outputs, labels, task_config):
    metrics = {}
    with torch.no_grad():
        for task in task_config:
            task_name = task['name']
            pred = outputs[task_name].cpu()
            true = labels[task_name].cpu()

            # 过滤无效样本
            valid_mask = true >= 0
            if task['type'] == 'binary':
                valid_mask &= (true <= 1)
            
            valid_pred = pred[valid_mask]
            valid_true = true[valid_mask]
            
            if valid_true.numel() == 0:
                continue

            if task['type'] == 'binary':
                try:
                    auc = roc_auc_score(valid_true.numpy(), torch.sigmoid(valid_pred).numpy())
                    acc = accuracy_score(valid_true.numpy(), (valid_pred > 0).float().numpy())
                    metrics[f"{task_name}_auc"] = auc
                    metrics[f"{task_name}_acc"] = acc
                except ValueError:
                    pass
            else:
                acc = accuracy_score(valid_true.numpy(), valid_pred.argmax(dim=1).numpy())
                metrics[f"{task_name}_acc"] = acc

                # 计算多分类AUC（使用One-vs-Rest策略）
                prob = torch.softmax(valid_pred, dim=1).numpy()
                auc = roc_auc_score(
                    valid_true.numpy(), 
                    prob,
                    multi_class='ovr',  # 使用One-vs-Rest策略
                    average='macro'     # 使用宏平均
                )
                metrics[f"{task_name}_auc"] = auc
                
    return metrics

def validate(model, val_loader, task_config, criterion, device):
    model.eval()
    total_loss = 0.0
    all_outputs = {task['name']: [] for task in task_config}
    all_labels = {task['name']: [] for task in task_config}
    
    with torch.no_grad():
        for features, batch_labels in tqdm(val_loader):
            features = {'features': features.to(device)}
            outputs = model(features) # , device
            
            # 收集输出和标签
            for task in task_config:
                task_name = task['name']
                all_outputs[task_name].append(outputs[task_name].cpu())
                all_labels[task_name].append(batch_labels[task_name].cpu())
            
            # 计算损失
            batch_loss = 0.0
            for task in task_config:
                task_name = task['name']
                labels = batch_labels[task_name].to(device)
                
                if task['type'] == 'binary':
                    valid_mask = (labels >= 0) & (labels <= 1)
                else:
                    valid_mask = labels >= 0
                
                valid_labels = labels[valid_mask]
                if valid_labels.numel() == 0:
                    continue
                
                task_output = outputs[task_name][valid_mask]
                
                if task['type'] == 'binary':
                    batch_loss += criterion['binary'](task_output, valid_labels) * task['weight']
                else:
                    batch_loss += criterion['multiclass'](task_output, valid_labels) * task['weight']
            
            total_loss += batch_loss.item()
    
    # 合并结果
    final_outputs = {}
    final_labels = {}
    for task in task_config:
        task_name = task['name']
        if len(all_outputs[task_name]) > 0:
            final_outputs[task_name] = torch.cat(all_outputs[task_name])
            final_labels[task_name] = torch.cat(all_labels[task_name])
    
    metrics = calculate_metrics(final_outputs, final_labels, task_config)
    return total_loss / len(val_loader), metrics



def validate2(model, val_loader, task_config, criterion, device):
    model.eval()
    total_loss = 0.0
    all_outputs = {task['name']: [] for task in task_config}
    all_labels = {task['name']: [] for task in task_config}
    
    with torch.no_grad():
        for features, batch_labels in tqdm(val_loader):
            features = {'features': features.to(device)}
            outputs = model(features)
            
            # 收集输出和标签（保持维度）
            for task in task_config:
                task_name = task['name']
                # 保持原始维度存储
                all_outputs[task_name].append(outputs[task_name].detach().cpu())
                all_labels[task_name].append(batch_labels[task_name].cpu())
            
            # 计算损失
            batch_loss = 0.0
            for task in task_config:
                task_name = task['name']
                labels = batch_labels[task_name].to(device)
                
                if task['type'] == 'binary':
                    # 计算损失时压缩维度
                    task_output = outputs[task_name].squeeze(-1)
                    valid_mask = (labels >= 0) & (labels <= 1)
                else:
                    task_output = outputs[task_name]
                    valid_mask = labels >= 0
                
                valid_labels = labels[valid_mask]
                if valid_labels.numel() == 0:
                    continue
                
                if task['type'] == 'binary':
                    loss = criterion['binary'](task_output[valid_mask], valid_labels.float())
                else:
                    loss = criterion['multiclass'](task_output[valid_mask], valid_labels.long())
                
                batch_loss += loss * task['weight']
            
            total_loss += batch_loss.item()
    
    # 合并结果（关键修正部分）
    final_outputs = {}
    final_labels = {}
    for task in task_config:
        task_name = task['name']
        if len(all_outputs[task_name]) == 0:
            continue
            
        # 处理二分类维度
        if task['type'] == 'binary':
            # 统一转换为1D且保留维度
            processed = []
            for t in all_outputs[task_name]:
                if t.dim() == 1 and t.size(0) == 1:  # 处理单个样本的情况
                    processed.append(t.unsqueeze(0))  # [1] -> [1,1]
                else:
                    processed.append(t.squeeze(-1))
            final_outputs[task_name] = torch.cat(processed)
        else:
            final_outputs[task_name] = torch.cat(all_outputs[task_name])
        
        final_labels[task_name] = torch.cat(all_labels[task_name])
    
    metrics = calculate_metrics(final_outputs, final_labels, task_config)
    return total_loss / len(val_loader), metrics


# def validate_single_task(model, loader, criterion, device, task_config):
#     model.eval()
#     total_loss = 0.0
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for features, labels in loader:
#             features = {'features': features.to(device)}
#             labels = labels.to(device)
            
#             outputs = model(features)
#             valid_mask = labels >= 0
            
#             if valid_mask.any():
#                 loss = criterion(outputs[valid_mask], labels[valid_mask])
#                 total_loss += loss.item()
                
#                 probs = torch.softmax(outputs, dim=1)
#                 all_preds.extend(probs[valid_mask].cpu().numpy())
#                 all_labels.extend(labels[valid_mask].cpu().numpy())
    
#     metrics = calculate_single_metrics(np.array(all_preds), np.array(all_labels))
#     return total_loss/len(loader), metrics

# def calculate_single_metrics(preds, labels):
#     from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
#     pred_labels = np.argmax(preds, axis=1)
#     return {
#         'accuracy': accuracy_score(labels, pred_labels),
#         'f1': f1_score(labels, pred_labels, average='weighted'),
#         'confusion_matrix': confusion_matrix(labels, pred_labels)
#     }

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import warnings

# def validate_single_task(model, loader, criterion, device, task_config):
#     model.eval()
#     total_loss = 0.0
#     all_preds = []
#     all_labels = []
    
#     num_classes = task_config['num_classes']  # 从配置获取类别数
    
#     with torch.no_grad():
#         for features, labels in tqdm(loader):
#             features = {'features': features.to(device)}
#             labels = labels.to(device)
            
#             outputs = model(features)
#             valid_mask = labels >= 0
            
#             if valid_mask.any():
#                 loss = criterion(outputs[valid_mask], labels[valid_mask])
#                 total_loss += loss.item()
                
#                 probs = torch.softmax(outputs, dim=1)
#                 all_preds.extend(probs[valid_mask].cpu().numpy())
#                 all_labels.extend(labels[valid_mask].cpu().numpy())
    
#     metrics = calculate_single_metrics(np.array(all_preds), 
#                                      np.array(all_labels),
#                                      num_classes=num_classes)  # 传入类别数
#     return total_loss/len(loader), metrics

# def calculate_single_metrics(preds, labels, num_classes):
#     metrics = {}
    
#     # 原始指标计算
#     pred_labels = np.argmax(preds, axis=1)
#     metrics['accuracy'] = accuracy_score(labels, pred_labels)
#     metrics['f1'] = f1_score(labels, pred_labels, average='weighted')
#     metrics['confusion_matrix'] = confusion_matrix(labels, pred_labels)
    
#     # 新增AUC计算
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         try:
#             # One-vs-Rest AUC
#             y_true_bin = label_binarize(labels, classes=np.arange(num_classes))
            
#             # 计算每个类别的AUC
#             auc_scores = []
#             for i in range(num_classes):
#                 if np.sum(y_true_bin[:, i]) > 0 and np.sum(1 - y_true_bin[:, i]) > 0:
#                     auc = roc_auc_score(y_true_bin[:, i], preds[:, i])
#                     auc_scores.append(auc)
            
#             # # 宏观平均AUC（跳过无效类别）
#             # metrics['auc_macro'] = np.mean(auc_scores) if auc_scores else 0.0
            
#             # # OVR整体AUC
#             # metrics['auc_ovr'] = roc_auc_score(labels, preds, 
#             #                                 multi_class='ovr',
#             #                                 average='macro')
            
#             metrics['auc_macro_ovr'] = roc_auc_score(
#                 labels, preds, multi_class='ovr', average='macro'
#             )
#             metrics['auc_micro_ovr'] = roc_auc_score(
#                 labels, preds, multi_class='ovr', average='micro'
#             )
            
#         except ValueError as e:
#             print(f"AUC calculation skipped: {str(e)}")
#             metrics['auc_macro'] = 0.0
#             metrics['auc_ovr'] = 0.0
    
#     return metrics


def validate_single_task(model, loader, criterion, device, task_config):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    num_classes = task_config['num_classes']
    
    with torch.no_grad():
        for features, labels in tqdm(loader):
            features = {'features': features.to(device)}
            labels = labels.to(device)
            
            outputs = model(features)
            valid_mask = labels >= 0  # 假设-1表示需要忽略的样本
            
            if valid_mask.any():
                loss = criterion(outputs[valid_mask], labels[valid_mask])
                total_loss += loss.item()
                
                # 二分类时使用sigmoid替代softmax
                # probs = torch.sigmoid(outputs).squeeze()  # 假设模型输出单值
                # print(torch.softmax(outputs, dim=1))
                probs = torch.softmax(outputs, dim=1)[:, 1]  # 取正类概率
                all_preds.extend(probs[valid_mask].cpu().numpy())
                all_labels.extend(labels[valid_mask].cpu().numpy())
    
    metrics = calculate_single_metrics(np.array(all_preds), 
                                     np.array(all_labels),
                                     num_classes=num_classes)
    return total_loss/len(loader), metrics

def calculate_single_metrics(preds, labels, num_classes):
    metrics = {}
    
    # 二分类阈值设为0.5
    pred_labels = (preds >= 0.5).astype(int)
    
    # 基础指标计算
    metrics['accuracy'] = accuracy_score(labels, pred_labels)
    metrics['f1'] = f1_score(labels, pred_labels, average='binary')  # 使用binary模式
    metrics['confusion_matrix'] = confusion_matrix(labels, pred_labels)
    
    # AUC计算（处理单一类别的情况）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if len(np.unique(labels)) >= 2:
                metrics['auc'] = roc_auc_score(labels, preds)
            else:
                metrics['auc'] = 0.0
        except Exception as e:
            print(f"AUC calculation error: {str(e)}")
            metrics['auc'] = 0.0
    
    return metrics