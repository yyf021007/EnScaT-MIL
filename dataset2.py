import numpy as np
import torch
import csv
import os
import torch.nn as nn
import torch.optim as optim
import h5py
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# from trident.slide_encoder_models import ABMILSlideEncoder, CHIEFSlideEncoder
from model.TransMIL import TransMIL
from sklearn.metrics import roc_auc_score, accuracy_score
from functools import lru_cache
from collections import defaultdict


import random
from pathlib import Path
import torch.utils.data as data
from torch.utils.data import dataloader


# class FenziDataset(Dataset):
#     def __init__(self, feats_paths, df, split):
#         self.df = df[df["split"] == split]
#         self.feats_paths = feats_paths
#         self.split = split
        
#         # 预加载元数据（不含实际特征）
#         self.slide_info = []
#         for _, row in tqdm(self.df.iterrows(), desc="Caching metadata"):
#             self.slide_info.append({
#                 'slide_id': row['slide_id'],
#                 'dataset': row['dataset'],
#                 'label': torch.tensor(-1 if pd.isna(row['fenzi']) else row['fenzi'], dtype=torch.long)
#             })

#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         info = self.slide_info[idx]
#         pt_path = os.path.join(self.feats_paths[info['dataset']], f"{info['slide_id']}.h5")
        
#         # 动态加载特征保持原始维度
#         # features = torch.load(pt_path)
#         with h5py.File(pt_path, "r") as f:
#             features = torch.from_numpy(f["features"][:])
#         return features, info['label']


# class FenziDataset(Dataset):
#     def __init__(self, feats_paths, df, split):
#         self.df = df[df["split"] == split]
#         self.feats_paths = feats_paths
#         self.split = split

#         # 预加载元数据（不含实际特征）
#         self.slide_info = []
#         for _, row in tqdm(self.df.iterrows(), desc="Caching metadata"):
#             self.slide_info.append({
#                 'slide_id': row['slide_id'],
#                 'dataset': row['dataset'],
#                 'label': torch.tensor(-1 if pd.isna(row['fenzi']) else row['fenzi'], dtype=torch.long)
#             })
        
#         # 提取唯一slide_id (关键修改)
#         unique_slide_ids = self.df['slide_id'].unique()
#         # 预加载所有特征到内存
#         self.feature_cache = {}
#         # 预加载唯一特征
#         for slide_id in tqdm(unique_slide_ids, desc="Caching features"):
#             # 获取该slide对应的dataset（假设一个slide只属于一个dataset）
#             dataset_name = self.df[self.df['slide_id'] == slide_id]['dataset'].iloc[0]
#             h5_path = os.path.join(feats_paths[dataset_name], f"{slide_id}.h5")
#             # pt_path = os.path.join(feats_paths[dataset_name], f"{slide_id}.pt")
#             with h5py.File(h5_path, "r") as f:
#                 self.feature_cache[slide_id] = torch.from_numpy(f["features"][:])
#             # self.feature_cache[slide_id] = torch.load(pt_path)

#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         info = self.slide_info[idx]
#         dataset_name = row['dataset']
#         feats_path = self.feats_paths[dataset_name]

#         features = self.feature_cache[row['slide_id']]

#         return features, info['label']

def collate_single_task(batch):
    """处理可变长度特征的collate函数"""
    # 当batch_size=1时直接返回原始格式
    if len(batch) == 1:
        return batch[0][0].unsqueeze(0), batch[0][1].unsqueeze(0)
    
    # 支持未来扩展的通用处理
    features = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return features, labels




# class FenziDataset(Dataset):
#     def __init__(self, feats_paths, df, split, num_features=1024, seed=42):
#         self.df = df[df["split"] == split]
#         self.feats_paths = feats_paths
#         self.split = split
#         self.num_features = num_features
#         self.seed = seed
#         self.slide_info = []
        
#         # 构建slide_id到dataset的映射字典
#         self.slide_to_dataset = self.df.groupby('slide_id')['dataset'].first().to_dict()
        
#         # 预加载所有特征到内存（核心修改部分）
#         self.feature_cache = self._preload_features()
        
#         # 预存元数据
#         for _, row in self.df.iterrows():
#             self.slide_info.append({
#                 'slide_id': row['slide_id'],
#                 'label': torch.tensor(-1 if pd.isna(row['fenzi']) else row['fenzi'], dtype=torch.long)
#             })

#     def _preload_features(self):
#         """预加载所有特征到内存"""
#         cache = {}
#         unique_slides = self.df.slide_id.unique()
        
#         # 进度条显示
#         from tqdm import tqdm
#         print(f"Preloading {len(unique_slides)} slides into memory...")
        
#         for slide_id in tqdm(unique_slides):
#             dataset_name = self.slide_to_dataset[slide_id]
#             h5_path = os.path.join(self.feats_paths[dataset_name], f"{slide_id}.h5")
            
#             # 加载并转换数据
#             with h5py.File(h5_path, "r") as f:
#                 # 保持原始数据精度为float16以节省内存
#                 features = torch.from_numpy(f["features"][:].astype(np.float16))
                
#             # 使用字典存储，键为slide_id
#             cache[slide_id] = features
            
#         return cache

#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         info = self.slide_info[idx]
#         slide_id = info['slide_id']
        
#         # 直接从内存获取特征（不再需要文件IO）
#         features = self.feature_cache[slide_id].clone()  # 使用clone避免原地修改
        
#         # 训练时动态采样特征（保持原有逻辑）
#         if self.split == 'train':
#             num_available = features.shape[0]
#             generator = torch.Generator().manual_seed(self.seed + idx)  # 加入idx作为随机因子
            
#             if num_available >= self.num_features:
#                 indices = torch.randperm(num_available, generator=generator)[:self.num_features]
#             else:
#                 indices = torch.randint(num_available, (self.num_features,), generator=generator)
            
#             features = features[indices]
            
#         return features, info['label']
    


# class FenziDataset(Dataset):
#     def __init__(self, feats_paths, df, split, num_features=1024, seed=42):  # 新增num_features和seed参数
#         self.df = df[df["split"] == split]
#         self.feats_paths = feats_paths
#         self.split = split
#         self.num_features = num_features  # 特征采样数
#         self.seed = seed                  # 随机种子
#         self.slide_info = []
#         # 构建slide_id到dataset的映射字典
#         self.slide_to_dataset = self.df.groupby('slide_id')['dataset'].first().to_dict()
#         # 预存元数据
#         for _, row in self.df.iterrows():
#             self.slide_info.append({
#                 'slide_id': row['slide_id'],
#                 'label': torch.tensor(-1 if pd.isna(row['fenzi']) else row['fenzi'], dtype=torch.long)
#             })
#         # 每个数据集实例维护自己的缓存
#         self.feature_cache = {}

#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         info = self.slide_info[idx]
#         slide_id = info['slide_id']
#         dataset_name = self.slide_to_dataset[slide_id]
        
#         # 缓存检查
#         if slide_id not in self.feature_cache:
#             h5_path = os.path.join(self.feats_paths[dataset_name], f"{slide_id}.h5")
#             with h5py.File(h5_path, "r") as f:
#                 features = torch.from_numpy(f["features"][:].astype(np.float16))
#             self.feature_cache[slide_id] = features
        
#         features = self.feature_cache[slide_id]  # 从缓存获取原始特征
        
#         # 训练时动态采样特征
#         if self.split == 'train':
#             num_available = features.shape[0]
#             generator = torch.Generator().manual_seed(self.seed)  # 固定种子
            
#             if num_available >= self.num_features:
#                 # 随机选择不重复的特征
#                 indices = torch.randperm(num_available, generator=generator)[:self.num_features]
#             else:
#                 # 允许重复的过采样
#                 indices = torch.randint(num_available, (self.num_features,), generator=generator)
            
#             features = features[indices]  # 采样后的特征

#         return features, info['label']





class FenziDataset(Dataset):
    def __init__(self, feats_paths, df, split, num_features=4000, seed=42):
        # 原有初始化代码保持不变...
        self.df = df[df["split"] == split]
        self.feats_paths = feats_paths
        self.split = split
        self.num_features = num_features
        self.seed = seed
        self.slide_info = []
         
        # 构建slide_id到dataset的映射字典
        self.slide_to_dataset = self.df.groupby('slide_id')['dataset'].first().to_dict()
        
        # 预加载所有特征到内存（核心修改部分）
        self.feature_cache = self._preload_features()
        
        # 预存元数据
        for _, row in self.df.iterrows():
            self.slide_info.append({
                'slide_id': row['slide_id'],
                'label': torch.tensor(-1 if pd.isna(row['p53']) else row['p53'], dtype=torch.long)
            })
        
        # 新增数据增强相关参数

        self.augmenter = OnlineFeatureAugmenter(
            minority_classes=[0, 2, 3],
            augment_prob=0.6,
            noise_std=0.1,
            channel_scale_range=(0.8, 1.2),  # 新增参数
            block_size=32                     # 新增块大小参数
        )
    
    def _preload_features(self):
        """预加载所有特征到内存"""
        cache = {}
        unique_slides = self.df.slide_id.unique()
        
        # 进度条显示
        from tqdm import tqdm
        print(f"Preloading {len(unique_slides)} slides into memory...")
        
        for slide_id in tqdm(unique_slides):
            dataset_name = self.slide_to_dataset[slide_id]
            h5_path = os.path.join(self.feats_paths[dataset_name], f"{slide_id}.h5")
            
            # # 加载并转换数据
            with h5py.File(h5_path, "r") as f:
                # 保持原始数据精度为float16以节省内存
                features = torch.from_numpy(f["features"][:].astype(np.float16))
            
            # pt_path = os.path.join(self.feats_paths[dataset_name], f"{slide_id}.pt")
            # features = torch.load(pt_path)
                
            # 使用字典存储，键为slide_id
            cache[slide_id] = features
            
        return cache

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        info = self.slide_info[idx]
        slide_id = info['slide_id']
        label = info['label']
        
        features = self.feature_cache[slide_id].clone()
        
        # 在线数据增强（仅在训练时应用）
        if self.split == 'train':
            # 获取原始特征和标签
            # features, label = self.augmenter.augment(features, label)
            
            # 保持原有的特征采样逻辑
            num_available = features.shape[0]
            generator = torch.Generator().manual_seed(self.seed + idx)  # 加入idx作为随机因子
            
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=generator)[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,), generator=generator)
            
            features = features[indices]
        
            
        return features, info['label']
    

class testFenziDataset(Dataset):
    def __init__(self, feats_paths, df, split, num_features=3072, seed=10):
        # 原有初始化代码保持不变...
        self.df = df[df["split"] == split]
        self.feats_paths = feats_paths
        self.split = split
        self.num_features = num_features
        self.seed = seed
        self.slide_info = []
         
        # 构建slide_id到dataset的映射字典
        self.slide_to_dataset = self.df.groupby('slide_id')['dataset'].first().to_dict()
        
        # 预加载所有特征到内存（核心修改部分）
        self.feature_cache = self._preload_features()
        
        # 预存元数据
        for _, row in self.df.iterrows():
            self.slide_info.append({
                'slide_id': row['slide_id'],
                'label': torch.tensor(-1 if pd.isna(row['weixing']) else row['weixing'], dtype=torch.long)
            })
        
        # 新增数据增强相关参数

        self.augmenter = OnlineFeatureAugmenter(
            minority_classes=[0, 2, 3],
            augment_prob=0.6,
            noise_std=0.1,
            channel_scale_range=(0.8, 1.2),  # 新增参数
            block_size=32                     # 新增块大小参数
        )
    
    def _preload_features(self):
        """预加载所有特征到内存"""
        cache = {}
        unique_slides = self.df.slide_id.unique()
        
        # 进度条显示
        from tqdm import tqdm
        print(f"Preloading {len(unique_slides)} slides into memory...")
        
        for slide_id in tqdm(unique_slides):
            dataset_name = self.slide_to_dataset[slide_id]
            h5_path = os.path.join(self.feats_paths[dataset_name], f"{slide_id}.h5")
            
            # 加载并转换数据
            with h5py.File(h5_path, "r") as f:
                # 保持原始数据精度为float16以节省内存
                features = torch.from_numpy(f["features"][:].astype(np.float16))
                
            # 使用字典存储，键为slide_id
            cache[slide_id] = features
            
        return cache

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        info = self.slide_info[idx]
        slide_id = info['slide_id']
        label = info['label']
        
        features = self.feature_cache[slide_id].clone()
        
        # 在线数据增强（仅在训练时应用）
        if self.split == 'train':
            # 获取原始特征和标签
            features, label = self.augmenter.augment(features, label)
            
            # 保持原有的特征采样逻辑
            num_available = features.shape[0]
            generator = torch.Generator().manual_seed(self.seed + idx)  # 加入idx作为随机因子
            
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=generator)[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,), generator=generator)
            
            features = features[indices]
        
            
        return features, info['label'], slide_id


class OnlineFeatureAugmenter:
    """改进后的在线特征增强器（移除Mixup）"""
    def __init__(self, minority_classes, augment_prob=0.7, noise_std=0.1, 
                 channel_scale_range=(0.5, 1.5), block_size=16):
        self.minority_classes = set(minority_classes)
        self.augment_prob = augment_prob
        self.noise_std = noise_std
        self.scale_range = channel_scale_range
        self.block_size = block_size
        
        # 配置新的增强方法库
        self.augment_methods = [
            self.add_adaptive_noise,
            self.channel_wise_scaling
            # self.random_block_shuffle
            # self.feature_space_projection
        ]
    
    def add_adaptive_noise(self, features, label):
        """自适应噪声增强"""
        # 基于特征幅度的噪声
        feature_std = features.std(dim=0, keepdim=True)
        noise = torch.randn_like(features) * feature_std * self.noise_std
        return features + noise, label
    
    def channel_wise_scaling(self, features, label):
        """通道随机缩放"""
        # 为每个特征通道生成随机缩放因子
        scaling_factors = torch.empty(features.size(-1)).uniform_(*self.scale_range)
        return features * scaling_factors.to(features.device), label
    
    def random_block_shuffle(self, features, label):
        """随机块置换增强"""
        if len(features) < self.block_size * 2:
            return features, label
        
        # 将特征分成多个块
        num_blocks = len(features) // self.block_size
        blocks = [features[i*self.block_size:(i+1)*self.block_size] 
                 for i in range(num_blocks)]
        
        # 随机打乱块顺序
        random.shuffle(blocks)
        return torch.cat(blocks, dim=0), label
    
    def feature_space_projection(self, features, label):
        """特征空间随机投影"""
        # 生成随机投影矩阵
        in_dim = features.size(-1)
        proj_matrix = torch.randn(in_dim, in_dim).to(features.device)
        proj_matrix /= torch.norm(proj_matrix, dim=1, keepdim=True)
        
        # 应用投影并保留原始维度
        return torch.matmul(features, proj_matrix), label

    # 保持原有基础方法
    def should_augment(self, label):
        return label.item() in self.minority_classes and random.random() < self.augment_prob
    
    def augment(self, features, label):
        if self.should_augment(label):
            aug_method = random.choice(self.augment_methods)
            return aug_method(features, label)
        return features, label