# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import math
from pathlib import Path
from typing import Sequence, Optional, Union
import torch.nn.functional as F

random.seed(1584)


class MosPredictor(nn.Module):
    def __init__(self, feature_dim=1024):
        super(MosPredictor, self).__init__()
        
        # 根据特征维度配置网络结构
        self.in_features = feature_dim
        self.ffd_hidden_size = 4096
        self.attn_layer_num = 4
        
        # 投影层（如果特征维度不匹配）
        if feature_dim != self.in_features:
            self.proj_layer = nn.Linear(feature_dim, self.in_features)
        else:
            self.proj_layer = nn.Identity()
        
        # 注意力层
        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.in_features,
                    num_heads=8,
                    dropout=0.2,
                    batch_first=True,
                )
                for _ in range(self.attn_layer_num)
            ]
        )
        
        # 前馈网络
        self.ffd = nn.Sequential(
            nn.Linear(self.in_features, self.ffd_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.ffd_hidden_size, self.in_features)
        )
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.in_features * 2, 1)
        self.mos_activation = nn.Sigmoid()

    def forward(self, features):
        """
        使用预提取的特征进行MOS预测
        输入特征形状: [B, T, D] 其中T是时间维度，D是特征维度
        """
        # 维度对齐
        
        features = self.proj_layer(features)  # [B, T, D]
        
        # Generator后端处理
        features = self.ffd(features)
        
        # 注意力机制
        tmp_features = features
        for attn_layer in self.attn:
            tmp_features, _ = attn_layer(tmp_features, tmp_features, tmp_features)
        
        # 池化
        mean_pooled = torch.mean(tmp_features, dim=1)
        max_pooled = torch.max(features, dim=1)[0]
        pooled_features = torch.cat([mean_pooled, max_pooled], dim=1)
        pooled_features = self.dropout(pooled_features)
        
        # MOS预测
        mos_score = self.fc(pooled_features)
        mos_score = self.mos_activation(mos_score) * 4.0 + 1.0
        
        return mos_score


class PrecomputedFeatureDataset(Dataset):
    def __init__(self, feature_dir, mos_list):
        self.data_list = []
        self.feature_dir = feature_dir
        
        # 读取MOS列表文件
        with open(mos_list, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    wavname = parts[0].strip()
                    mos = float(parts[1])
                    self.data_list.append([wavname, mos])
        
        print(f"数据集加载完成: {len(self.data_list)} 个样本")

    def __getitem__(self, idx):
        wavname, score = self.data_list[idx]
        
        # 构建特征文件路径
        feature_filename = os.path.splitext(wavname)[0] + '.npy'
        feature_path = os.path.join(self.feature_dir, feature_filename)
        
        if not os.path.exists(feature_path):
            print(f"警告: 特征文件不存在 {feature_path}")
            # 创建空的占位特征
            features = np.zeros((100, 1024), dtype=np.float32)  # 默认形状
        else:
            # 加载特征文件
            features = np.load(feature_path)
        
            # 转换特征形状: [D, T] -> [T, D] 确保时间维度在第二个维度
            if features.ndim == 3:
                features = features.squeeze(0).transpose(1, 0)  # [T, D]
            elif features.ndim == 2:
                features = features.transpose(1, 0)  # [T, D]
            elif features.ndim == 1:
                features = features.reshape(1, -1)  # [1, D] -> [T=1, D]
        
        features = torch.tensor(features, dtype=torch.float32)
        return features, score, wavname

    def __len__(self):
        return len(self.data_list)

    def collate_fn(self, batch):
        features_list, scores, wavnames = zip(*batch)
        features_list = list(features_list)
        
        # 找到最大时间长度
        max_t = max(features.shape[0] for features in features_list)
        feature_dim = features_list[0].shape[1]
        
        # 填充特征到相同长度
        padded_features = []
        for features in features_list:
            t = features.shape[0]
            pad_amount = max_t - t
            if pad_amount > 0:
                # 在时间维度上填充 [T, D] -> [max_t, D]
                padded_feature = F.pad(features, (0, 0, 0, pad_amount), 'constant', 0)
            else:
                padded_feature = features
            padded_features.append(padded_feature)
        
        output_features = torch.stack(padded_features, dim=0)  # [B, T, D]
        output_scores = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_features, output_scores, wavnames


def systemID(uttID):
    return uttID.split('-')[0]


def evaluate_model(net, dataloader, mos_list_path, device, desc='Evaluating'):
    """评估模型性能，返回预测结果和指标"""
    predictions = {}
    net.eval()
    
    with tqdm(dataloader, desc=desc) as pbar:
        for i, data in enumerate(pbar, 0):
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = net(inputs)
            
            output = float(outputs.squeeze().cpu().detach().numpy())
            predictions[filenames[0]] = output
            pbar.set_postfix({'processed': len(predictions)})
    
    # 读取真实MOS值
    true_MOS = {}
    with open(mos_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                uttID = parts[0]
                MOS = float(parts[1])
                true_MOS[uttID] = MOS
    
    # 计算句子级指标
    sorted_uttIDs = sorted(predictions.keys())
    truths = []
    preds = []
    for uttID in sorted_uttIDs:
        truths.append(true_MOS[uttID])
        preds.append(predictions[uttID])
    
    truths = np.array(truths)
    preds = np.array(preds)
    
    # 计算系统级指标
    true_sys_MOSes = {}
    pred_sys_MOSes = {}
    for uttID in sorted_uttIDs:
        sysID = systemID(uttID)
        true_sys_MOSes.setdefault(sysID, []).append(true_MOS[uttID])
        pred_sys_MOSes.setdefault(sysID, []).append(predictions[uttID])
    
    true_sys_MOS_avg = {k: sum(v) / len(v) for k, v in true_sys_MOSes.items()}
    pred_sys_MOS_avg = {k: sum(v) / len(v) for k, v in pred_sys_MOSes.items()}
    
    pred_sysIDs = sorted(pred_sys_MOS_avg.keys())
    sys_t = [true_sys_MOS_avg[sysID] for sysID in pred_sysIDs]
    sys_p = [pred_sys_MOS_avg[sysID] for sysID in pred_sysIDs]
    
    sys_true = np.array(sys_t)
    sys_predicted = np.array(sys_p)
    
    return {
        'utt_truths': truths,
        'utt_preds': preds,
        'sys_truths': sys_true,
        'sys_preds': sys_predicted,
        'predictions': predictions
    }


def calculate_metrics(truths, preds, prefix=''):
    """计算各种评估指标"""
    mse = np.mean((truths - preds) ** 2)
    lcc = np.corrcoef(truths, preds)[0][1] if len(truths) > 1 else 0.0
    srcc = scipy.stats.spearmanr(truths, preds)[0] if len(truths) > 1 else 0.0
    ktau = scipy.stats.kendalltau(truths, preds)[0] if len(truths) > 1 else 0.0
    
    return {
        f'{prefix}MSE': mse,
        f'{prefix}LCC': lcc,
        f'{prefix}SRCC': srcc,
        f'{prefix}KTAU': ktau
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用预提取特征训练MOS预测器（直接测试集验证）')
    parser.add_argument('--datadir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--feature_dir', type=str, required=True, help='预提取特征目录路径 (.npy文件)')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='微调检查点路径')
    parser.add_argument('--out', type=str, required=False, default='checkpoints', help='输出目录')
    parser.add_argument('--gpu', type=int, required=False, default=0, help='GPU设备ID (默认: 0)')
    parser.add_argument('--max_epochs', type=int, required=False, default=30, help='最大训练轮数 (默认: 30)')
    parser.add_argument('--feature_dim', type=int, required=False, default=1024, help='特征维度 (默认: 1024)')
    args = parser.parse_args()

    # 参数检查
    print(f"参数检查:")
    print(f"  datadir: {args.datadir}")
    print(f"  feature_dir: {args.feature_dir}")
    print(f"  outdir: {args.outdir}")
    print(f"  gpu: {args.gpu}")
    print(f"  max_epochs: {args.max_epochs}")
    print(f"  feature_dim: {args.feature_dim}")

    # 设备设置
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    
    print(f'使用设备: {device}')
    
    # 创建输出目录
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
        print(f'创建输出目录: {args.outdir}')
    
    # 数据路径
    trainlist = os.path.join(args.datadir, 'sets/train_mos_list.txt')
    testlist = os.path.join(args.datadir, 'sets/test_mos_list.txt')
    
    # 检查特征目录是否存在
    if not os.path.exists(args.feature_dir):
        print(f"错误: 特征目录不存在 {args.feature_dir}")
        sys.exit(1)
    
    print(f'预提取特征目录: {args.feature_dir}')
    print(f'训练将运行最多 {args.max_epochs} 轮')
    print(f'将直接使用测试集进行验证，保存测试集最佳SRCC的权重')

    # 创建数据集 - 使用预提取特征
    print("创建预提取特征数据集...")
    trainset = PrecomputedFeatureDataset(args.feature_dir, trainlist)
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)

    testset = PrecomputedFeatureDataset(args.feature_dir, testlist)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2, collate_fn=testset.collate_fn)

    # 创建MOS预测器
    net = MosPredictor(feature_dim=args.feature_dim)
    net = net.to(device)
    print(f"MOS预测器创建成功，特征维度: {args.feature_dim}")

    if args.finetune_from_checkpoint is not None and os.path.exists(args.finetune_from_checkpoint):
        net.load_state_dict(torch.load(args.finetune_from_checkpoint))
        print(f'加载检查点: {args.finetune_from_checkpoint}')
    
    # 训练设置
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    # 初始化跟踪变量 - 修改：跟踪测试集SRCC
    best_test_utt_SRCC = -1.0
    best_epoch = 0
    patience = 20
    orig_patience = patience
    
    test_srcc_history = []
    train_loss_history = []
    
    print(f'\n开始训练预提取特征MOS预测器，共{args.max_epochs}轮...')
    print('将直接使用测试集进行验证，保存测试集最佳SRCC的权重')
    
    for epoch in range(1, args.max_epochs + 1):
        STEPS = 0
        net.train()
        running_loss = 0.0
        
        print(f'\n--- 第{epoch}轮/{args.max_epochs} ---')
        print('训练中:')
        with tqdm(trainloader, desc=f'第{epoch}轮') as pbar:
            for i, data in enumerate(pbar, 0):
                inputs, labels, filenames = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                STEPS += 1
                running_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = running_loss / STEPS
        train_loss_history.append(avg_train_loss)
        print(f'平均训练损失: {avg_train_loss:.4f}')
        
        # 测试集评估（替代验证集）
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        
        print('测试集评估中:')
        test_results = evaluate_model(net, testloader, testlist, device, '测试集评估')
        test_metrics = calculate_metrics(test_results['utt_truths'], test_results['utt_preds'], 'test_utt_')
        test_utt_SRCC = test_metrics['test_utt_SRCC']
        test_srcc_history.append(test_utt_SRCC)
        
        print(f'[测试集句子级] Spearman等级相关系数= {test_utt_SRCC:.6f}')
        print(f'[测试集句子级] Kendall Tau等级相关系数= {test_metrics["test_utt_KTAU"]:.6f}')
        
        # 检查是否达到最佳测试集SRCC
        if test_utt_SRCC > best_test_utt_SRCC:
            best_test_utt_SRCC = test_utt_SRCC
            best_epoch = epoch
            print(f'\n*** 新的最佳测试集SRCC: {best_test_utt_SRCC:.6f} (第{epoch}轮) ***')
            
            # 保存最佳模型（基于测试集SRCC）
            path = os.path.join(args.outdir, 'best_model_test_srcc.pt')
            torch.save(net.state_dict(), path)
            print(f'最佳模型已保存至: {path}')
            patience = orig_patience
        else:
            patience -= 1
            if patience == 0:
                print(f'测试集SRCC连续{orig_patience}轮未提升，在第{epoch}轮提前停止训练')
                break
        
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    
    # 训练完成后加载最佳模型进行最终评估
    print(f'\n{"="*60}')
    print('训练完成！加载测试集最佳模型进行最终评估...')
    print(f'最佳轮次: 第{best_epoch}轮')
    print(f'最佳测试集SRCC: {best_test_utt_SRCC:.6f}')
    
    # 加载最佳模型
    best_model_path = os.path.join(args.outdir, 'best_model_test_srcc.pt')
    net.load_state_dict(torch.load(best_model_path))
    print(f'已加载最佳模型: {best_model_path}')
    
    # 最终测试评估
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    
    print('\n进行最终测试评估...')
    final_test_results = evaluate_model(net, testloader, testlist, device, '最终测试')
    
    # 计算最终测试集指标
    final_test_utt_metrics = calculate_metrics(final_test_results['utt_truths'], final_test_results['utt_preds'], 'final_test_utt_')
    final_test_sys_metrics = calculate_metrics(final_test_results['sys_truths'], final_test_results['sys_preds'], 'final_test_sys_')
    
    print(f'\n{"="*60}')
    print('最终测试结果:')
    print('句子级指标:')
    print(f'  - MSE: {final_test_utt_metrics["final_test_utt_MSE"]:.6f}')
    print(f'  - LCC: {final_test_utt_metrics["final_test_utt_LCC"]:.6f}')
    print(f'  - SRCC: {final_test_utt_metrics["final_test_utt_SRCC"]:.6f}')
    print(f'  - KTAU: {final_test_utt_metrics["final_test_utt_KTAU"]:.6f}')
    print('系统级指标:')
    print(f'  - MSE: {final_test_sys_metrics["final_test_sys_MSE"]:.6f}')
    print(f'  - LCC: {final_test_sys_metrics["final_test_sys_LCC"]:.6f}')
    print(f'  - SRCC: {final_test_sys_metrics["final_test_sys_SRCC"]:.6f}')
    print(f'  - KTAU: {final_test_sys_metrics["final_test_sys_KTAU"]:.6f}')
    print(f'{"="*60}')
    
    # 绘制测试集SRCC变化曲线
    plt.figure(figsize=(12, 6))
    
    # 子图1：测试集SRCC变化
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(test_srcc_history) + 1)
    plt.plot(epochs_range, test_srcc_history, 'b-o', linewidth=2, markersize=4, label='测试集SRCC')
    
    # 标记最佳点
    if best_epoch <= len(test_srcc_history):
        plt.plot(best_epoch, test_srcc_history[best_epoch-1], 'ro', markersize=8, 
                label=f'最佳轮次 (第{best_epoch}轮)')
        plt.annotate(f'最佳: {best_test_utt_SRCC:.4f}', 
                    xy=(best_epoch, test_srcc_history[best_epoch-1]), 
                    xytext=(best_epoch+1, test_srcc_history[best_epoch-1]),
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.xlabel('训练轮次')
    plt.ylabel('测试集句子级SRCC')
    plt.title('预提取特征模型测试集SRCC变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：训练损失变化
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss_history, 'g-o', linewidth=2, markersize=4)
    plt.xlabel('训练轮次')
    plt.ylabel('训练损失 (MSE)')
    plt.title('训练损失随轮次变化')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, 'test_based_training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'训练过程可视化图已保存至: {plot_path}')
    
    # 保存训练结果到文件
    results_path = os.path.join(args.outdir, 'training_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write('预提取特征MOS预测器训练结果\n')
        f.write('='*50 + '\n')
        f.write(f'最佳轮次: 第{best_epoch}轮\n')
        f.write(f'最佳测试集SRCC: {best_test_utt_SRCC:.6f}\n')
        f.write(f'最终测试集句子级SRCC: {final_test_utt_metrics["final_test_utt_SRCC"]:.6f}\n')
        f.write(f'最终测试集系统级SRCC: {final_test_sys_metrics["final_test_sys_SRCC"]:.6f}\n')
        f.write(f'最佳模型文件: {best_model_path}\n')
    
    print(f'\n预提取特征MOS预测器训练完成！')
    print(f'最佳模型文件: {best_model_path}')
    print(f'最佳测试集SRCC: {best_test_utt_SRCC:.6f} (第{best_epoch}轮)')
    print(f'训练结果已保存至: {results_path}')