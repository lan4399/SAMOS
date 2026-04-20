#!/usr/bin/env python3
"""
预提取特征MOS预测器 - 仅推理测试集
直接使用训练好的权重对测试集进行推理
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from datetime import datetime
import sys




class MosPredictor(nn.Module):
    def __init__(self, feature_dim=1024):
        super(MosPredictor, self).__init__()
        
        self.in_features = feature_dim
        self.ffd_hidden_size = 4096
        self.attn_layer_num = 4
        
        if feature_dim != self.in_features:
            self.proj_layer = nn.Linear(feature_dim, self.in_features)
        else:
            self.proj_layer = nn.Identity()
        
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
        features = self.proj_layer(features)
        features = self.ffd(features)
        
        tmp_features = features
        for attn_layer in self.attn:
            tmp_features, _ = attn_layer(tmp_features, tmp_features, tmp_features)
        
        mean_pooled = torch.mean(tmp_features, dim=1)
        max_pooled = torch.max(features, dim=1)[0]
        pooled_features = torch.cat([mean_pooled, max_pooled], dim=1)
        pooled_features = self.dropout(pooled_features)
        
        mos_score = self.fc(pooled_features)
        mos_score = self.mos_activation(mos_score) * 4.0 + 1.0
        
        return mos_score


class PrecomputedFeatureDataset(Dataset):
    def __init__(self, feature_dir, mos_list):
        self.data_list = []
        self.feature_dir = feature_dir
        
        with open(mos_list, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    wavname = parts[0].strip()
                    mos = float(parts[1])
                    self.data_list.append([wavname, mos])
        
        print(f"测试集加载完成: {len(self.data_list)} 个样本")

    def __getitem__(self, idx):
        wavname, score = self.data_list[idx]
        
        feature_filename = os.path.splitext(wavname)[0] + '.npy'
        feature_path = os.path.join(self.feature_dir, feature_filename)
        
        if not os.path.exists(feature_path):
            print(f"警告: 特征文件不存在 {feature_path}")
            features = np.zeros((100, 1024), dtype=np.float32)
        else:
            features = np.load(feature_path)
        
            if features.ndim == 3:
                features = features.squeeze(0).transpose(1, 0)
            elif features.ndim == 2:
                features = features.transpose(1, 0)
            elif features.ndim == 1:
                features = features.reshape(1, -1)
        
        features = torch.tensor(features, dtype=torch.float32)
        return features, score, wavname

    def __len__(self):
        return len(self.data_list)

    def collate_fn(self, batch):
        features_list, scores, wavnames = zip(*batch)
        features_list = list(features_list)
        
        max_t = max(features.shape[0] for features in features_list)
        feature_dim = features_list[0].shape[1]
        
        padded_features = []
        for features in features_list:
            t = features.shape[0]
            pad_amount = max_t - t
            if pad_amount > 0:
                padded_feature = F.pad(features, (0, 0, 0, pad_amount), 'constant', 0)
            else:
                padded_feature = features
            padded_features.append(padded_feature)
        
        output_features = torch.stack(padded_features, dim=0)
        output_scores = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_features, output_scores, wavnames


def systemID(uttID):
    return uttID.split('-')[0]


def evaluate_model(net, dataloader, mos_list_path, device, desc='Evaluating'):
    """评估模型性能"""
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
    """计算评估指标"""
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


def main():
    parser = argparse.ArgumentParser(description='预提取特征MOS预测器 - 仅推理测试集')
    parser.add_argument('--datadir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--feature_dir', type=str, required=True, help='预提取特征目录路径')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型权重路径')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='输出目录')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--feature_dim', type=int, default=1024, help='特征维度')
    
    args = parser.parse_args()

    print("=" * 60)
    print("预提取特征MOS预测器 - 测试集推理模式")
    print("=" * 60)
    
    # 参数检查
    print("参数检查:")
    print(f"  数据目录: {args.datadir}")
    print(f"  特征目录: {args.feature_dir}")
    print(f"  模型路径: {args.model_path}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  特征维度: {args.feature_dim}")
    print(f"  GPU设备: {args.gpu}")
    
    # 设备设置
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f'使用设备: {device}')
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在 {args.model_path}")
        return
    
    if not os.path.exists(args.feature_dir):
        print(f"错误: 特征目录不存在 {args.feature_dir}")
        return
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f'创建输出目录: {args.output_dir}')
    
    # 数据路径
    testlist = os.path.join(args.datadir, 'sets/test_mos_list.txt')
    if not os.path.exists(testlist):
        print(f"错误: 测试集MOS列表不存在 {testlist}")
        return
    
    # 创建模型
    net = MosPredictor(feature_dim=args.feature_dim)
    net = net.to(device)
    
    # 加载预训练权重
    print(f"加载预训练权重: {args.model_path}")
    try:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        print("权重加载成功!")
    except Exception as e:
        print(f"权重加载失败: {e}")
        return
    
    # 创建测试数据集
    print("创建测试数据集...")
    testset = PrecomputedFeatureDataset(args.feature_dir, testlist)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2, 
                          collate_fn=testset.collate_fn)
    
    print(f"测试集样本数量: {len(testset)}")
    
    # 进行推理
    print('\n开始推理测试集...')
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    
    test_results = evaluate_model(net, testloader, testlist, device, '测试集推理')
    
    # 计算测试集指标
    test_utt_metrics = calculate_metrics(test_results['utt_truths'], test_results['utt_preds'], 'test_utt_')
    test_sys_metrics = calculate_metrics(test_results['sys_truths'], test_results['sys_preds'], 'test_sys_')
    
    # 打印结果
    print(f'\n{"="*60}')
    print('测试集推理结果:')
    print(f'样本数量: {len(test_results["utt_truths"])}')
    print(f'系统数量: {len(test_results["sys_truths"])}')
    print(f'{"="*60}')
    
    print('句子级指标:')
    print(f'  - MSE: {test_utt_metrics["test_utt_MSE"]:.6f}')
    print(f'  - LCC: {test_utt_metrics["test_utt_LCC"]:.6f}')
    print(f'  - SRCC: {test_utt_metrics["test_utt_SRCC"]:.6f}')
    print(f'  - KTAU: {test_utt_metrics["test_utt_KTAU"]:.6f}')
    
    if len(test_results["sys_truths"]) > 0:
        print('系统级指标:')
        print(f'  - MSE: {test_sys_metrics["test_sys_MSE"]:.6f}')
        print(f'  - LCC: {test_sys_metrics["test_sys_LCC"]:.6f}')
        print(f'  - SRCC: {test_sys_metrics["test_sys_SRCC"]:.6f}')
        print(f'  - KTAU: {test_sys_metrics["test_sys_KTAU"]:.6f}')
    print(f'{"="*60}')
    
    # 保存详细结果
    result_data = {
        'inference_config': {
            'model_path': args.model_path,
            'feature_dir': args.feature_dir,
            'test_datadir': args.datadir,
            'feature_dim': args.feature_dim,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'statistics': {
            'total_samples': len(test_results["utt_truths"]),
            'system_count': len(test_results["sys_truths"])
        },
        'metrics': {
            'utterance_level': test_utt_metrics,
            'system_level': test_sys_metrics
        },
        'predictions': test_results['predictions']
    }
    
    # 保存JSON结果
    result_file = os.path.join(args.output_dir, 'inference_results.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f'详细结果已保存至: {result_file}')
    
    # 保存CSV预测结果
    csv_file = os.path.join(args.output_dir, 'predictions.csv')
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write('filename,predicted_mos,true_mos\n')
        for uttID, pred_mos in test_results['predictions'].items():
            # 查找真实MOS值
            true_mos = None
            with open(testlist, 'r', encoding='utf-8') as mos_file:
                for line in mos_file:
                    parts = line.strip().split(',')
                    if len(parts) >= 2 and parts[0] == uttID:
                        true_mos = float(parts[1])
                        break
            
            if true_mos is not None:
                f.write(f'{uttID},{pred_mos:.4f},{true_mos}\n')
            else:
                f.write(f'{uttID},{pred_mos:.4f},\n')
    
    print(f'预测结果CSV已保存至: {csv_file}')
    

    plt.tight_layout()
    plot_file = os.path.join(args.output_dir, 'prediction_scatter.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f'预测结果散点图已保存至: {plot_file}')
    
    print(f'\n推理完成!')
    print(f'测试集句子级SRCC: {test_utt_metrics["test_utt_SRCC"]:.6f}')
    if len(test_results["sys_truths"]) > 0:
        print(f'测试集系统级SRCC: {test_sys_metrics["test_sys_SRCC"]:.6f}')


if __name__ == '__main__':
    main()