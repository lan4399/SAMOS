# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import argparse
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
import scipy.stats
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WavLMModel
import json
random.seed(42)


class MosPredictor(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, fine_tune_ssl=False):
        super(MosPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.fine_tune_ssl = fine_tune_ssl
        
        # 根据fine_tune_ssl参数设置WavLM是否需要梯度
        for param in self.ssl_model.parameters():
            param.requires_grad = fine_tune_ssl
        
        # WavLM基础模型的隐藏层维度是768，WavLM-large是1024
        self.num_layers = self.ssl_model.config.num_hidden_layers
        self.hidden_size = self.ssl_model.config.hidden_size
        
        # 修改：使用平均特征而不是拼接，特征维度保持为hidden_size
        self.in_features = self.hidden_size
        self.ffd_hidden_size = 4096
        self.attn_layer_num = 4
        
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

    def forward(self, wav):
        # 输入wav形状: [B, 1, T]
        wav = wav.squeeze(1)  # [B, T]
        
        # 根据fine_tune_ssl参数决定是否计算WavLM的梯度
        if self.fine_tune_ssl:
            # WavLM参与训练，需要计算梯度
            outputs = self.ssl_model(wav, output_hidden_states=True)
        else:
            # WavLM不参与训练，不计算梯度
            with torch.no_grad():
                outputs = self.ssl_model(wav, output_hidden_states=True)
        
        # 修改：获取所有隐藏层的特征并计算平均
        # hidden_states是一个元组，包含所有层的输出 [layer0, layer1, ..., layerN]
        # 每个层的输出形状: [B, T, hidden_size]
        hidden_states = outputs.hidden_states  # 包含输入嵌入和所有隐藏层
        
        # 跳过输入嵌入层（第0层），只使用隐藏层
        all_layer_features = []
        for layer_idx in range(1, len(hidden_states)):  # 从第1层开始（第0层是输入嵌入）
            layer_features = hidden_states[layer_idx]  # [B, T, hidden_size]
            all_layer_features.append(layer_features)
        
        # 修改：将所有层的特征堆叠并在层维度上取平均
        # stacked_features形状: [B, T, num_layers, hidden_size]
        stacked_features = torch.stack(all_layer_features, dim=2)
        
        # 在层维度上取平均，得到形状: [B, T, hidden_size]
        ssl_feature = torch.mean(stacked_features, dim=2)
        
        # 后续处理保持不变
        ssl_feature = self.ffd(ssl_feature)
        
        tmp_ssl_feature = ssl_feature
        for attn_layer in self.attn:
            tmp_ssl_feature, _ = attn_layer(tmp_ssl_feature, tmp_ssl_feature, tmp_ssl_feature)
        
        mean_pooled = torch.mean(tmp_ssl_feature, dim=1)
        max_pooled = torch.max(ssl_feature, dim=1)[0]
        pooled_features = torch.cat([mean_pooled, max_pooled], dim=1)
        pooled_features = self.dropout(pooled_features)
        
        mos_score = self.fc(pooled_features)
        mos_score = self.mos_activation(mos_score) * 4.0 + 1.0
        
        return mos_score.squeeze(-1)


class InferenceDataset(Dataset):
    """用于推理的数据集，支持有标签和无标签数据"""
    def __init__(self, wavdir, mos_list=None, file_list=None):
        self.data_list = []
        self.wavdir = wavdir
        self.has_labels = mos_list is not None
        
        if self.has_labels:
            # 有标签的情况
            with open(mos_list, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    wavname = parts[0]
                    mos = float(parts[1])
                    self.data_list.append([wavname, mos])
        else:
            # 无标签的情况，从文件列表或目录读取
            if file_list:
                with open(file_list, 'r') as f:
                    for line in f:
                        wavname = line.strip()
                        self.data_list.append([wavname, None])
            else:
                # 从目录读取所有wav文件
                for filename in os.listdir(wavdir):
                    if filename.endswith('.wav'):
                        self.data_list.append([filename, None])

    def __getitem__(self, idx):
        wavname, score = self.data_list[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        
        if not os.path.exists(wavpath):
            raise FileNotFoundError(f"Audio file not found: {wavpath}")
            
        wav, sr = torchaudio.load(wavpath)
        
        # 确保音频是16kHz，WavLM需要16kHz输入
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        
        return wav, score, wavname

    def __len__(self):
        return len(self.data_list)

    def collate_fn(self, batch):
        wavs, scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wav.shape[1] for wav in wavs)
        output_wavs = []
        
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            if amount_to_pad > 0:
                padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            else:
                padded_wav = wav
            output_wavs.append(padded_wav)
        
        output_wavs = torch.stack(output_wavs, dim=0)
        
        # 如果有标签，转换为tensor
        if scores[0] is not None:
            scores = torch.tensor(scores, dtype=torch.float32)
        else:
            scores = None
            
        return output_wavs, scores, wavnames


def systemID(uttID):
    """从文件名提取系统ID"""
    if '-' in uttID:
        return uttID.split('-')[0]
    elif '_' in uttID:
        return uttID.split('_')[0]
    else:
        return "unknown"


def inference(net, dataloader, device, has_labels=False, desc='Inference'):
    """推理函数"""
    predictions = {}
    all_scores = [] if has_labels else None
    net.eval()
    
    with tqdm(dataloader, desc=desc) as pbar:
        for i, data in enumerate(pbar, 0):
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = net(inputs)
            
            # 处理输出
            if outputs.dim() == 0:
                output = float(outputs.cpu().detach().numpy())
            else:
                output = float(outputs.squeeze().cpu().detach().numpy())
            
            predictions[filenames[0]] = {
                'predicted_mos': output
            }
            
            if has_labels and labels is not None:
                true_score = float(labels.squeeze().cpu().numpy())
                predictions[filenames[0]]['true_mos'] = true_score
                all_scores.append((true_score, output))
            
            pbar.set_postfix({'processed': len(predictions)})
    
    return predictions, all_scores


def calculate_metrics(truths, preds, prefix=''):
    """计算评估指标"""
    if len(truths) == 0:
        return {}
        
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


def save_results(predictions, output_dir, dataset_name, has_labels=False):
    """保存推理结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存为CSV
    csv_path = os.path.join(output_dir, f'{dataset_name}_predictions.csv')
    rows = []
    
    for filename, data in predictions.items():
        row = {'filename': filename, 'predicted_mos': data['predicted_mos']}
        if has_labels and 'true_mos' in data:
            row['true_mos'] = data['true_mos']
            row['error'] = abs(data['true_mos'] - data['predicted_mos'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f'Predictions saved to: {csv_path}')
    
    # 保存为JSON
    json_path = os.path.join(output_dir, f'{dataset_name}_predictions.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f'Detailed results saved to: {json_path}')
    
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description='MOS Prediction Inference on Out-of-Domain Test Sets')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to the trained model checkpoint')
    parser.add_argument('--wavlm_model', type=str, default='microsoft/wavlm-base-plus',
                       help='HuggingFace WavLM model name')
    parser.add_argument('--test_wavdir', type=str, required=True,
                       help='Directory containing test audio files')
    parser.add_argument('--test_list', type=str, required=False,
                       help='Path to test MOS list file (if available)')
    parser.add_argument('--file_list', type=str, required=False,
                       help='Path to file list (if no MOS labels)')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory for inference results')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use')
    
    args = parser.parse_args()

    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f'Using GPU: {args.gpu}')
    else:
        device = torch.device("cpu")
        print('Using CPU')
    
    print(f'Device: {device}')
    
    # 检查文件存在性
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    if not os.path.exists(args.test_wavdir):
        raise FileNotFoundError(f"Test directory not found: {args.test_wavdir}")
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f'Created output directory: {args.output_dir}')
    
    # 加载WavLM模型
    print(f'Loading WavLM model: {args.wavlm_model}...')
    ssl_model = WavLMModel.from_pretrained(args.wavlm_model)
    hidden_size = ssl_model.config.hidden_size
    
    # 创建MOS预测器
    print('Creating MOS predictor...')
    net = MosPredictor(ssl_model, hidden_size, fine_tune_ssl=False)
    
    # 加载训练好的权重
    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 处理可能的权重键不匹配
    if all(key.startswith('module.') for key in checkpoint.keys()):
        # 如果权重是以DataParallel方式保存的
        from collections import OrderedDict
        new_checkpoint = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # 去掉 'module.' 前缀
            new_checkpoint[name] = v
        checkpoint = new_checkpoint
    
    net.load_state_dict(checkpoint)
    net = net.to(device)
    net.eval()
    print('Model loaded successfully!')
    
    # 确定是否有标签
    has_labels = args.test_list is not None
    dataset_name = os.path.basename(args.test_wavdir)
    
    if has_labels:
        print(f'Testing with labels from: {args.test_list}')
    else:
        print('Testing without labels (inference only)')
    
    # 创建数据集和数据加载器
    print('Creating dataset...')
    testset = InferenceDataset(
        wavdir=args.test_wavdir,
        mos_list=args.test_list,
        file_list=args.file_list
    )
    
    testloader = DataLoader(
        testset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2, 
        collate_fn=testset.collate_fn
    )
    
    print(f'Found {len(testset)} audio files')
    
    # 进行推理
    print('\nStarting inference...')
    predictions, all_scores = inference(
        net, testloader, device, 
        has_labels=has_labels, 
        desc=f'Processing {dataset_name}'
    )
    
    # 计算指标（如果有标签）
    metrics = {}
    if has_labels and all_scores:
        truths = np.array([s[0] for s in all_scores])
        preds = np.array([s[1] for s in all_scores])
        
        metrics = calculate_metrics(truths, preds, 'test_')
        
        print('\n' + '='*60)
        print('EVALUATION RESULTS:')
        print('='*60)
        print(f'MSE:  {metrics.get("test_MSE", 0):.6f}')
        print(f'LCC:  {metrics.get("test_LCC", 0):.6f}')
        print(f'SRCC: {metrics.get("test_SRCC", 0):.6f}')
        print(f'KTAU: {metrics.get("test_KTAU", 0):.6f}')
        print('='*60)
    
    # 保存结果
    print('\nSaving results...')
    csv_path, json_path = save_results(
        predictions, args.output_dir, dataset_name, has_labels
    )
    
    # 保存指标（如果有）
    if metrics:
        metrics_path = os.path.join(args.output_dir, f'{dataset_name}_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f'Metrics saved to: {metrics_path}')
    
    # 打印统计信息
    print('\nPREDICTION STATISTICS:')
    pred_scores = [data['predicted_mos'] for data in predictions.values()]
    print(f'Number of files: {len(predictions)}')
    print(f'Predicted MOS range: {min(pred_scores):.3f} - {max(pred_scores):.3f}')
    print(f'Predicted MOS mean: {np.mean(pred_scores):.3f} ± {np.std(pred_scores):.3f}')
    
    if has_labels:
        true_scores = [data['true_mos'] for data in predictions.values() if 'true_mos' in data]
        errors = [abs(data['true_mos'] - data['predicted_mos']) for data in predictions.values() if 'true_mos' in data]
        print(f'True MOS range: {min(true_scores):.3f} - {max(true_scores):.3f}')
        print(f'True MOS mean: {np.mean(true_scores):.3f} ± {np.std(true_scores):.3f}')
        print(f'Mean absolute error: {np.mean(errors):.3f}')
    
    print(f'\nInference completed! Results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()