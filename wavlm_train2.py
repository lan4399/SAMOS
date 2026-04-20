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
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import WavLMModel
random.seed(1584)

class QualityQueryPooling(nn.Module):
    """
    可学习的质量查询池化层 (借鉴自 CVPR Q-Former 和 ViT [CLS] Token)
    替代 mean()，利用注意力机制主动聚焦于音频序列中决定 MOS 分数的局部关键帧（如爆音、伪影片段）。
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        # 初始化一个可学习的 "MOS 质量查询向量"
        # 它的作用类似于一个"考官"，去审视整个音频序列
        self.quality_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 使用多头注意力，不同的 head 可以学习关注不同的瑕疵类型（如：底噪、截断、高频失真）
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, 
                                         dropout=dropout, batch_first=True)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, T, D]
        batch_size = x.size(0)
        
        # 扩展 query 以匹配当前的 batch size -> [B, 1, D]
        q = self.quality_query.expand(batch_size, -1, -1)
        
        # Cross-Attention: Query 寻找 Key(音频序列) 中的关键信息，并聚合 Value
        # attn_out: [B, 1, D]
        attn_out, _ = self.mha(query=q, key=x, value=x)
        
        # 残差连接 (将 query 自身信息保留) + 归一化
        out = self.layer_norm(q + self.dropout(attn_out))
        
        # 压缩掉序列维度 (变成了 1)，得到最终的定长向量 [B, D]
        return out.squeeze(1)

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
        
        self.ffd = nn.Sequential(
            nn.Linear(self.in_features, self.ffd_hidden_size),
            nn.GELU(), # 将 ReLU 换成近两年更常用的 GELU
            nn.Dropout(0.1),
            nn.Linear(self.ffd_hidden_size, self.in_features)
        )
        
        # --- 核心改进：引入质量查询注意力池化 ---
        self.query_pooling = QualityQueryPooling(d_model=self.in_features, num_heads=8)
        
        
       
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.in_features , 1)
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
        #features = self.ffd(ssl_feature) # [B, T, D]
        
        # --- 核心操作：不再使用 mean_pooled ---
        # 使用 Query 主动在 T 个帧中提取最重要的特征
        pooled_features = self.query_pooling(ssl_feature) # [B, D]
        
        pooled_features = self.dropout(pooled_features)
        
        # MOS预测
        mos_score = self.fc(pooled_features)
        mos_score = self.mos_activation(mos_score) * 4.0 + 1.0  # 映射到1-5范围
        
        return mos_score.squeeze(-1)  # [B,]2

# 以下所有代码保持不变...
class MyDataset1(Dataset):  # train
    def __init__(self, wavdir, mos_list):
        self.data_list = []
        with open(mos_list, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                wavname = parts[0]
                mos = float(parts[1])
                self.data_list.append([wavname, mos])
        self.wavdir = wavdir

    def __getitem__(self, idx):
        wavname, score = self.data_list[idx]
        wavpath = os.path.join(self.wavdir, wavname)
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
        scores = torch.tensor(scores, dtype=torch.float32)
        return output_wavs, scores, wavnames


class MyDataset2(Dataset):  # dev
    def __init__(self, wavdir, mos_list):
        self.data_list = []
        with open(mos_list, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                wavname = parts[0]
                mos = float(parts[1])
                self.data_list.append([wavname, mos])
        self.wavdir = wavdir

    def __getitem__(self, idx):
        wavname, score = self.data_list[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav, sr = torchaudio.load(wavpath)
        
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
        scores = torch.tensor(scores, dtype=torch.float32)
        return output_wavs, scores, wavnames


class MyDataset3(Dataset):  # test
    def __init__(self, wavdir, mos_list):
        self.data_list = []
        with open(mos_list, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                wavname = parts[0]
                mos = float(parts[1])
                self.data_list.append([wavname, mos])
        self.wavdir = wavdir

    def __getitem__(self, idx):
        wavname, score = self.data_list[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav, sr = torchaudio.load(wavpath)
        
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
        scores = torch.tensor(scores, dtype=torch.float32)
        return output_wavs, scores, wavnames


def systemID(uttID):
    return uttID.split('-')[0]


def evaluate_model(net, dataloader, mos_list_path, device, desc='Evaluating'):
    predictions = {}
    net.eval()
    
    with tqdm(dataloader, desc=desc) as pbar:
        for i, data in enumerate(pbar, 0):
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = net(inputs)
            
            if outputs.dim() == 0:
                output = float(outputs.cpu().detach().numpy())
            else:
                output = float(outputs.squeeze().cpu().detach().numpy())
            
            predictions[filenames[0]] = output
            pbar.set_postfix({'processed': len(predictions)})
    
    true_MOS = {}
    with open(mos_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            uttID = parts[0]
            MOS = float(parts[1])
            true_MOS[uttID] = MOS
    
    sorted_uttIDs = sorted(predictions.keys())
    truths = []
    preds = []
    for uttID in sorted_uttIDs:
        truths.append(true_MOS[uttID])
        preds.append(predictions[uttID])
    
    truths = np.array(truths)
    preds = np.array(preds)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--wavlm_model', type=str, default='/home/duhuipeng/SAMOS/wavlm-base-plus', 
                        help='HuggingFace WavLM model name (default: microsoft/wavlm-base)')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, 
                        help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='checkpoints', 
                        help='Output directory for your trained checkpoints')
    parser.add_argument('--gpu', type=int, required=False, default=0, 
                        help='GPU device ID to use (default: 0)')
    parser.add_argument('--max_epochs', type=int, required=False, default=30, 
                        help='Maximum number of epochs to train (default: 30)')
    parser.add_argument('--batch_size', type=int, required=False, default=2, 
                        help='Batch size for training (default: 2)')
    parser.add_argument('--learning_rate', type=float, required=False, default=0.0001, 
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--fine_tune_ssl', action='store_true', 
                        help='Fine-tune WavLM parameters (default: freeze WavLM)')
    
    args = parser.parse_args()

    datadir = args.datadir
    ckptdir = args.outdir
    my_checkpoint = args.finetune_from_checkpoint
    gpu_id = args.gpu
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    wavlm_model_name = args.wavlm_model
    fine_tune_ssl = args.fine_tune_ssl
    
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir, exist_ok=True)
        print(f'Created output directory: {ckptdir}')
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f'Using GPU: {gpu_id}')
    else:
        device = torch.device("cpu")
        print('Using CPU (GPU not available)')
    
    print('DEVICE: ' + str(device))
    print(f'Checkpoints will be saved to: {ckptdir}')
    print(f'Training will run for maximum {max_epochs} epochs')
    print(f'WavLM model: {wavlm_model_name}')
    print(f'Fine-tune WavLM: {fine_tune_ssl}')
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {learning_rate}')

    wavdir = os.path.join(datadir, 'wav')
    trainlist = os.path.join(datadir, 'sets/train_mos_list.txt')
    validlist = os.path.join(datadir, 'sets/val_mos_list.txt')
    testlist = os.path.join(datadir, 'sets/test_mos_list.txt')

    # 加载WavLM模型
    print(f'Loading WavLM model: {wavlm_model_name}...')
    ssl_model = WavLMModel.from_pretrained(wavlm_model_name)
    
    # 获取模型配置信息
    hidden_size = ssl_model.config.hidden_size
    num_layers = ssl_model.config.num_hidden_layers
    
    print(f'WavLM model loaded:')
    print(f'  - Hidden size: {hidden_size}')
    print(f'  - Number of layers: {num_layers}')
    print(f'  - Feature extraction: Average pooling of all layers')
    print(f'  - WavLM participates in training: {fine_tune_ssl}')
    
    # 创建数据集
    trainset = MyDataset1(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                           num_workers=2, collate_fn=trainset.collate_fn)

    validset = MyDataset2(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, 
                           num_workers=2, collate_fn=validset.collate_fn)

    testset = MyDataset3(wavdir, testlist)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, 
                          num_workers=2, collate_fn=testset.collate_fn)

    # 创建MOS预测器
    net = MosPredictor(ssl_model, hidden_size, fine_tune_ssl=fine_tune_ssl)
    net = net.to(device)

    if my_checkpoint is not None:
        net.load_state_dict(torch.load(my_checkpoint, map_location=device))
        print(f'Loaded checkpoint from: {my_checkpoint}')
    
    # 计算可训练参数
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)')
    
    # 设置优化器
    # 如果WavLM参与训练，优化器需要包含WavLM的参数
    if fine_tune_ssl:
        # 优化所有参数（包括WavLM）
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        print('Optimizer: SGD (all parameters including WavLM)')
    else:
        # 只优化MOS预测器的参数（不包括WavLM）
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        print('Optimizer: SGD (only MOS predictor parameters, WavLM frozen)')
    
    criterion = nn.MSELoss()

    # 初始化跟踪变量
    best_val_utt_SRCC = -1.0
    best_epoch = 0
    patience = 20
    orig_patience = patience
    
    val_srcc_history = []
    train_loss_history = []
    
    print(f'\n开始训练，共{max_epochs}轮...')
    
    for epoch in range(1, max_epochs + 1):
        net.train()
        running_loss = 0.0
        STEPS = 0
        
        print(f'\n--- Epoch {epoch}/{max_epochs} ---')
        print('Training:')
        with tqdm(trainloader, desc=f'Epoch {epoch}') as pbar:
            for i, data in enumerate(pbar, 0):
                inputs, labels, filenames = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                STEPS += 1
                running_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = running_loss / STEPS
        train_loss_history.append(avg_train_loss)
        print(f'Average training loss: {avg_train_loss:.4f}')
        
        # 验证集评估
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        
        print('Validating:')
        val_results = evaluate_model(net, validloader, validlist, device, 'Validation')
        val_metrics = calculate_metrics(val_results['utt_truths'], val_results['utt_preds'], 'val_utt_')
        val_utt_SRCC = val_metrics['val_utt_SRCC']
        val_srcc_history.append(val_utt_SRCC)
        
        print(f'[Validation Utterance] SRCC: {val_utt_SRCC:.6f}')
        print(f'[Validation Utterance] LCC: {val_metrics["val_utt_LCC"]:.6f}')
        print(f'[Validation Utterance] MSE: {val_metrics["val_utt_MSE"]:.6f}')
        
        # 检查是否达到最佳验证集SRCC
        if val_utt_SRCC > best_val_utt_SRCC:
            best_val_utt_SRCC = val_utt_SRCC
            best_epoch = epoch
            print(f'\n*** New best validation SRCC: {best_val_utt_SRCC:.6f} (Epoch {epoch}) ***')
            
            # 保存最佳模型
            path = os.path.join(ckptdir, 'best_model.pt')
            torch.save(net.state_dict(), path)
            print(f'Best model saved to: {path}')
            patience = orig_patience
        else:
            patience -= 1
            if patience == 0:
                print(f'Validation SRCC not improved for {orig_patience} epochs. Early stopping at epoch {epoch}')
                break
        
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    
    # 训练完成后加载最佳模型进行最终测试
    print(f'\n{"="*60}')
    print('Training completed! Loading best model for final testing...')
    print(f'Best epoch: {best_epoch}')
    print(f'Best validation SRCC: {best_val_utt_SRCC:.6f}')
    
    # 加载最佳模型
    best_model_path = os.path.join(ckptdir, 'best_model.pt')
    net.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f'Loaded best model: {best_model_path}')
    
    # 最终测试评估
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    
    print('\nRunning final test evaluation...')
    test_results = evaluate_model(net, testloader, testlist, device, 'Final Testing')
    
    # 计算测试集指标
    test_utt_metrics = calculate_metrics(test_results['utt_truths'], test_results['utt_preds'], 'test_utt_')
    test_sys_metrics = calculate_metrics(test_results['sys_truths'], test_results['sys_preds'], 'test_sys_')
    
    print(f'\n{"="*60}')
    print('FINAL TEST RESULTS:')
    print('Utterance-level metrics:')
    print(f'  - MSE: {test_utt_metrics["test_utt_MSE"]:.6f}')
    print(f'  - LCC: {test_utt_metrics["test_utt_LCC"]:.6f}')
    print(f'  - SRCC: {test_utt_metrics["test_utt_SRCC"]:.6f}')
    print(f'  - KTAU: {test_utt_metrics["test_utt_KTAU"]:.6f}')
    print('System-level metrics:')
    print(f'  - MSE: {test_sys_metrics["test_sys_MSE"]:.6f}')
    print(f'  - LCC: {test_sys_metrics["test_sys_LCC"]:.6f}')
    print(f'  - SRCC: {test_sys_metrics["test_sys_SRCC"]:.6f}')
    print(f'  - KTAU: {test_sys_metrics["test_sys_KTAU"]:.6f}')
    print(f'{"="*60}')
    
    # 保存训练结果
    results_path = os.path.join(ckptdir, 'training_results.txt')
    with open(results_path, 'w') as f:
        f.write('WavLM MOS Prediction Training Results\n')
        f.write('='*50 + '\n')
        f.write(f'Best epoch: {best_epoch}\n')
        f.write(f'Best validation SRCC: {best_val_utt_SRCC:.6f}\n')
        f.write(f'Final test SRCC: {test_utt_metrics["test_utt_SRCC"]:.6f}\n')
        f.write(f'WavLM model: {wavlm_model_name}\n')
        f.write(f'WavLM participates in training: {fine_tune_ssl}\n')
        f.write(f'Feature extraction: Average pooling of all layers\n')
        f.write(f'Total parameters: {total_params:,}\n')
        f.write(f'Trainable parameters: {trainable_params:,}\n')
    
    print(f'Training results saved to: {results_path}')
    print(f'\nTraining completed! Best model: {best_model_path}')