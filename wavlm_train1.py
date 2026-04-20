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

class MDTA1D(nn.Module):
    """
    多Dconv头转置注意力 (Multi-Dconv Head Transposed Attention)
    来源：CVPR 顶级图像恢复模型 Restormer
    跨界优势：提取全局通道协方差（噪声/信号比例），不丢失时间维度信息。
    """
    def __init__(self, channels, num_heads=8):
        super(MDTA1D, self).__init__()
        self.num_heads = num_heads
        # 可学习的温度系数，用于动态调节 softmax 的尖锐度
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1x1 卷积代替 Linear，便于直接处理序列 [B, C, T]
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        
        # 3x3 深度可分离卷积，在提取全局注意力前，先捕获局部的时域瞬态伪影 (如爆音)
        self.qkv_dwconv = nn.Conv1d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: [B, T, C] -> 转换为 [B, C, T] 适应卷积
        x = x.transpose(1, 2)
        b, c, t = x.shape

        # 1. 局部时域上下文注入生成 Q, K, V
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 2. 重塑为多头维度: [B, num_heads, C//num_heads, T]
        q = q.view(b, self.num_heads, c // self.num_heads, t)
        k = k.view(b, self.num_heads, c // self.num_heads, t)
        v = v.view(b, self.num_heads, c // self.num_heads, t)

        # --- 核心颠覆点：Transposed Attention (特征通道协方差注意力) ---
        # 沿着时间轴 T 进行 L2 归一化。
        # 作用：让注意力权重对音频的长短变幻免疫，严格衡量通道间的相关性！
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 注意力矩阵大小为 [C/h, C/h]，完全摆脱了 T 的限制
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 注意力矩阵作用于 V，完成特征重组
        out = (attn @ v)

        # 3. 恢复形状并输出
        out = out.view(b, c, t)
        out = self.project_out(out)

        # 返回形状: [B, T, C]
        return out.transpose(1, 2)

class GDFN1D(nn.Module):
    """
    门控深度卷积前馈网络 (Gated-Dconv Feed-Forward Network)
    来源：CVPR Restormer
    替代传统的 Linear-ReLU-Linear，门控机制能在非线性阶段有效抑制无用的背景特征。
    """
    def __init__(self, channels, expansion_factor=2.66):
        super(GDFN1D, self).__init__()
        hidden_channels = int(channels * expansion_factor)

        self.project_in = nn.Conv1d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        # 分割成两条支路进行门控融合
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        # 一半走 GELU 激活并作为门开关，控制另一半特征的流出
        x = F.gelu(x1) * x2
        
        x = self.project_out(x)
        return x.transpose(1, 2)

class RestormerBlock1D(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA1D(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN1D(channels)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class MGLU(nn.Module):
    """
    混合门控线性单元 (Mixed Gated Linear Unit)
    针对序列特征优化，引入门控机制与局部时域混合
    """
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        # 门控路径和值路径
        self.w1 = nn.Linear(d_model, d_ffn) # 门控支路
        self.w2 = nn.Linear(d_model, d_ffn) # 值支路
        self.w3 = nn.Linear(d_ffn, d_model) # 输出投影
        
        # 局部混合器：通过深度卷积捕获音频特征在时间维度(T)上的局部关联
        # kernel_size=3 可以让模型观察到当前帧及其前后帧的变化
        self.local_mixer = nn.Conv1d(d_ffn, d_ffn, kernel_size=3, padding=1, groups=d_ffn)
        
        self.act = nn.SiLU()  # 使用 SiLU (Swish) 激活，梯度更平滑
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [B, T, D]
        residual = x
        x = self.layer_norm(x)
        
        # 投影到中间空间
        gate = self.w1(x)  # [B, T, D_ffn]
        value = self.w2(x) # [B, T, D_ffn]
        
        # 对门控支路进行局部时域混合 (Mixing)
        # 需要转置维度以符合 Conv1d 的输入 [B, C, T]
        gate = gate.transpose(1, 2)
        gate = self.local_mixer(gate)
        gate = gate.transpose(1, 2)
        
        # 混合门控操作：门控信号 * 激活后的值
        # 这允许模型动态选择哪些特征对质量评估更重要
        x = gate * self.act(value)
        
        x = self.w3(x)
        x = self.dropout(x)
        
        return x + residual


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
        
        self.mglu = MGLU(d_model=self.in_features, d_ffn=self.ffd_hidden_size)
        self.restormer_blocks = nn.Sequential(
            RestormerBlock1D(self.in_features, num_heads=8),
            RestormerBlock1D(self.in_features, num_heads=8)
        )
        
       
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.in_features*2 , 1)
        self.mos_activation = nn.Sigmoid()

    def forward(self, wav):
        # 输入wav形状: [B, 1, T]
        wav = wav.squeeze(1)  # [B, T]

        # WavLM前向
        if self.fine_tune_ssl:
            outputs = self.ssl_model(wav, output_hidden_states=True)
        else:
            with torch.no_grad():
                outputs = self.ssl_model(wav, output_hidden_states=True)

        # 多层隐藏状态平均（保持你原来的逻辑）
        hidden_states = outputs.hidden_states
        all_layer_features = hidden_states[1:]  # 去掉embedding层
        stacked_features = torch.stack(all_layer_features, dim=2)
        ssl_feature = torch.mean(stacked_features, dim=2)  # [B, T, D]

        # ================= 两路并行 =================
        # 第一路：MGLU
        mglu_out = self.mglu(ssl_feature)              # [B, T, D]
        mglu_pooled = torch.mean(mglu_out, dim=1)     # [B, D]

        # 第二路：Restormer
        restormer_out = self.restormer_blocks(ssl_feature)  # [B, T, D]
        restormer_pooled = torch.mean(restormer_out, dim=1)  # [B, D]

        # ================= 拼接 =================
        pooled_features = torch.cat(
            [mglu_pooled, restormer_pooled], dim=1
        )  # [B, 2D]

        pooled_features = self.dropout(pooled_features)

        # MOS预测（fc 输入维度已为 2*hidden_size）
        mos_score = self.fc(pooled_features)
        mos_score = self.mos_activation(mos_score) * 4.0 + 1.0

        return mos_score.squeeze(-1)

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