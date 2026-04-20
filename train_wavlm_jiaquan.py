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
        
        # 保持原有参数设置
        self.in_features = self.hidden_size
        self.ffd_hidden_size = 4096
        self.attn_layer_num = 4
        
        # 层注意力机制 - 为每一层学习注意力权重
        self.layer_attention = nn.Sequential(
            nn.Linear(self.hidden_size, 128),  # 压缩维度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # 输出每一层的注意力分数
        )
        
        # 投影层：将每一层的特征投影到统一空间
        self.layer_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 保持原有的自注意力层
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
        
        # 保持原有的前馈网络
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
        
        # 获取所有隐藏层的特征
        hidden_states = outputs.hidden_states
        
        # 收集所有隐藏层特征（跳过输入嵌入层）
        layer_features_list = []
        for layer_idx in range(1, len(hidden_states)):
            layer_feat = hidden_states[layer_idx]  # [B, T, D]
            # 应用投影
            proj_feat = self.layer_projection(layer_feat)
            layer_features_list.append(proj_feat)
        
        # 计算每一层的注意力权重
        batch_size, seq_len, feat_dim = layer_features_list[0].shape
        num_layers = len(layer_features_list)
        
        # 对每一层特征计算平均表示
        layer_reps = []
        for feat in layer_features_list:
            layer_rep = torch.mean(feat, dim=1)  # [B, D]
            layer_reps.append(layer_rep)
        
        # 堆叠层表示
        stacked_reps = torch.stack(layer_reps, dim=1)  # [B, L, D]
        
        # 计算注意力分数
        attention_scores = self.layer_attention(stacked_reps)  # [B, L, 1]
        attention_scores = attention_scores.squeeze(-1)  # [B, L]
        
        # 应用softmax得到注意力权重
        layer_weights = F.softmax(attention_scores, dim=1)  # [B, L]
        
        # 加权融合各层特征
        fused_feature = torch.zeros(batch_size, seq_len, feat_dim, 
                                   device=wav.device, dtype=wav.dtype)
        
        for i in range(num_layers):
            # 扩展权重维度
            weight_expanded = layer_weights[:, i:i+1, None]  # [B, 1, 1]
            weight_expanded = weight_expanded.expand(-1, seq_len, feat_dim)  # [B, T, D]
            
            # 加权累加
            fused_feature = fused_feature + weight_expanded * layer_features_list[i]
        
        # 后续处理（保持与原代码一致）
        ssl_feature = self.ffd(fused_feature)
        
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