# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import argparse
import fairseq
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
random.seed(1584)


class MosPredictor(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super(MosPredictor, self).__init__()
        self.ssl_model = ssl_model
        
        # 根据Generator配置设置参数
        self.in_features = 1024
        self.ffd_hidden_size = 4096
        self.attn_layer_num = 4
        for param in self.ssl_model.parameters():
            param.requires_grad = False  # 不计算梯度
        # HuBERT Large的输出维度已经是1024，不需要投影层
        if ssl_out_dim != self.in_features:
            self.proj_layer = nn.Linear(ssl_out_dim, self.in_features)
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

    def forward(self, wav):
        """
        前端：HuBERT特征提取
        后端：Generator架构
        """
        # 处理音频输入：确保是单声道，采样率16kHz
        wav = wav.squeeze(1)  # [B, T]
        
        # HuBERT特征提取
        res = self.ssl_model(wav, mask=False, features_only=True)
        ssl_feature = res['x']  # [B, T, 1024]
        
        # 维度对齐
        ssl_feature = self.proj_layer(ssl_feature)  # [B, T, 1024]
        
        # Generator后端处理
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
        
        return mos_score


class MyDataset1(Dataset):  # train
    def __init__(self, wavdir, mos_list):
        self.data_list = []
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos = float(parts[1])
            self.data_list.append([wavname, mos])
        self.wavdir = wavdir

    def __getitem__(self, idx):
        wavname, score = self.data_list[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        return wav, score, wavname

    def __len__(self):
        return len(self.data_list)

    def collate_fn(self, batch):
        wavs, scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key=lambda x: x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        output_wavs = torch.stack(output_wavs, dim=0)
        scores = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_wavs, scores, wavnames


class MyDataset2(Dataset):  # dev
    def __init__(self, wavdir, mos_list):
        self.data_list = []
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos = float(parts[1])
            self.data_list.append([wavname, mos])
        self.wavdir = wavdir

    def __getitem__(self, idx):
        wavname, score = self.data_list[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        return wav, score, wavname

    def __len__(self):
        return len(self.data_list)

    def collate_fn(self, batch):
        wavs, scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key=lambda x: x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        output_wavs = torch.stack(output_wavs, dim=0)
        scores = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_wavs, scores, wavnames


class MyDataset3(Dataset):  # test
    def __init__(self, wavdir, mos_list):
        self.data_list = []
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos = float(parts[1])
            self.data_list.append([wavname, mos])
        self.wavdir = wavdir

    def __getitem__(self, idx):
        wavname, score = self.data_list[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        return wav, score, wavname

    def __len__(self):
        return len(self.data_list)

    def collate_fn(self, batch):
        wavs, scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key=lambda x: x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        output_wavs = torch.stack(output_wavs, dim=0)
        scores = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_wavs, scores, wavnames


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
            outputs = net(inputs)
            
            if outputs.dim() == 2 and outputs.shape[1] == 1:
                output = float(outputs.squeeze().cpu().detach().numpy())
            else:
                scores = torch.mean(outputs.squeeze(-1), dim=1, keepdim=True)
                output = float(scores.cpu().detach().numpy()[0])
            
            predictions[filenames[0]] = output
            pbar.set_postfix({'processed': len(predictions)})
    
    # 读取真实MOS值
    true_MOS = {}
    with open(mos_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
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


def load_hubert_model(model_path, device):
    """专门加载HuBERT模型"""
    print(f"Loading HuBERT model from: {model_path}")
    
    # 加载模型
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
    model = models[0]
    
    # 移除预训练相关的模块（如分类头）
    if hasattr(model, 'remove_pretraining_modules'):
        model.remove_pretraining_modules()
    
    # 设置模型为评估模式
    model.eval()
    model.to(device)
    
    print(f"HuBERT model loaded successfully")
    print(f"Model configuration: {cfg.model}")
    
    return model, cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--fairseq_base_model', type=str, required=True, help='Path to HuBERT model (e.g., hubert_large_ll60k.pt)')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='checkpoints', help='Output directory for your trained checkpoints')
    parser.add_argument('--gpu', type=int, required=False, default=1, help='GPU device ID to use (default: 1)')
    parser.add_argument('--max_epochs', type=int, required=False, default=30, help='Maximum number of epochs to train (default: 30)')
    args = parser.parse_args()

    cp_path = args.fairseq_base_model
    datadir = args.datadir
    ckptdir = args.outdir
    my_checkpoint = args.finetune_from_checkpoint
    gpu_id = args.gpu
    max_epochs = args.max_epochs
    
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

    wavdir = os.path.join(datadir, 'wav')
    trainlist = os.path.join(datadir, 'sets/train_mos_list.txt')
    validlist = os.path.join(datadir, 'sets/val_mos_list.txt')
    testlist = os.path.join(datadir, 'sets/test_mos_list.txt')

    # 专门处理HuBERT模型
    ssl_model_type = cp_path.split('/')[-1]
    
    # HuBERT模型的特征维度配置
    if 'hubert' in ssl_model_type.lower():
        if 'large' in ssl_model_type.lower():
            SSL_OUT_DIM = 1024  # HuBERT Large的隐藏层维度
        elif 'base' in ssl_model_type.lower():
            SSL_OUT_DIM = 768   # HuBERT Base的隐藏层维度
        else:
            SSL_OUT_DIM = 1024  # 默认假设为Large版本
    else:
        # 保持对原有Wav2Vec2模型的支持
        if ssl_model_type == 'wav2vec_small.pt':
            SSL_OUT_DIM = 768
        elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
            SSL_OUT_DIM = 1024
        else:
            print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
            exit()
    
    print(f'Detected SSL model type: {ssl_model_type}')
    print(f'Using SSL output dimension: {SSL_OUT_DIM}')

    # 加载HuBERT模型
    ssl_model, ssl_cfg = load_hubert_model(cp_path, device)
    
    trainset = MyDataset1(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)

    validset = MyDataset2(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=2, collate_fn=validset.collate_fn)

    testset = MyDataset3(wavdir, testlist)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2, collate_fn=testset.collate_fn)

    # 创建MOS预测器
    net = MosPredictor(ssl_model, SSL_OUT_DIM)
    net = net.to(device)

    if my_checkpoint is not None:
        net.load_state_dict(torch.load(my_checkpoint))
        print(f'Loaded checkpoint from: {my_checkpoint}')
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    # 初始化跟踪变量
    best_val_utt_SRCC = -1.0
    best_epoch = 0
    patience = 20
    orig_patience = patience
    
    # 用于记录每轮的验证集SRCC
    val_srcc_history = []
    train_loss_history = []
    
    print(f'\n开始训练HuBERT-based MOS预测器，共{max_epochs}轮...')
    
    for epoch in range(1, max_epochs + 1):
        STEPS = 0
        net.train()
        running_loss = 0.0
        
        print(f'\n--- Epoch {epoch}/{max_epochs} ---')
        print('Training:')
        with tqdm(trainloader, desc=f'Epoch {epoch}') as pbar:
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
        print(f'AVG EPOCH TRAIN LOSS: {avg_train_loss:.4f}')
        
        # 验证集评估
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        
        print('Validating:')
        val_results = evaluate_model(net, validloader, validlist, device, 'Validation')
        val_metrics = calculate_metrics(val_results['utt_truths'], val_results['utt_preds'], 'val_utt_')
        val_utt_SRCC = val_metrics['val_utt_SRCC']
        val_srcc_history.append(val_utt_SRCC)
        
        print(f'[val_UTTERANCE] Spearman rank correlation coefficient= {val_utt_SRCC:.6f}')
        print(f'[val_UTTERANCE] Kendall Tau rank correlation coefficient= {val_metrics["val_utt_KTAU"]:.6f}')
        
        # 检查是否达到最佳验证集SRCC
        if val_utt_SRCC > best_val_utt_SRCC:
            best_val_utt_SRCC = val_utt_SRCC
            best_epoch = epoch
            print(f'\n*** 新的最佳验证集SRCC: {best_val_utt_SRCC:.6f} (Epoch {epoch}) ***')
            
            # 保存最佳模型
            path = os.path.join(ckptdir, 'best_model.pt')
            torch.save(net.state_dict(), path)
            print(f'最佳模型已保存至: {path}')
            patience = orig_patience
        else:
            patience -= 1
            if patience == 0:
                print(f'验证集SRCC连续{orig_patience}轮未提升，在第{epoch}轮提前停止训练')
                break
        
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    
    # 训练完成后加载最佳模型进行最终测试
    print(f'\n{"="*60}')
    print('训练完成！加载最佳模型进行最终测试...')
    print(f'最佳轮次: Epoch {best_epoch}')
    print(f'最佳验证集SRCC: {best_val_utt_SRCC:.6f}')
    
    # 加载最佳模型
    best_model_path = os.path.join(ckptdir, 'best_model.pt')
    net.load_state_dict(torch.load(best_model_path))
    print(f'已加载最佳模型: {best_model_path}')
    
    # 最终测试评估
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    
    print('\n进行最终测试评估...')
    test_results = evaluate_model(net, testloader, testlist, device, 'Final Testing')
    
    # 计算测试集指标
    test_utt_metrics = calculate_metrics(test_results['utt_truths'], test_results['utt_preds'], 'test_utt_')
    test_sys_metrics = calculate_metrics(test_results['sys_truths'], test_results['sys_preds'], 'test_sys_')
    
    print(f'\n{"="*60}')
    print('最终测试结果:')
    print('句子级指标:')
    print(f'  - MSE: {test_utt_metrics["test_utt_MSE"]:.6f}')
    print(f'  - LCC: {test_utt_metrics["test_utt_LCC"]:.6f}')
    print(f'  - SRCC: {test_utt_metrics["test_utt_SRCC"]:.6f}')
    print(f'  - KTAU: {test_utt_metrics["test_utt_KTAU"]:.6f}')
    print('系统级指标:')
    print(f'  - MSE: {test_sys_metrics["test_sys_MSE"]:.6f}')
    print(f'  - LCC: {test_sys_metrics["test_sys_LCC"]:.6f}')
    print(f'  - SRCC: {test_sys_metrics["test_sys_SRCC"]:.6f}')
    print(f'  - KTAU: {test_sys_metrics["test_sys_KTAU"]:.6f}')
    print(f'{"="*60}')
    
    # 绘制验证集SRCC变化曲线
    plt.figure(figsize=(12, 6))
    
    # 子图1：验证集SRCC变化
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(val_srcc_history) + 1)
    plt.plot(epochs_range, val_srcc_history, 'b-o', linewidth=2, markersize=4, label='验证集SRCC')
    
    # 标记最佳点
    if best_epoch <= len(val_srcc_history):
        plt.plot(best_epoch, val_srcc_history[best_epoch-1], 'ro', markersize=8, 
                label=f'最佳轮次 (Epoch {best_epoch})')
        plt.annotate(f'最佳: {best_val_utt_SRCC:.4f}', 
                    xy=(best_epoch, val_srcc_history[best_epoch-1]), 
                    xytext=(best_epoch+1, val_srcc_history[best_epoch-1]),
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.xlabel('训练轮次')
    plt.ylabel('验证集句子级SRCC')
    plt.title('HuBERT-based模型验证集SRCC变化')
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
    plot_path = os.path.join(ckptdir, 'hubert_training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'训练过程可视化图已保存至: {plot_path}')
    
    print(f'\nHuBERT-based MOS预测器训练完成！')
    print(f'最佳模型文件: {best_model_path}')
    print(f'最佳验证集SRCC: {best_val_utt_SRCC:.6f} (Epoch {best_epoch})')