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
        #self.fc = nn.Linear(self.in_features * 2, 1)
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
        #ssl_feature = self.ffd(ssl_feature)
        
        #tmp_ssl_feature = ssl_feature
        #for attn_layer in self.attn:
        #    tmp_ssl_feature, _ = attn_layer(tmp_ssl_feature, tmp_ssl_feature, tmp_ssl_feature)
        mean_pooled = torch.mean(ssl_feature, dim=1)
        #mean_pooled = torch.mean(tmp_ssl_feature, dim=1)
        #max_pooled = torch.max(ssl_feature, dim=1)[0]
        #pooled_features = torch.cat([mean_pooled, max_pooled], dim=1)
        pooled_features = self.dropout(mean_pooled)
        
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


def test_external_dataset(net, wavdir, mos_list_path, wavlm_model_name, device, output_dir='external_test_results'):
    """
    使用训练好的模型测试外部数据集
    
    Args:
        net: 加载了权重的模型
        wavdir: 外部数据集wav文件目录
        mos_list_path: 外部数据集MOS列表文件路径
        wavlm_model_name: WavLM模型名称
        device: 计算设备
        output_dir: 输出结果目录
    """
    print(f"\n{'='*60}")
    print("开始测试外部数据集")
    print(f"音频目录: {wavdir}")
    print(f"MOS列表: {mos_list_path}")
    print(f"WavLM模型: {wavlm_model_name}")
    print(f"设备: {device}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建外部测试集
    testset = MyDataset3(wavdir, mos_list_path)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, 
                          num_workers=2, collate_fn=testset.collate_fn)
    
    # 运行测试
    test_results = evaluate_model(net, testloader, mos_list_path, device, '测试外部数据集')
    
    # 计算指标
    test_utt_metrics = calculate_metrics(test_results['utt_truths'], 
                                        test_results['utt_preds'], 'test_utt_')
    test_sys_metrics = calculate_metrics(test_results['sys_truths'], 
                                       test_results['sys_preds'], 'test_sys_')
    
    # 打印结果
    print(f"\n{'='*60}")
    print("外部数据集测试结果:")
    print(f"{'='*60}")
    print(f"数据集: {mos_list_path}")
    print(f"音频数量: {len(test_results['utt_truths'])}")
    print(f"系统数量: {len(test_results['sys_truths'])}")
    print(f"-"*60)
    print("音频级别指标:")
    print(f"  - MSE:  {test_utt_metrics['test_utt_MSE']:.6f}")
    print(f"  - LCC:  {test_utt_metrics['test_utt_LCC']:.6f}")
    print(f"  - SRCC: {test_utt_metrics['test_utt_SRCC']:.6f}")
    print(f"  - KTAU: {test_utt_metrics['test_utt_KTAU']:.6f}")
    print("系统级别指标:")
    print(f"  - MSE:  {test_sys_metrics['test_sys_MSE']:.6f}")
    print(f"  - LCC:  {test_sys_metrics['test_sys_LCC']:.6f}")
    print(f"  - SRCC: {test_sys_metrics['test_sys_SRCC']:.6f}")
    print(f"  - KTAU: {test_sys_metrics['test_sys_KTAU']:.6f}")
    print(f"{'='*60}")
    
    # 保存详细结果
    results_path = os.path.join(output_dir, 'detailed_results.txt')
    with open(results_path, 'w') as f:
        f.write('外部数据集MOS预测测试结果\n')
        f.write('='*60 + '\n')
        f.write(f'音频目录: {wavdir}\n')
        f.write(f'MOS列表: {mos_list_path}\n')
        f.write(f'WavLM模型: {wavlm_model_name}\n')
        f.write(f'音频数量: {len(test_results["utt_truths"])}\n')
        f.write(f'系统数量: {len(test_results["sys_truths"])}\n\n')
        
        f.write('音频级别指标:\n')
        f.write(f'  MSE:  {test_utt_metrics["test_utt_MSE"]:.6f}\n')
        f.write(f'  LCC:  {test_utt_metrics["test_utt_LCC"]:.6f}\n')
        f.write(f'  SRCC: {test_utt_metrics["test_utt_SRCC"]:.6f}\n')
        f.write(f'  KTAU: {test_utt_metrics["test_utt_KTAU"]:.6f}\n\n')
        
        f.write('系统级别指标:\n')
        f.write(f'  MSE:  {test_sys_metrics["test_sys_MSE"]:.6f}\n')
        f.write(f'  LCC:  {test_sys_metrics["test_sys_LCC"]:.6f}\n')
        f.write(f'  SRCC: {test_sys_metrics["test_sys_SRCC"]:.6f}\n')
        f.write(f'  KTAU: {test_sys_metrics["test_sys_KTAU"]:.6f}\n\n')
        
        f.write('='*60 + '\n')
        f.write('详细预测结果:\n')
        f.write('文件名,预测MOS,真实MOS\n')
        sorted_names = sorted(test_results['predictions'].keys())
        for wavname in sorted_names:
            pred = test_results['predictions'][wavname]
            true_mos = test_results['utt_truths'][sorted_names.index(wavname)]
            f.write(f'{wavname},{pred:.6f},{true_mos}\n')
    
    print(f"\n详细结果已保存到: {results_path}")
    
    # 保存CSV格式的预测结果
    csv_path = os.path.join(output_dir, 'predictions.csv')
    with open(csv_path, 'w') as f:
        f.write('filename,predicted_MOS,true_MOS\n')
        sorted_names = sorted(test_results['predictions'].keys())
        for wavname in sorted_names:
            pred = test_results['predictions'][wavname]
            true_mos = test_results['utt_truths'][sorted_names.index(wavname)]
            f.write(f'{wavname},{pred:.6f},{true_mos}\n')
    
    print(f"CSV格式预测结果已保存到: {csv_path}")
    
    return test_results


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
    
    # 添加测试外部数据集的参数
    parser.add_argument('--test_only', action='store_true',
                       help='Only test the model without training')
    parser.add_argument('--test_wavdir', type=str, default=None,
                       help='Directory containing test wav files (for external test)')
    parser.add_argument('--test_mos_list', type=str, default=None,
                       help='Path to test MOS list file (for external test)')
    parser.add_argument('--external_output_dir', type=str, default='external_test_results',
                       help='Output directory for external test results')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                       help='Path to trained model checkpoint (required for test_only mode)')
    
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
    
    # 测试外部数据集参数
    test_only = args.test_only
    test_wavdir = args.test_wavdir
    test_mos_list = args.test_mos_list
    external_output_dir = args.external_output_dir
    model_checkpoint = args.model_checkpoint if args.test_only else None
    
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
    print(f'WavLM model: {wavlm_model_name}')
    print(f'Fine-tune WavLM: {fine_tune_ssl}')
    
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
    
    # 创建MOS预测器
    net = MosPredictor(ssl_model, hidden_size, fine_tune_ssl=fine_tune_ssl)
    net = net.to(device)
    
    # 检查是否只进行测试
    if test_only:
        if not model_checkpoint:
            print("Error: --model_checkpoint is required in test_only mode")
            exit(1)
        
        if not test_wavdir or not test_mos_list:
            print("Error: --test_wavdir and --test_mos_list are required in test_only mode")
            exit(1)
        
        # 加载训练好的模型权重
        print(f'Loading model checkpoint: {model_checkpoint}')
        net.load_state_dict(torch.load(model_checkpoint, map_location=device))
        
        # 测试外部数据集
        test_external_dataset(
            net=net,
            wavdir=test_wavdir,
            mos_list_path=test_mos_list,
            wavlm_model_name=wavlm_model_name,
            device=device,
            output_dir=external_output_dir
        )
        exit(0)  # 测试完成后退出
    
    # 以下是原有的训练逻辑
    wavdir = os.path.join(datadir, 'wav')
    trainlist = os.path.join(datadir, 'sets/train_mos_list.txt')
    validlist = os.path.join(datadir, 'sets/val_mos_list.txt')
    testlist = os.path.join(datadir, 'sets/test_mos_list.txt')
    
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
    
    # 训练完成后，如果指定了外部测试集，也进行测试
    if test_wavdir and test_mos_list and os.path.exists(test_wavdir) and os.path.exists(test_mos_list):
        print(f'\n{"="*60}')
        print("检测到外部测试集参数，开始测试外部数据集...")
        test_external_dataset(
            net=net,
            wavdir=test_wavdir,
            mos_list_path=test_mos_list,
            wavlm_model_name=wavlm_model_name,
            device=device,
            output_dir=external_output_dir
        )
    else:
        print(f'\n{"="*60}')
        print("注意：要测试外部数据集，请使用以下参数运行：")
        print(f"python your_script.py --test_only --model_checkpoint best_model.pt --test_wavdir /path/to/wavs --test_mos_list /path/to/mos_list.txt")
        print(f"{'='*60}")