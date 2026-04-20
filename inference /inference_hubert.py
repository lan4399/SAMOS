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
import json
from datetime import datetime
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


def inference_only_mode(args):
    """仅推理模式：使用训练好的权重测试域外数据集"""
    print("=" * 60)
    print("推理模式：测试域外数据集")
    print("=" * 60)
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    
    print(f'使用设备: {device}')
    
    # 加载HuBERT模型
    ssl_model, ssl_cfg = load_hubert_model(args.fairseq_base_model, device)
    
    # 确定SSL输出维度
    ssl_model_type = args.fairseq_base_model.split('/')[-1]
    if 'hubert' in ssl_model_type.lower():
        if 'large' in ssl_model_type.lower():
            SSL_OUT_DIM = 1024
        elif 'base' in ssl_model_type.lower():
            SSL_OUT_DIM = 768
        else:
            SSL_OUT_DIM = 1024
    else:
        if ssl_model_type == 'wav2vec_small.pt':
            SSL_OUT_DIM = 768
        elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
            SSL_OUT_DIM = 1024
        else:
            print('不支持的SSL模型类型: ' + ssl_model_type)
            exit()
    
    print(f'使用SSL输出维度: {SSL_OUT_DIM}')
    
    # 创建模型并加载权重
    net = MosPredictor(ssl_model, SSL_OUT_DIM)
    net = net.to(device)
    
    # 加载训练好的权重
    if not os.path.exists(args.pretrained_checkpoint):
        print(f"错误: 预训练权重文件不存在: {args.pretrained_checkpoint}")
        return
    
    print(f"加载预训练权重: {args.pretrained_checkpoint}")
    net.load_state_dict(torch.load(args.pretrained_checkpoint, map_location=device))
    print("权重加载成功!")
    
    # 准备测试数据
    test_wavdir = os.path.join(args.test_datadir, 'wav')
    test_mos_list = os.path.join(args.test_datadir, 'sets/test_mos_list.txt')
    
    # 检查文件是否存在
    if not os.path.exists(test_wavdir):
        print(f"警告: wav目录不存在，尝试直接使用测试数据目录")
        test_wavdir = args.test_datadir
    
    if not os.path.exists(test_mos_list):
        # 尝试其他可能的MOS列表文件位置
        possible_locations = [
            os.path.join(args.test_datadir, 'test_mos_list.txt'),
            os.path.join(args.test_datadir, 'mos_list.txt'),
            os.path.join(args.test_datadir, 'sets/mos_list.txt')
        ]
        for loc in possible_locations:
            if os.path.exists(loc):
                test_mos_list = loc
                break
        else:
            print(f"错误: 未找到MOS列表文件")
            return
    
    print(f"测试数据目录: {test_wavdir}")
    print(f"测试MOS列表: {test_mos_list}")
    
    # 创建测试数据集
    testset = MyDataset3(test_wavdir, test_mos_list)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2, 
                          collate_fn=testset.collate_fn)
    
    print(f"测试集样本数量: {len(testset)}")
    
    # 进行推理测试
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    
    print('\n开始推理测试...')
    test_results = evaluate_model(net, testloader, test_mos_list, device, '域外测试')
    
    # 计算测试集指标
    test_utt_metrics = calculate_metrics(test_results['utt_truths'], test_results['utt_preds'], 'test_utt_')
    test_sys_metrics = calculate_metrics(test_results['sys_truths'], test_results['sys_preds'], 'test_sys_')
    
    # 打印详细结果
    print(f'\n{"="*60}')
    print('域外测试结果:')
    print(f'测试数据集: {args.test_datadir}')
    print(f'模型权重: {args.pretrained_checkpoint}')
    print(f'样本数量: {len(test_results["utt_truths"])}')
    print(f'系统数量: {len(test_results["sys_truths"])}')
    print(f'{"="*60}')
    
    if len(test_results["utt_truths"]) > 0:
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
        else:
            print('系统级指标: 无有效系统数据')
    else:
        print('警告: 没有有效的测试结果!')
    
    print(f'{"="*60}')
    
    # 保存详细结果到文件
    result_data = {
        'test_datadir': args.test_datadir,
        'pretrained_checkpoint': args.pretrained_checkpoint,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sample_count': len(test_results["utt_truths"]),
        'system_count': len(test_results["sys_truths"]),
        'utterance_level': test_utt_metrics,
        'system_level': test_sys_metrics,
        'predictions': test_results['predictions']
    }
    
    # 创建结果文件名
    base_name = os.path.basename(args.pretrained_checkpoint).replace('.pt', '')
    test_dir_name = os.path.basename(os.path.normpath(args.test_datadir))
    result_file = f'inference_results_{base_name}_{test_dir_name}.json'
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f'详细结果已保存至: {result_file}')
    
    return result_data


def main():
    parser = argparse.ArgumentParser(description='HuBERT-based MOS预测器')
    
    # 训练模式参数
    parser.add_argument('--datadir', type=str, help='训练数据目录路径')
    parser.add_argument('--fairseq_base_model', type=str, help='HuBERT模型路径')
    parser.add_argument('--finetune_from_checkpoint', type=str, help='微调检查点路径')
    parser.add_argument('--outdir', type=str, default='checkpoints', help='输出目录')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--max_epochs', type=int, default=30, help='最大训练轮次')
    
    # 推理模式参数
    parser.add_argument('--inference', action='store_true', help='启用推理模式')
    parser.add_argument('--pretrained_checkpoint', type=str, help='预训练权重路径')
    parser.add_argument('--test_datadir', type=str, help='测试数据集目录')
    
    args = parser.parse_args()
    
    # 判断运行模式
    if args.inference:
        # 推理模式
        if not args.pretrained_checkpoint or not args.test_datadir:
            print("错误: 推理模式需要指定 --pretrained_checkpoint 和 --test_datadir")
            exit(1)
        
        inference_only_mode(args)
    else:
        # 训练模式
        if not args.datadir or not args.fairseq_base_model:
            print("错误: 训练模式需要指定 --datadir 和 --fairseq_base_model")
            exit(1)
        
        # 原有的训练代码
        train_mode(args)


def train_mode(args):
    """训练模式"""
    cp_path = args.fairseq_base_model
    datadir = args.datadir
    ckptdir = args.outdir
    my_checkpoint = args.finetune_from_checkpoint
    gpu_id = args.gpu
    max_epochs = args.max_epochs
    
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir, exist_ok=True)
        print(f'创建输出目录: {ckptdir}')
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f'使用GPU: {gpu_id}')
    else:
        device = torch.device("cpu")
        print('使用CPU (GPU不可用)')
    
    print('设备: ' + str(device))
    print(f'检查点保存目录: {ckptdir}')
    print(f'训练轮次: {max_epochs}')
    
    wavdir = os.path.join(datadir, 'wav')
    trainlist = os.path.join(datadir, 'sets/train_mos_list.txt')
    validlist = os.path.join(datadir, 'sets/val_mos_list.txt')
    testlist = os.path.join(datadir, 'sets/test_mos_list.txt')
    
    # 确定SSL模型类型和输出维度
    ssl_model_type = cp_path.split('/')[-1]
    if 'hubert' in ssl_model_type.lower():
        if 'large' in ssl_model_type.lower():
            SSL_OUT_DIM = 1024
        elif 'base' in ssl_model_type.lower():
            SSL_OUT_DIM = 768
        else:
            SSL_OUT_DIM = 1024
    else:
        if ssl_model_type == 'wav2vec_small.pt':
            SSL_OUT_DIM = 768
        elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
            SSL_OUT_DIM = 1024
        else:
            print('不支持的SSL模型类型: ' + ssl_model_type)
            exit()
    
    print(f'检测到SSL模型类型: {ssl_model_type}')
    print(f'使用SSL输出维度: {SSL_OUT_DIM}')
    
    # 加载HuBERT模型
    ssl_model, ssl_cfg = load_hubert_model(cp_path, device)
    
    # 创建数据集和数据加载器
    trainset = MyDataset1(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2, 
                           collate_fn=trainset.collate_fn)
    
    validset = MyDataset2(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=2, 
                           collate_fn=validset.collate_fn)
    
    testset = MyDataset3(wavdir, testlist)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2, 
                           collate_fn=testset.collate_fn)
    
    # 创建模型
    net = MosPredictor(ssl_model, SSL_OUT_DIM)
    net = net.to(device)
    
    if my_checkpoint is not None:
        net.load_state_dict(torch.load(my_checkpoint))
        print(f'加载检查点: {my_checkpoint}')
    
    # 训练循环（原有训练代码）
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    
    best_val_utt_SRCC = -1.0
    best_epoch = 0
    patience = 20
    orig_patience = patience
    
    val_srcc_history = []
    train_loss_history = []
    
    print(f'\n开始训练HuBERT-based MOS预测器，共{max_epochs}轮...')
    
    for epoch in range(1, max_epochs + 1):
        # 训练代码（保持原有逻辑）
        STEPS = 0
        net.train()
        running_loss = 0.0
        
        print(f'\n--- 第{epoch}轮/{max_epochs} ---')
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
        
        # 验证和测试代码（保持原有逻辑）
        # ...（原有验证和测试代码）


if __name__ == '__main__':
    main()