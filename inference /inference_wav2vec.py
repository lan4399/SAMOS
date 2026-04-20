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
        wav = wav.squeeze(1)
        res = self.ssl_model(wav, mask=False, features_only=True)
        ssl_feature = res['x']
        
        ssl_feature = self.proj_layer(ssl_feature)
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
    

class MyDataset(Dataset):
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
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        output_wavs = torch.stack(output_wavs, dim=0)
        scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
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


def main_inference(checkpoint_path, test_datadir, fairseq_model_path, gpu_id=0):
    """直接推理模式：加载现有权重并测试指定数据集"""
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f'使用GPU: {gpu_id}')
    else:
        device = torch.device("cpu")
        print('使用CPU (GPU不可用)')
    
    print(f'加载fairseq模型: {fairseq_model_path}')
    ssl_model_type = fairseq_model_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('错误: 不支持的SSL模型类型: ' + ssl_model_type)
        exit()
    
    # 加载fairseq模型
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([fairseq_model_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()
    
    # 初始化MOS预测器
    net = MosPredictor(ssl_model, SSL_OUT_DIM)
    net = net.to(device)
    
    # 加载训练好的权重
    print(f'加载训练好的权重: {checkpoint_path}')
    if not os.path.exists(checkpoint_path):
        print(f'错误: 权重文件不存在: {checkpoint_path}')
        return
    
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print('权重加载成功!')
    
    # 准备测试数据
    test_wavdir = os.path.join(test_datadir, 'wav')
    test_mos_list = os.path.join(test_datadir, 'sets/test_mos_list.txt')
    
    if not os.path.exists(test_wavdir):
        print(f'错误: 测试数据目录不存在: {test_wavdir}')
        return
    
    if not os.path.exists(test_mos_list):
        print(f'错误: 测试MOS列表不存在: {test_mos_list}')
        return
    
    print(f'测试数据目录: {test_wavdir}')
    print(f'测试MOS列表: {test_mos_list}')
    
    # 创建测试数据集
    testset = MyDataset(test_wavdir, test_mos_list)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2, collate_fn=testset.collate_fn)
    
    print(f'测试集样本数量: {len(testset)}')
    
    # 进行推理测试
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    
    print('\n开始推理测试...')
    test_results = evaluate_model(net, testloader, test_mos_list, device, '推理测试')
    
    # 计算测试集指标
    test_utt_metrics = calculate_metrics(test_results['utt_truths'], test_results['utt_preds'], 'test_utt_')
    test_sys_metrics = calculate_metrics(test_results['sys_truths'], test_results['sys_preds'], 'test_sys_')
    
    # 打印详细结果
    print(f'\n{"="*60}')
    print('推理测试结果:')
    print(f'测试数据集: {test_datadir}')
    print(f'模型权重: {checkpoint_path}')
    print(f'{"="*60}')
    
    print('句子级指标:')
    print(f'  - 样本数量: {len(test_results["utt_truths"])}')
    print(f'  - MSE: {test_utt_metrics["test_utt_MSE"]:.6f}')
    print(f'  - LCC: {test_utt_metrics["test_utt_LCC"]:.6f}')
    print(f'  - SRCC: {test_utt_metrics["test_utt_SRCC"]:.6f}')
    print(f'  - KTAU: {test_utt_metrics["test_utt_KTAU"]:.6f}')
    
    print('系统级指标:')
    print(f'  - 系统数量: {len(test_results["sys_truths"])}')
    print(f'  - MSE: {test_sys_metrics["test_sys_MSE"]:.6f}')
    print(f'  - LCC: {test_sys_metrics["test_sys_LCC"]:.6f}')
    print(f'  - SRCC: {test_sys_metrics["test_sys_SRCC"]:.6f}')
    print(f'  - KTAU: {test_sys_metrics["test_sys_KTAU"]:.6f}')
    print(f'{"="*60}')
    
    # 保存详细结果到文件
    import json
    from datetime import datetime
    
    result_data = {
        'test_datadir': test_datadir,
        'checkpoint_path': checkpoint_path,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sample_count': len(test_results["utt_truths"]),
        'system_count': len(test_results["sys_truths"]),
        'utterance_level': test_utt_metrics,
        'system_level': test_sys_metrics,
        'predictions': test_results['predictions']
    }
    
    # 创建结果文件名
    base_name = os.path.basename(checkpoint_path).replace('.pt', '')
    test_dir_name = os.path.basename(os.path.normpath(test_datadir))
    result_file = f'inference_results_{base_name}_{test_dir_name}.json'
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f'详细结果已保存至: {result_file}')
    
    return result_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MOS预测模型推理')
    
    # 推理模式参数
    parser.add_argument('--inference-mode', action='store_true', 
                       help='启用推理模式（直接测试现有模型）')
    parser.add_argument('--checkpoint-path', type=str, 
                       help='训练好的模型权重路径')
    parser.add_argument('--test-datadir', type=str, 
                       help='测试数据集目录（包含wav/和sets/目录）')
    parser.add_argument('--fairseq-model', type=str, 
                       default='w2v_large_lv_fsh_swbd_cv.pt',
                       help='fairseq基础模型路径')
    parser.add_argument('--gpu', type=int, default=0, 
                       help='GPU设备ID')
    
    # 原始训练模式参数（保持兼容）
    parser.add_argument('--datadir', type=str, 
                       help='训练数据目录（传统训练模式使用）')
    parser.add_argument('--fairseq_base_model', type=str, 
                       help='fairseq基础模型路径（传统训练模式使用）')
    parser.add_argument('--finetune_from_checkpoint', type=str, 
                       help='微调检查点路径')
    parser.add_argument('--outdir', type=str, default='checkpoints', 
                       help='输出目录')
    parser.add_argument('--max_epochs', type=int, default=30, 
                       help='最大训练轮次')
    
    args = parser.parse_args()
    
    # 判断运行模式
    if args.inference_mode:
        # 推理模式
        if not args.checkpoint_path or not args.test_datadir:
            print("错误: 推理模式需要指定 --checkpoint-path 和 --test-datadir")
            exit(1)
            
        print("=" * 60)
        print("MOS预测模型推理模式")
        print("=" * 60)
        
        main_inference(
            checkpoint_path=args.checkpoint_path,
            test_datadir=args.test_datadir,
            fairseq_model_path=args.fairseq_model,
            gpu_id=args.gpu
        )
        
    else:
        # 传统训练模式（保持原有逻辑）
        print("传统训练模式（原有逻辑）")
        # ... 这里保留原有的训练代码逻辑
        # 由于代码较长，这里省略训练部分的重复代码
        # 实际使用时需要将原来的训练代码放在这里
        print("注意: 传统训练模式代码需要单独实现")