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