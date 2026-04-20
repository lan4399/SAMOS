import torch
import torch.nn as nn
import torch.nn.functional as F




class Loss(nn.Module):
    """
    Implements the clipped MSE loss and categorical loss.
    """
    def __init__(self, alpha, tau, beta, gama):#alpha, tau为0.1，0.25，beta和gama为1，0.5
        super(Loss, self).__init__()
        self.alpha = alpha
        self.tau = tau
        self.beta = beta
        self.gama = gama
        criterion = torch.nn.MSELoss#this
        #criterion2 = torch.max
        self.main_criterion = criterion(reduction="none")
        #self.contrastive_criterion = criterion2()

    def clipped_criterion(self, pred_score, gt_score, criterion_module):#criterion_module=MSE
        # might investigate how to combine masked loss with categorical output
        if pred_score.dim() > 1:
            pred_score = pred_score.squeeze(-1)#压缩predict_score的最后一个维度，变为（B,T)
        loss = criterion_module(pred_score, gt_score)#求MSE,shape=(B,T)
        threshold = torch.abs(pred_score - gt_score) > self.tau#若差值>0.5，则算入loss（防止过拟合），这里threshold返回了一个True/False矩阵，shape=（B，T）
        loss = torch.mean(threshold * loss)#mean函数返回矩阵(B,T)内的所有值的平均
        return loss

    def contra_criterion(self, pred_score, gt_score):
        if pred_score.dim() > 2:
            pred_score = pred_score.mean(dim=1).squeeze(1)
        # pred_score, gt_score: tensor, [batch_size]  
        gt_diff = gt_score.unsqueeze(1) - gt_score.unsqueeze(0)
        pred_diff = pred_score.unsqueeze(1) - pred_score.unsqueeze(0)
        loss = torch.maximum(torch.zeros(gt_diff.shape).to(gt_diff.device), torch.abs(pred_diff - gt_diff) - self.alpha) 
        loss = loss.mean().div(2)
        return loss
       

    def forward(self,pred_score, gt_score, contraloss = 'true'):
        """   gt = ground truth
        Args:
            pred_mean, pred_score: [batch, time, 1]
        """
        # repeat for frame level loss
        #batch = pred_score.shape[0]
        if contraloss =='true':
            contra_loss = self.contra_criterion(pred_score, gt_score)
        elif contraloss =='false':
            contra_loss = 0
        if pred_score.dim() > 1:
            time = pred_score.shape[1]#pred_score=(B,T,1)
            gt_score = gt_score.unsqueeze(1).repeat(1, time)#从（B）变为(B,)变为（B，T），在第二个维度上重复T次
        clipped_mse = self.clipped_criterion(pred_score, gt_score, self.main_criterion)#self.main_criterion=MSE
        return clipped_mse,  contra_loss ,self.beta * clipped_mse + self.gama * contra_loss

#####################################################################################

# Categorical loss was not useful in initial experiments, but I keep it here for future reference


