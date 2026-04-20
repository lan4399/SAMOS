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
from lossfunction import Loss
from conformer import ConformerBlock
import matplotlib.pyplot as plt
random.seed(1584)

class MosPredictor(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super(MosPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        # listner id embedding
        self.num_judges = 289
        self.judge_embedding = nn.Embedding(num_embeddings = self.num_judges, embedding_dim = 128)#128
        #lstm
        self.lstm = nn.LSTM(input_size = 960,
                                       hidden_size = 128,
                                       num_layers = 3, batch_first = True, bidirectional = True)#

        #conformer
        self.conformer = ConformerBlock(dim=64, n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)
        
        
        self.relu = nn.ReLU()
        self.dense1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.average_layer = nn.AdaptiveAvgPool1d(1)


    def forward(self, wav, ids, codecs):#不对codec处理，直接拼
        #ssl feature
        wav = wav.squeeze(1)  
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']#(B,T,768)
        # codecs
        t1 = x.shape[1]
        t2 = codecs.shape[1]
        if t1 == t2:
            T = t1
        elif t1 > t2:
            x = x[:, :t2-t1, :]
            T = t2
        elif t1 < t2:
            codecs = codecs[:, :t1-t2, :]
            T = t1
        codecs = self.conformer(codecs)#(b,t,64)
        #get judge embedding
        ids=ids.long()
        judge_feat = self.judge_embedding(ids) # (b, 128)    
        judge_feat = torch.stack([judge_feat for i in range(T)], dim = 1) #(b, t, 128)
        lstm_input = torch.cat([x, codecs, judge_feat], dim = -1) # concat along feature dimension(B,T,F=128+768+64=960)
        
        #lstm
        lstm_output, (h, c) = self.lstm(lstm_input)#(B,T,256)
        frame_score= self.dense1(lstm_output)#(B,T,1)
        weight = self.dense2(lstm_output)
        frame_score = frame_score * weight
        return frame_score

    
class MyDataset1(Dataset):#train
    def __init__(self, wavdir, mos_list):
        self.data_list = []
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos = float(parts[1])
            id = int(288)
            self.data_list.append([wavname, mos,id])
       
        self.wavdir = wavdir
        

        
    def __getitem__(self, idx):
        wavname,  score, id = self.data_list[idx]
        wav_codec_name = wavname.split('.')[0] + '.npy'
        wav_codec_path = os.path.join('/home/tylan/SAMOS/APCcoder/train/',wav_codec_name)
        #wav_codec_path = os.path.join('/home/yfshi/mnt_102/MOSANet/MOSA-Net-Cross-Domain-main/data/phase1-ood/APCcoder/train/',wav_codec_name)
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        wav_codec = np.load(wav_codec_path)#(64,t)
        return wav, score, wavname, id, wav_codec
    

    def __len__(self):
        return len(self.data_list)


    def collate_fn(self, batch):  ## zero padding
        wavs, scores, wavnames, ids, codecs= zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        max_len2 = max(codecs,key =lambda x : x.shape[1]).shape[1]
        output_codecs = []
        for codec in codecs:
            amount_to_pad2 = max_len2 - codec.shape[1]
            codec = torch.from_numpy(codec)#(64,t)
            padded_codec = torch.nn.functional.pad(codec, (0, amount_to_pad2), 'constant', 0)
            padded_codec = torch.transpose(padded_codec, 0, 1)#(t,64)
            output_codecs.append(padded_codec)
        output_wavs = torch.stack(output_wavs, dim=0)
        output_codecs = torch.stack(output_codecs, dim=0)
        scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)#(B)
        ids  = torch.stack([torch.tensor(x) for x in list(ids)], dim=0)#(B)
        return output_wavs, scores, wavnames, ids, output_codecs
    

class MyDataset2(Dataset):#dev
    def __init__(self, wavdir, mos_list):
        self.data_list = []
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos = float(parts[1])
            id = int(288)
            self.data_list.append([wavname, mos,id])
       
        self.wavdir = wavdir

        
    def __getitem__(self, idx):
        wavname,  score, id = self.data_list[idx]
        wav_codec_name = wavname.split('.')[0] + '.npy'
        wav_codec_path = os.path.join('/home/tylan/SAMOS/APCcoder/val/',wav_codec_name)
        #wav_codec_path = os.path.join('/home/yfshi/mnt_102/MOSANet/MOSA-Net-Cross-Domain-main/data/phase1-ood/APCcoder/val/',wav_codec_name)
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        wav_codec = np.load(wav_codec_path)
        return wav, score, wavname, id, wav_codec
    

    def __len__(self):
        return len(self.data_list)


    def collate_fn(self, batch):  ## zero padding
        wavs, scores, wavnames, ids, codecs= zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        max_len2 = max(codecs,key =lambda x : x.shape[1]).shape[1]
        output_codecs = []
        for codec in codecs:
            amount_to_pad2 = max_len2 - codec.shape[1]
            codec = torch.from_numpy(codec)#(64,t)
            padded_codec = torch.nn.functional.pad(codec, (0, amount_to_pad2), 'constant', 0)
            padded_codec = torch.transpose(padded_codec, 0, 1)#(t,64)
            output_codecs.append(padded_codec)
        output_wavs = torch.stack(output_wavs, dim=0)
        output_codecs = torch.stack(output_codecs, dim=0)
        scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)#(B)
        ids  = torch.stack([torch.tensor(x) for x in list(ids)], dim=0)#(B)
        return output_wavs, scores, wavnames, ids, output_codecs

class MyDataset3(Dataset):#test
    def __init__(self, wavdir, mos_list):
        self.data_list = []
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos = float(parts[1])
            id = int(288)
            self.data_list.append([wavname, mos,id])
       
        self.wavdir = wavdir
        

        
    def __getitem__(self, idx):
        wavname,  score, id = self.data_list[idx]
        wav_codec_name = wavname.split('.')[0] + '.npy'
        wav_codec_path = os.path.join('/home/tylan/SAMOS/APCcoder/test/',wav_codec_name)
        #wav_codec_path = os.path.join('/home/yfshi/mnt_102/MOSANet/MOSA-Net-Cross-Domain-main/data/phase1-ood/APCcoder/test/',wav_codec_name)
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        wav_codec = np.load(wav_codec_path)
        return wav, score, wavname, id, wav_codec
    

    def __len__(self):
        return len(self.data_list)


    def collate_fn(self, batch):  ## zero padding
        wavs, scores, wavnames, ids, codecs= zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        max_len2 = max(codecs,key =lambda x : x.shape[1]).shape[1]
        output_codecs = []
        for codec in codecs:
            amount_to_pad2 = max_len2 - codec.shape[1]
            codec = torch.from_numpy(codec)#(64,t)
            padded_codec = torch.nn.functional.pad(codec, (0, amount_to_pad2), 'constant', 0)
            padded_codec = torch.transpose(padded_codec, 0, 1)#(t,64)
            output_codecs.append(padded_codec)
        output_wavs = torch.stack(output_wavs, dim=0)
        output_codecs = torch.stack(output_codecs, dim=0)
        scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)#(B)
        ids  = torch.stack([torch.tensor(x) for x in list(ids)], dim=0)#(B)
        return output_wavs, scores, wavnames, ids, output_codecs


def systemID(uttID):
    return uttID.split('-')[0]




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--fairseq_base_model', type=str, required=True, help='Path to pretrained fairseq base model')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='checkpoints', help='Output directory for your trained checkpoints')
    args = parser.parse_args()

    cp_path = args.fairseq_base_model
    datadir = args.datadir
    ckptdir = args.outdir+'/SAMOS'
    my_checkpoint = args.finetune_from_checkpoint
    
    if not os.path.exists(ckptdir):
        os.system('mkdir -p ' + ckptdir)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    wavdir = os.path.join(datadir, 'wav')
    trainlist = os.path.join(datadir, 'sets/train_mos_ID_list2.txt')
    #trainlist = os.path.join(datadir, 'sets/train_mos_list.txt')
    validlist = os.path.join(datadir, 'sets/val_mos_list.txt')
    testlist = os.path.join(datadir, 'sets/test_mos_list.txt')

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()
    
    trainset = MyDataset1(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)

    validset = MyDataset2(wavdir, validlist )
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    testset = MyDataset3(wavdir, testlist)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=2, collate_fn=testset.collate_fn)

    net = MosPredictor(ssl_model, SSL_OUT_DIM)
    net = net.to(device)

    if my_checkpoint != None:  ## do (further) finetuning
        net.load_state_dict(torch.load(my_checkpoint))
    
    criterion = nn.L1Loss()
    criterion = Loss(0.1, 0.25, 1, 0.5)
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    PREV_VAL_LOSS=9999999999
    PREV_TEST_SYS_SRCC=0
    PREV_TEST_UTT_SRCC=0
    PREV_VAL_SYS_SRCC=0
    PREV_VAL_UTT_SRCC=0
    PREV_VAL_UTT_KTAU=0
    test_sys_SRCC_values = []
    test_utt_SRCC_values = []
    val_sys_SRCC_values = []
    val_utt_SRCC_values = []
    val_utt_KTAU_values = []
    orig_patience=20
    patience=orig_patience
    for epoch in range(1,1001):
        STEPS=0
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, filenames, ids, codecs = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            ids = ids.to(device)
            codecs = codecs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs, ids, codecs)
            reg_loss, contra_loss, total_loss = criterion(outputs, labels, contraloss = 'true')
            total_loss.backward()
            optimizer.step()
            STEPS += 1
            running_loss += total_loss.item()
        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))
        epoch_val_loss = 0.0
        net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        ## validation
        #get val_set_system_srcc
        predictions = { }  # 字典，内容为filename : prediction
        for i, data in enumerate(validloader, 0):
            inputs, labels, filenames, ids, codecs = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            ids = ids.to(device)
            codecs = codecs.to(device)
            outputs = net(inputs,ids,codecs)
            scores = torch.mean(outputs.squeeze(-1),dim=1,keepdim=True)
            output = float(scores.cpu().detach().numpy()[0])
            predictions[filenames[0]] = output

        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        true_MOS = { }
        testf = open(validlist, 'r')
        for line in testf:
            parts = line.strip().split(',')#如parts=['sys64e2f-utt491a78a.wav', '4.0']
            uttID = parts[0]#wav_name
            MOS = float(parts[1])
            true_MOS[uttID] = MOS# 

        sorted_uttIDs = sorted(predictions.keys())#
        #UTT_level
        ts = []
        ps = []
        for uttID in sorted_uttIDs:
            t = true_MOS[uttID]
            p = predictions[uttID]
            ts.append(t)
            ps.append(p)

        truths = np.array(ts)
        preds = np.array(ps)
        val_utt_SRCC=scipy.stats.spearmanr(truths.T, preds.T)[0]
        val_utt_MSE=np.mean((truths-preds)**2)
        val_utt_KTAU=scipy.stats.kendalltau(truths, preds)[0]

        val_utt_SRCC_values.append(val_utt_SRCC)
        val_utt_KTAU_values.append(val_utt_KTAU)
    ### SYSTEM
        true_sys_MOSes = { }
        for uttID in sorted_uttIDs:
            sysID = systemID(uttID)
            noop1 = true_sys_MOSes.setdefault(sysID, [ ])
            true_sys_MOSes[sysID].append(true_MOS[uttID])
        true_sys_MOS_avg = { }
        for k, v in true_sys_MOSes.items():
            avg_MOS = sum(v) / (len(v) * 1.0)
            true_sys_MOS_avg[k] = avg_MOS

        pred_sys_MOSes = { }
        for uttID in sorted_uttIDs:
            sysID = systemID(uttID)
            noop = pred_sys_MOSes.setdefault(sysID, [ ])
            pred_sys_MOSes[sysID].append(predictions[uttID])
        pred_sys_MOS_avg = { }
        for k, v in pred_sys_MOSes.items():
            avg_MOS = sum(v) / (len(v) * 1.0)
            pred_sys_MOS_avg[k] = avg_MOS

    ## make lists sorted by system
        pred_sysIDs = sorted(pred_sys_MOS_avg.keys())
        sys_p = [ ]
        sys_t = [ ]
        for sysID in pred_sysIDs:
            sys_p.append(pred_sys_MOS_avg[sysID])
            sys_t.append(true_sys_MOS_avg[sysID])

        sys_true = np.array(sys_t)
        sys_predicted = np.array(sys_p)
        val_sys_SRCC=scipy.stats.spearmanr(sys_true.T, sys_predicted.T)[0]

#save checkpoints
        #print('[val_UTTERANCE] Spearman rank correlation coefficient= %f' % val_utt_SRCC)
        print('[val_system] Spearman rank correlation coefficient= %f' % val_sys_SRCC)
        val_sys_SRCC_values.append(val_sys_SRCC)
        
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        #get test_set_system_srcc
        predictions = { }  # 字典，内容为filename : prediction
        for i, data in enumerate(testloader, 0):
            inputs, labels, filenames, ids, codecs = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            ids = ids.to(device)
            codecs = codecs.to(device)
            outputs = net(inputs,ids,codecs)
            scores = torch.mean(outputs.squeeze(-1),dim=1,keepdim=True)
            output = float(scores.cpu().detach().numpy()[0])
            predictions[filenames[0]] = output 

        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        true_MOS = { }
        testf = open(testlist, 'r')
        for line in testf:
            parts = line.strip().split(',')
            uttID = parts[0]#wav_name
            MOS = float(parts[1])
            true_MOS[uttID] = MOS# 

        sorted_uttIDs = sorted(predictions.keys())
        #UTT_level
        ts = []
        ps = []
        for uttID in sorted_uttIDs:
            t = true_MOS[uttID]
            p = predictions[uttID]
            ts.append(t)
            ps.append(p)

        truths = np.array(ts)
        preds = np.array(ps)
        test_utt_SRCC=scipy.stats.spearmanr(truths.T, preds.T)[0]
        test_utt_MSE=np.mean((truths-preds)**2)
        test_utt_LCC=np.corrcoef(truths, preds)[0][1]
        test_utt_KTAU=scipy.stats.kendalltau(truths, preds)[0]

    ### SYSTEM
        true_sys_MOSes = { }
        for uttID in sorted_uttIDs:
            sysID = systemID(uttID)
            noop1 = true_sys_MOSes.setdefault(sysID, [ ])
            true_sys_MOSes[sysID].append(true_MOS[uttID])
        true_sys_MOS_avg = { }
        for k, v in true_sys_MOSes.items():
            avg_MOS = sum(v) / (len(v) * 1.0)
            true_sys_MOS_avg[k] = avg_MOS

        pred_sys_MOSes = { }
        for uttID in sorted_uttIDs:
            sysID = systemID(uttID)
            noop = pred_sys_MOSes.setdefault(sysID, [ ])
            pred_sys_MOSes[sysID].append(predictions[uttID])
        pred_sys_MOS_avg = { }
        for k, v in pred_sys_MOSes.items():
            avg_MOS = sum(v) / (len(v) * 1.0)
            pred_sys_MOS_avg[k] = avg_MOS

    ## make lists sorted by system
        pred_sysIDs = sorted(pred_sys_MOS_avg.keys())
        sys_p = [ ]
        sys_t = [ ]
        for sysID in pred_sysIDs:
            sys_p.append(pred_sys_MOS_avg[sysID])
            sys_t.append(true_sys_MOS_avg[sysID])

        sys_true = np.array(sys_t)
        sys_predicted = np.array(sys_p)
        test_sys_SRCC=scipy.stats.spearmanr(sys_true.T, sys_predicted.T)[0]
        test_sys_MSE=np.mean((sys_true-sys_predicted)**2)
        test_sys_LCC=np.corrcoef(sys_true, sys_predicted)[0][1]
        test_sys_KTAU=scipy.stats.kendalltau(sys_true, sys_predicted)[0]

#save checkpoints
        print('[test_UTTERANCE] Test error= %f' % test_utt_MSE)
        print('[test_UTTERANCE] Linear correlation coefficient= %f' % test_utt_LCC)
        print('[test_UTTERANCE] Spearman rank correlation coefficient= %f' % test_utt_SRCC)
        print('[test_UTTERANCE] Kendall Tau rank correlation coefficient= %f' % test_utt_KTAU)
        print('[test_SYSTEM] Test error= %f' % test_sys_MSE)
        print('[test_SYSTEM] Linear correlation coefficient= %f' % test_sys_LCC)
        print('[test_SYSTEM] Spearman rank correlation coefficient= %f' % test_sys_SRCC)
        print('[test_SYSTEM] Kendall Tau rank correlation coefficient= %f' % test_sys_KTAU)

        test_sys_SRCC_values.append(test_sys_SRCC)
        test_utt_SRCC_values.append(test_utt_SRCC)


        if val_sys_SRCC > PREV_VAL_SYS_SRCC:
            PREV_VAL_SYS_SRCC=val_sys_SRCC
            print('sys_SRCC has decreased')
            path = os.path.join(ckptdir, 'ckpt_' +str(epoch) )
            torch.save(net.state_dict(), path)
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('SRCC has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
        # plot 
            
        plt.figure()
        plt.plot(range(1, len(test_sys_SRCC_values) + 1), test_sys_SRCC_values, label='test_sys_SRCC')
        plt.plot(range(1, len(val_sys_SRCC_values) + 1), val_sys_SRCC_values, label='val_sys_SRCC')
        plt.plot(range(1, len(val_utt_KTAU_values) + 1), val_utt_KTAU_values, label='val_utt_KTAU')

        plt.legend()
        plt.title('Test & val SRCC Values over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('SRCC Value')
        for i, sys_SRCC in enumerate(test_sys_SRCC_values):
            plt.text(i+1, sys_SRCC, f'{sys_SRCC:.3f}', ha='center', va='bottom')
        for i, sys_SRCC in enumerate(val_sys_SRCC_values):
            plt.text(i+1, sys_SRCC, f'{sys_SRCC:.3f}', ha='center', va='bottom')
            
        plt.savefig('/home/tylan/SAMOS/plot/srcc.png')

    print('Finished Training')


                            


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       