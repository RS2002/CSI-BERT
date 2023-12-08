from model import CSIBERT,Token_Classifier,Sequence_Classifier
from transformers import BertConfig,AdamW
import argparse
import tqdm
import torch
from dataset import load_zero_people,load_all
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import numpy as np
import random

pad=np.array([-1000]*52)

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mask_percent', type=float, default=0.15)
    parser.add_argument('--normal', action="store_true", default=False) # whether to use norm layer
    parser.add_argument('--hs', type=int, default=64)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=100) # max input length
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--position_embedding_type', type=str, default="absolute")
    parser.add_argument('--time_embedding', action="store_true", default=False) # whether to use time embedding
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument("--carrier_dim", type=int, default=52)
    parser.add_argument("--carrier_attn", action="store_true",default=False)
    parser.add_argument('--lr', type=float, default=0.0005)
    # parser.add_argument("--test_people", type=int, nargs='+', default=[0,1])
    parser.add_argument('--epoch', type=int, default=100)
    args = parser.parse_args()
    return args

def get_mask_ind(seq_len=100,mask_percent=0.15):
    Lseq=[i for i in range(seq_len)]
    mask_ind = random.sample(Lseq, round(seq_len * mask_percent))
    mask85 = random.sample(mask_ind, round(len(mask_ind)*0.85))
    cur15 = list(set(mask_ind)-set(mask85))
    return mask85, cur15

def main():
    args=get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    bertconfig=BertConfig(max_position_embeddings=args.max_len, hidden_size=args.hs, position_embedding_type=args.position_embedding_type,num_hidden_layers=args.layers,num_attention_heads=args.heads)
    csibert=CSIBERT(bertconfig,args.carrier_dim,args.carrier_attn,args.time_embedding)
    csi_dim=args.carrier_dim
    model=Token_Classifier(csibert,args.carrier_dim)
    model=model.to(device)
    classifier=Sequence_Classifier(csibert,2)
    classifier=classifier.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters:', total_params)
    optim = AdamW(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=0.01)
    # train_data,test_data=load_zero_people(args.test_people)
    train_data=load_all()
    train_data,valid_data=train_test_split(train_data, test_size=0.1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    loss_func = nn.MSELoss(reduction='none')
    loss_func_cls = nn.CrossEntropyLoss()
    best_loss=None

    for j in range(args.epoch):
        model.train()
        torch.set_grad_enabled(True)
        loss_list=[]
        err_list=[]
        pbar = tqdm.tqdm(train_loader, disable=False)
        for x,_,_,_,timestamp in pbar:
            x=x.to(device)
            timestamp=timestamp.to(device)
            input = copy.deepcopy(x)
            max_values, _ = torch.max(input, dim=-2, keepdim=True)
            input[input == pad[0]] = -pad[0]
            min_values, _ = torch.min(input, dim=-2, keepdim=True)
            input[input == -pad[0]] = pad[0]

            if args.normal: # 在时间维度归一化
                non_pad = (input != pad[0]).float()
                avg = copy.deepcopy(input)
                avg[input == pad[0]] = 0
                avg = torch.sum(avg, dim=-2, keepdim=True) / torch.sum(non_pad, dim=-2, keepdim=True)
                std = (input - avg) ** 2
                std[input == pad[0]] = 0
                std = torch.sum(std, dim=-2, keepdim=True) / torch.sum(non_pad, dim=-2, keepdim=True)
                std = torch.sqrt(std)
                input = (input - avg) / std

            batch_size,seq_len,carrier_num=input.shape
            loss_mask = torch.zeros(batch_size, seq_len)

            if args.normal:
                rand_word = torch.tensor(csibert.mask(batch_size, std=std, avg=avg)).to(device)
            else:
                rand_word = torch.tensor(csibert.mask(batch_size, min=min_values, max=max_values)).to(device)
            for b in range(batch_size):
                # get index for masking
                mask85, cur15 = get_mask_ind(seq_len,args.mask_percent)
                # apply mask, random, remain current token
                for i in mask85:
                    if x[b][i][0] == pad[0]:
                        continue
                    input[b][i] = rand_word[b][i]
                    loss_mask[b][i] = 1
                for i in cur15:
                    if x[b][i][0] == pad[0]:
                        continue
                    loss_mask[b][i] = 1

            loss_mask = loss_mask.to(device)
            attn_mask = (x[:, :, 0] != pad[0]).float().to(device)  # (batch, seq_len)
            if args.time_embedding:
                y = model(input, attn_mask)
            else:
                y = model(input, attn_mask, timestamp)

            cls=classifier(input,attn_mask,adversarial=True)
            truth=torch.zeros(batch_size,dtype=torch.long).to(device)
            loss_truth=loss_func_cls(cls,truth)
            cls=classifier(y,adversarial=True)
            false=torch.zeros(batch_size,dtype=torch.long).to(device)+1
            loss_false=loss_func_cls(cls,false)

            if args.normal:
                y = y * std + avg

            # print(x[0,:,0])
            # print(y[0,:,0])

            # loss1: MASK loss
            loss1 = torch.sum(loss_func(y,x), dim=-1)
            loss1 = torch.sum(loss1 * loss_mask) / torch.sum(loss_mask) / csi_dim
            # loss2: Total loss
            loss2 = torch.sum(loss_func(y, x), dim=-1)
            loss2 = torch.sum(loss2 * attn_mask) / torch.sum(attn_mask) / csi_dim
            # loss3: Min loss
            min_values_hat, _ = torch.min(y, dim=-2, keepdim=True)
            loss3 = torch.mean(loss_func(min_values_hat, min_values))
            # loss4: Max loss
            max_values_hat, _ = torch.max(y, dim=-2, keepdim=True)
            loss4 = torch.mean(loss_func(max_values_hat,max_values))
            if args.normal:
                #loss5,6: Avg&Std loss
                avg_hat = torch.sum(y, dim=-2, keepdim=True) / seq_len
                std_hat = (y - avg_hat) ** 2
                std_hat = torch.sum(std_hat, dim=-2, keepdim=True) / seq_len
                std_hat = torch.sqrt(std_hat)
                loss5 = torch.mean(loss_func(avg_hat, avg))
                loss6 = torch.mean(loss_func(std_hat, std))

                loss=loss1+loss2+loss3+loss4+loss5+loss6
            else:
                loss=loss1#+loss2+loss3+loss4


            loss+=(loss_truth+loss_false)*0.1


            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
            optim.step()

            loss_mask=loss_mask.unsqueeze(2)
            loss_mask=loss_mask.repeat(1,1,csi_dim)
            loss_mask[x==0]=0
            x[x==0]=1
            error = torch.sum(torch.abs(y - x) / x * loss_mask) / torch.sum(loss_mask)
            loss_list.append(loss.item())
            err_list.append(error.item())
        log="Epoch {} | Train Loss {:06f}, Train Error {:06f}, ".format(j+1,np.mean(loss_list),np.mean(err_list))
        print(log)
        with open("Pretrain.txt", 'a') as file:
            file.write(log)

        model.eval()
        torch.set_grad_enabled(False)
        loss_list=[]
        err_list=[]
        pbar = tqdm.tqdm(valid_loader, disable=False)
        for x,_,_,_,timestamp in pbar:
            x=x.to(device)
            timestamp=timestamp.to(device)
            input = copy.deepcopy(x)
            input=input.to(device)
            max_values, _ = torch.max(input, dim=-2, keepdim=True)
            input[input == pad[0]] = -pad[0]
            min_values, _ = torch.min(input, dim=-2, keepdim=True)
            input[input == -pad[0]] = pad[0]

            if args.normal: # 在时间维度归一化
                non_pad = (input != pad[0]).float()
                avg = copy.deepcopy(input)
                avg[input == pad[0]] = 0
                avg = torch.sum(avg, dim=-2, keepdim=True) / torch.sum(non_pad, dim=-2, keepdim=True)
                std = (input - avg) ** 2
                std[input == pad[0]] = 0
                std = torch.sum(std, dim=-2, keepdim=True) / torch.sum(non_pad, dim=-2, keepdim=True)
                std = torch.sqrt(std)
                input = (input - avg) / std

            batch_size,seq_len,carrier_num=input.shape
            loss_mask = torch.zeros(batch_size, seq_len)

            if args.normal:
                rand_word = torch.tensor(csibert.mask(batch_size, std=std, avg=avg)).to(device)
            else:
                rand_word = torch.tensor(csibert.mask(batch_size, min=min_values, max=max_values)).to(device)
            for b in range(batch_size):
                # get index for masking
                mask85, cur15 = get_mask_ind(seq_len,args.mask_percent)
                # apply mask, random, remain current token
                for i in mask85:
                    if x[b][i][0] == pad[0]:
                        continue
                    input[b][i] = rand_word[b][i]
                    loss_mask[b][i] = 1
                for i in cur15:
                    if x[b][i][0] == pad[0]:
                        continue
                    loss_mask[b][i] = 1

            loss_mask = loss_mask.to(device)
            attn_mask = (x[:, :, 0] != pad[0]).float().to(device)  # (batch, seq_len)
            if args.time_embedding:
                y = model(input, attn_mask)
            else:
                y = model(input, attn_mask, timestamp)

            if args.normal:
                y = y * std + avg

            # loss1: MASK loss
            loss1 = torch.sum(loss_func(y,x), dim=-1)
            loss1 = torch.sum(loss1 * loss_mask) / torch.sum(loss_mask) / csi_dim
            # loss2: Total loss
            loss2 = torch.sum(loss_func(y, x), dim=-1)
            loss2 = torch.sum(loss2 * attn_mask) / torch.sum(attn_mask) / csi_dim
            # loss3: Min loss
            min_values_hat, _ = torch.min(y, dim=-2, keepdim=True)
            loss3 = torch.mean(loss_func(min_values_hat, min_values))
            # loss4: Max loss
            max_values_hat, _ = torch.max(y, dim=-2, keepdim=True)
            loss4 = torch.mean(loss_func(max_values_hat,max_values))
            if args.normal:
                # loss5,6: Avg&Std loss
                avg_hat = torch.sum(y, dim=-2, keepdim=True) / seq_len
                std_hat = (y - avg_hat) ** 2
                std_hat = torch.sum(std_hat, dim=-2, keepdim=True) / seq_len
                std_hat = torch.sqrt(std_hat)
                loss5 = torch.mean(loss_func(avg_hat, avg))
                loss6 = torch.mean(loss_func(std_hat, std))

                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            else:
                loss = loss1 #+ loss2 + loss3 + loss4


            loss_mask=loss_mask.unsqueeze(2)
            loss_mask=loss_mask.repeat(1,1,csi_dim)
            loss_mask[x==0]=0
            x[x==0]=1
            error = torch.sum(torch.abs(y - x) / x * loss_mask) / torch.sum(loss_mask)
            loss_list.append(loss.item())
            err_list.append(error.item())
        log="Test Loss {:06f}, Test Error {:06f} ".format(np.mean(loss_list),np.mean(err_list))
        print(log)
        with open("Pretrain.txt", 'a') as file:
            file.write(log+"\n")

        torch.save(csibert.state_dict(), "csibert_pretrain.pth")
        torch.save(model.state_dict(), "pretrain.pth")




if __name__ == '__main__':
    main()