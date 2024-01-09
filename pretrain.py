from model import CSIBERT,Token_Classifier,Sequence_Classifier
from transformers import BertConfig,AdamW
import argparse
import tqdm
import torch
from dataset import load_zero_people,load_all,load_data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import numpy as np
import random
import math

pad=np.array([-1000]*52)

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mask_percent', type=float, default=0.15)
    parser.add_argument('--random_mask_percent', action="store_true",default=False)
    parser.add_argument('--normal', action="store_true", default=False) # whether to use norm layer
    parser.add_argument('--hs', type=int, default=64)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=100) # max input length
    parser.add_argument('--intermediate_size', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--position_embedding_type', type=str, default="absolute")
    parser.add_argument('--time_embedding', action="store_true", default=False) # whether to use time embedding
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument("--carrier_dim", type=int, default=52)
    parser.add_argument("--carrier_attn", action="store_true",default=False)
    parser.add_argument("--adversarial", action="store_true",default=False) # whether to use adversarial learning
    parser.add_argument('--lr', type=float, default=0.0005)
    # parser.add_argument("--test_people", type=int, nargs='+', default=[0,1])
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--data_path', type=str, default="./data/magnitude.npy")
    parser.add_argument('--parameter', type=str, default=None)
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
    adv = args.adversarial
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    bertconfig=BertConfig(max_position_embeddings=args.max_len, hidden_size=args.hs, position_embedding_type=args.position_embedding_type,num_hidden_layers=args.layers,num_attention_heads=args.heads, intermediate_size=args.intermediate_size)
    csibert=CSIBERT(bertconfig,args.carrier_dim,args.carrier_attn,args.time_embedding)
    csi_dim=args.carrier_dim
    model=Token_Classifier(csibert,args.carrier_dim)
    model=model.to(device)

    if args.parameter is not None:
        model.load_state_dict(torch.load(args.parameter),strict=False)

    classifier=Sequence_Classifier(csibert,2)
    classifier=classifier.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters:', total_params)
    optim = AdamW(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=0.01)
    # train_data,test_data=load_zero_people(args.test_people)
    train_data=load_all(magnitude_path=args.data_path)
    train_data,valid_data=train_test_split(train_data, test_size=0.1, random_state=113)
    # train_data,valid_data=load_data(train_prop=0.1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    loss_smooth = nn.MSELoss(reduction='none')
    loss_func = nn.MSELoss(reduction='none')
    loss_func_cls = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss(reduction='none')

    best_mse = 100
    best_mape = 1
    best_loss = 10000
    mse_epoch = 0
    mape_epoch = 0
    loss_epoch = 0
    j = 0
    while True:
        j+=1
        model.train()
        torch.set_grad_enabled(True)
        loss_list=[]
        mse_list=[]
        err_list=[]
        truth_total=[]
        truth_mask=[]
        false_total=[]
        false_mask=[]

        pbar = tqdm.tqdm(train_loader, disable=False)
        for x,_,_,_,timestamp in pbar:
            x=x.to(device)
            timestamp=timestamp.to(device)
            input = copy.deepcopy(x)
            max_values, _ = torch.max(input, dim=-2, keepdim=True)
            input[input == pad[0]] = -pad[0]
            min_values, _ = torch.min(input, dim=-2, keepdim=True)
            input[input == -pad[0]] = pad[0]

            non_pad = (input != pad[0]).float()
            avg = copy.deepcopy(input)
            avg[input == pad[0]] = 0
            avg = torch.sum(avg, dim=-2, keepdim=True) / (torch.sum(non_pad, dim=-2, keepdim=True)+1e-8)
            std = (input - avg) ** 2
            std[input == pad[0]] = 0
            std = torch.sum(std, dim=-2, keepdim=True) / (torch.sum(non_pad, dim=-2, keepdim=True)+1e-8)
            std = torch.sqrt(std)

            if args.normal:
                input = (input - avg) / (std+1e-8)

            batch_size,seq_len,carrier_num=input.shape
            loss_mask = torch.zeros(batch_size, seq_len)

            if args.normal:
                rand_word = torch.tensor(csibert.mask(batch_size, std=torch.tensor([1]).to(device), avg=torch.tensor([0]).to(device))).to(device)
            else:
                rand_word = torch.tensor(csibert.mask(batch_size, std=std.to(device), avg=avg.to(device))).to(device)
            for b in range(batch_size):
                # get index for masking
                if args.random_mask_percent:
                    mask_percent = random.uniform(0.15, 0.7)
                else:
                    mask_percent=args.mask_percent
                mask85, cur15 = get_mask_ind(seq_len,mask_percent)
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

            input[x==pad[0]]=rand_word[x==pad[0]]
            loss_mask = loss_mask.to(device)
            attn_mask = (x[:, :, 0] != pad[0]).float().to(device)  # (batch, seq_len)
            if args.time_embedding:
                y = model(input, attn_mask)
                # y = model(input, None)
            else:
                y = model(input, attn_mask, timestamp)
                # y = model(input, None, timestamp)

            if args.normal:
                y = y * std + avg
            avg_hat = torch.mean(y, dim=-2, keepdim=True)
            std_hat = torch.sqrt(torch.mean((y - avg_hat)** 2, dim=-2, keepdim=True))
            if args.normal:
                y_standard = (y - avg_hat) / (std_hat + 1e-8)
            else:
                y_standard=y

            # Adversarial
            num=300
            if j>num:
                alpha=1.0
            else:
                alpha=2.0/(1.0+math.exp(-10*j/num))-1


            truth=torch.zeros(batch_size,dtype=torch.long).to(device)
            cls_total=classifier(input,attn_mask,adversarial=True,alpha=alpha)
            output = torch.argmax(cls_total, dim=-1)
            acc = torch.mean((output == truth).float())
            truth_total.append(acc.item())
            cls_mask=classifier(input,loss_mask,adversarial=True,alpha=alpha)
            output = torch.argmax(cls_mask, dim=-1)
            acc = torch.mean((output == truth).float())
            truth_mask.append(acc.item())
            loss_truth_total=loss_func_cls(cls_total,truth)
            loss_truth_mask=loss_func_cls(cls_mask,truth)

            false=torch.zeros(batch_size,dtype=torch.long).to(device)+1
            cls_total=classifier(y_standard,adversarial=True,alpha=alpha)
            output = torch.argmax(cls_total, dim=-1)
            acc = torch.mean((output == false).float())
            false_total.append(acc.item())
            cls_mask=classifier(y_standard,loss_mask,adversarial=True,alpha=alpha)
            output = torch.argmax(cls_mask, dim=-1)
            acc = torch.mean((output == false).float())
            false_mask.append(acc.item())
            loss_false_total=loss_func_cls(cls_total,false)
            loss_false_mask=loss_func_cls(cls_mask,false)


            # loss1: MASK loss
            loss1 = torch.sum(loss_func(y,x), dim=-1)
            loss1 = torch.sum(loss1 * loss_mask) / torch.sum(loss_mask) / csi_dim
            current_mse = torch.sum(loss_mse(y,x), dim=-1)
            current_mse = torch.sum(current_mse * loss_mask) / torch.sum(loss_mask) / csi_dim
            # loss2: Total loss
            loss2 = torch.sum(loss_func(y, x), dim=-1)
            loss2 = torch.sum(loss2 * attn_mask) / torch.sum(attn_mask) / csi_dim
            loss = loss1 + loss2
            # loss3,4: Total Avg&Std loss
            loss3 = torch.mean(loss_smooth(avg_hat, avg))
            loss4 = torch.mean(loss_smooth(std_hat, std))
            loss+=loss3+loss4
            # loss5,6: Mask Avg&Std loss
            mask = loss_mask.unsqueeze(2).repeat(1, 1, csi_dim)
            x_mask = x * mask
            y_mask = y * mask
            x_mask_mean = torch.sum(x_mask, dim=1, keepdim=True) / (torch.sum(mask, dim=1, keepdim=True) + 1e-3)
            y_mask_mean = torch.sum(y_mask, dim=1, keepdim=True) / (torch.sum(mask, dim=1, keepdim=True) + 1e-3)
            x_mask_std = torch.sqrt(torch.sum(((x_mask_mean - x_mask)*mask) ** 2, dim=1) / (torch.sum(mask, dim=1) + 1e-3))
            y_mask_std = torch.sqrt(torch.sum(((y_mask_mean - y_mask)*mask) ** 2, dim=1) / (torch.sum(mask, dim=1) + 1e-3))
            loss5 = torch.mean(loss_smooth(x_mask_mean, y_mask_mean))
            loss6 = torch.mean(loss_smooth(x_mask_std, y_mask_std))
            loss +=  loss5 + loss6
            loss_list.append(loss.item())
            if adv:
                loss += loss_truth_total + loss_false_total + loss_truth_mask + loss_false_mask

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸

            has_nan=False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan=True
                        break
            if has_nan:
                # print("NAN Gradient->Skip")
                continue

            optim.step()

            loss_mask=loss_mask.unsqueeze(2)
            loss_mask=loss_mask.repeat(1,1,csi_dim)
            loss_mask[x==0]=0
            x[x==0]=1
            error = torch.sum(torch.abs(y - x) / x * loss_mask) / torch.sum(loss_mask)
            mse_list.append(current_mse.item())
            err_list.append(error.item())
        log = "Epoch {} | Train Loss {:06f}, Train MAPE {:06f}, Train MSE {:06f}, ".format(j, np.mean(loss_list),np.mean(err_list),np.mean(mse_list))
        print(log)
        print("Discrimination | Truth(Mask) {:06f}, Truth(Total) {:06f}, False(Mask) {:06f}, False(Total) {:06f}".format(np.mean(truth_total),np.mean(truth_mask),np.mean(false_total),np.mean(false_mask)))
        with open("Pretrain.txt", 'a') as file:
            file.write(log)

        model.eval()
        torch.set_grad_enabled(False)
        loss_list=[]
        mse_list=[]
        err_list=[]
        truth_total=[]
        truth_mask=[]
        false_total=[]
        false_mask=[]

        pbar = tqdm.tqdm(valid_loader, disable=False)
        for x,_,_,_,timestamp in pbar:
            x=x.to(device)
            timestamp=timestamp.to(device)
            input = copy.deepcopy(x)
            max_values, _ = torch.max(input, dim=-2, keepdim=True)
            input[input == pad[0]] = -pad[0]
            min_values, _ = torch.min(input, dim=-2, keepdim=True)
            input[input == -pad[0]] = pad[0]

            non_pad = (input != pad[0]).float()
            avg = copy.deepcopy(input)
            avg[input == pad[0]] = 0
            avg = torch.sum(avg, dim=-2, keepdim=True) / (torch.sum(non_pad, dim=-2, keepdim=True)+1e-5)
            std = (input - avg) ** 2
            std[input == pad[0]] = 0
            std = torch.sum(std, dim=-2, keepdim=True) / (torch.sum(non_pad, dim=-2, keepdim=True)+1e-5)
            std = torch.sqrt(std)

            if args.normal:
                input = (input - avg) / (std+1e-5)

            batch_size,seq_len,carrier_num=input.shape
            loss_mask = torch.zeros(batch_size, seq_len)

            if args.normal:
                rand_word = torch.tensor(csibert.mask(batch_size, std=torch.tensor([1]).to(device), avg=torch.tensor([0]).to(device))).to(device)
            else:
                rand_word = torch.tensor(csibert.mask(batch_size, std=std.to(device), avg=avg.to(device))).to(device)
            for b in range(batch_size):
                # get index for masking
                if args.random_mask_percent:
                    mask_percent = random.uniform(0.15, 0.8)
                else:
                    mask_percent = args.mask_percent
                mask85, cur15 = get_mask_ind(seq_len, mask_percent)
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

            input[x==pad[0]]=rand_word[x==pad[0]]
            loss_mask = loss_mask.to(device)
            attn_mask = (x[:, :, 0] != pad[0]).float().to(device)  # (batch, seq_len)
            if args.time_embedding:
                y = model(input, attn_mask)
                # y = model(input, None)
            else:
                y = model(input, attn_mask, timestamp)
                # y = model(input, None, timestamp)

            if args.normal:
                y = y * std + avg
            avg_hat = torch.mean(y, dim=-2, keepdim=True)
            std_hat = torch.sqrt(torch.mean((y - avg_hat)** 2, dim=-2, keepdim=True))
            if args.normal:
                y_standard = (y - avg_hat) / (std_hat + 1e-5)
            else:
                y_standard=y

            # Adversarial
            if j>num:
                alpha=1.0
            else:
                alpha=2.0/(1.0+math.exp(-10*j/num))-1

            truth = torch.zeros(batch_size, dtype=torch.long).to(device)
            cls_total = classifier(input, attn_mask, adversarial=True, alpha=alpha)
            output = torch.argmax(cls_total, dim=-1)
            acc = torch.mean((output == truth).float())
            truth_total.append(acc.item())
            cls_mask = classifier(input, loss_mask, adversarial=True, alpha=alpha)
            output = torch.argmax(cls_mask, dim=-1)
            acc = torch.mean((output == truth).float())
            truth_mask.append(acc.item())
            loss_truth_total = loss_func_cls(cls_total, truth)
            loss_truth_mask = loss_func_cls(cls_mask, truth)

            false = torch.zeros(batch_size, dtype=torch.long).to(device) + 1
            cls_total = classifier(y_standard, adversarial=True, alpha=alpha)
            output = torch.argmax(cls_total, dim=-1)
            acc = torch.mean((output == false).float())
            false_total.append(acc.item())
            cls_mask = classifier(y_standard, loss_mask, adversarial=True, alpha=alpha)
            output = torch.argmax(cls_mask, dim=-1)
            acc = torch.mean((output == false).float())
            false_mask.append(acc.item())
            loss_false_total = loss_func_cls(cls_total, false)
            loss_false_mask = loss_func_cls(cls_mask, false)

            # loss1: MASK loss
            loss1 = torch.sum(loss_func(y, x), dim=-1)
            loss1 = torch.sum(loss1 * loss_mask) / torch.sum(loss_mask) / csi_dim
            current_mse = torch.sum(loss_mse(y, x), dim=-1)
            current_mse = torch.sum(current_mse * loss_mask) / torch.sum(loss_mask) / csi_dim
            # loss2: Total loss
            loss2 = torch.sum(loss_func(y, x), dim=-1)
            loss2 = torch.sum(loss2 * attn_mask) / torch.sum(attn_mask) / csi_dim
            loss = loss1 + loss2
            # loss3,4: Total Avg&Std loss
            loss3 = torch.mean(loss_smooth(avg_hat, avg))
            loss4 = torch.mean(loss_smooth(std_hat, std))
            loss += loss3 + loss4
            # loss5,6: Mask Avg&Std loss
            mask = loss_mask.unsqueeze(2).repeat(1, 1, csi_dim)
            x_mask = x * mask
            y_mask = y * mask
            x_mask_mean = torch.sum(x_mask, dim=1, keepdim=True) / (torch.sum(mask, dim=1, keepdim=True) + 1e-5)
            y_mask_mean = torch.sum(y_mask, dim=1, keepdim=True) / (torch.sum(mask, dim=1, keepdim=True) + 1e-5)
            x_mask_std = torch.sqrt(
                torch.sum(((x_mask_mean - x_mask) * mask) ** 2, dim=1) / (torch.sum(mask, dim=1) + 1e-5))
            y_mask_std = torch.sqrt(
                torch.sum(((y_mask_mean - y_mask) * mask) ** 2, dim=1) / (torch.sum(mask, dim=1) + 1e-5))
            loss5 = torch.mean(loss_smooth(x_mask_mean, y_mask_mean))
            loss6 = torch.mean(loss_smooth(x_mask_std, y_mask_std))
            loss += loss5 + loss6
            loss_list.append(loss.item())
            if adv:
                loss += loss_truth_total + loss_false_total + loss_truth_mask + loss_false_mask


            loss_mask=loss_mask.unsqueeze(2)
            loss_mask=loss_mask.repeat(1,1,csi_dim)
            loss_mask[x==0]=0
            x[x==0]=1
            error = torch.sum(torch.abs(y - x) / x * loss_mask) / torch.sum(loss_mask)
            mse_list.append(current_mse.item())
            err_list.append(error.item())
        log = "Test Loss {:06f}, Test MAPE {:06f}, Test MSE {:06f} ".format(np.mean(loss_list), np.mean(err_list), np.mean(mse_list))
        print(log)
        print("Discrimination | Truth(Mask) {:06f}, Truth(Total) {:06f}, False(Mask) {:06f}, False(Total) {:06f}".format(np.mean(truth_total),np.mean(truth_mask),np.mean(false_total),np.mean(false_mask)))
        with open("Pretrain.txt", 'a') as file:
            file.write(log + "\n")

        mape,mse,loss=np.mean(err_list), np.mean(mse_list),np.mean(loss_list)
        if mape<best_mape or mse<best_mse or loss<best_loss:
            torch.save(csibert.state_dict(), "csibert_pretrain.pth")
            torch.save(model.state_dict(), "pretrain.pth")
        if mape<best_mape:
            best_mape=mape
            mape_epoch=0
        else:
            mape_epoch+=1
        if mse<best_mse:
            best_mse=mse
            mse_epoch=0
        else:
            mse_epoch+=1
        if loss<best_loss:
            best_loss=loss
            loss_epoch=0
        else:
            loss_epoch+=1
        if mape_epoch>=args.epoch and mse_epoch>args.epoch and loss_epoch>args.epoch:
            break
        print("MAPE Epoch {:}, MSE Epoch {:}, Loss Epcoh {:}".format(mape_epoch,mse_epoch,loss_epoch))


if __name__ == '__main__':
    main()