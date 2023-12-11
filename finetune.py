from model import CSIBERT,Sequence_Classifier,Classification
from transformers import BertConfig
import argparse
import tqdm
import torch
from dataset import load_all,load_zero_people
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import numpy as np
from sklearn.model_selection import train_test_split

pad=np.array([-1000]*52)

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mask_percent', type=float, default=0.15)
    parser.add_argument('--normal', action="store_true", default=False)
    parser.add_argument('--hs', type=int, default=64)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--intermediate_size', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--position_embedding_type', type=str, default="absolute")
    parser.add_argument('--time_embedding', action="store_true", default=False) # whether to use time embedding
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument("--carrier_dim", type=int, default=52)
    parser.add_argument("--carrier_attn", action="store_true",default=False)
    parser.add_argument('--lr', type=float, default=0.0005)
    # parser.add_argument("--test_people", type=int, nargs='+', default=[0,1])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--class_num', type=int, default=6) #action:6, people:8
    parser.add_argument('--task', type=str, default="action") # "action" or "people"
    parser.add_argument("--path", type=str, default='./csibert_pretrain.pth')
    args = parser.parse_args()
    return args

def main():
    args=get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    bertconfig=BertConfig(max_position_embeddings=args.max_len, hidden_size=args.hs, position_embedding_type=args.position_embedding_type,num_hidden_layers=args.layers,num_attention_heads=args.heads, intermediate_size=args.intermediate_size)
    csibert=CSIBERT(bertconfig,args.carrier_dim,args.carrier_attn, args.time_embedding)
    csibert.load_state_dict(torch.load(args.path))
    csi_dim=args.carrier_dim
    # model=Sequence_Classifier(csibert,args.class_num)
    model = Classification(csibert, args.class_num)
    model=model.to(device)


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters:', total_params)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    dataset=load_all()
    train_data,test_data=train_test_split(dataset, test_size=0.1, random_state=42)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    loss_func = nn.CrossEntropyLoss()
    best_loss=None

    for j in range(args.epoch):
        model.train()
        torch.set_grad_enabled(True)
        loss_list=[]
        acc_list=[]
        pbar = tqdm.tqdm(train_loader, disable=False)
        for x,_,action,people,timestamp in pbar:
            x=x.to(device)
            timestamp=timestamp.to(device)
            if args.task=="action":
                label=action.to(device)
            elif args.task=="people":
                label=people.to(device)
            else:
                print("ERROR")
                exit(-1)
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
            attn_mask = (x[:, :, 0] != pad[0]).float().to(device)  # (batch, seq_len)
            if args.position_embedding_type=="absolute":
                y = model(input, attn_mask)
            else:
                y = model(input, attn_mask, timestamp)

            loss = loss_func(y,label)
            output = torch.argmax(y, dim=-1)
            acc=torch.sum(output==label)/batch_size

            model.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
            optim.step()

            loss_list.append(loss.item())
            acc_list.append(acc.item())
        log="Epoch {} | Train Loss {:06f}, Train Acc {:06f}, ".format(j+1,np.mean(loss_list),np.mean(acc_list))
        print(log)
        with open("Finetune.txt", 'a') as file:
            file.write(log)

        model.eval()
        torch.set_grad_enabled(False)
        loss_list=[]
        acc_list=[]
        pbar = tqdm.tqdm(test_loader, disable=False)
        for x,_,action,people,timestamp in pbar:
            x=x.to(device)
            timestamp=timestamp.to(device)
            if args.task=="action":
                label=action.to(device)
            elif args.task=="people":
                label=people.to(device)
            else:
                print("ERROR")
                exit(-1)
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
            attn_mask = (x[:, :, 0] != pad[0]).float().to(device)  # (batch, seq_len)
            if args.position_embedding_type=="absolute":
                y = model(input, attn_mask)
            else:
                y = model(input, attn_mask, timestamp)

            loss = loss_func(y,label)
            output = torch.argmax(y, dim=-1)
            acc=torch.sum(output==label)/batch_size

            loss_list.append(loss.item())
            acc_list.append(acc.item())
        log="Test Loss {:06f}, Test Acc {:06f}, ".format(np.mean(loss_list),np.mean(acc_list))
        print(log)
        with open("Finetune.txt", 'a') as file:
            file.write(log+"\n")
        if best_loss is None or np.mean(loss_list)<best_loss:
            best_loss=np.mean(loss_list)
            torch.save(model.state_dict(), "finetune.pth")

if __name__ == '__main__':
    main()