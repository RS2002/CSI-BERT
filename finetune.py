from model import CSIBERT,Sequence_Classifier,Classification
from transformers import BertConfig
import argparse
import tqdm
import torch
from dataset import load_all,load_data
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
    parser.add_argument("--freeze", action="store_true",default=False)
    parser.add_argument('--lr', type=float, default=0.0005)
    # parser.add_argument("--test_people", type=int, nargs='+', default=[0,1])
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--class_num', type=int, default=6) #action:6, people:8
    parser.add_argument('--task', type=str, default="action") # "action" or "people"
    parser.add_argument("--path", type=str, default='./csibert_pretrain.pth')
    parser.add_argument("--no_pretrain", action="store_true",default=False)
    parser.add_argument('--data_path', type=str, default="./data/magnitude.npy")
    args = parser.parse_args()
    return args

def main():
    ACC=[]

    args=get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    bertconfig=BertConfig(max_position_embeddings=args.max_len, hidden_size=args.hs, position_embedding_type=args.position_embedding_type,num_hidden_layers=args.layers,num_attention_heads=args.heads, intermediate_size=args.intermediate_size)
    csibert=CSIBERT(bertconfig,args.carrier_dim,args.carrier_attn, args.time_embedding)
    if not args.no_pretrain:
        csibert.load_state_dict(torch.load(args.path))
    # model=Sequence_Classifier(csibert,args.class_num)
    model = Classification(csibert, args.class_num)
    model=model.to(device)


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters:', total_params)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    # dataset=load_all()
    # train_data,test_data=train_test_split(dataset, test_size=0.1, random_state=42)
    train_data,test_data=load_data(args.data_path, train_prop=0.9)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    loss_func = nn.CrossEntropyLoss()
    best_acc=0
    best_epoch=0

    j=0
    while True:
        j+=1
        model.train()
        torch.set_grad_enabled(True)
        if args.freeze:
            csibert.eval()
            for param in csibert.parameters():
                if param.requires_grad:
                    param.requires_grad = False
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

            non_pad = (input != pad[0]).float()
            avg = copy.deepcopy(input)
            avg[input == pad[0]] = 0
            avg = torch.sum(avg, dim=-2, keepdim=True) / (torch.sum(non_pad, dim=-2, keepdim=True)+1e-8)
            std = (input - avg) ** 2
            std[input == pad[0]] = 0
            std = torch.sum(std, dim=-2, keepdim=True) / (torch.sum(non_pad, dim=-2, keepdim=True)+1e-8)
            std = torch.sqrt(std)

            if args.normal:
                input = (input - avg) / (std+1e-5)

            batch_size,seq_len,carrier_num=input.shape
            attn_mask = (x[:, :, 0] != pad[0]).float().to(device)
            if args.normal:
                rand_word = torch.tensor(csibert.mask(batch_size, std=torch.tensor([1]).to(device), avg=torch.tensor([0]).to(device))).to(device)
            else:
                rand_word = torch.tensor(csibert.mask(batch_size, std=std.to(device), avg=avg.to(device))).to(device)
            input[x==pad[0]]=rand_word[x==pad[0]]
            if args.time_embedding:
                y = model(input, attn_mask)
            else:
                y = model(input, attn_mask, timestamp)

            loss = loss_func(y,label)
            output = torch.argmax(y, dim=-1)
            acc=torch.sum(output==label)/batch_size

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
            optim.step()

            loss_list.append(loss.item())
            acc_list.append(acc.item())
        log="Epoch {} | Train Loss {:06f}, Train Acc {:06f}, ".format(j,np.mean(loss_list),np.mean(acc_list))
        print(log)
        with open(args.task+".txt", 'a') as file:
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

            non_pad = (input != pad[0]).float()
            avg = copy.deepcopy(input)
            avg[input == pad[0]] = 0
            avg = torch.sum(avg, dim=-2, keepdim=True) / (torch.sum(non_pad, dim=-2, keepdim=True)+1e-8)
            std = (input - avg) ** 2
            std[input == pad[0]] = 0
            std = torch.sum(std, dim=-2, keepdim=True) / (torch.sum(non_pad, dim=-2, keepdim=True)+1e-8)
            std = torch.sqrt(std)

            if args.normal:
                input = (input - avg) / (std+1e-5)

            batch_size,seq_len,carrier_num=input.shape
            attn_mask = (x[:, :, 0] != pad[0]).float().to(device)
            if args.normal:
                rand_word = torch.tensor(csibert.mask(batch_size, std=torch.tensor([1]).to(device), avg=torch.tensor([0]).to(device))).to(device)
            else:
                rand_word = torch.tensor(csibert.mask(batch_size, std=std.to(device), avg=avg.to(device))).to(device)
            input[x==pad[0]]=rand_word[x==pad[0]]
            if args.time_embedding:
                y = model(input, attn_mask)
            else:
                y = model(input, attn_mask, timestamp)

            loss = loss_func(y,label)
            output = torch.argmax(y, dim=-1)
            acc=torch.sum(output==label)/batch_size

            loss_list.append(loss.item())
            acc_list.append(acc.item())
        log="Test Loss {:06f}, Test Acc {:06f}".format(np.mean(loss_list),np.mean(acc_list))
        print(log)
        ACC.append(np.mean(acc_list))
        with open(args.task+".txt", 'a') as file:
            file.write(log+"\n")
        if np.mean(acc_list)>=best_acc:
            best_acc=np.mean(acc_list)
            torch.save(model.state_dict(), args.task+".pth")
            best_epoch=0
        else:
            best_epoch+=1
        if best_epoch>=args.epoch:
            break

    print("Acc Max:",np.max(ACC))
    print("Acc Mean:",np.max(ACC[-30:]))
    print("Acc Std:",np.max(ACC[-30:]))


if __name__ == '__main__':
    main()