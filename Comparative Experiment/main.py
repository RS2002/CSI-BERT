from models import FFN,CNN,RNN,LSTM,Resnet
import torch
import argparse
from dataset import load_data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import tqdm
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument("--carrier_dim", type=int, default=52)
    parser.add_argument("--method",type=str,default="LSTM")
    parser.add_argument("--daata_path",type=str,default="../data/magnitude.npy")
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.0005)
    # parser.add_argument("--test_people", type=int, nargs='+', default=[0,1])
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--class_num', type=int, default=6) #action:6, people:8
    parser.add_argument('--task', type=str, default="action") # "action" or "people"
    args = parser.parse_args()
    return args

def main():
    args=get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    model_name=args.method
    max_len=args.max_len
    carrier_dim=args.carrier_dim
    class_num=args.class_num
    task=args.task
    if model_name=="FFN":
        model=FFN(input_dim=max_len*carrier_dim,class_num=class_num)
    elif model_name=="CNN":
        model=CNN(class_num=class_num)
    elif model_name=="RNN":
        model=RNN(input_dim=carrier_dim,class_num=class_num)
    elif model_name=="LSTM":
        model=LSTM(input_dim=carrier_dim,class_num=class_num)
    elif model_name=="Resnet":
        model=Resnet(class_num=class_num,pretrained=True)
    else:
        print("ERROR")
        exit(-1)
    model=model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters:', total_params)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    dataset=load_data(args.data_path)
    train_data,test_data=train_test_split(dataset, test_size=0.1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    loss_func = nn.CrossEntropyLoss()
    best_acc=0

    for j in range(args.epoch):
        model.train()
        torch.set_grad_enabled(True)
        loss_list=[]
        acc_list=[]
        pbar = tqdm.tqdm(train_loader, disable=False)
        for x,_,action,people,_ in pbar:
            x=x.to(device)
            if args.task=="action":
                label=action.to(device)
            elif args.task=="people":
                label=people.to(device)
            else:
                print("ERROR")
                exit(-1)
            y=model(x)
            loss = loss_func(y,label)
            output = torch.argmax(y, dim=-1)
            acc=torch.sum(output==label)/x.shape[0]

            model.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
            optim.step()

            loss_list.append(loss.item())
            acc_list.append(acc.item())
            log = "Epoch {} | Train Loss {:06f}, Train Acc {:06f}, ".format(j + 1, np.mean(loss_list),np.mean(acc_list))
            print(log)
            with open(model_name+".txt", 'a') as file:
                file.write(log)

        model.eval()
        torch.set_grad_enabled(False)
        loss_list=[]
        acc_list=[]
        pbar = tqdm.tqdm(test_loader, disable=False)
        for x,_,action,people,_ in pbar:
            x=x.to(device)
            if args.task=="action":
                label=action.to(device)
            elif args.task=="people":
                label=people.to(device)
            else:
                print("ERROR")
                exit(-1)
            y=model(x)
            loss = loss_func(y,label)
            output = torch.argmax(y, dim=-1)
            acc=torch.sum(output==label)/x.shape[0]

            loss_list.append(loss.item())
            acc_list.append(acc.item())
        log = "Test Loss {:06f}, Test Acc {:06f}, ".format(np.mean(loss_list), np.mean(acc_list))
        print(log)
        with open(model_name+".txt", 'a') as file:
            file.write(log + "\n")
        if best_acc is None or np.mean(acc_list) > best_acc:
            best_acc = np.mean(acc_list)
            torch.save(model.state_dict(), model_name+".pth")


if __name__ == '__main__':
    main()


