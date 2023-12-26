from model import CSIBERT,Token_Classifier
from transformers import BertConfig
import argparse
import tqdm
import torch
from dataset import load_all
from torch.utils.data import DataLoader
import copy
import numpy as np

pad=np.array([-1000]*52)

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
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
    parser.add_argument("--path", type=str, default='./pretrain.pth')
    parser.add_argument('--data_path', type=str, default="./data/magnitude.npy")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    device_name = "cuda:" + args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    bertconfig=BertConfig(max_position_embeddings=args.max_len, hidden_size=args.hs, position_embedding_type=args.position_embedding_type,num_hidden_layers=args.layers,num_attention_heads=args.heads, intermediate_size=args.intermediate_size)
    csibert = CSIBERT(bertconfig, args.carrier_dim, args.carrier_attn, args.time_embedding)
    csi_dim=args.carrier_dim
    model=Token_Classifier(csibert,csi_dim)
    model.load_state_dict(torch.load(args.path))
    model = model.to(device)
    data = load_all(magnitude_path=args.data_path)
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    model.eval()
    torch.set_grad_enabled(False)
    pbar = tqdm.tqdm(data_loader, disable=False)
    output1 = None
    output2 = None
    for x, _, _, _, timestamp in pbar:
        x = x.to(device)
        timestamp = timestamp.to(device)
        attn_mask = (x[:, :, 0] != pad[0]).float().to(device)
        input = copy.deepcopy(x)
        batch_size, seq_len, carrier_num = input.shape
        max_values, _ = torch.max(input, dim=-2, keepdim=True)
        input[input == pad[0]] = -pad[0]
        min_values, _ = torch.min(input, dim=-2, keepdim=True)
        input[input == -pad[0]] = pad[0]
        if args.normal:
            non_pad = (input != pad[0]).float()
            avg = copy.deepcopy(input)
            avg[input == pad[0]] = 0
            avg = torch.sum(avg, dim=-2, keepdim=True) / torch.sum(non_pad, dim=-2, keepdim=True)
            std = (input - avg) ** 2
            std[input == pad[0]] = 0
            std = torch.sum(std, dim=-2, keepdim=True) / torch.sum(non_pad, dim=-2, keepdim=True)
            std = torch.sqrt(std)
            input = (input - avg) / std

        if args.normal:
            rand_word = torch.tensor(csibert.mask(batch_size, std=torch.tensor([1]).to(device), avg=torch.tensor([0]).to(device))).to(device)
        else:
            rand_word = torch.tensor(csibert.mask(batch_size, min=min_values, max=max_values)).to(device)
        input[x==pad[0]]=rand_word[x==pad[0]]
        if args.time_embedding:
            y = model(input, None)
        else:
            y = model(input, None, timestamp)
        if args.normal:
            y = y * std + avg

        attn_mask=attn_mask.unsqueeze(2)
        y2 = x*attn_mask+y*(1-attn_mask)

        if output1 is None:
            output1=y
            output2=y2
        else:
            output1=torch.cat([output1,y],dim=0)
            output2=torch.cat([output2,y2],dim=0)

    replace=output1.cpu().numpy()
    recover=output2.cpu().numpy()
    np.save("replace.npy",replace)
    np.save("recover.npy", recover)

if __name__ == '__main__':
    main()