from transformers import BertConfig,BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class CSIBERT(nn.Module):
    def __init__(self,bertconfig,input_dim,carrier_attention=False, time_emb=True):
        super().__init__()
        self.bertconfig=bertconfig
        self.auto_pos=time_emb
        self.bert=BertModel(bertconfig)
        self.hidden_dim=bertconfig.hidden_size
        self.input_dim=input_dim
        self.carrier_attention=carrier_attention
        if carrier_attention:
            # self.Linear1=nn.Linear(1, 64)
            # self.query=nn.Linear(64,64)
            # self.key=nn.Linear(64,64)
            # self.value=nn.Linear(64,64)
            # self.self_attention=nn.MultiheadAttention(embed_dim=64, num_heads=4, dropout=0, batch_first=True)
            # self.norm1=nn.BatchNorm1d(64)
            # self.Linear2=nn.Linear(64, 64)
            # self.norm2=nn.BatchNorm1d(input_dim*64)
            # self.emb = nn.Sequential(
            #     nn.Linear(input_dim * 64, 64),
            #     nn.ReLU(),
            #     nn.Linear(64, self.hidden_dim)
            # )

            self.attention = SelfAttention(bertconfig.max_position_embeddings, 128, input_dim)
            self.emb=nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim)
            )

        else:
            self.emb=nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim)
            )


    def forward(self,x,attn_mask=None,timestamp=None):
        if self.carrier_attention:
            x=x.permute(0,2,1)
            attn_mat = self.attention(x)
            x = torch.bmm(attn_mat, x)
            x = x.permute(0, 2, 1)

            # batch_size,length,input_dim=x.shape
            # x=x.reshape(-1,input_dim,1)
            # x_emb=self.Linear1(x)
            # x,_=self.self_attention(self.query(x_emb),self.key(x_emb),self.value(x_emb))
            # x+=x_emb
            # x=x.reshape(-1,64)
            # x=self.norm1(x)
            # x=self.Linear2(x)
            # x=x.reshape(batch_size*length,-1)
            # x=self.norm2(x)
            # x=x.reshape(batch_size,length,-1)
        # print(torch.max(x))
        # print(torch.min(x))
        x=self.emb(x)
        if timestamp is not None:
            pos_emb=self.positional_embedding(timestamp)
            x=x+pos_emb
        y=self.bert(inputs_embeds=x,attention_mask=attn_mask, output_hidden_states=True)
        y=y.hidden_states[-1]
        # print(torch.max(y))
        # print(torch.min(y))
        return y

    def mask(self,batch_size=1,min=None,max=None,std=None,avg=None):
        if std is not None and avg is not None:
            device=std.device
            result=torch.randn((batch_size, self.bertconfig.max_position_embeddings ,self.input_dim)).to(device)
            result=result*std+avg
        else:
            result=torch.rand((batch_size, self.bertconfig.max_position_embeddings ,self.input_dim))
            if min is not None and max is not None:
                device = max.device
                result=result.to(device)
                result=result*(max-min)+min
        return result

    def positional_embedding(self,timestamp,t=1):
        timestamp**=t
        device=timestamp.device
        min=torch.min(timestamp,dim=-1,keepdim=True)[0]
        max=torch.max(timestamp,dim=-1,keepdim=True)[0]
        ran=timestamp.shape[-1]
        timestamp=(timestamp-min)/(max-min)*ran
        d_model=self.hidden_dim
        dim=torch.tensor(list(range(d_model))).to(device)
        batch_size,length=timestamp.shape
        timestamp=timestamp.unsqueeze(2).repeat(1, 1, d_model)
        dim=dim.reshape([1,1,-1]).repeat(batch_size,length,1)
        sin_emb = torch.sin(timestamp/10000**(dim//2*2/d_model))
        cos_emb = torch.cos(timestamp/10000**(dim//2*2/d_model))
        mask=torch.zeros(d_model).to(device)
        mask[::2]=1
        emb=sin_emb*mask+cos_emb*(1-mask)
        return emb

class Token_Classifier(nn.Module):
    def __init__(self,bert,class_num=52):
        super().__init__()
        self.bert=bert
        self.classifier=nn.Sequential(
            nn.Linear(bert.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, class_num)
        )

    def forward(self,x,attn_mask=None,timestamp=None):
        x=self.bert(x,attn_mask,timestamp)
        x=self.classifier(x)
        return x

# GRL
# 梯度反转层，这一层正向表现为恒等变换，反向传播是改变梯度的符号，alpha用来平衡域损失的权重。
class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Sequence_Classifier(nn.Module):
    def __init__(self,bert,class_num=6):
        super().__init__()
        self.bert=bert
        self.query = nn.Linear(bert.hidden_dim, 64)
        self.key = nn.Linear(bert.hidden_dim, 64)
        self.value = nn.Linear(bert.hidden_dim, 64)
        self.self_attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, dropout=0, batch_first=True)
        self.norm1=nn.BatchNorm1d(64)
        self.Linear=nn.Linear(64, 64)
        self.norm2 = nn.BatchNorm1d(bert.bertconfig.max_position_embeddings * 64)
        self.classifier=nn.Sequential(
            nn.Linear(bert.bertconfig.max_position_embeddings * 64, 64),
            nn.ReLU(),
            nn.Linear(64, class_num)
        )
        self.GRL = GRL()


    def forward(self,x,attn_mask=None,timestamp=None,adversarial=False,alpha=1):
        x=self.bert(x,attn_mask,timestamp)
        if adversarial:
            x = self.GRL.apply(x,alpha)
        batch_size,length,hidden_dim=x.shape
        x_attn, _ = self.self_attention(self.query(x), self.key(x), self.value(x))
        x = x + x_attn
        x1 = x.reshape(-1, 64)
        x1 = self.norm1(x1)
        x1 = self.Linear(x1)
        x2 = x1.reshape(batch_size, -1)
        x2 = self.norm2(x2)
        x2=self.classifier(x2)
        return x2

class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat

class Classification(nn.Module):
    def __init__(self, csibert, class_num, hs=64, da=128, r=4):
        super().__init__()
        self.bert = csibert
        self.attention = SelfAttention(hs, da, r)
        self.classifier = nn.Sequential(
            nn.Linear(hs * r, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )
        self.GRL = GRL()


    def forward(self, x, attn=None, timestamp=None,adversarial=False):
        x = self.bert(x, attn, timestamp)
        if adversarial:
            x = self.GRL.apply(x)
        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.classifier(flatten)
        return res


#test
if __name__ == '__main__':
    configuration = BertConfig(max_position_embeddings=100, hidden_size=64, num_hidden_layers=4,num_attention_heads=4)
    model=CSIBERT(configuration,52,True,True)
    print(model.mask(2))
    #print(model)
    x=torch.randn([2,100,52])
    t=torch.randn([2,100])
    y=model(x,None,t)
    print(y.shape)
    token_classifier=Token_Classifier(model,52)
    y2=token_classifier(x,None,t)
    print(y2.shape)
    seq_classifier=Sequence_Classifier(model,6)
    y3=seq_classifier(x,None,t)
    print(y3.shape)
