import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_network import BaseMIL, Mlp

class Attention(nn.Module):
    def __init__(self,input_dim=512,act='relu',bias=False,dropout=False):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = 128 
        self.K = 1

        self.attention = [nn.Linear(self.L, self.D,bias=bias)]

        if act == 'gelu': 
            self.attention += [nn.GELU()]
        elif act == 'relu':
            self.attention += [nn.ReLU()]
        elif act == 'tanh':
            self.attention += [nn.Tanh()]

        if dropout:
            self.attention += [nn.Dropout(0.25)]

        self.attention += [nn.Linear(self.D, self.K,bias=bias)]

        self.attention = nn.Sequential(*self.attention)

    def forward(self,x):
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        return A

class AttentionGated(nn.Module):
    def __init__(self, input_dim=512, act='relu',bias=False,dropout=False):
        super(AttentionGated, self).__init__()
        self.L = input_dim
        self.D = 128 
        self.K = 1

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        return A

class DAttention(BaseMIL):
    def __init__(self, instance_dim, embed_instance_dim, bag_dim, n_classes, num_attention_heads, attention_dim, gated, bias, dropout, act):
        super().__init__(instance_dim, embed_instance_dim, bag_dim, n_classes)
        self.embed_instance_dim = embed_instance_dim
        self.D = attention_dim 
        self.num_attention_heads = num_attention_heads
        self.gated = gated
        
        if act.lower() == "gelu":
            act_layer = nn.GELU
        else:
            act_layer = nn.ReLU
                
        if gated:
            self.attention = AttentionGated(self.embed_instance_dim, act, bias, dropout)
        else:
            self.attention = Attention(self.embed_instance_dim, act,bias, dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.bag_dim * self.num_attention_heads, n_classes),
        )
        
    def forward(self, x, return_attn=True, return_bag_feat=True):
        h = x.squeeze() 
        A = self.attention(h)
        A = F.softmax(A, dim=-1)  
        bag_feat = torch.mm(A, h) 
        
        logits = self.classifier(bag_feat)

        ret = {'logits': logits}

        if return_attn:
            ret['attn'] = A
        if return_bag_feat:
            ret['bag_feat'] = bag_feat
        
        return ret

