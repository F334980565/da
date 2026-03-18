import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..base_network import BaseMIL, Mlp
from copy import deepcopy

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm = norm_layer(in_features) if norm_layer is not None else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    def forward_feature(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class Classifier(nn.Module):    
    def __init__(self, in_features, hidden_features, n_classes, act_layer=nn.ReLU, norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        self.mlp = Mlp(in_features, hidden_features, hidden_features, act_layer, norm_layer, drop)
        self.linear = nn.Linear(hidden_features, n_classes)
    def forward(self, x):
        x = self.mlp(x)
        x = self.linear(x)
        return x
    def forward_feature(self, x):
        x = self.mlp(x)
        return x
    
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
        A = torch.transpose(A, -1, -2) 
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

        A = torch.transpose(A, -1, -2) 
        return A      

class DAttention(BaseMIL):
    def __init__(self, instance_dim, embed_instance_dim, bag_dim, n_classes, 
                 num_attention_heads, attention_dim, gated, bias, dropout, act):
        super().__init__(instance_dim, embed_instance_dim, bag_dim, n_classes)
        
        self.embed_instance_dim = embed_instance_dim
        self.D = attention_dim
        self.num_attention_heads = num_attention_heads
        self.gated = gated
        
        if act.lower() == "gelu":
            act_layer = nn.GELU
        else:
            act_layer = nn.ReLU
            
        self.feature_emb = Mlp(instance_dim, 
                               embed_instance_dim,
                               embed_instance_dim,
                               act_layer,
                               nn.LayerNorm,
                               0.0) 
        
        if gated:
            self.attention = AttentionGated(self.embed_instance_dim, act, bias, dropout)
        else:
            self.attention = Attention(self.embed_instance_dim, act, bias, dropout)
        
        input_dim = self.bag_dim * self.num_attention_heads
        
        self.classifier = Classifier(input_dim, input_dim, n_classes, act_layer=nn.ReLU, norm_layer=nn.LayerNorm, drop=0.0)
    def forward(self, x, return_bag_feat = True):
        embbed_ins_feat = self.feature_emb(x)
        h = embbed_ins_feat.squeeze()
        A = self.attention(h)
        A = F.softmax(A, dim=-1)
        bag_feat = torch.mm(A, h).view(1, -1) 
        
        logits = self.classifier(bag_feat)
        ret = {'logits': logits}
            
        if return_bag_feat:
            ret['bag_feat'] = bag_feat
            
        return ret
    def forward_feat(self, x, return_bag_feat = True):
        embbed_ins_feat = self.feature_emb(x)
        h = embbed_ins_feat.squeeze()
        A = self.attention(h)
        A = F.softmax(A, dim=-1)
        bag_feat = torch.mm(A, h).view(1, -1)
        ret = {}
        ret['bag_feat'] = bag_feat
            
        return ret

class DAttMIL(BaseMIL):
    def __init__(self, instance_dim, embed_instance_dim, bag_dim, n_classes, num_attention_heads, attention_dim, gated, bias, dropout, act):
        super().__init__(instance_dim, embed_instance_dim, bag_dim, n_classes)
        self.alpha_base = 0.99
        self.current_iter = 0
        self.warmup_iter = 200
        
        self.s = DAttention(instance_dim, embed_instance_dim, bag_dim, n_classes, num_attention_heads, attention_dim, gated, bias, dropout, act)
        self.t = deepcopy(self.s)
        self.bag_dim = bag_dim
        for param in self.t.parameters():
            param.requires_grad = False
    def set_bank_size(self, size):
        self.bank_size = size
        device = next(self.parameters()).device 
        
        self.register_buffer("src_mean", torch.zeros(self.bank_size, self.bag_dim, device=device))
        self.register_buffer("src_bag", torch.zeros(self.bank_size, self.bag_dim, device=device))
        self.register_buffer("src_labels", torch.full((self.bank_size,), -1, dtype=torch.long, device=device))
        self.src_ptr = 0
        
        self.register_buffer("tgt_mean", torch.zeros(self.bank_size, self.bag_dim, device=device))
        self.register_buffer("tgt_bag", torch.zeros(self.bank_size, self.bag_dim, device=device))
        self.is_full = False 
        self.tgt_ptr = 0
    @torch.no_grad()
    def update_memory_bank(self, mean_feat, bag_feat, domain, labels=None):
        batch_size = mean_feat.size(0)
        if domain == 'src':
            ptr = int(self.src_ptr)
            for i in range(batch_size):
                self.src_mean[ptr] = mean_feat[i]
                self.src_bag[ptr] = bag_feat[i]
                self.src_labels[ptr] = labels[i]
                ptr = (ptr + 1) % self.bank_size
                if ptr >= self.bank_size - 1: self.is_full = True
            self.src_ptr = ptr
        else:
            ptr = int(self.tgt_ptr)
            for i in range(batch_size):
                self.tgt_mean[ptr] = mean_feat[i]
                self.tgt_bag[ptr] = bag_feat[i]
                ptr = (ptr + 1) % self.bank_size
                if ptr >= self.bank_size - 1: self.is_full = True
            self.tgt_ptr = ptr
    
    def forward(self, x, **kwargs):
        if self.training:
            return self.s(x, **kwargs)
        else:
            ret = self.s.forward_feat(x)
            bag_feat = ret['bag_feat']
            logits = self.t.classifier(bag_feat)
            ret = {'logits': logits}
            
            return ret 
        
    def update_teacher(self):
        progress = min(1.0, 2 * self.current_iter / self.warmup_iter)
        alpha = self.alpha_base * (1 - np.exp(-5 * progress))
        self.current_iter += 1
        
        with torch.no_grad():
            for param_s, param_t in zip(self.s.parameters(), self.t.parameters()):
                param_t.data.mul_(alpha).add_(param_s.data, alpha=1 - alpha)
                
def get_pca_loss(model, frozen_prototypes, num_classes, assign_mode='prototype'):
    f_t = model.tgt_mean
    f_s = model.src_mean

    with torch.no_grad():
        proto_s = model.s.feature_emb(frozen_prototypes)
    
    f_t = model.s.feature_emb(f_t)
    if assign_mode == 'classifier':
        logits_t = model.s.classifier(f_t)
        probs_t = F.softmax(logits_t, dim=1)
        conf_threshold = 0.8
    elif assign_mode == 'prototype':
        f_t_norm = F.normalize(f_t, p=2, dim=1)
        proto_s_norm = F.normalize(proto_s, p=2, dim=1)
        cos_sim = torch.mm(f_t_norm, proto_s_norm.t()) 
        tau = 0.1 
        probs_t = F.softmax(cos_sim / tau, dim=1)
        conf_threshold = 0.85

    max_probs, pseudo_labels_t = torch.max(probs_t, dim=1)
    conf_mask = (max_probs > conf_threshold).float()

    proto_t_list = []
    for c in range(num_classes):
        mask_t = ((pseudo_labels_t == c).float() * conf_mask)
        if mask_t.sum() > 0:
            c_proto = (f_t * mask_t.unsqueeze(1)).sum(0) / mask_t.sum()
            proto_t_list.append(c_proto)
        else:
            proto_t_list.append(proto_s[c].detach())
    
    proto_t = torch.stack(proto_t_list) 
    cos_loss = 1.0 - F.cosine_similarity(proto_s, proto_t, dim=1).mean()
    return cos_loss 
def pairwise_cosine_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return 1.0 - torch.matmul(x_norm, y_norm.T)

def get_awpd_loss(model, temperature=1.0): 
    w_t = model.t.classifier.linear.weight.detach() 
    f_s = model.src_bag                  
    f_t = model.tgt_bag                  
    y_s_gt = model.src_labels

    f_s = model.s.classifier.forward_feature(f_s)
    f_t = model.s.classifier.forward_feature(f_t)

    tgt_logits = model.s.classifier.linear(f_t) 
    tgt_assign = F.softmax(tgt_logits / temperature, dim=1)
    cost_mat_t = pairwise_cosine_dist(f_t, w_t) 
    loss_target = (tgt_assign * cost_mat_t).sum(dim=1).mean()

    src_logits = model.s.classifier.linear(f_s)
    src_assign = F.softmax(src_logits / temperature, dim=1)
    cost_mat_s = pairwise_cosine_dist(f_s, w_t)
    loss_source_transport = (src_assign * cost_mat_s).sum(dim=1).mean()

    return (loss_target + loss_source_transport) * 0.5

def cal_energy(logits_t, T=1.0):
    energy_t = -T * torch.logsumexp(logits_t / T, dim=1)
    return torch.mean(energy_t)

def get_energy_loss(model, frozen_prototypes, temperture=1.0):
    with torch.no_grad():
        proto_emb = model.s.feature_emb(frozen_prototypes) 
        proto_logits = model.s.classifier(proto_emb)
        tau = - torch.logsumexp(proto_logits, dim=-1).mean()        
                            
    logits_s = model.s.classifier(model.src_bag)
    logits_t = model.s.classifier(model.tgt_bag)
    src_energy = cal_energy(logits_s).mean()
    tgt_energy = cal_energy(logits_t).mean()
    
    src_gap = torch.relu(tau - src_energy) 
    tgt_gap = torch.relu(tau - tgt_energy) 
    gap_loss = torch.pow(src_gap, 2) + torch.pow(tgt_gap, 2)

    dist_loss = torch.pow(tgt_energy - src_energy, 2)
    energy_loss = dist_loss + 0.1 * gap_loss 
    return energy_loss