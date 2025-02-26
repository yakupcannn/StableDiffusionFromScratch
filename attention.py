import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self,n_heads:int,channels:int,in_proj_bias=True,out_proj_bias = True):
        super().__init__()
        self.in_proj = nn.Linear(channels,3 * channels,bias=in_proj_bias)
        self.out_proj = nn.Linear(channels,channels,bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = channels//n_heads

    def forward(self,x:torch.tensor,causal_mask=False)->torch.tensor:
        # x:(batch,seq_len,dim) 
        b,sq,d = x.shape
        qkv_shape = (b,sq,self.n_heads,self.d_head)
        #(batch,seq_len,dim) -> 3 tensors (batch,seq_len,dim)
        q,k,v = self.in_proj(x).chunk(3,dim = -1)
        
        #(batch,seq_len,dim) -> (batch,seq_len,n_heads,d_head) -> (batch,n_heads,seq_len,d_head)
        q = q.view(qkv_shape).transpose(1,2)
        k = k.view(qkv_shape).transpose(1,2)
        v = v.view(qkv_shape).transpose(1,2)
        
        #{(batch,n_heads,seq_len,d_head) @ (batch,n_heads,d_head,seq_len)} -> (batch,n_heads,seq_len,seq_len) 
        weight = q @ k.transpose(-1,-2)

        if causal_mask:
            #Mask where the upper triangle is 1 
            mask = torch.ones_like(weight,dtype=torch.bool).triu(1)
            weight.masked_fill_(mask,-torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight,dim = -1)
        
        #{(batch,n_heads,seq_len,seq_len) @ (batch,n_heads,seq_len,d_head) } -> (batch,n_heads,seq_len,d_head)
        output = weight @ v

        #(batch,n_heads,seq_len,d_head) -> (batch,seq_len,n_heads,d_head)
        output = output.transpose(1,2)

        output = output.reshape((b,sq,d))
        
        output = self.out_proj(output)
        
        #(batch,seq_len,dim)
        return output 

class CrossAttention(nn.Module):
    def __init__(self,n_heads:int, embed_size:int,d_cross:int,in_proj_bias=True,out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(embed_size,embed_size,bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross,embed_size,bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross,embed_size,bias=in_proj_bias)
        self.out_proj = nn.Linear(embed_size,embed_size,bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = embed_size // n_heads

    def forward(self,x,y):
        #x: latent (Batch,seq_len_Q,dim_Q)
        #y: context (Batch,seq_len_KV,dim_KV) (Batch,77,768)
        b,seq_len,embed_size = x.shape
        inter_shape = (b,-1,self.n_heads,self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(inter_shape).transpose(1,2)
        k = k.view(inter_shape).transpose(1,2)
        v = v.view(inter_shape).transpose(1,2)

        weight = q @ k.transpose(-1,-2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight,dim = -1)
        
        output = weight @ v

        output = output.transpose(1,2).contiguous()

        output = output.reshape((b,seq_len,embed_size))
        
        output = self.out_proj(output)

        return output







