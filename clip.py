import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention

class ClipEmbedding(nn.Module):
    def __init__(self,vocab_size:int,emded_size:int,n_tokens:int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,emded_size)
        #This positional embedding vectors are gonna be learned while training
        self.position_embedding = nn.Parameter(torch.zeros((n_tokens,emded_size)))

    def forward(self,tokens):
        #(Batch,seq_len) -> (Batch,seq_len,dim)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x 
    

class ClipLayer(nn.Module):
    
    def __init__(self,n_head:int,emd_size:int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(emd_size)
        self.attention = SelfAttention(n_head,emd_size)
        self.layernorm_2 = nn.LayerNorm(emd_size)
        self.linear_1 = nn.Linear(emd_size,4*emd_size)
        self.linear_2 = nn.Linear(4*emd_size,emd_size)


    def forward(self,x):
        # x:(Batch,seq_len,dim)
        
        ##Self Attention Block
        res_x = x 
        x = self.layernorm_1(x)
        x = self.attention(x,causal_mask=True)
        x += res_x

        ## Feed Forward Block
        res_x = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        # Quick Gelu Activation Func
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear_2(x)
        x+=res_x
        
        return x


        

        

class Clip(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = ClipEmbedding(49408,768,77)
        self.layers = nn.ModuleList([
            ClipLayer(12,768) for i in range(12)
        ]) 
        self.layernorm = nn.LayerNorm(768)


    def forward(self,tokens:torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        #(Batch,seq_len) -> (Batch,seq_len,dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
            
        output = self.layernorm(state)
        #(Batch,seq_len,dim)
        return output