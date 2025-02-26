import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention,CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self,embed_size:int):
        super().__init__()
        self.linear_1 = nn.Linear(embed_size,4*embed_size)
        self.linear_2 = nn.Linear(4*embed_size,4*embed_size)

    def forward(self,x:torch.tensor) ->torch.tensor:
        #x:(1,320)
        x=self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        return x
    
class UpSample(nn.Module):
    def __init__(self,channels:int):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x):
        #Double Size
        x = F.interpolate(x,scale_factor=2,mode="nearest")
        return self.conv(x)

class Unet_Residual_Block(nn.Module):
    def __init__(self,in_channels,out_channels,timesteps=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32,in_channels)
        self.conv_feature = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.linear_time = nn.Linear(timesteps,out_channels)

        self.groupnorm_merged = nn.GroupNorm(32,out_channels)
        self.conv_merged = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)

    def forward(self,x,time):
        #x:(Batch,in_channels,H/8,W/8)
        #time:(1,1280)
        res_x = x
        x = self.groupnorm_feature(x)
        x= F.silu(x)
        x = self.conv_feature(x)
        time = F.silu(time)
        time = self.linear_time(time)

        merged = x + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(res_x)

class Unet_Attention_Block(nn.Module):
    def __init__(self,n_heads:int,embed_size:int,d_context=768):
        super().__init__()
        channels = n_heads*embed_size

        self.groupnorm = nn.GroupNorm(32,channels,eps=1e-6)
        self.conv_input = nn.Conv2d(channels,channels,kernel_size=1,padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1   = SelfAttention(n_heads,channels,in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads,channels,d_context,in_proj_bias =False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels,4*channels*2)
        self.linear_geglu_2  = nn.Linear(4*channels,channels)

        self.conv_output = nn.Conv2d(channels,channels,kernel_size=1,padding=0)

    def forward(self,x,context):
        #x:(Batch,channels,h,w)
        #context:(batch,seq_len,dim)

        final_res_x = x 
        x = self.groupnorm(x)
        x = self.conv_input(x)

        b,c,h,w = x.shape
        #(Batch,Channels,H,W) -> (Batch,Channels,H*W)
        x = x.view((b,c,h*w))
        #(Batch,Channels,H*W) -> (Batch,H*W,Channels)
        x=x.transpose(-1,-2)
      
        res_x = x
        x = self.layernorm_1(x)
        #Self Attention
        x = self.attention_1(x)
        x += res_x

        res_x = x
        x = self.layernorm_2(x)
        #Cross Attention
        x = self.attention_2(x,context)
        x += res_x

        res_x = x
        x = self.layernorm_3(x)
        x,gate = self.linear_geglu_1(x).chunk(2,dim = -1)
        x *= F.gelu(gate)
        x = self.linear_geglu_2(x)
        x +=res_x

        #(Batch,H*W,Channels) ->  (Batch,Channels,H*W)
        x = x.transpose(-1,-2)
        #(Batch,Channels,H*W) -> (Batch,Channels,H,W)
        x = x.view((b,c,h,w))

        return self.conv_output(x) + final_res_x



class SwitchSequential(nn.Sequential):
    
    def forward (self,x:torch.tensor,context:torch.tensor,time:torch.tensor) ->torch.tensor:
        for layer in self:
            if isinstance(layer,Unet_Residual_Block):
                x = layer(x,time)
            elif isinstance(layer,Unet_Attention_Block):
                x = layer(x,context)
            else:
                x = layer(x)
        return x 


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4,320,kernel_size=3,padding=1)),
            SwitchSequential(Unet_Residual_Block(320,320),Unet_Attention_Block(8,40)),
            SwitchSequential(Unet_Residual_Block(320,320),Unet_Attention_Block(8,40)),
            SwitchSequential(nn.Conv2d(320,320,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(Unet_Residual_Block(320,640),Unet_Attention_Block(8,80)),
            SwitchSequential(Unet_Residual_Block(640,640),Unet_Attention_Block(8,80)),
            SwitchSequential(nn.Conv2d(640,640,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(Unet_Residual_Block(640,1280),Unet_Attention_Block(8,160)),
            SwitchSequential(Unet_Residual_Block(1280,1280),Unet_Attention_Block(8,160)),
            SwitchSequential(nn.Conv2d(1280,1280,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(Unet_Residual_Block(1280,1280)),
            SwitchSequential(Unet_Residual_Block(1280,1280)),
        ])
        self.bottleneck = SwitchSequential(
            Unet_Residual_Block(1280,1280),
            Unet_Attention_Block(8,160),
            Unet_Residual_Block(1280,1280),
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(Unet_Residual_Block(2560,1280)),
            SwitchSequential(Unet_Residual_Block(2560,1280)),
            SwitchSequential(Unet_Residual_Block(2560,1280),UpSample(1280)),
            SwitchSequential(Unet_Residual_Block(2560,1280),Unet_Attention_Block(8,160)),
            SwitchSequential(Unet_Residual_Block(2560,1280),Unet_Attention_Block(8,160)),
            SwitchSequential(Unet_Residual_Block(1920,1280),Unet_Attention_Block(8,160),UpSample(1280)),
            SwitchSequential(Unet_Residual_Block(1920,640),Unet_Attention_Block(8,80)),
            SwitchSequential(Unet_Residual_Block(1280,640),Unet_Attention_Block(8,80)),
            SwitchSequential(Unet_Residual_Block(960,640),Unet_Attention_Block(8,80),UpSample(640)),
            SwitchSequential(Unet_Residual_Block(960,320),Unet_Attention_Block(8,40)),
            SwitchSequential(Unet_Residual_Block(640,320),Unet_Attention_Block(8,40)),
            SwitchSequential(Unet_Residual_Block(640,320),Unet_Attention_Block(8,40)),

        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class LastUnetLayer(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,in_channels)
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)

    def forward(self,x):
        #x:(Batch,320,H/8,W/8)
        x = self.groupnorm(x)
        x = F.silu(x)
        #x:(Batch,4,H/8,W/8)
        return self.conv(x)




class Diffusion(nn.Module):
    def __init__(self,):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = Unet()
        self.final = LastUnetLayer(320,4)

    def forward(self,latent:torch.tensor,context:torch.tensor,time:torch.tensor) ->torch.tensor:
        #latent:(Batch,4,H/8,W/8)
        #context:(Batch,seq_len,dim)
        #time:(1,320)

        #(1,320) -> (1,1280)
        time = self.time_embedding(time)
        
        #(Batch,4,H/8,W/8) -> (Batch,320,H/8,W/8)
        output = self.unet(latent,context,time)
        #(Batch,320,H/8,W/8) -> (Batch,4,H/8,W/8)
        output = self.final(output)
        
        #(Batch,4,H/8,W/8)
        return output







