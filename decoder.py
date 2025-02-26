import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention

G_NUMBER = 32
SCALING_FACTOR = 0.18215

class VAE_AttentionBlock(nn.Module):
    def __init__(self,channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(G_NUMBER,channels)
        self.attention = SelfAttention(1,channels)
    

    def forward(self,x:torch.tensor)->torch.tensor:
        # x:(Batch,Channels,Height,Width)
        res_x = x

        x = self.groupnorm(x)

        b,c,h,w = x.shape
        
        #(Batch,Channels,Height,Width) -> (Batch,Channels,Height * Width)
        x = x.view((b,c,h*w))
        
        #(Batch,Channels,Height*Width) -> (Batch,Height * Width,Channels)
        x = x.transpose(-1,-2)
        
        #(Batch,Height * Width,Channels) -> (Batch,Height * Width,Channels)
        x = self.attention(x)

        #(Batch,Height * Width,Channels) -> (Batch,Channels,Height * Width)
        x = x.transpose(-1,-2)
        
        #(Batch,Channels,Height * Width) -> (Batch,Channels,Height,Width)
        x = x.view((b,c,h,w))

        return x + res_x



class VAE_ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        
        self.groupnorm_1 = nn.GroupNorm(G_NUMBER,in_channels)
        self.conv_1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)

        self.groupnorm_2 = nn.GroupNorm(G_NUMBER,out_channels)
        self.conv_2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
            

    def forward(self,x:torch.tensor) -> torch.tensor:
        #x:(Batch,in_channels,height,width)
        res_x = x
        
        x= self.groupnorm_1(x)
        x = F.silu(x)
        x= self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x =self.conv_2(x) 

        return x + self.residual_layer(res_x)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4, kernel_size=1,padding=0),
            nn.Conv2d(4,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            
            #(Batch,512,H/8,W/8) -> (Batch,512,H/8,W/8)
            VAE_ResidualBlock(512,512),
            
            #(Batch,512,H/8,W/8) -> (Batch,512,H/4,W/4)
            nn.Upsample(scale_factor=2),
            
            #(Batch,512,H/4,W/4) ->(Batch,512,H/4,W/4)
            nn.Conv2d(512,512,kernel_size=3,padding=1),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            #(Batch,512,H/4,W/4) ->(Batch,512,H/2,W/2)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 

            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),

            #(Batch,256,H/2,W/2) ->(Batch,256,H,W)
            nn.Upsample(scale_factor=2),

             #(Batch,256,H,W) ->(Batch,256,H,W)
            nn.Conv2d(256,256,kernel_size=3,padding=1),

            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            nn.GroupNorm(G_NUMBER,128),

            nn.SiLU(),
            #To convert the image RGB channels
            nn.Conv2d(128,3,kernel_size=3,padding=1)
        )
    def forward(self, x:torch.tensor) -> torch.tensor:
        # x:(Batch,4,H/8,W/8)
        x /= SCALING_FACTOR
        for module in self:
            x= module(x)


        #(Batch,3,H,W)
        return x

