import torch
import torch.nn as nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock,VAE_ResidualBlock

SCALING_FACTOR = 0.18215  # Scaling factor for multivariate distribution

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            #(Batch,3,H,W) -> (Batch,128,H,W)
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            
            #(Batch,128,H,W) -> (Batch,128,H,W)
            VAE_ResidualBlock(128,128),
            
            #(Batch,128,H,W) -> (Batch,128,H,W)
            VAE_ResidualBlock(128,128),

            #(Batch,128,H,W) -> (Batch,128,H/2,W/2)
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
            
            #(Batch,128,H/2,W/2) -> (Batch,256,H/2,W/2)
            VAE_ResidualBlock(128,256),

            #(Batch,256,H/2,W/2) ->(Batch,256,H/2,W/2)
            VAE_ResidualBlock(256,256),

            #(Batch,256,H/2,W/2) -> (Batch,256,H/4,W/4)
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),
            
            #(Batch,256,H/4,W/4) -> (Batch,512,H/4,W/4)
            VAE_ResidualBlock(256,512),
            
            #(Batch,512,H/4,W/4) -> (Batch,512,H/4,W/4)
            VAE_ResidualBlock(512,512),

            #(Batch,512,H/4,W/4) -> (Batch,512,H/8,W/8)
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),

            #(Batch,512,H/8,W/8) -> (Batch,512,H/8,W/8)
            VAE_ResidualBlock(512,512),

            #(Batch,512,H/8,W/8) -> (Batch,512,H/8,W/8)
            VAE_ResidualBlock(512,512),

            #(Batch,512,H/8,W/8) -> (Batch,512,H/8,W/8)
            VAE_ResidualBlock(512,512),

            #(Batch,512,H/8,W/8) -> (Batch,512,H/8,W/8)
            VAE_AttentionBlock(512),

            #(Batch,512,H/8,W/8) -> (Batch,512,H/8,W/8)
            VAE_ResidualBlock(512,512),
            
            #(Batch,512,H/8,W/8) -> (Batch,512,H/8,W/8)
            nn.GroupNorm(32,512),
            
            #Activation Function
            nn.SiLU(),

            #Bottleneck (Batch,512,H/8,W/8) ->(Batch,8,H/8,W/8)
            nn.Conv2d(512,8,kernel_size=3,padding=1),

            #(Batch,8,H/8,W/8) -> (Batch,8,H/8,W/8)
            nn.Conv2d(8,8,kernel_size=1,padding=0),
        )
    def forward(self, x:torch.tensor, noise:torch.tensor) -> torch.tensor:
        #x:(Batch_size,Channel,Height,Width)
        #noise: (Batch_size,output_channels,height,width), N(0,1)
        
        
        for module in self:
            if getattr(module,"stride",None) == (2,2):
                #Add Padding to Particular loc
                x = F.pad(x,(0,1,0,1))
            x = module(x)
        #(Batch,8,H/8,W/8) -> 2 tensors (Batch,4,H/8,W/8)
        mean, log_variance = torch.chunk(x,2,dim=1)
        log_variance.clamp(min=-30,max=20)
        variance = log_variance.exp()
        std = variance.sqrt()

        # Z=N(0,1) -> X=N(mean,variance)??
        # X = mean + std*Z
        x = mean + std * noise
        
        #Scaling factor
        x *= SCALING_FACTOR
        return x
    


