import torch
import torch.nn as nn 
from torch.nn import functional as F
import numpy as np
from ddpm import DDPMSampler
from tqdm import tqdm

WIDTH = 512
HEIGHT = 512
LATENT_WITDH = WIDTH // 8
LATENT_HEIGHT = HEIGHT //8

def generate_image(prompt:str,uncond_prompt:str,input_image:None,
                   in_img_att=0.8,do_cfg=True,promt_cond_att=7.5,
                   sampler_name="ddpm",n_infer_steps=50,models={},
                   seed = None,device = None,idle_device=None,tokenizer= None):
    with torch.no_grad():
        if device == None or idle_device == None :
            raise ValueError("Device or Idle Device cannot be None")

        if not (0< in_img_att <= 1):
            raise ValueError("input image attention value must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device = device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            #turn prompt into tokens
            cond_tokens = tokenizer.batch_encode_plus([prompt],padding="max_length",max_length=77).input_ids
            #(Batch,seq_len)
            cond_tokens = torch.tensor(cond_tokens,dtype=torch.long,device = device)
            #(Batch,seq_len) --> (Batch,seq_len,dim)
            cond_emd = clip(cond_tokens)

            #turn prompt into tokens
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt],padding="max_length",max_length=77).input_ids
            #(Batch,seq_len)
            uncond_tokens = torch.tensor(uncond_tokens,dtype=torch.long,device = device)
            #(Batch,seq_len) --> (Batch,seq_len,dim)
            uncond_emd = clip(uncond_tokens)
            #(2,seq_len,dim)
            context_emd = torch.cat([cond_emd,uncond_emd])


        else:
            tokens = tokenizer.batch_encode_plus([prompt],padding="max_length",max_length=77).input_ids
            tokens = torch.tensor(tokens,dtype=torch.long,device=device)
            context_emd = clip(tokens)

        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_infer_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        latent_shape = (1,4,LATENT_HEIGHT,LATENT_WITDH)
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image = input_image.resize((WIDTH,HEIGHT))
            input_image_tensor = np.array(input_image)
            input_image_tensor = torch.tensor(input_image_tensor,dtype=torch.float32,device=device)

            input_image_tensor = transform_img(input_image_tensor,(0,255),(-1,1))
            #(Height,Width,Channel) --> (Batch,Height,Width,Channels)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            #(Batch,Height,Width,Channels) --> (Batch,Channel,Height,Width)
            input_image_tensor = input_image_tensor.permute(0,3,1,2)

            encoder_noise = torch.randn(latent_shape,generator=generator,device=device)
            #Send the image to encoder of VAE
            latents = encoder(input_image_tensor,encoder_noise)
            sampler.set_img_attention(att_amount=in_img_att)
            latents = sampler.add_noise(latents,sampler.time_steps[0])
            to_idle(encoder)

        else:
            latents = torch.randn(latent_shape,generator=generator,device=device)
            
        diffusion = models["diffusion"]
        diffusion.to(device)
        time_steps = tqdm(sampler.time_steps)
        for i,time_step in enumerate(time_steps):
            #(1,320) 
            time_emd = get_time_embedding(time_step).to(device)
            #(Batch,4,latent_height,latent_width)
            model_input = latents
            
            if do_cfg:
                #(Batch,4,latent_height,latent_width) -> (2*Batch,4,latent_height,latent_width)
                model_input = model_input.repeat(2,1,1,1)
                #Predicted Noise by UNET
            model_output = diffusion(model_input,context_emd,time_emd)

            if do_cfg:
                cond_output, uncond_output = model_output.chunk(2)
                model_output = promt_cond_att * (cond_output-uncond_output) + uncond_output
            # Remove predicted noise from the image 
            latents = sampler.step(time_step,latents,model_output)
        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = transform_img(images,(-1,1),(0,255),clamp=True)
        #(Batch,Channels,Height,Width) -> (Batch,Height,Width,Channels)
        images = images.permute(0,2,3,1)
        images = images.to("cpu",torch.uint8).numpy()
        return images[0]
    

def transform_img(x:torch.tensor,first_scale,second_scale,clamp=False):
    first_min,first_max = first_scale
    second_min,second_max = second_scale
    x -= first_min
    x *= (second_max-second_min)/(first_max-first_min)
    x+=second_min
    if clamp:
        x = x.clamp(second_min,second_max)
    return x


def get_time_embedding(time_step):
    #(160,)
    freq = torch.pow(10000,-torch.arange(start=0,end=160,dtype=torch.float32)/160)
    #(1,160)
    x = torch.tensor([time_step],dtype=torch.float32)[:,None] * freq[None]
    #(1,320)
    return torch.cat([torch.cos(x),torch.sin(x)],dim=-1)

    
        


















        

        
