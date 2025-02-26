import torch
import numpy as np

class DDPMSampler:
    def __init__(self,generator:torch.Generator,training_steps:int = 1000,beta_start:float=0.00085,beta_end:float=0.0120):
        self.betas = torch.linspace(beta_start ** 0.5,beta_end ** 0.5,training_steps,dtype = torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas,dim=0)
        self.one_tensor = torch.tensor(1.0)

        self.generator = generator
        self.training_steps = training_steps
        self.time_steps = torch.from_numpy(np.arange(0,training_steps)[::-1].copy())

    def set_inference_steps(self,n_infer_steps=50):
        self.num_inference_steps = n_infer_steps
        step_ratio = self.training_steps // self.num_inference_steps
        time_steps = (np.arange(0, self.num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.time_steps = torch.from_numpy(time_steps)

    def _get_prev_time_step(self,time_step:int)->int:
        return time_step - self.training_steps//self.num_inference_steps
    
    def set_img_attention(self,att_amount = 1.0):
        beginning_step = self.num_inference_steps- int( self.num_inference_steps * att_amount)
        self.time_steps = self.time_steps[beginning_step:]
        self.beginning_step = beginning_step



    def step(self,time_step:int,latents:torch.tensor,pred_model_out:torch.tensor):
        prev_t = self._get_prev_time_step(time_step)
        alpha_cumprod_t = self.alphas_cumprod[time_step]
        beta_cumprod_t = 1 - alpha_cumprod_t
        alpha_prev_cumprod_t = self.alphas_cumprod[prev_t] if prev_t >=0 else self.one_tensor
        beta_prev_cumprod_t = 1 - alpha_prev_cumprod_t 
        curr_alpha_t = alpha_cumprod_t / alpha_prev_cumprod_t
        curr_beta_t = 1 - curr_alpha_t

        #Compute predicted original samples using DDPM Paper Formula
        pred_original_sample =(latents - beta_cumprod_t ** (0.5) * pred_model_out) / alpha_cumprod_t ** (0.5)

        # Compute predicted original samples and current sample coefficients
        pred_original_sample_coeff =  (alpha_prev_cumprod_t ** 0.5 * curr_beta_t) / (beta_cumprod_t)
        curr_sample_coeff = (curr_alpha_t ** (0.5) * beta_prev_cumprod_t) / beta_cumprod_t
        # Compute the mean of prev sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + curr_sample_coeff * latents
        var = 0 
        if time_step > 0:
            noise = torch.randn(pred_model_out.shape,generator=self.generator,device=pred_model_out.device,dtype = pred_model_out.dtype)
            var = (self._get_variance(time_step)**0.5) *noise
        
        return pred_prev_sample +  var 
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_prev_time_step(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one_tensor
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        return variance




    def add_noise(self,original_samples:torch.FloatTensor,curr_time_step:torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alphas_cumprod.to(device = original_samples.device,dtype = original_samples.dtype)
        curr_time_step = curr_time_step.to(device = original_samples.device)
        sqrt_alpha_prod = alpha_cumprod[curr_time_step] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        stdev = (1-alpha_cumprod[curr_time_step])** 0.5
        stdev = stdev.flatten()
        while len(stdev.shape) < len(original_samples.shape):
            stdev = stdev.unsqueeze(-1)

        
        return sqrt_alpha_prod*original_samples + stdev * torch.randn(original_samples.shape,generator=self.generator,device = original_samples.device,dtype = original_samples.dtype)
    



