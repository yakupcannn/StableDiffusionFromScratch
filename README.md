# StableDiffusionFromScratch

## About 
Stable Diffusion is a strategy to create new images from random noise or given image

### How It Works
- Basically we get an input either text or image
- Add some random noise on it until the image looks like noise
- Find Noise Loss using basic matrix substraction
- Backpropagate the loss until getting closer 0
- Update Weights
- Create new image according to text prompt or given image

## How To Install ?
- create a folder which is called "StableDiffusionFromScratch"
- go to the directory using cd StableDiffusionFromScratch
- clone the repository using git clone "repository link"
- download requirements using requirements.txt (pip install -r requirements.txt)
- Ready to use

## Download Pretrained Weights and Tokenizer Files
- Download "vocab.json" and "merges.txt" from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer and save them in the data folder
- Download "v1-5-pruned-emaonly.ckpt" from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main and save it in the data folder

  ## Sources
  - "https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=17510s&ab_channel=UmarJamil"
  - "https://arxiv.org/pdf/2006.11239"
  



  
