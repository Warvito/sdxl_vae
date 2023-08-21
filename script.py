from PIL import Image
from diffusers.models.autoencoder_kl import AutoencoderKL
import numpy as np
import torch

image_path = "/home/walter/Downloads/Breast_MRI_009_0000_slice95.png"
image = Image.open(image_path)
# image to torch
image = torch.from_numpy(np.array(image)).float()
image = image.unsqueeze(0).unsqueeze(0)
image = image.repeat_interleave(3, dim=1)
image = image / 255.0
image = image * 2.0 - 1.0

device = torch.device("cuda")
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae")
vae = vae.to(device)
vae.eval()

with torch.no_grad():
    latent = vae.encode(image.to(device)).latent_dist.sample()
    recon = vae.decode(latent, return_dict=False)[0]

recon = recon.cpu().numpy()
recon = recon.squeeze(0)[0]
image = image.cpu().numpy()
image = image.squeeze(0)[0]
image = np.concatenate((image, recon), axis=1)
image = (image + 1.0) / 2.0
image = image * 255.0
image = image.astype(np.uint8)
image = Image.fromarray(image)
image.save("image.png")
