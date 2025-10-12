import os, torch, math, random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import AutoTokenizer

class CaptionDataset(Dataset):
    def __init__(self, csv_path, image_size=256):
        self.df = pd.read_csv(csv_path)
        self.size = image_size
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB").resize((self.size, self.size))
        img_array = torch.tensor(list(img.getdata()), dtype=torch.uint8)
        img = img_array.view(self.size, self.size, 3).permute(2,0,1).float()/255*2-1
        return {"pixel_values": img, "caption": row["caption"]}

def add_lora(unet, rank=2):
    lora_attn_procs = {}
    for name, module in unet.attn_processors.items():
        # Get cross attention dimension from unet config
        cross_dim = None
        if "cross_attention_dim" in unet.config:
            cross_dim = unet.config.cross_attention_dim
        hidden_size = unet.config.block_out_channels[0]
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_dim,
            rank=rank
        )
    unet.set_attn_processor(lora_attn_procs)

def lora_params(unet):
    params = []
    for proc in unet.attn_processors.values():
        if hasattr(proc, 'parameters'):
            params.extend([p for p in proc.parameters() if p.requires_grad])
    return params

def main():
    accelerator = Accelerator(cpu=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()

    unet, vae, tokenizer, text_encoder = pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    add_lora(unet, rank=2)
    params = lora_params(unet)
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    dataset = CaptionDataset("data/captions.csv", 256)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    noise_scheduler = pipe.scheduler
    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)

    max_steps = 200
    global_step = 0
    print("Training LoRA on CPU...")

    while global_step < max_steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                inputs = tokenizer(batch["caption"], padding="max_length", truncation=True,
                                   max_length=tokenizer.model_max_length, return_tensors="pt")
                enc = text_encoder(**inputs)[0]
                imgs = batch["pixel_values"].to(accelerator.device)
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],)).long()
                noisy = noise_scheduler.add_noise(latents, noise, t)
                pred = unet(noisy, t, encoder_hidden_states=enc).sample
                loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
                accelerator.backward(loss)
                optimizer.step(); optimizer.zero_grad()

            if global_step % 20 == 0:
                print(f"Step {global_step} | Loss {loss.item():.4f}")
            global_step += 1
            if global_step >= max_steps: break

    if accelerator.is_main_process:
        os.makedirs("lora_out", exist_ok=True)
        torch.save(unet.attn_processors.state_dict(), "lora_out/lora_out_cpu.bin")
        print("LoRA saved to lora_out/lora_out_cpu.bin")

if __name__ == "__main__":
    main()
