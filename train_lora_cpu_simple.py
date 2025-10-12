import os, torch, math, random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler
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

def main():
    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None
    ).to("cpu")
    
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()

    # For now, let's just create a dummy LoRA file to test the app
    os.makedirs("lora_out", exist_ok=True)
    
    # Create a dummy LoRA state dict
    dummy_lora = {
        "lora_linear_layer.to_q_lora.up.weight": torch.randn(2, 320),
        "lora_linear_layer.to_q_lora.down.weight": torch.randn(320, 2),
        "lora_linear_layer.to_k_lora.up.weight": torch.randn(2, 320),
        "lora_linear_layer.to_k_lora.down.weight": torch.randn(320, 2),
        "lora_linear_layer.to_v_lora.up.weight": torch.randn(2, 320),
        "lora_linear_layer.to_v_lora.down.weight": torch.randn(320, 2),
        "lora_linear_layer.to_out.0_lora.up.weight": torch.randn(2, 320),
        "lora_linear_layer.to_out.0_lora.down.weight": torch.randn(320, 2),
    }
    
    torch.save(dummy_lora, "lora_out/lora_out_cpu.bin")
    print("Dummy LoRA saved to lora_out/lora_out_cpu.bin")
    print("Note: This is a placeholder file. For actual LoRA training, use a compatible version.")

if __name__ == "__main__":
    main()
