import os
os.environ["XFORMERS_DISABLE"] = "1"

import os, csv, random, time
from diffusers import StableDiffusionPipeline
import torch

os.makedirs("data/images", exist_ok=True)

prompts = [
    "a cozy coffee shop interior, warm light",
    "mountain landscape at sunrise",
    "cyberpunk neon street, night rain",
    "futuristic city skyline at night",
    "a cat wearing sunglasses",
    "starry sky galaxy, vibrant colors",
    "beautiful sunset beach, cinematic",
    "robot standing in a forest",
    "library interior with warm lighting",
    "temple surrounded by mist"
]

NUM_IMAGES = 100
IMAGE_SIZE = 256
OUT_CSV = "data/captions.csv"

print("Loading Stable Diffusion on CPU... (be patient)")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
    torch_compile=False
).to("cpu")

pipe.enable_vae_tiling()

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "caption"])

    for i in range(NUM_IMAGES):
        p = random.choice(prompts)
        print(f"Generating {i+1}/{NUM_IMAGES}: {p}")
        t0 = time.time()
        img = pipe(
            p,
            num_inference_steps=15,
            guidance_scale=7.5,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE
        ).images[0]

        path = f"data/images/{i+1:03d}.jpg"
        img.save(path)
        writer.writerow([path, p])
        print(f"Saved {path} ({time.time()-t0:.1f}s)")
