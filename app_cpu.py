import os
os.environ["XFORMERS_DISABLE"] = "1"
os.environ["DIFFUSERS_NO_ONNX"] = "1"


import torch
import streamlit as st
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="Text-to-Image CPU", page_icon="🎨")

@st.cache_resource
def load_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-base",
        torch_dtype=torch.float32,
        safety_checker=None
    ).to("cpu")
    pipe.enable_vae_tiling()
    if os.path.exists("lora_out/lora_out_cpu.bin"):
        try:
            pipe.load_lora_weights("lora_out", weight_name="lora_out_cpu.bin")
        except Exception as e:
            st.warning(f"Could not load LoRA weights: {e}")
    return pipe

pipe = load_pipe()
st.title("🎨 Text-to-Image CPU with LoRA")

prompt = st.text_area("Prompt", "a cozy coffee shop interior")
neg = st.text_input("Negative prompt", "blurry")
steps = st.slider("Steps", 10, 40, 20)
guidance = st.slider("CFG Scale", 1.0, 15.0, 7.5)
width = st.selectbox("Width", [256, 384, 512], index=0)
height = st.selectbox("Height", [256, 384, 512], index=0)

if st.button("Generate"):
    with torch.autocast("cpu", enabled=False):
        img = pipe(
            prompt,
            negative_prompt=neg,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height)
        ).images[0]
    st.image(img, caption="Generated Image", use_column_width=True)
