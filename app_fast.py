import os
import sys

# Environment setup
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load .env
from pathlib import Path
env_path = Path(".env")
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

import time
import torch
import streamlit as st
from diffusers import DiffusionPipeline
from PIL import Image, ImageEnhance
import io
import base64
import random

HF_TOKEN = os.environ.get("HF_TOKEN", "")

st.set_page_config(page_title="AI Image Generator - Fast", page_icon="🎨", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
.title {font-size: 3rem; font-weight: 800; text-align: center; color: white; margin: 2rem 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    """Load TINY fast model - only 1.6GB, downloads in 2-3 minutes"""
    try:
        # Use Segmind Tiny SD - smallest and fastest!
        pipe = DiffusionPipeline.from_pretrained(
            "segmind/tiny-sd",
            torch_dtype=torch.float32,
            use_safetensors=True,
            safety_checker=None
        ).to("cpu")
        
        pipe.enable_attention_slicing(slice_size=1)
        return pipe
    except Exception as e:
        st.error(f"Error: {e}")
        return None

st.markdown('<h1 class="title">🎨 Fast AI Image Generator</h1>', unsafe_allow_html=True)
st.info("⚡ Using Tiny-SD model - Small download (1.6GB), Fast generation (15-30 sec)")

with st.spinner("📥 Loading model (first time: 2-3 min download, then instant)..."):
    model = load_model()

if model is None:
    st.error("Failed to load model")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Settings")
    
    mode = st.selectbox("Mode", ["Text-to-Image", "🎨 Random Art"])
    
    if mode == "🎨 Random Art":
        styles = ["abstract", "landscape", "portrait", "fantasy", "scifi", "anime"]
        subjects = ["sunset", "forest", "city", "ocean", "mountains", "space"]
        
        if st.button("🎲 Generate Random", use_container_width=True):
            style = random.choice(styles)
            subject = random.choice(subjects)
            prompt = f"beautiful {subject}, {style} style, detailed, high quality"
            st.session_state.prompt = prompt
        
        prompt = st.text_area("Prompt", st.session_state.get("prompt", "beautiful sunset, detailed"), height=100)
    else:
        prompt = st.text_area("Prompt", "beautiful sunset over mountains, detailed, high quality", height=100)
    
    negative = st.text_input("Negative", "blurry, bad quality, ugly")
    
    with st.expander("⚙️ Settings"):
        steps = st.slider("Steps", 10, 30, 15)
        guidance = st.slider("Guidance", 5.0, 10.0, 7.5, 0.5)
        seed = st.number_input("Seed (-1 = random)", -1, 999999, -1)
    
    generate = st.button("🎨 Generate", use_container_width=True)

with col2:
    st.markdown("### 🖼️ Result")
    
    if generate and prompt:
        start = time.time()
        
        with st.spinner(f"Generating... (~{steps * 2} seconds)"):
            try:
                if seed >= 0:
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                else:
                    generator = None
                
                with torch.inference_mode():
                    image = model(
                        prompt=prompt,
                        negative_prompt=negative,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        width=512,
                        height=512,
                        generator=generator
                    ).images[0]
                
                # Enhance
                image = ImageEnhance.Sharpness(image).enhance(1.2)
                
                elapsed = time.time() - start
                st.success(f"✅ Done in {elapsed:.1f}s!")
                
                st.image(image, use_container_width=True)
                
                # Download
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                st.markdown(f'<a href="data:image/png;base64,{img_str}" download="image.png" style="color: #667eea; font-weight: 600;">📥 Download</a>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("👈 Enter a prompt and click Generate")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white;">
<p><strong>Fast AI Image Generator</strong> - Optimized for CPU</p>
<p>Model: Segmind Tiny-SD (1.6GB) • Speed: 15-30 seconds • Quality: Good</p>
</div>
""", unsafe_allow_html=True)
