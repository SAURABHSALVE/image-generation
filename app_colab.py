import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XFORMERS_DISABLE"] = "1"
os.environ["DIFFUSERS_NO_ONNX"] = "1"

# Streamlit config for Colab
import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('server.headless', True)
st.set_option('server.enableCORS', False)
st.set_option('server.enableXsrfProtection', False)

import time
import torch
import random
from diffusers import StableDiffusionPipeline, LCMScheduler
from PIL import Image, ImageEnhance
import io
import base64

# Page config
st.set_page_config(
    page_title="AI Image Studio - Colab",
    page_icon="🎨",
    layout="wide"
)

# Minimal CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.main-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    color: white;
    margin: 2rem 0;
    text-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    """Load LCM model - fastest for Colab"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipe = pipe.to(device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        
        # Optimizations
        if device == "cpu":
            pipe.enable_attention_slicing(slice_size=1)
        else:
            # GPU optimizations
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        
        return pipe, device
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

def enhance_image(image):
    """Quick image enhancement"""
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    return image

def get_download_link(img, filename="generated.png"):
    """Create download link"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/png;base64,{img_str}" download="{filename}">📥 Download</a>'

# Initialize session state
if "images" not in st.session_state:
    st.session_state.images = []
if "count" not in st.session_state:
    st.session_state.count = 0

# Main UI
st.markdown('<h1 class="main-title">🎨 AI Image Studio</h1>', unsafe_allow_html=True)

# Load model
with st.spinner("🚀 Loading AI model..."):
    model, device = load_model()

if model is None:
    st.error("Failed to load model")
    st.stop()

# Show device info
device_emoji = "🚀" if device == "cuda" else "💻"
st.info(f"{device_emoji} Running on: **{device.upper()}** | Model: **LCM (Super Fast)**")

# Input section
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area(
        "Your Prompt",
        "a beautiful mountain landscape at sunset, highly detailed, 8k",
        height=100
    )

with col2:
    mode = st.selectbox("Mode", ["Text-to-Image", "Random Art"])
    num_images = st.slider("Images", 1, 4, 1)

# Advanced settings
with st.expander("⚙️ Settings"):
    col_a, col_b = st.columns(2)
    with col_a:
        width = st.select_slider("Width", [256, 384, 512, 640], value=512)
        height = st.select_slider("Height", [256, 384, 512, 640], value=512)
    with col_b:
        steps = st.slider("Steps (LCM: 4-8 optimal)", 4, 10, 6)
        guidance = st.slider("Guidance (LCM: 1.0-1.5)", 1.0, 2.0, 1.0, 0.1)

# Generate button
if st.button("🎨 Generate Images"):
    if mode == "Random Art":
        styles = ["abstract", "surreal", "fantasy", "sci-fi", "landscape"]
        subjects = ["cosmic", "mystical", "futuristic", "ethereal", "dreamlike"]
        prompt = f"{random.choice(subjects)} {random.choice(styles)} art, highly detailed, masterpiece"
    
    if not prompt:
        st.error("Please enter a prompt")
    else:
        start_time = time.time()
        progress_bar = st.progress(0)
        
        generated = []
        
        try:
            for i in range(num_images):
                progress_bar.progress((i + 1) / num_images)
                
                seed = random.randint(0, 999999)
                generator = torch.Generator(device=device).manual_seed(seed)
                
                with torch.inference_mode():
                    image = model(
                        prompt=prompt,
                        negative_prompt="blurry, low quality, distorted, ugly",
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        width=width,
                        height=height,
                        generator=generator
                    ).images[0]
                
                image = enhance_image(image)
                generated.append(image)
            
            # Update session
            st.session_state.images.extend(generated)
            st.session_state.count += len(generated)
            
            gen_time = time.time() - start_time
            
            progress_bar.empty()
            st.success(f"✅ Generated {len(generated)} image(s) in {gen_time:.1f}s ({gen_time/len(generated):.1f}s each)")
            
            # Display images
            if len(generated) == 1:
                st.image(generated[0], use_container_width=True)
                st.markdown(get_download_link(generated[0]), unsafe_allow_html=True)
            else:
                cols = st.columns(2)
                for idx, img in enumerate(generated):
                    cols[idx % 2].image(img, use_container_width=True)
                    cols[idx % 2].markdown(get_download_link(img, f"image_{idx}.png"), unsafe_allow_html=True)
            
            # Show prompt
            st.info(f"**Prompt:** {prompt}")
            
        except Exception as e:
            st.error(f"Generation failed: {e}")
            progress_bar.empty()

# Stats
if st.session_state.count > 0:
    st.markdown("---")
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Total Images", st.session_state.count)
    col_s2.metric("In History", len(st.session_state.images))
    col_s3.metric("Device", device.upper())

# History
if st.session_state.images:
    st.markdown("---")
    st.markdown("### 🖼️ Recent Images")
    
    if st.button("🗑️ Clear History"):
        st.session_state.images = []
        st.rerun()
    
    cols = st.columns(4)
    for idx, img in enumerate(reversed(st.session_state.images[-8:])):
        cols[idx % 4].image(img, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 1rem;">
    <p><strong>AI Image Studio</strong> - Optimized for Google Colab</p>
    <p>Powered by LCM + Stable Diffusion</p>
</div>
""", unsafe_allow_html=True)
