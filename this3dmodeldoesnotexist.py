import os
import sys

# Environment setup - MUST be first
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XFORMERS_DISABLE"] = "1"
os.environ["DIFFUSERS_NO_ONNX"] = "1"

# Load .env file
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
from diffusers import StableDiffusionPipeline, LCMScheduler
from PIL import Image, ImageEnhance
import io
import base64
import random
import logging

# Configuration
logging.basicConfig(filename="app_errors.log", level=logging.ERROR,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Page config
st.set_page_config(
    page_title="This 3D Model Does Not Exist",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

* { 
    font-family: 'Inter', sans-serif; 
    margin: 0;
    padding: 0;
}

.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    background-attachment: fixed;
}

.main-title {
    font-size: 4rem;
    font-weight: 800;
    text-align: center;
    color: white;
    margin: 2rem 0 1rem 0;
    text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    letter-spacing: -0.02em;
}

.subtitle {
    font-size: 1.3rem;
    text-align: center;
    color: rgba(255,255,255,0.9);
    margin-bottom: 3rem;
    font-weight: 300;
}

.model-container {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    margin: 2rem auto;
    max-width: 1200px;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 1rem 3rem;
    font-weight: 700;
    font-size: 1.3rem;
    transition: all 0.3s ease;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    width: 100%;
    margin: 2rem 0;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(102, 126, 234, 0.7);
}

.info-box {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 1.5rem;
    color: white;
    margin: 2rem 0;
    border: 1px solid rgba(255,255,255,0.2);
}

.stat-box {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 1.5rem 2rem;
    color: white;
    text-align: center;
    margin: 0.5rem;
    border: 1px solid rgba(255,255,255,0.2);
}

.stat-value {
    font-size: 2.5rem;
    font-weight: 800;
    display: block;
}

.stat-label {
    font-size: 0.9rem;
    opacity: 0.9;
    margin-top: 0.5rem;
}

.prompt-display {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    border-radius: 10px;
    padding: 1rem;
    color: white;
    margin: 1rem 0;
    font-style: italic;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
}

.angle-label {
    background: rgba(102, 126, 234, 0.8);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 10px;
    font-weight: 600;
    text-align: center;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# 3D Model Categories
MODEL_TYPES = [
    "character", "creature", "robot", "vehicle", "weapon", "prop",
    "building", "furniture", "plant", "crystal", "artifact", "tool",
    "spaceship", "mech", "dragon", "monster", "alien", "fantasy item"
]

MODEL_STYLES = [
    "low poly", "high poly", "stylized", "realistic", "cartoon",
    "sci-fi", "fantasy", "steampunk", "cyberpunk", "medieval",
    "futuristic", "organic", "mechanical", "geometric", "abstract"
]

MODEL_MATERIALS = [
    "metallic", "wooden", "stone", "glass", "crystal", "plastic",
    "leather", "fabric", "glowing", "rusty", "polished", "matte"
]

MODEL_DETAILS = [
    "highly detailed", "intricate design", "clean topology", "game ready",
    "cinematic quality", "professional", "optimized", "textured",
    "with normal maps", "PBR materials", "studio lighting"
]

CAMERA_ANGLES = [
    "front view",
    "side view", 
    "three-quarter view",
    "back view"
]

def generate_random_3d_prompt(angle):
    """Generate random 3D model prompt for specific angle"""
    model_type = random.choice(MODEL_TYPES)
    style = random.choice(MODEL_STYLES)
    material = random.choice(MODEL_MATERIALS)
    details = random.sample(MODEL_DETAILS, 2)
    
    prompt = f"{style} {material} {model_type}, {angle}, {', '.join(details)}, 3D render, white background, product shot, studio lighting, 8k"
    return prompt, model_type

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the AI model optimized for CPU/GPU - Using LCM for speed!"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True
        ).to(device)
        
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        
        # Optimizations
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing(slice_size=1)
        
        if device == "cuda":
            pipe.enable_vae_slicing()
        
        return pipe, device
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        return None, None

def enhance_image(image):
    """Enhance the generated image"""
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    
    return image

def get_image_download_link(img, filename="3dmodel.png"):
    """Create download link"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/png;base64,{img_str}" download="{filename}" style="color: #667eea; text-decoration: none; font-weight: 600; font-size: 1.1rem;">📥 Download</a>'

# Initialize session state
if "model_count" not in st.session_state:
    st.session_state.model_count = 0
if "total_time" not in st.session_state:
    st.session_state.total_time = 0
if "current_images" not in st.session_state:
    st.session_state.current_images = []
if "current_prompts" not in st.session_state:
    st.session_state.current_prompts = []
if "current_model_type" not in st.session_state:
    st.session_state.current_model_type = ""

# Main UI
st.markdown('<h1 class="main-title">🎲 This 3D Model Does Not Exist</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Generated 3D Models • Fully Random • No Prompts Needed • Just Click!</p>', unsafe_allow_html=True)

# Load model
with st.spinner("🚀 Loading AI model..."):
    model, device = load_model()

if model is None:
    st.error("Failed to load AI model. Please check your setup.")
    st.stop()

# Show device info
device_emoji = "🚀" if device == "cuda" else "💻"
st.info(f"{device_emoji} Running on: **{device.upper()}** | Model: **LCM (Super Fast)** | Every click = New random 3D model!")

# Main content
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # Auto-generate on page load or button click
    if st.button("🎲 Generate Another Random 3D Model", key="generate") or len(st.session_state.current_images) == 0:
        start_time = time.time()
        
        with st.spinner("🎲 Creating unique 3D model from 4 angles... (20-60 seconds)..."):
            try:
                generated_images = []
                generated_prompts = []
                
                # Generate 4 different angles of the same model
                for angle in CAMERA_ANGLES:
                    prompt, model_type = generate_random_3d_prompt(angle)
                    generated_prompts.append((angle, prompt))
                    
                    # Use same seed for all angles to maintain consistency
                    if angle == CAMERA_ANGLES[0]:
                        seed = random.randint(0, 999999)
                    
                    generator = torch.Generator(device=device).manual_seed(seed)
                    
                    with torch.inference_mode():
                        image = model(
                            prompt=prompt,
                            negative_prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy, multiple objects, duplicate",
                            num_inference_steps=6,
                            guidance_scale=1.0,
                            width=512,
                            height=512,
                            generator=generator
                        ).images[0]
                    
                    # Enhance image
                    image = enhance_image(image)
                    generated_images.append(image)
                
                # Update session state
                st.session_state.current_images = generated_images
                st.session_state.current_prompts = generated_prompts
                st.session_state.current_model_type = model_type
                st.session_state.model_count += 1
                
                generation_time = time.time() - start_time
                st.session_state.total_time += generation_time
                
                st.success(f"✅ 3D Model created in {generation_time:.1f} seconds! ({generation_time/4:.1f}s per angle)")
                
            except Exception as e:
                logging.error(f"Generation failed: {e}")
                st.error(f"Generation failed: {e}")
    
    # Display current images
    if st.session_state.current_images:
        st.markdown('<div class="model-container">', unsafe_allow_html=True)
        
        st.markdown(f"<h2 style='text-align: center; color: #667eea; margin-bottom: 2rem;'>🎲 Random {st.session_state.current_model_type.title()}</h2>", unsafe_allow_html=True)
        
        # Display 4 angles in 2x2 grid
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        cols = [row1_col1, row1_col2, row2_col1, row2_col2]
        
        for idx, (image, (angle, prompt)) in enumerate(zip(st.session_state.current_images, st.session_state.current_prompts)):
            with cols[idx]:
                st.markdown(f'<div class="angle-label">{angle.upper()}</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.markdown(get_download_link(image, f"3dmodel_{angle.replace(' ', '_')}.png"), unsafe_allow_html=True)
        
        # Optional: Show prompts (hidden by default)
        if st.checkbox("� Shoiw Technical Details", value=False):
            st.markdown("**Prompts used for each angle:**")
            for angle, prompt in st.session_state.current_prompts:
                st.markdown(f"**{angle.title()}:**")
                st.text(prompt)
                st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Placeholder
        st.markdown("""
        <div class="info-box">
            <h2 style="text-align: center; margin-bottom: 1rem;">🎲 Generating Your First Model...</h2>
            <p style="text-align: center; font-size: 1.1rem;">
                Please wait while we create a completely unique 3D model for you.<br/>
                This will take 20-60 seconds...<br/>
                <br/>
                <strong>🎲 Fully Random:</strong> No prompts needed - just pure AI creativity!<br/>
                <strong>📐 4 Angles:</strong> Front, side, three-quarter, and back views<br/>
                <strong>⚡ Fast:</strong> Using LCM model<br/>
                <strong>📥 Download:</strong> Save any angle you like
            </p>
        </div>
        """, unsafe_allow_html=True)

# Statistics
if st.session_state.model_count > 0:
    avg_time = st.session_state.total_time / st.session_state.model_count
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown(f"""
        <div class="stat-box">
            <span class="stat-value">{st.session_state.model_count}</span>
            <span class="stat-label">Models Created</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_s2:
        st.markdown(f"""
        <div class="stat-box">
            <span class="stat-value">{avg_time:.1f}s</span>
            <span class="stat-label">Avg Generation Time</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_s3:
        st.markdown(f"""
        <div class="stat-box">
            <span class="stat-value">{st.session_state.model_count * 4}</span>
            <span class="stat-label">Total Angles Generated</span>
        </div>
        """, unsafe_allow_html=True)

# Info section
st.markdown("""
<div class="info-box">
    <h3 style="margin-bottom: 1rem;">ℹ️ About This Project</h3>
    <p style="line-height: 1.8;">
        <strong>This 3D Model Does Not Exist</strong> generates unique 3D model concepts 
        from multiple angles using AI.<br/><br/>
        
        <strong>How it works:</strong><br/>
        • Fully automatic - no prompts or settings needed<br/>
        • Randomly combines model types, styles, and materials<br/>
        • Generates 4 different camera angles (front, side, three-quarter, back)<br/>
        • Uses same seed for consistency across angles<br/>
        • Uses LCM for super fast generation<br/>
        • Every model is completely unique<br/>
        • Just like "This Person Does Not Exist" but for 3D models!<br/><br/>
        
        <strong>Model Types:</strong><br/>
        Characters, Creatures, Robots, Vehicles, Weapons, Props, Buildings, 
        Furniture, Plants, Crystals, Artifacts, Spaceships, Mechs, Dragons, 
        Monsters, Aliens, Fantasy Items, and more!<br/><br/>
        
        <strong>Technology:</strong><br/>
        • AI Model: LCM Dreamshaper v7 (10x faster!)<br/>
        • Framework: PyTorch + Diffusers<br/>
        • Interface: Streamlit<br/>
        • Optimization: CPU/GPU optimized with LCM scheduler
    </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.7); margin: 3rem 0 2rem 0; font-size: 0.9rem;">
    <p>
        🎲 <strong>This 3D Model Does Not Exist</strong> - AI-Generated 3D Model Concepts<br/>
        Powered by Stable Diffusion LCM • Built with Python & Streamlit<br/>
        Part of AI Image Studio - College Project
    </p>
</div>
""", unsafe_allow_html=True)
