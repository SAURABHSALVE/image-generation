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
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageEnhance
import io
import base64
import random
import logging

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", "")
logging.basicConfig(filename="app_errors.log", level=logging.ERROR,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Page config
st.set_page_config(
    page_title="This Art Does Not Exist",
    page_icon="🎨",
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
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
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

.image-container {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    margin: 2rem auto;
    max-width: 800px;
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

.stats-container {
    display: flex;
    justify-content: space-around;
    margin: 2rem 0;
    flex-wrap: wrap;
}

.stat-box {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 1.5rem 2rem;
    color: white;
    text-align: center;
    min-width: 150px;
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

.footer {
    text-align: center;
    color: rgba(255,255,255,0.7);
    margin: 3rem 0 2rem 0;
    font-size: 0.9rem;
}

.loading-text {
    color: white;
    text-align: center;
    font-size: 1.2rem;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# Art prompt templates
ART_STYLES = [
    "abstract expressionism",
    "surrealism",
    "impressionism",
    "cubism",
    "pop art",
    "minimalism",
    "art nouveau",
    "baroque",
    "renaissance",
    "contemporary art",
    "digital art",
    "concept art",
    "fantasy art",
    "sci-fi art",
    "cyberpunk art",
    "steampunk art",
    "watercolor painting",
    "oil painting",
    "acrylic painting",
    "mixed media art"
]

ART_SUBJECTS = [
    "cosmic landscape",
    "mystical forest",
    "futuristic cityscape",
    "abstract composition",
    "ethereal portrait",
    "dreamlike scenery",
    "geometric patterns",
    "organic forms",
    "celestial bodies",
    "underwater world",
    "mountain vista",
    "desert landscape",
    "tropical paradise",
    "winter wonderland",
    "autumn forest",
    "spring meadow",
    "urban architecture",
    "ancient ruins",
    "magical realm",
    "alien planet"
]

ART_MOODS = [
    "vibrant and energetic",
    "calm and serene",
    "dark and mysterious",
    "bright and cheerful",
    "dramatic and intense",
    "peaceful and tranquil",
    "chaotic and dynamic",
    "elegant and refined",
    "bold and striking",
    "subtle and delicate"
]

QUALITY_KEYWORDS = [
    "masterpiece",
    "highly detailed",
    "professional",
    "award winning",
    "stunning",
    "breathtaking",
    "intricate",
    "beautiful",
    "artistic",
    "creative"
]

def generate_random_prompt():
    """Generate a random art prompt"""
    style = random.choice(ART_STYLES)
    subject = random.choice(ART_SUBJECTS)
    mood = random.choice(ART_MOODS)
    quality = ", ".join(random.sample(QUALITY_KEYWORDS, 3))
    
    prompt = f"{subject} in {style} style, {mood}, {quality}, 8k, high resolution"
    return prompt

def generate_negative_prompt():
    """Generate negative prompt for better quality"""
    return "ugly, blurry, low quality, distorted, deformed, bad anatomy, watermark, signature, text, worst quality, low resolution, grainy, amateur, duplicate"

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the AI model optimized for CPU - Using LCM for 10x speed!"""
    try:
        from diffusers import LCMScheduler
        
        # Use LCM model - 10x faster, only needs 4-8 steps!
        pipe = StableDiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=torch.float32,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True
        ).to("cpu")
        
        # LCM scheduler - super fast
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        
        # CPU optimizations
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing(slice_size=1)
        
        return pipe
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        return None

def enhance_image(image):
    """Enhance the generated image"""
    # Slight sharpness boost
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)
    
    # Slight contrast boost
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    
    return image

def get_image_download_link(img, filename="thisartdoesnotexist.png"):
    """Create download link"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/png;base64,{img_str}" download="{filename}" style="color: #667eea; text-decoration: none; font-weight: 600; font-size: 1.1rem;">📥 Download This Art</a>'

# Initialize session state
if "art_count" not in st.session_state:
    st.session_state.art_count = 0
if "total_time" not in st.session_state:
    st.session_state.total_time = 0
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = None

# Main UI
st.markdown('<h1 class="main-title">🎨 This Art Does Not Exist</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Generated Art • Every Piece is Unique • Powered by Stable Diffusion</p>', unsafe_allow_html=True)

# Load model
with st.spinner("🚀 Loading AI model..."):
    model = load_model()

if model is None:
    st.error("Failed to load AI model. Please check your setup.")
    st.stop()

# Main content
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # Generate button
    if st.button("🎨 Generate New Art", key="generate"):
        prompt = generate_random_prompt()
        negative_prompt = generate_negative_prompt()
        
        st.session_state.current_prompt = prompt
        
        start_time = time.time()
        
        with st.spinner("🎨 Creating unique art... Using LCM for super fast generation (10-30 seconds)..."):
            try:
                # Generate with random seed
                seed = random.randint(0, 999999)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                
                with torch.inference_mode():
                    image = model(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=6,  # LCM only needs 4-8 steps!
                        guidance_scale=1.0,  # LCM uses lower guidance
                        width=512,
                        height=512,
                        generator=generator
                    ).images[0]
                
                # Enhance image
                image = enhance_image(image)
                
                # Update session state
                st.session_state.current_image = image
                st.session_state.art_count += 1
                
                generation_time = time.time() - start_time
                st.session_state.total_time += generation_time
                
                st.success(f"✅ Art created in {generation_time:.1f} seconds!")
                
            except Exception as e:
                logging.error(f"Generation failed: {e}")
                st.error(f"Generation failed: {e}")
    
    # Display current image
    if st.session_state.current_image:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(st.session_state.current_image, use_container_width=True)
        
        # Prompt display
        if st.session_state.current_prompt:
            st.markdown(f'<div class="prompt-display">"{st.session_state.current_prompt}"</div>', unsafe_allow_html=True)
        
        # Download link
        st.markdown(get_image_download_link(st.session_state.current_image), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Placeholder
        st.markdown("""
        <div class="info-box">
            <h2 style="text-align: center; margin-bottom: 1rem;">Welcome! 👋</h2>
            <p style="text-align: center; font-size: 1.1rem;">
                Click the button above to generate a completely unique piece of art.<br/>
                Every artwork is created by AI and has never existed before.<br/>
                <br/>
                <strong>⚡ SUPER FAST:</strong> Using LCM model (10-30 seconds per artwork!)<br/>
                <strong>🎨 Infinite Variety:</strong> Random styles, subjects, and moods<br/>
                <strong>📥 Download:</strong> Save any artwork you like
            </p>
        </div>
        """, unsafe_allow_html=True)

# Statistics
if st.session_state.art_count > 0:
    avg_time = st.session_state.total_time / st.session_state.art_count
    
    st.markdown("""
    <div class="stats-container">
        <div class="stat-box">
            <span class="stat-value">{}</span>
            <span class="stat-label">Artworks Created</span>
        </div>
        <div class="stat-box">
            <span class="stat-value">{:.1f}s</span>
            <span class="stat-label">Avg Generation Time</span>
        </div>
        <div class="stat-box">
            <span class="stat-value">{:.1f}m</span>
            <span class="stat-label">Total Time</span>
        </div>
    </div>
    """.format(
        st.session_state.art_count,
        avg_time,
        st.session_state.total_time / 60
    ), unsafe_allow_html=True)

# Info section
st.markdown("""
<div class="info-box">
    <h3 style="margin-bottom: 1rem;">ℹ️ About This Project</h3>
    <p style="line-height: 1.8;">
        <strong>This Art Does Not Exist</strong> is inspired by thispersondoesnotexist.com, 
        but instead of faces, it generates unique artworks using AI.<br/><br/>
        
        <strong>How it works:</strong><br/>
        • Randomly combines art styles, subjects, and moods<br/>
        • Uses LCM (Latent Consistency Model) for super fast generation<br/>
        • Optimized for CPU with FAST generation (10-30 seconds!)<br/>
        • Every artwork is completely unique and never existed before<br/><br/>
        
        <strong>Technology:</strong><br/>
        • AI Model: LCM Dreamshaper v7 (10x faster than SD v1.5!)<br/>
        • Framework: PyTorch + Diffusers<br/>
        • Interface: Streamlit<br/>
        • Optimization: CPU-optimized with attention slicing + LCM scheduler
    </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>
        🎨 <strong>This Art Does Not Exist</strong> - AI-Generated Art Project<br/>
        Powered by Stable Diffusion • Built with Python & Streamlit<br/>
        Part of Advanced AI Image Studio - College Project
    </p>
</div>
""", unsafe_allow_html=True)
