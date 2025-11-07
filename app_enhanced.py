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
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LCMScheduler
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import logging
import numpy as np
import cv2
import random

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", "")
logging.basicConfig(filename="app_errors.log", level=logging.ERROR,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Page config
st.set_page_config(
    page_title="AI Image Studio - Fast Edition",
    page_icon="🎨",
    layout="wide"
)

# Enhanced CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
}

.title-gradient {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3rem;
    text-align: center;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}

.feature-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    margin: 0.25rem;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin: 0.5rem 0;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
}

.metric-label {
    font-size: 0.9rem;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# Fast Model Loading - LCM ONLY
@st.cache_resource(show_spinner=False)
def load_models():
    """Load LCM model - 10x faster than SD v1.5"""
    models = {}
    status_messages = []
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        status_messages.append(("info", f"Loading LCM model on {device.upper()}..."))
        
        # Load LCM - super fast model
        base_pipe = StableDiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True
        ).to(device)
        
        # LCM scheduler - only needs 4-8 steps!
        base_pipe.scheduler = LCMScheduler.from_config(base_pipe.scheduler.config)
        
        # Memory optimizations
        base_pipe.enable_vae_tiling()
        base_pipe.enable_attention_slicing(slice_size=1)
        
        if device == "cuda":
            base_pipe.enable_vae_slicing()
        
        models['txt2img'] = base_pipe
        models['img2img'] = StableDiffusionImg2ImgPipeline(**base_pipe.components)
        models['device'] = device
        
        status_messages.append(("success", f"✅ LCM loaded on {device.upper()} (10x faster - only needs 4-8 steps!)"))
        
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        status_messages.append(("error", f"Failed to load model: {e}"))
        return None, status_messages
    
    return models, status_messages

# Image Processing Functions
def enhance_image(image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0):
    """Fast image enhancement"""
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation)
    
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
    
    return image

def apply_style_filter(image, style="none"):
    """Apply artistic style filters"""
    if style == "none":
        return image
    
    img_array = np.array(image)
    
    if style == "oil_painting":
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
    elif style == "watercolor":
        img_array = cv2.stylization(img_array, sigma_s=60, sigma_r=0.6)
    elif style == "pencil_sketch":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        img_array = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    elif style == "cartoon":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img_array, 9, 300, 300)
        img_array = cv2.bitwise_and(color, color, mask=edges)
    
    return Image.fromarray(img_array)

def get_image_download_link(img, filename="image.png"):
    """Create download link for image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/png;base64,{img_str}" download="{filename}">📥 Download</a>'

# Random art generation
ART_STYLES = [
    "abstract expressionism", "surrealism", "impressionism", "cubism", "pop art",
    "minimalism", "art nouveau", "digital art", "concept art", "fantasy art",
    "sci-fi art", "cyberpunk art", "watercolor painting", "oil painting"
]

ART_SUBJECTS = [
    "cosmic landscape", "mystical forest", "futuristic cityscape", "abstract composition",
    "ethereal portrait", "dreamlike scenery", "geometric patterns", "organic forms",
    "celestial bodies", "underwater world", "mountain vista", "desert landscape",
    "tropical paradise", "winter wonderland", "magical realm", "alien planet"
]

ART_MOODS = [
    "vibrant and energetic", "calm and serene", "dark and mysterious",
    "bright and cheerful", "dramatic and intense", "peaceful and tranquil",
    "bold and striking", "elegant and refined"
]

def generate_random_art_prompt():
    """Generate random art prompt"""
    style = random.choice(ART_STYLES)
    subject = random.choice(ART_SUBJECTS)
    mood = random.choice(ART_MOODS)
    quality_words = ["masterpiece", "highly detailed", "professional", "stunning", "intricate"]
    quality = ", ".join(random.sample(quality_words, 3))
    return f"{subject} in {style} style, {mood}, {quality}, 8k"

# 3D Model generation
MODEL_TYPES = [
    "character", "creature", "robot", "vehicle", "weapon", "prop",
    "building", "furniture", "plant", "crystal", "artifact", "tool",
    "spaceship", "mech", "dragon", "monster", "alien", "fantasy item"
]

MODEL_STYLES = [
    "low poly", "high poly", "stylized", "realistic", "cartoon",
    "sci-fi", "fantasy", "steampunk", "cyberpunk", "medieval",
    "futuristic", "organic", "mechanical", "geometric"
]

MODEL_MATERIALS = [
    "metallic", "wooden", "stone", "glass", "crystal", "plastic",
    "leather", "glowing", "rusty", "polished"
]

MODEL_DETAILS = [
    "highly detailed", "intricate design", "clean topology", "game ready",
    "cinematic quality", "professional", "textured", "PBR materials"
]

CAMERA_ANGLES = ["front view", "side view", "three-quarter view", "back view"]

def generate_random_3d_prompt(angle):
    """Generate random 3D model prompt for specific angle"""
    model_type = random.choice(MODEL_TYPES)
    style = random.choice(MODEL_STYLES)
    material = random.choice(MODEL_MATERIALS)
    details = random.sample(MODEL_DETAILS, 2)
    
    prompt = f"{style} {material} {model_type}, {angle}, {', '.join(details)}, 3D render, white background, product shot, studio lighting, 8k"
    return prompt, model_type

# Main UI
st.markdown('<h1 class="title-gradient">🎨 AI Image Studio - Fast Edition</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: white; margin-bottom: 2rem;">⚡ Powered by LCM - 10x Faster Generation!</p>', unsafe_allow_html=True)

# Feature badges
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <span class="feature-badge">⚡ LCM Model (Super Fast)</span>
    <span class="feature-badge">🎨 Text-to-Image</span>
    <span class="feature-badge">🖼️ Image-to-Image</span>
    <span class="feature-badge">🎭 Artistic Filters</span>
    <span class="feature-badge">🎲 Random Art</span>
    <span class="feature-badge">🎲 Random 3D Models</span>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "image_history" not in st.session_state:
    st.session_state.image_history = []
if "generation_stats" not in st.session_state:
    st.session_state.generation_stats = {"total": 0, "total_time": 0}

# Load model
with st.spinner("🚀 Loading LCM model (fast!)..."):
    models, status_messages = load_models()
    
    if models:
        for msg_type, msg in status_messages:
            if msg_type == "success":
                st.sidebar.success(msg)
            elif msg_type == "error":
                st.sidebar.error(msg)
    else:
        st.error("Failed to load model. Please check your setup.")
        st.stop()

# Sidebar controls
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Generation Settings")

# Mode selection
mode = st.sidebar.selectbox(
    "Generation Mode",
    ["Text-to-Image", "Image-to-Image", "🎲 Random Art Generator", "🎲 Random 3D Model (4 Angles)"],
    help="Select the type of generation"
)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Prompt Configuration")
    
    # Special handling for Random Art mode
    if mode == "🎲 Random Art Generator":
        st.info("🎲 **Random Art Mode**: Click generate to create a unique artwork!")
        
        if st.button("🎲 Generate New Random Prompt", use_container_width=True):
            st.session_state.random_prompt = generate_random_art_prompt()
        
        if "random_prompt" not in st.session_state:
            st.session_state.random_prompt = generate_random_art_prompt()
        
        prompt = st.session_state.random_prompt
        st.text_area("Generated Prompt", prompt, height=100, disabled=True)
    
    # Special handling for Random 3D Model mode
    elif mode == "🎲 Random 3D Model (4 Angles)":
        st.info("🎲 **Random 3D Model Mode**: Fully automatic! Just click generate!")
        st.warning("⚠️ Generates 4 angles automatically (front, side, 3/4, back). Takes 20-60 seconds.")
        
        st.markdown("**What you'll get:**")
        st.markdown("- 🎲 Completely random 3D model")
        st.markdown("- 📐 4 different camera angles")
        st.markdown("- 🎨 Random type, style, and materials")
        st.markdown("- ⚡ No prompts needed!")
        
        prompt = ""  # Will be generated per angle
    
    else:
        prompt = st.text_area(
            "Your Prompt",
            "a serene mountain landscape at sunset, golden hour lighting, highly detailed",
            height=100,
            help="Describe what you want to generate"
        )
    
    # Negative prompt
    negative_prompt = st.text_area(
        "Negative Prompt (Optional)",
        "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark",
        height=60,
        help="Things to avoid in the generation"
    )
    
    # File upload for img2img
    input_image = None
    if mode == "Image-to-Image":
        uploaded_file = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Reference Image", use_container_width=True)
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            if mode == "🎲 Random Art Generator":
                num_images = 1
                st.info("Random Art: 1 image per generation")
            elif mode == "🎲 Random 3D Model (4 Angles)":
                num_images = 4
                st.info("3D Model: 4 angles (front, side, 3/4, back)")
            else:
                num_images = st.slider("Number of Images", 1, 4, 1)
            
            width = st.select_slider("Width", [256, 384, 512, 640, 768], value=512)
            height = st.select_slider("Height", [256, 384, 512, 640, 768], value=512)
        
        with col_b:
            steps = st.slider("Steps (LCM: 4-8 optimal)", 4, 12, 6)
            guidance_scale = st.slider("Guidance (LCM: 1.0-1.5)", 1.0, 2.0, 1.0, 0.1)
            
            if mode == "Image-to-Image":
                strength = st.slider("Transformation Strength", 0.3, 1.0, 0.75, 0.05)
            
            if mode in ["🎲 Random Art Generator", "🎲 Random 3D Model (4 Angles)"]:
                seed = -1
                st.info("Random seed for unique generation")
            else:
                seed = st.number_input("Seed (-1 for random)", -1, 999999, -1)
    
    # Post-processing options
    with st.expander("✨ Post-Processing"):
        col_c, col_d = st.columns(2)
        
        with col_c:
            brightness = st.slider("Brightness", 0.7, 1.3, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.7, 1.3, 1.0, 0.1)
        
        with col_d:
            saturation = st.slider("Saturation", 0.7, 1.3, 1.0, 0.1)
            sharpness = st.slider("Sharpness", 0.7, 1.5, 1.0, 0.1)
    
    # Artistic filters
    with st.expander("🎨 Artistic Filters"):
        artistic_style = st.selectbox(
            "Apply Artistic Style",
            ["none", "oil_painting", "watercolor", "pencil_sketch", "cartoon"],
            help="Apply artistic filters to the generated image"
        )
    
    # Generate button
    if mode == "🎲 Random Art Generator":
        generate_btn = st.button("🎲 Generate Random Art", use_container_width=True)
    elif mode == "🎲 Random 3D Model (4 Angles)":
        generate_btn = st.button("� Geneerate Random 3D Model", use_container_width=True)
    else:
        generate_btn = st.button("🎨 Generate Images", use_container_width=True)

with col2:
    st.markdown("### 🖼️ Generated Images")
    
    if generate_btn:
        if not prompt and mode not in ["🎲 Random Art Generator", "🎲 Random 3D Model (4 Angles)"]:
            st.error("Please enter a prompt")
        else:
            # For Random Art mode, generate new prompt
            if mode == "🎲 Random Art Generator":
                prompt = generate_random_art_prompt()
                st.session_state.random_prompt = prompt
            
            # For Random 3D Model mode, generate prompts for each angle
            if mode == "🎲 Random 3D Model (4 Angles)":
                # Generate seed once for consistency across angles
                model_seed = random.randint(0, 999999)
                angle_prompts = []
                model_type = ""
                for angle in CAMERA_ANGLES:
                    angle_prompt, model_type = generate_random_3d_prompt(angle)
                    angle_prompts.append((angle, angle_prompt))
                st.session_state.model_type_preview = model_type
            
            start_time = time.time()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            generated_images = []
            
            try:
                pipe = models['txt2img'] if mode != "Image-to-Image" else models['img2img']
                device = models['device']
                
                with torch.inference_mode():
                    for i in range(num_images):
                        # For 3D Model mode, use angle-specific prompts
                        if mode == "🎲 Random 3D Model (4 Angles)":
                            angle, current_prompt = angle_prompts[i]
                            status_text.text(f"Generating {angle}... ({i+1}/{num_images})")
                            generator = torch.Generator(device=device).manual_seed(model_seed)
                        else:
                            current_prompt = prompt
                            status_text.text(f"Generating {i+1}/{num_images}... (LCM is fast!)")
                            # Set seed
                            if seed >= 0:
                                generator = torch.Generator(device=device).manual_seed(seed + i)
                            else:
                                generator = None
                        
                        progress_bar.progress((i + 1) / num_images)
                        
                        if mode != "Image-to-Image":
                            image = pipe(
                                prompt=current_prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                                width=width,
                                height=height,
                                generator=generator
                            ).images[0]
                        else:
                            if input_image is None:
                                st.error("Please upload an image")
                                break
                            
                            # Resize input image
                            input_resized = input_image.resize((width, height), Image.Resampling.LANCZOS)
                            
                            image = pipe(
                                prompt=current_prompt,
                                image=input_resized,
                                negative_prompt=negative_prompt,
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                                strength=strength,
                                generator=generator
                            ).images[0]
                        
                        # Post-processing
                        image = enhance_image(image, brightness, contrast, saturation, sharpness)
                        
                        if artistic_style != "none":
                            image = apply_style_filter(image, artistic_style)
                        
                        generated_images.append(image)
                
                # Calculate stats
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Update session state
                st.session_state.image_history.extend([(img, prompt) for img in generated_images])
                st.session_state.generation_stats["total"] += len(generated_images)
                st.session_state.generation_stats["total_time"] += generation_time
                
                # Display results
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Generated {len(generated_images)} image(s) in {generation_time:.2f}s ({generation_time/len(generated_images):.2f}s each)")
                
                # Display images
                if len(generated_images) == 1:
                    st.image(generated_images[0], use_container_width=True)
                    st.markdown(get_image_download_link(generated_images[0], "generated_0.png"), unsafe_allow_html=True)
                elif mode == "🎲 Random 3D Model (4 Angles)":
                    # Display 3D model angles with labels
                    st.markdown(f"### 🎲 Random {model_type.title()} - 4 Angles")
                    cols = st.columns(2)
                    for idx, (img, (angle, _)) in enumerate(zip(generated_images, angle_prompts)):
                        with cols[idx % 2]:
                            st.markdown(f"**{angle.title()}**")
                            st.image(img, use_container_width=True)
                            st.markdown(get_image_download_link(img, f"3dmodel_{angle.replace(' ', '_')}.png"), unsafe_allow_html=True)
                else:
                    cols = st.columns(2)
                    for idx, img in enumerate(generated_images):
                        cols[idx % 2].image(img, use_container_width=True)
                        cols[idx % 2].markdown(get_image_download_link(img, f"generated_{idx}.png"), unsafe_allow_html=True)
                
                # Show generation info
                with st.expander("📊 Generation Details"):
                    st.write(f"**Model:** LCM Dreamshaper v7 (Fast!)")
                    st.write(f"**Device:** {device.upper()}")
                    if mode == "🎲 Random 3D Model (4 Angles)":
                        st.write(f"**Model Type:** {model_type.title()}")
                        st.write(f"**Angles Generated:** {len(CAMERA_ANGLES)}")
                        st.write("**Prompts:**")
                        for angle, angle_prompt in angle_prompts:
                            st.write(f"  - {angle.title()}: {angle_prompt}")
                    else:
                        st.write(f"**Prompt:** {prompt}")
                    st.write(f"**Negative:** {negative_prompt}")
                    st.write(f"**Settings:** {width}x{height}, {steps} steps, CFG {guidance_scale}")
                    st.write(f"**Time:** {generation_time:.2f}s ({generation_time/len(generated_images):.2f}s per image)")
                    if seed >= 0:
                        st.write(f"**Seeds:** {seed} to {seed + len(generated_images) - 1}")
                
            except Exception as e:
                logging.error(f"Generation failed: {e}")
                st.error(f"Generation failed: {e}")
                progress_bar.empty()
                status_text.empty()
    
    # Show placeholder if no generation yet
    if not st.session_state.image_history:
        st.info("👈 Configure your settings and click 'Generate' to start creating!")

# Statistics Dashboard
st.markdown("---")
st.markdown("### 📊 Generation Statistics")

col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{st.session_state.generation_stats['total']}</div>
        <div class="metric-label">Total Images</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat2:
    avg_time = (st.session_state.generation_stats['total_time'] / 
                st.session_state.generation_stats['total']) if st.session_state.generation_stats['total'] > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{avg_time:.1f}s</div>
        <div class="metric-label">Avg Time/Image</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(st.session_state.image_history)}</div>
        <div class="metric-label">In History</div>
    </div>
    """, unsafe_allow_html=True)

with col_stat4:
    device_name = models['device'].upper() if models else "N/A"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{device_name}</div>
        <div class="metric-label">Device</div>
    </div>
    """, unsafe_allow_html=True)

# Image History Gallery
if st.session_state.image_history:
    st.markdown("---")
    st.markdown("### 🖼️ Image History Gallery")
    
    # Gallery controls
    col_g1, col_g2 = st.columns([3, 1])
    with col_g1:
        show_count = st.selectbox("Show Recent", [8, 16, 24, "All"], index=0)
    with col_g2:
        if st.button("🗑️ Clear History"):
            st.session_state.image_history = []
            st.rerun()
    
    # Display gallery
    if show_count == "All":
        display_images = st.session_state.image_history
    else:
        display_images = st.session_state.image_history[-show_count:]
    
    cols = st.columns(4)
    for idx, (img, prompt) in enumerate(reversed(display_images)):
        with cols[idx % 4]:
            st.image(img, use_container_width=True)
            st.caption(prompt[:40] + "..." if len(prompt) > 40 else prompt)
            st.markdown(get_image_download_link(img, f"history_{idx}.png"), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 2rem;">
    <p><strong>AI Image Studio - Fast Edition</strong></p>
    <p>Powered by LCM (Latent Consistency Model) - 10x Faster than SD v1.5!</p>
    <p style="font-size: 0.9rem;">⚡ Only 4-8 steps needed • 🚀 GPU optimized • 💻 CPU friendly</p>
</div>
""", unsafe_allow_html=True)
