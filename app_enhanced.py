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
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler
)
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import logging
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from huggingface_hub import InferenceClient
import numpy as np
import cv2
import tempfile

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", "")
logging.basicConfig(filename="app_errors.log", level=logging.ERROR,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Page config
st.set_page_config(
    page_title="Advanced AI Image Studio - College Project",
    page_icon="🎨",
    layout="wide"
)

# Enhanced CSS with modern design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
}

.main-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
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

.image-card {
    background: white;
    border-radius: 16px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    transition: transform 0.3s ease;
}

.image-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.2);
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

# Advanced Model Loading with Multiple Options
@st.cache_resource(show_spinner=False)
def load_models(model_choice="sd-v1-5"):
    """Load AI models with advanced schedulers and optimizations"""
    models = {}
    status_messages = []
    
    common_args = {
        "torch_dtype": torch.float32,
        "use_safetensors": True,
        "safety_checker": None,
        "requires_safety_checker": False,
        "low_cpu_mem_usage": True
    }
    
    try:
        if model_choice == "lcm":
            status_messages.append(("info", "Loading LCM (Latent Consistency Model) - FASTEST..."))
            
            # LCM is 10x faster - only needs 4-8 steps!
            from diffusers import LCMScheduler
            
            base_pipe = StableDiffusionPipeline.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7",
                **common_args
            ).to("cpu")
            
            # LCM scheduler - super fast
            base_pipe.scheduler = LCMScheduler.from_config(base_pipe.scheduler.config)
            
            # Memory optimizations
            base_pipe.enable_vae_tiling()
            base_pipe.enable_attention_slicing(slice_size=1)
            
            models['txt2img'] = base_pipe
            models['img2img'] = StableDiffusionImg2ImgPipeline(**base_pipe.components)
            
            status_messages.append(("success", "✅ LCM loaded (10x faster - only needs 4-8 steps!)"))
            
        elif model_choice == "sd-v1-5":
            status_messages.append(("info", "Loading Stable Diffusion v1.5..."))
            
            # Load base pipeline
            base_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                **common_args
            ).to("cpu")
            
            # Use advanced scheduler for better quality
            base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                base_pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
                solver_order=2
            )
            
            # Memory optimizations
            base_pipe.enable_vae_tiling()
            base_pipe.enable_attention_slicing(slice_size=1)
            
            models['txt2img'] = base_pipe
            models['img2img'] = StableDiffusionImg2ImgPipeline(**base_pipe.components)
            
            status_messages.append(("success", "✅ SD v1.5 loaded with DPM++ scheduler"))
            
        elif model_choice == "realistic-vision":
            status_messages.append(("info", "Loading Realistic Vision v5.1..."))
            
            base_pipe = StableDiffusionPipeline.from_pretrained(
                "SG161222/Realistic_Vision_V5.1_noVAE",
                **common_args
            ).to("cpu")
            
            base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                base_pipe.scheduler.config,
                use_karras_sigmas=True
            )
            
            base_pipe.enable_vae_tiling()
            base_pipe.enable_attention_slicing(slice_size=1)
            
            models['txt2img'] = base_pipe
            models['img2img'] = StableDiffusionImg2ImgPipeline(**base_pipe.components)
            
            status_messages.append(("success", "✅ Realistic Vision loaded (photorealistic)"))
            
        elif model_choice == "dreamshaper":
            status_messages.append(("info", "Loading DreamShaper v8..."))
            
            base_pipe = StableDiffusionPipeline.from_pretrained(
                "Lykon/DreamShaper",
                **common_args
            ).to("cpu")
            
            base_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                base_pipe.scheduler.config
            )
            
            base_pipe.enable_vae_tiling()
            base_pipe.enable_attention_slicing(slice_size=1)
            
            models['txt2img'] = base_pipe
            models['img2img'] = StableDiffusionImg2ImgPipeline(**base_pipe.components)
            
            status_messages.append(("success", "✅ DreamShaper loaded (artistic)"))
            
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        status_messages.append(("error", f"Failed to load model: {e}"))
        return None, status_messages
    
    return models, status_messages

@st.cache_resource
def load_clip_model():
    """Load CLIP for advanced prompt analysis"""
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to("cpu")
    return processor, model

# Advanced Image Processing Functions
def advanced_upscale(image, scale_factor=2, method="lanczos"):
    """Advanced upscaling with multiple methods"""
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    
    if method == "lanczos":
        return image.resize(new_size, Image.Resampling.LANCZOS)
    elif method == "bicubic":
        return image.resize(new_size, Image.Resampling.BICUBIC)
    else:
        return image.resize(new_size, Image.Resampling.LANCZOS)

def apply_style_transfer(image, style="none"):
    """Apply artistic style filters"""
    if style == "oil_painting":
        # Simulate oil painting effect
        img_array = np.array(image)
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        return Image.fromarray(img_array)
    
    elif style == "watercolor":
        # Simulate watercolor effect
        img_array = np.array(image)
        img_array = cv2.stylization(img_array, sigma_s=60, sigma_r=0.6)
        return Image.fromarray(img_array)
    
    elif style == "pencil_sketch":
        # Create pencil sketch effect
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(sketch_rgb)
    
    elif style == "cartoon":
        # Create cartoon effect
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img_array, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return Image.fromarray(cartoon)
    
    return image

def enhance_image_advanced(image, brightness=1.0, contrast=1.0, saturation=1.0, 
                          sharpness=1.0, denoise=False):
    """Advanced image enhancement with multiple parameters"""
    # Brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    # Contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    # Color saturation
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation)
    
    # Sharpness
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
    
    # Denoise
    if denoise:
        img_array = np.array(image)
        img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        image = Image.fromarray(img_array)
    
    return image

def create_image_grid(images, cols=2):
    """Create a grid of images"""
    if not images:
        return None
    
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new('RGB', (w * cols, h * rows), color='white')
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * w, row * h))
    
    return grid

def extract_first_frame(video_bytes):
    """Extract first frame from video"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_bytes)
        temp_path = temp_file.name
    
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    success, frame = cap.read()
    cap.release()
    os.unlink(temp_path)
    
    if not success:
        raise ValueError("Could not read frame from video")
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def get_image_download_link(img, filename="image.png"):
    """Create download link for image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/png;base64,{img_str}" download="{filename}">📥 Download</a>'

# Advanced Prompt Engineering
def enhance_prompt(prompt, style="none", quality_boost=True):
    """Enhance prompts with style and quality keywords"""
    enhanced = prompt.strip()
    
    # Add style-specific keywords
    style_keywords = {
        "photorealistic": "photorealistic, 8k uhd, high quality, detailed, professional photography",
        "artistic": "artistic, painterly, creative, expressive, masterpiece",
        "cinematic": "cinematic lighting, dramatic, film grain, depth of field, bokeh",
        "anime": "anime style, manga, cel shaded, vibrant colors, detailed",
        "fantasy": "fantasy art, magical, ethereal, mystical, enchanted",
        "scifi": "sci-fi, futuristic, cyberpunk, neon, high tech",
        "vintage": "vintage, retro, nostalgic, film photography, aged",
        "minimalist": "minimalist, clean, simple, modern, elegant"
    }
    
    if style in style_keywords:
        enhanced += f", {style_keywords[style]}"
    
    # Add quality boosters
    if quality_boost:
        enhanced += ", highly detailed, sharp focus, professional"
    
    return enhanced

def generate_negative_prompt(quality_level="high"):
    """Generate comprehensive negative prompts"""
    base_negative = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
    
    if quality_level == "high":
        return base_negative + ", watermark, signature, text, jpeg artifacts, worst quality, low resolution, grainy, pixelated, amateur"
    elif quality_level == "medium":
        return base_negative + ", watermark, low resolution"
    else:
        return base_negative

# Prompt templates for different use cases
PROMPT_TEMPLATES = {
    "Portrait": "portrait of {subject}, professional photography, studio lighting, detailed face, 8k",
    "Landscape": "beautiful landscape of {subject}, golden hour, dramatic sky, highly detailed, 8k",
    "Product": "product photography of {subject}, white background, professional lighting, commercial",
    "Architecture": "architectural photography of {subject}, modern design, clean lines, professional",
    "Food": "food photography of {subject}, appetizing, professional lighting, macro detail",
    "Abstract": "abstract art of {subject}, creative, colorful, artistic, unique perspective",
    "Character": "character design of {subject}, full body, detailed, concept art, professional",
    "Interior": "interior design of {subject}, modern, elegant, well-lit, architectural digest"
}

# Random art generation for "This Art Does Not Exist" mode
ART_STYLES = [
    "abstract expressionism", "surrealism", "impressionism", "cubism", "pop art",
    "minimalism", "art nouveau", "baroque", "renaissance", "contemporary art",
    "digital art", "concept art", "fantasy art", "sci-fi art", "cyberpunk art",
    "steampunk art", "watercolor painting", "oil painting", "acrylic painting"
]

ART_SUBJECTS = [
    "cosmic landscape", "mystical forest", "futuristic cityscape", "abstract composition",
    "ethereal portrait", "dreamlike scenery", "geometric patterns", "organic forms",
    "celestial bodies", "underwater world", "mountain vista", "desert landscape",
    "tropical paradise", "winter wonderland", "autumn forest", "spring meadow",
    "urban architecture", "ancient ruins", "magical realm", "alien planet"
]

ART_MOODS = [
    "vibrant and energetic", "calm and serene", "dark and mysterious",
    "bright and cheerful", "dramatic and intense", "peaceful and tranquil",
    "chaotic and dynamic", "elegant and refined", "bold and striking"
]

def generate_random_art_prompt():
    """Generate random art prompt for 'This Art Does Not Exist' mode"""
    import random
    style = random.choice(ART_STYLES)
    subject = random.choice(ART_SUBJECTS)
    mood = random.choice(ART_MOODS)
    quality_words = ["masterpiece", "highly detailed", "professional", "stunning", "intricate"]
    quality = ", ".join(random.sample(quality_words, 3))
    return f"{subject} in {style} style, {mood}, {quality}, 8k"

# Main UI
st.markdown('<h1 class="title-gradient">🎨 Advanced AI Image Studio</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Professional-Grade Image Generation for College Projects</p>', unsafe_allow_html=True)

# Feature badges
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <span class="feature-badge">🎨 Multiple AI Models</span>
    <span class="feature-badge">✨ Style Transfer</span>
    <span class="feature-badge">🔧 Advanced Controls</span>
    <span class="feature-badge">📊 Quality Analytics</span>
    <span class="feature-badge">🚀 API Integration</span>
    <span class="feature-badge">🎭 Artistic Filters</span>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "image_history" not in st.session_state:
    st.session_state.image_history = []
if "generation_stats" not in st.session_state:
    st.session_state.generation_stats = {"total": 0, "total_time": 0}

# Load models
with st.spinner("🚀 Loading AI models..."):
    model_choice_map = {
        "⚡ LCM (FASTEST - 4-8 steps, 10-30 sec)": "lcm",
        "Stable Diffusion v1.5 (Balanced)": "sd-v1-5",
        "Realistic Vision (Photorealistic)": "realistic-vision",
        "DreamShaper (Artistic)": "dreamshaper"
    }
    
    selected_model = st.sidebar.selectbox(
        "🤖 AI Model",
        list(model_choice_map.keys()),
        help="Choose the AI model based on your desired output style. LCM is 10x faster!"
    )
    
    model_key = model_choice_map[selected_model]
    models, status_messages = load_models(model_key)
    
    if models:
        clip_processor, clip_model = load_clip_model()
        for msg_type, msg in status_messages:
            if msg_type == "success":
                st.sidebar.success(msg)
            elif msg_type == "error":
                st.sidebar.error(msg)
    else:
        st.error("Failed to load models. Please check your setup.")
        st.stop()

# Sidebar controls
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Generation Settings")

# Mode selection
mode = st.sidebar.selectbox(
    "Generation Mode",
    ["Text-to-Image", "Image-to-Image", "Video-to-Image", "Batch Generation", "🎨 This Art Does Not Exist"],
    help="Select the type of generation you want to perform"
)

# API option
use_api = st.sidebar.checkbox(
    "🚀 Use HuggingFace API (Fast)",
    value=False,
    help="Use cloud GPU for 10x faster generation (Text-to-Image only)"
)

if use_api and not HF_TOKEN:
    st.sidebar.error("⚠️ HF_TOKEN required in .env file")
    use_api = False

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Prompt Configuration")
    
    # Special handling for "This Art Does Not Exist" mode
    if mode == "🎨 This Art Does Not Exist":
        st.info("🎨 **Random Art Mode**: Click generate to create a unique, random artwork!")
        
        if st.button("🎲 Generate Random Prompt", use_container_width=True):
            st.session_state.random_prompt = generate_random_art_prompt()
        
        if "random_prompt" not in st.session_state:
            st.session_state.random_prompt = generate_random_art_prompt()
        
        prompt = st.session_state.random_prompt
        st.text_area("Generated Prompt", prompt, height=100, disabled=True)
        
        st.markdown("**✨ Every generation creates a completely unique artwork!**")
    else:
        # Prompt template
        use_template = st.checkbox("Use Prompt Template", value=False)
        if use_template:
            template_type = st.selectbox("Template Type", list(PROMPT_TEMPLATES.keys()))
            subject = st.text_input("Subject", "a beautiful sunset")
            prompt = PROMPT_TEMPLATES[template_type].format(subject=subject)
            st.info(f"Generated prompt: {prompt}")
        else:
            prompt = st.text_area(
                "Your Prompt",
                "a serene mountain landscape at sunset, golden hour lighting, highly detailed",
                height=100,
                help="Describe what you want to generate"
            )
    
    # Style enhancement
    style_preset = st.selectbox(
        "Style Preset",
        ["none", "photorealistic", "artistic", "cinematic", "anime", "fantasy", "scifi", "vintage", "minimalist"],
        help="Apply style-specific enhancements to your prompt"
    )
    
    # Negative prompt
    neg_quality = st.selectbox("Negative Prompt Quality", ["high", "medium", "low"])
    negative_prompt = st.text_area(
        "Negative Prompt (Optional)",
        generate_negative_prompt(neg_quality),
        height=80,
        help="Things to avoid in the generation"
    )
    
    # File uploads for img2img and video2img
    input_image = None
    if mode == "Image-to-Image":
        uploaded_file = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Reference Image", use_container_width=True)
    
    elif mode == "Video-to-Image":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            try:
                input_image = extract_first_frame(uploaded_video.getvalue())
                st.image(input_image, caption="Extracted Frame", use_container_width=True)
            except Exception as e:
                st.error(f"Error extracting frame: {e}")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            if mode == "🎨 This Art Does Not Exist":
                num_images = 1  # Always 1 for random art mode
                st.info("Random Art Mode: 1 image per generation")
            else:
                num_images = st.slider("Number of Images", 1, 4, 1)
            
            width = st.select_slider("Width", [384, 448, 512, 576, 640], value=512)
            height = st.select_slider("Height", [384, 448, 512, 576, 640], value=512)
            steps = st.slider("Inference Steps", 10, 50, 15 if mode == "🎨 This Art Does Not Exist" else 20)
        
        with col_b:
            guidance_scale = st.slider("Guidance Scale", 5.0, 15.0, 7.5, 0.5)
            if mode in ["Image-to-Image", "Video-to-Image"]:
                strength = st.slider("Transformation Strength", 0.0, 1.0, 0.75, 0.05)
            
            if mode == "🎨 This Art Does Not Exist":
                seed = -1  # Always random for this mode
                st.info("Random Art Mode: Random seed")
            else:
                seed = st.number_input("Seed (-1 for random)", -1, 999999, -1)
    
    # Post-processing options
    with st.expander("✨ Post-Processing"):
        col_c, col_d = st.columns(2)
        
        with col_c:
            upscale_factor = st.selectbox("Upscale", ["None", "1.5x", "2x", "3x"])
            brightness = st.slider("Brightness", 0.5, 1.5, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.5, 1.5, 1.0, 0.1)
        
        with col_d:
            saturation = st.slider("Saturation", 0.5, 1.5, 1.0, 0.1)
            sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
            denoise = st.checkbox("Apply Denoising", value=False)
    
    # Artistic filters
    with st.expander("🎨 Artistic Filters"):
        artistic_style = st.selectbox(
            "Apply Artistic Style",
            ["none", "oil_painting", "watercolor", "pencil_sketch", "cartoon"],
            help="Apply artistic filters to the generated image"
        )
    
    # Generate button
    if mode == "🎨 This Art Does Not Exist":
        generate_btn = st.button("🎨 Generate Random Art", use_container_width=True)
    else:
        generate_btn = st.button("🎨 Generate Images", use_container_width=True)

with col2:
    st.markdown("### 🖼️ Generated Images")
    
    if generate_btn:
        if not prompt:
            st.error("Please enter a prompt")
        else:
            # For "This Art Does Not Exist" mode, generate new random prompt
            if mode == "🎨 This Art Does Not Exist":
                prompt = generate_random_art_prompt()
                st.session_state.random_prompt = prompt
                enhanced_prompt = prompt  # Already enhanced
            else:
                # Enhance prompt
                enhanced_prompt = enhance_prompt(prompt, style_preset, quality_boost=True)
            
            start_time = time.time()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            generated_images = []
            
            try:
                if use_api and mode == "Text-to-Image":
                    # Use HuggingFace API with updated endpoint
                    try:
                        from huggingface_hub import InferenceClient
                        client = InferenceClient(token=HF_TOKEN)
                        
                        for i in range(num_images):
                            status_text.text(f"Generating via API {i+1}/{num_images}...")
                            progress_bar.progress((i + 1) / num_images)
                            
                            # Use faster SDXL Turbo model
                            image = client.text_to_image(
                                prompt=enhanced_prompt,
                                model="stabilityai/sdxl-turbo",
                                negative_prompt=negative_prompt,
                                num_inference_steps=min(steps, 4),  # Turbo needs only 1-4 steps
                                guidance_scale=0.0,  # Turbo doesn't use guidance
                                width=width,
                                height=height
                            )
                    except Exception as api_error:
                        st.error(f"API Error: {api_error}")
                        st.info("Falling back to local generation...")
                        use_api = False
                        
                        # Post-processing
                        if upscale_factor != "None":
                            scale = float(upscale_factor.replace("x", ""))
                            image = advanced_upscale(image, scale)
                        
                        image = enhance_image_advanced(
                            image, brightness, contrast, saturation, sharpness, denoise
                        )
                        
                        if artistic_style != "none":
                            image = apply_style_transfer(image, artistic_style)
                        
                        generated_images.append(image)
                
                else:
                    # Use local model
                    pipe = models['txt2img'] if mode == "Text-to-Image" else models['img2img']
                    
                    with torch.inference_mode():
                        for i in range(num_images):
                            status_text.text(f"Generating locally {i+1}/{num_images}...")
                            progress_bar.progress((i + 1) / num_images)
                            
                            # Set seed
                            if seed >= 0:
                                generator = torch.Generator(device="cpu").manual_seed(seed + i)
                            else:
                                generator = None
                            
                            if mode == "Text-to-Image":
                                image = pipe(
                                    prompt=enhanced_prompt,
                                    negative_prompt=negative_prompt,
                                    num_inference_steps=steps,
                                    guidance_scale=guidance_scale,
                                    width=width,
                                    height=height,
                                    generator=generator
                                ).images[0]
                            else:
                                if input_image is None:
                                    st.error("Please upload an image/video")
                                    break
                                
                                # Resize input image
                                input_resized = input_image.resize((width, height), Image.Resampling.LANCZOS)
                                
                                image = pipe(
                                    prompt=enhanced_prompt,
                                    image=input_resized,
                                    negative_prompt=negative_prompt,
                                    num_inference_steps=steps,
                                    guidance_scale=guidance_scale,
                                    strength=strength,
                                    generator=generator
                                ).images[0]
                            
                            # Post-processing
                            if upscale_factor != "None":
                                scale = float(upscale_factor.replace("x", ""))
                                image = advanced_upscale(image, scale)
                            
                            image = enhance_image_advanced(
                                image, brightness, contrast, saturation, sharpness, denoise
                            )
                            
                            if artistic_style != "none":
                                image = apply_style_transfer(image, artistic_style)
                            
                            generated_images.append(image)
                
                # Calculate stats
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Update session state
                st.session_state.image_history.extend([(img, enhanced_prompt) for img in generated_images])
                st.session_state.generation_stats["total"] += len(generated_images)
                st.session_state.generation_stats["total_time"] += generation_time
                
                # Display results
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Generated {len(generated_images)} image(s) in {generation_time:.2f}s")
                
                # Display images
                if len(generated_images) == 1:
                    st.image(generated_images[0], use_container_width=True)
                    st.markdown(get_image_download_link(generated_images[0], "generated_0.png"), unsafe_allow_html=True)
                else:
                    cols = st.columns(2)
                    for idx, img in enumerate(generated_images):
                        cols[idx % 2].image(img, use_container_width=True)
                        cols[idx % 2].markdown(get_image_download_link(img, f"generated_{idx}.png"), unsafe_allow_html=True)
                
                # Show generation info
                with st.expander("📊 Generation Details"):
                    st.write(f"**Model:** {selected_model}")
                    st.write(f"**Prompt:** {enhanced_prompt}")
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
        st.info("👈 Configure your settings and click 'Generate Images' to start creating!")

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
    total_time_min = st.session_state.generation_stats['total_time'] / 60
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_time_min:.1f}m</div>
        <div class="metric-label">Total Time</div>
    </div>
    """, unsafe_allow_html=True)

# Image History Gallery
if st.session_state.image_history:
    st.markdown("---")
    st.markdown("### 🖼️ Image History Gallery")
    
    # Gallery controls
    col_g1, col_g2, col_g3 = st.columns([2, 1, 1])
    with col_g1:
        gallery_view = st.selectbox("View", ["Grid", "List", "Slideshow"])
    with col_g2:
        show_count = st.selectbox("Show", [8, 16, 24, "All"])
    with col_g3:
        if st.button("🗑️ Clear History"):
            st.session_state.image_history = []
            st.rerun()
    
    # Display gallery
    if show_count == "All":
        display_images = st.session_state.image_history
    else:
        display_images = st.session_state.image_history[-show_count:]
    
    if gallery_view == "Grid":
        cols = st.columns(4)
        for idx, (img, prompt) in enumerate(reversed(display_images)):
            with cols[idx % 4]:
                st.image(img, use_container_width=True)
                st.caption(prompt[:50] + "..." if len(prompt) > 50 else prompt)
                st.markdown(get_image_download_link(img, f"history_{idx}.png"), unsafe_allow_html=True)
    
    elif gallery_view == "List":
        for idx, (img, prompt) in enumerate(reversed(display_images)):
            col_l1, col_l2 = st.columns([1, 2])
            with col_l1:
                st.image(img, use_container_width=True)
            with col_l2:
                st.markdown(f"**Prompt:** {prompt}")
                st.markdown(get_image_download_link(img, f"history_{idx}.png"), unsafe_allow_html=True)
            st.markdown("---")
    
    elif gallery_view == "Slideshow":
        if display_images:
            slide_idx = st.slider("Image", 0, len(display_images) - 1, 0)
            img, prompt = display_images[-(slide_idx + 1)]
            st.image(img, use_container_width=True)
            st.markdown(f"**Prompt:** {prompt}")
            st.markdown(get_image_download_link(img, f"slideshow_{slide_idx}.png"), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Advanced AI Image Studio</strong> - College Project</p>
    <p>Powered by Stable Diffusion, HuggingFace Transformers, and Streamlit</p>
    <p style="font-size: 0.9rem;">Features: Multiple AI Models • Style Transfer • Advanced Post-Processing • API Integration</p>
</div>
""", unsafe_allow_html=True)
