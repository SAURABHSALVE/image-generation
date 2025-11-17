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
    page_title="AI Image Studio with Agentic AI",
    page_icon="🤖",
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

# ═══════════════════════════════════════════════════════════════
# 🤖 AGENTIC AI FEATURES
# ═══════════════════════════════════════════════════════════════

def ai_prompt_enhancement_agent(basic_prompt):
    """
    AI Agent that intelligently enhances user prompts for better image generation.
    
    This agent analyzes the input prompt and adds contextually appropriate
    quality boosters, lighting descriptions, and technical keywords.
    """
    if not basic_prompt or len(basic_prompt.strip()) == 0:
        return basic_prompt
    
    enhanced = basic_prompt.strip()
    words = enhanced.lower().split()
    
    # Agent's knowledge base
    quality_keywords = ['detailed', 'quality', 'professional', 'masterpiece', '8k', '4k', 'hd']
    lighting_keywords = ['light', 'lighting', 'lit', 'illuminated', 'glow']
    style_keywords = ['cinematic', 'artistic', 'photorealistic', 'stylized']
    
    # Agent Decision 1: Add quality boosters if missing
    has_quality = any(keyword in enhanced.lower() for keyword in quality_keywords)
    if not has_quality:
        enhanced += ", highly detailed, professional quality"
    
    # Agent Decision 2: Add lighting if missing
    has_lighting = any(keyword in enhanced.lower() for keyword in lighting_keywords)
    if not has_lighting:
        # Choose lighting based on content
        if any(word in enhanced.lower() for word in ['portrait', 'person', 'face']):
            enhanced += ", studio lighting"
        elif any(word in enhanced.lower() for word in ['landscape', 'outdoor', 'nature']):
            enhanced += ", natural lighting, golden hour"
        else:
            enhanced += ", perfect lighting"
    
    # Agent Decision 3: Add technical quality for short prompts
    if len(words) < 5:
        enhanced += ", 8k, sharp focus, masterpiece"
    
    # Agent Decision 4: Add style context if missing
    has_style = any(keyword in enhanced.lower() for keyword in style_keywords)
    if not has_style and len(words) < 10:
        enhanced += ", cinematic"
    
    return enhanced

def smart_negative_prompt_agent(prompt, mode="Text-to-Image"):
    """
    AI Agent that generates context-aware negative prompts.
    
    This agent analyzes the user's prompt and generation mode to create
    an optimized negative prompt that prevents common quality issues.
    """
    # Base negative prompt
    base_negative = "blurry, low quality, distorted, deformed, ugly, bad anatomy"
    
    if not prompt:
        return base_negative + ", watermark, text, worst quality"
    
    prompt_lower = prompt.lower()
    
    # Agent analyzes prompt content and adds specific negatives
    
    # For human/character content
    if any(word in prompt_lower for word in ['person', 'people', 'human', 'portrait', 'face', 'character']):
        return base_negative + ", extra limbs, disfigured face, bad proportions, mutated hands, poorly drawn face, mutation, bad hands, missing fingers, extra fingers"
    
    # For landscape/scenery
    elif any(word in prompt_lower for word in ['landscape', 'scenery', 'nature', 'mountain', 'forest', 'ocean', 'sky']):
        return base_negative + ", oversaturated, artificial, fake colors, unrealistic, cartoon, anime"
    
    # For 3D models
    elif any(word in prompt_lower for word in ['3d', 'model', 'render', 'object']):
        return base_negative + ", multiple objects, cluttered, bad topology, low poly, messy, duplicate"
    
    # For artistic content
    elif any(word in prompt_lower for word in ['art', 'painting', 'drawing', 'artistic']):
        return base_negative + ", amateur, sketch, unfinished, messy, childish, simple"
    
    # For animals
    elif any(word in prompt_lower for word in ['animal', 'cat', 'dog', 'bird', 'creature']):
        return base_negative + ", deformed body, extra legs, missing legs, bad anatomy, mutated"
    
    # For vehicles/mechanical
    elif any(word in prompt_lower for word in ['car', 'vehicle', 'robot', 'machine', 'mech']):
        return base_negative + ", broken, damaged, incomplete, asymmetric, bad design"
    
    # Default comprehensive negative
    else:
        return base_negative + ", watermark, signature, text, jpeg artifacts, worst quality, low resolution, grainy, amateur"

def quality_control_agent(image):
    """
    AI Agent that automatically detects and fixes quality issues in generated images.
    
    This agent analyzes the image for common problems like poor brightness,
    low contrast, or lack of sharpness, and applies appropriate corrections.
    """
    issues_found = []
    fixes_applied = []
    
    img_array = np.array(image)
    
    # Agent Analysis 1: Check brightness
    brightness = img_array.mean()
    if brightness < 85:
        image = ImageEnhance.Brightness(image).enhance(1.25)
        issues_found.append("Image too dark")
        fixes_applied.append("Increased brightness by 25%")
    elif brightness > 200:
        image = ImageEnhance.Brightness(image).enhance(0.85)
        issues_found.append("Image too bright")
        fixes_applied.append("Decreased brightness by 15%")
    
    # Agent Analysis 2: Check contrast
    contrast = img_array.std()
    if contrast < 45:
        image = ImageEnhance.Contrast(image).enhance(1.2)
        issues_found.append("Low contrast detected")
        fixes_applied.append("Enhanced contrast by 20%")
    
    # Agent Analysis 3: Apply intelligent sharpening
    image = ImageEnhance.Sharpness(image).enhance(1.15)
    fixes_applied.append("Applied smart sharpening")
    
    # Agent Analysis 4: Subtle color enhancement
    image = ImageEnhance.Color(image).enhance(1.05)
    fixes_applied.append("Enhanced color vibrancy")
    
    return image, issues_found, fixes_applied

def parameter_optimization_agent(prompt, mode, device, width, height):
    """
    AI Agent that optimizes generation parameters based on context.
    
    This agent analyzes the prompt complexity, device capabilities, and
    generation mode to suggest optimal parameters for best results.
    """
    suggestions = []
    optimal_params = {
        'steps': 6,
        'guidance': 1.0,
        'width': width,
        'height': height
    }
    
    # Agent Analysis 1: Prompt complexity
    word_count = len(prompt.split()) if prompt else 0
    
    if word_count > 20:
        optimal_params['steps'] = 8
        optimal_params['guidance'] = 1.5
        suggestions.append("Complex prompt detected: Increased steps to 8 and guidance to 1.5")
    elif word_count < 5:
        optimal_params['steps'] = 4
        suggestions.append("Simple prompt: Reduced steps to 4 for faster generation")
    
    # Agent Analysis 2: Device optimization
    if device == "cpu":
        if width > 512 or height > 512:
            optimal_params['width'] = 384
            optimal_params['height'] = 384
            suggestions.append("CPU detected: Reduced size to 384x384 for better performance")
    else:  # GPU
        if width < 512:
            optimal_params['width'] = 512
            optimal_params['height'] = 512
            suggestions.append("GPU detected: Increased size to 512x512 for better quality")
    
    # Agent Analysis 3: Mode-specific optimization
    if mode == "Image-to-Image":
        optimal_params['steps'] = max(optimal_params['steps'], 8)
        suggestions.append("Image-to-Image mode: Increased steps for better transformation")
    
    return optimal_params, suggestions

def style_detection_agent(prompt):
    """
    AI Agent that automatically detects the best artistic style for the prompt.
    
    This agent analyzes the prompt content and suggests the most appropriate
    artistic style, lighting, and rendering approach.
    """
    prompt_lower = prompt.lower() if prompt else ""
    
    detected_style = {
        'primary_style': 'realistic',
        'lighting': 'natural',
        'rendering': 'standard',
        'confidence': 0.5,
        'reasoning': []
    }
    
    # Agent Analysis 1: Content-based style detection
    if any(word in prompt_lower for word in ['portrait', 'person', 'face', 'human', 'people']):
        detected_style['primary_style'] = 'photorealistic'
        detected_style['lighting'] = 'studio'
        detected_style['rendering'] = 'high detail'
        detected_style['confidence'] = 0.9
        detected_style['reasoning'].append("Human subject detected → Photorealistic style")
    
    elif any(word in prompt_lower for word in ['fantasy', 'magic', 'dragon', 'wizard', 'mystical', 'enchanted']):
        detected_style['primary_style'] = 'fantasy art'
        detected_style['lighting'] = 'dramatic'
        detected_style['rendering'] = 'painterly'
        detected_style['confidence'] = 0.85
        detected_style['reasoning'].append("Fantasy elements detected → Fantasy art style")
    
    elif any(word in prompt_lower for word in ['robot', 'cyber', 'future', 'sci-fi', 'space', 'alien']):
        detected_style['primary_style'] = 'sci-fi'
        detected_style['lighting'] = 'neon/dramatic'
        detected_style['rendering'] = 'high tech'
        detected_style['confidence'] = 0.85
        detected_style['reasoning'].append("Sci-fi elements detected → Sci-fi style")
    
    elif any(word in prompt_lower for word in ['painting', 'art', 'canvas', 'brush', 'artistic']):
        detected_style['primary_style'] = 'artistic'
        detected_style['lighting'] = 'soft'
        detected_style['rendering'] = 'painterly'
        detected_style['confidence'] = 0.8
        detected_style['reasoning'].append("Artistic keywords detected → Artistic style")
    
    elif any(word in prompt_lower for word in ['cartoon', 'anime', 'manga', 'comic']):
        detected_style['primary_style'] = 'stylized'
        detected_style['lighting'] = 'cel-shaded'
        detected_style['rendering'] = 'clean lines'
        detected_style['confidence'] = 0.9
        detected_style['reasoning'].append("Cartoon/anime keywords → Stylized style")
    
    elif any(word in prompt_lower for word in ['landscape', 'nature', 'mountain', 'forest', 'ocean']):
        detected_style['primary_style'] = 'realistic'
        detected_style['lighting'] = 'natural/golden hour'
        detected_style['rendering'] = 'detailed'
        detected_style['confidence'] = 0.8
        detected_style['reasoning'].append("Landscape detected → Realistic with natural lighting")
    
    # Agent Analysis 2: Mood-based lighting adjustment
    if any(word in prompt_lower for word in ['dark', 'night', 'shadow', 'mysterious']):
        detected_style['lighting'] = 'low key/dramatic'
        detected_style['reasoning'].append("Dark mood → Low key lighting")
    elif any(word in prompt_lower for word in ['bright', 'sunny', 'cheerful', 'happy']):
        detected_style['lighting'] = 'bright/high key'
        detected_style['reasoning'].append("Bright mood → High key lighting")
    
    return detected_style

def composition_analysis_agent(prompt):
    """
    AI Agent that analyzes and suggests optimal image composition.
    
    This agent determines the best framing, aspect ratio, and composition
    rules based on the subject matter.
    """
    prompt_lower = prompt.lower() if prompt else ""
    
    composition = {
        'framing': 'medium shot',
        'aspect_ratio': '1:1',
        'rule': 'rule of thirds',
        'focus': 'center',
        'suggestions': []
    }
    
    # Agent Analysis 1: Subject-based framing
    if any(word in prompt_lower for word in ['portrait', 'face', 'headshot']):
        composition['framing'] = 'close-up'
        composition['aspect_ratio'] = '3:4 (portrait)'
        composition['focus'] = 'face/eyes'
        composition['suggestions'].append("Portrait detected → Close-up framing, vertical aspect")
    
    elif any(word in prompt_lower for word in ['full body', 'character', 'person standing']):
        composition['framing'] = 'full shot'
        composition['aspect_ratio'] = '3:4 (portrait)'
        composition['focus'] = 'subject center'
        composition['suggestions'].append("Full body → Full shot framing")
    
    elif any(word in prompt_lower for word in ['landscape', 'panorama', 'vista', 'scenery']):
        composition['framing'] = 'wide shot'
        composition['aspect_ratio'] = '16:9 (landscape)'
        composition['rule'] = 'rule of thirds + leading lines'
        composition['suggestions'].append("Landscape → Wide framing, horizontal aspect")
    
    elif any(word in prompt_lower for word in ['product', 'object', 'item']):
        composition['framing'] = 'close-up'
        composition['aspect_ratio'] = '1:1 (square)'
        composition['focus'] = 'product center'
        composition['rule'] = 'centered composition'
        composition['suggestions'].append("Product → Centered composition, square format")
    
    # Agent Analysis 2: Action-based composition
    if any(word in prompt_lower for word in ['action', 'running', 'jumping', 'flying', 'moving']):
        composition['rule'] = 'dynamic diagonal'
        composition['suggestions'].append("Action detected → Dynamic diagonal composition")
    
    return composition

def color_harmony_agent(prompt):
    """
    AI Agent that suggests optimal color palettes and harmony.
    
    This agent analyzes the prompt and recommends color schemes that
    enhance the visual appeal and mood of the image.
    """
    prompt_lower = prompt.lower() if prompt else ""
    
    color_scheme = {
        'palette': 'natural',
        'temperature': 'neutral',
        'saturation': 'medium',
        'harmony_type': 'analogous',
        'recommendations': []
    }
    
    # Agent Analysis 1: Mood-based color temperature
    if any(word in prompt_lower for word in ['warm', 'sunset', 'fire', 'autumn', 'cozy']):
        color_scheme['temperature'] = 'warm'
        color_scheme['palette'] = 'warm tones (orange, red, yellow)'
        color_scheme['recommendations'].append("Warm mood → Warm color palette")
    
    elif any(word in prompt_lower for word in ['cool', 'ice', 'winter', 'ocean', 'night']):
        color_scheme['temperature'] = 'cool'
        color_scheme['palette'] = 'cool tones (blue, cyan, purple)'
        color_scheme['recommendations'].append("Cool mood → Cool color palette")
    
    # Agent Analysis 2: Subject-based saturation
    if any(word in prompt_lower for word in ['vibrant', 'colorful', 'bright', 'vivid']):
        color_scheme['saturation'] = 'high'
        color_scheme['recommendations'].append("Vibrant keywords → High saturation")
    
    elif any(word in prompt_lower for word in ['muted', 'pastel', 'soft', 'subtle']):
        color_scheme['saturation'] = 'low'
        color_scheme['recommendations'].append("Soft keywords → Low saturation")
    
    # Agent Analysis 3: Style-based harmony
    if any(word in prompt_lower for word in ['fantasy', 'magical', 'dreamy']):
        color_scheme['harmony_type'] = 'complementary'
        color_scheme['recommendations'].append("Fantasy theme → Complementary colors for contrast")
    
    elif any(word in prompt_lower for word in ['natural', 'realistic', 'photo']):
        color_scheme['harmony_type'] = 'analogous'
        color_scheme['recommendations'].append("Realistic theme → Analogous colors for harmony")
    
    elif any(word in prompt_lower for word in ['dramatic', 'intense', 'bold']):
        color_scheme['harmony_type'] = 'triadic'
        color_scheme['recommendations'].append("Dramatic theme → Triadic colors for impact")
    
    return color_scheme

def content_safety_agent(prompt):
    """
    AI Agent that checks content appropriateness and suggests modifications.
    
    This agent ensures generated content is appropriate and suggests
    alternatives for potentially problematic prompts.
    """
    prompt_lower = prompt.lower() if prompt else ""
    
    safety_check = {
        'is_safe': True,
        'confidence': 1.0,
        'issues': [],
        'suggestions': [],
        'modified_prompt': prompt
    }
    
    # Agent Analysis 1: Check for inappropriate content
    inappropriate_keywords = ['violence', 'gore', 'explicit', 'nsfw', 'inappropriate']
    
    for keyword in inappropriate_keywords:
        if keyword in prompt_lower:
            safety_check['is_safe'] = False
            safety_check['confidence'] = 0.3
            safety_check['issues'].append(f"Potentially inappropriate keyword: '{keyword}'")
            safety_check['suggestions'].append("Consider using more appropriate terms")
    
    # Agent Analysis 2: Check for ambiguous terms
    ambiguous_terms = {
        'naked': 'unclothed statue/art',
        'weapon': 'historical artifact',
        'fight': 'martial arts demonstration'
    }
    
    for term, alternative in ambiguous_terms.items():
        if term in prompt_lower:
            safety_check['suggestions'].append(f"Ambiguous term '{term}' - consider '{alternative}' for clarity")
            safety_check['modified_prompt'] = prompt.replace(term, alternative)
    
    # Agent Analysis 3: Positive reinforcement
    if safety_check['is_safe']:
        safety_check['suggestions'].append("Prompt appears appropriate for generation")
    
    return safety_check

def iterative_refinement_agent(image, prompt, iteration=1, max_iterations=3):
    """
    AI Agent that analyzes generated images and suggests refinements.
    
    This agent evaluates the quality of generated images and provides
    actionable feedback for improvement in subsequent iterations.
    """
    img_array = np.array(image)
    
    analysis = {
        'iteration': iteration,
        'quality_score': 0.0,
        'issues_detected': [],
        'refinement_suggestions': [],
        'should_refine': False
    }
    
    # Agent Analysis 1: Technical quality metrics
    brightness = img_array.mean()
    contrast = img_array.std()
    
    quality_points = 0
    max_points = 5
    
    # Brightness check
    if 80 <= brightness <= 180:
        quality_points += 1
    else:
        analysis['issues_detected'].append(f"Brightness: {brightness:.1f} (optimal: 80-180)")
        analysis['refinement_suggestions'].append("Adjust prompt to specify lighting conditions")
    
    # Contrast check
    if contrast >= 45:
        quality_points += 1
    else:
        analysis['issues_detected'].append(f"Low contrast: {contrast:.1f}")
        analysis['refinement_suggestions'].append("Add 'high contrast' or 'vivid' to prompt")
    
    # Color distribution check
    color_variance = np.var(img_array, axis=(0, 1))
    if np.mean(color_variance) > 100:
        quality_points += 1
    else:
        analysis['issues_detected'].append("Limited color range")
        analysis['refinement_suggestions'].append("Add 'colorful' or 'vibrant' to prompt")
    
    # Detail check (edge detection simulation)
    if contrast > 40:
        quality_points += 1
    else:
        analysis['refinement_suggestions'].append("Add 'highly detailed' to prompt")
    
    # Overall composition (center focus check)
    center_region = img_array[img_array.shape[0]//4:3*img_array.shape[0]//4, 
                               img_array.shape[1]//4:3*img_array.shape[1]//4]
    if center_region.mean() > img_array.mean() * 0.9:
        quality_points += 1
    
    # Calculate quality score
    analysis['quality_score'] = (quality_points / max_points) * 100
    
    # Agent Decision: Should we refine?
    if analysis['quality_score'] < 70 and iteration < max_iterations:
        analysis['should_refine'] = True
        analysis['refinement_suggestions'].append(f"Quality score: {analysis['quality_score']:.1f}% - Refinement recommended")
    else:
        analysis['refinement_suggestions'].append(f"Quality score: {analysis['quality_score']:.1f}% - Acceptable quality")
    
    return analysis

# Main UI
st.markdown('<h1 class="title-gradient">🤖 AI Image Studio with Agentic AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: white; margin-bottom: 2rem;">⚡ Powered by LCM + Intelligent AI Agents!</p>', unsafe_allow_html=True)

# Feature badges
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <span class="feature-badge">🤖 AI Agents</span>
    <span class="feature-badge">⚡ LCM Model</span>
    <span class="feature-badge">🎨 Text-to-Image</span>
    <span class="feature-badge">🖼️ Image-to-Image</span>
    <span class="feature-badge">� Artisticr Filters</span>
    <span class="feature-badge">🎲 Random Art</span>
    <span class="feature-badge">🎲 3D Models</span>
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
st.sidebar.markdown("### 🤖 Core AI Agents")

# Core AI Agent toggles
use_prompt_agent = st.sidebar.checkbox("🤖 Prompt Enhancement", value=True, 
    help="Automatically improves your prompts")
use_negative_agent = st.sidebar.checkbox("🤖 Smart Negative Prompt", value=True,
    help="Generates context-aware negative prompts")
use_quality_agent = st.sidebar.checkbox("🤖 Quality Control", value=True,
    help="Automatically fixes quality issues")
use_param_agent = st.sidebar.checkbox("🤖 Parameter Optimization", value=False,
    help="Optimizes settings for best results")

st.sidebar.markdown("### 🎨 Advanced AI Agents")

# Advanced AI Agent toggles
use_style_agent = st.sidebar.checkbox("🎨 Style Detection", value=False,
    help="Automatically detects best artistic style")
use_composition_agent = st.sidebar.checkbox("📐 Composition Analysis", value=False,
    help="Suggests optimal framing and composition")
use_color_agent = st.sidebar.checkbox("🌈 Color Harmony", value=False,
    help="Recommends optimal color palettes")
use_safety_agent = st.sidebar.checkbox("🛡️ Content Safety", value=True,
    help="Checks content appropriateness")
use_refinement_agent = st.sidebar.checkbox("🔄 Iterative Refinement", value=False,
    help="Analyzes and suggests improvements")

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
        user_prompt = st.text_area(
            "Your Prompt",
            "a serene mountain landscape at sunset",
            height=100,
            help="Describe what you want to generate"
        )
        
        # Apply Content Safety Agent first
        if use_safety_agent and user_prompt:
            safety_result = content_safety_agent(user_prompt)
            if not safety_result['is_safe']:
                st.warning("🛡️ **Content Safety Agent** detected potential issues")
                for issue in safety_result['issues']:
                    st.warning(f"⚠️ {issue}")
                if safety_result['modified_prompt'] != user_prompt:
                    st.info(f"💡 Suggested alternative: {safety_result['modified_prompt']}")
                    user_prompt = safety_result['modified_prompt']
        
        # Apply Style Detection Agent
        if use_style_agent and user_prompt:
            style_info = style_detection_agent(user_prompt)
            st.success(f"🎨 **Style Detection Agent** analyzed your prompt!")
            with st.expander("🎨 Style Analysis"):
                st.markdown(f"**Detected Style:** {style_info['primary_style']}")
                st.markdown(f"**Lighting:** {style_info['lighting']}")
                st.markdown(f"**Rendering:** {style_info['rendering']}")
                st.markdown(f"**Confidence:** {style_info['confidence']*100:.0f}%")
                for reason in style_info['reasoning']:
                    st.info(f"• {reason}")
        
        # Apply Composition Analysis Agent
        if use_composition_agent and user_prompt:
            comp_info = composition_analysis_agent(user_prompt)
            st.success(f"📐 **Composition Analysis Agent** suggests optimal framing!")
            with st.expander("📐 Composition Suggestions"):
                st.markdown(f"**Framing:** {comp_info['framing']}")
                st.markdown(f"**Aspect Ratio:** {comp_info['aspect_ratio']}")
                st.markdown(f"**Composition Rule:** {comp_info['rule']}")
                st.markdown(f"**Focus Point:** {comp_info['focus']}")
                for suggestion in comp_info['suggestions']:
                    st.info(f"• {suggestion}")
        
        # Apply Color Harmony Agent
        if use_color_agent and user_prompt:
            color_info = color_harmony_agent(user_prompt)
            st.success(f"🌈 **Color Harmony Agent** recommends color scheme!")
            with st.expander("🌈 Color Recommendations"):
                st.markdown(f"**Palette:** {color_info['palette']}")
                st.markdown(f"**Temperature:** {color_info['temperature']}")
                st.markdown(f"**Saturation:** {color_info['saturation']}")
                st.markdown(f"**Harmony Type:** {color_info['harmony_type']}")
                for rec in color_info['recommendations']:
                    st.info(f"• {rec}")
        
        # Apply Prompt Enhancement Agent
        if use_prompt_agent and user_prompt:
            prompt = ai_prompt_enhancement_agent(user_prompt)
            if prompt != user_prompt:
                st.success("🤖 **Prompt Enhancement Agent** improved your prompt!")
                with st.expander("See what the agent did"):
                    st.markdown(f"**Original:** {user_prompt}")
                    st.markdown(f"**Enhanced:** {prompt}")
        else:
            prompt = user_prompt
    
    # Smart Negative Prompt Agent
    if use_negative_agent and prompt:
        auto_negative = smart_negative_prompt_agent(prompt, mode)
        st.info("🤖 **Smart Negative Prompt Agent** generated context-aware negative prompt")
        negative_prompt = st.text_area(
            "AI-Generated Negative Prompt (Editable)",
            auto_negative,
            height=80,
            help="AI agent created this based on your prompt"
        )
    else:
        negative_prompt = st.text_area(
            "Negative Prompt (Optional)",
            "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark",
            height=80,
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
        # Parameter Optimization Agent
        if use_param_agent and prompt:
            device = models.get('device', 'cpu') if models else 'cpu'
            optimal_params, agent_suggestions = parameter_optimization_agent(
                prompt, mode, device, 512, 512
            )
            
            if agent_suggestions:
                st.success("🤖 **Parameter Optimization Agent** analyzed your request!")
                for suggestion in agent_suggestions:
                    st.info(f"• {suggestion}")
            
            # Use agent's recommendations
            default_steps = optimal_params['steps']
            default_width = optimal_params['width']
            default_height = optimal_params['height']
            default_guidance = optimal_params['guidance']
        else:
            default_steps = 6
            default_width = 512
            default_height = 512
            default_guidance = 1.0
        
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
            
            width = st.select_slider("Width", [256, 384, 512, 640, 768], value=default_width)
            height = st.select_slider("Height", [256, 384, 512, 640, 768], value=default_height)
        
        with col_b:
            steps = st.slider("Steps (LCM: 4-8 optimal)", 4, 12, default_steps)
            guidance_scale = st.slider("Guidance (LCM: 1.0-1.5)", 1.0, 2.0, default_guidance, 0.1)
            
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
                        
                        # Apply Quality Control Agent
                        if use_quality_agent:
                            image, issues, fixes = quality_control_agent(image)
                            if issues:
                                status_text.text(f"🤖 Quality Agent: Fixed {len(issues)} issues")
                        
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
                
                # Apply Iterative Refinement Agent
                if use_refinement_agent and len(generated_images) > 0:
                    st.markdown("---")
                    st.markdown("### 🔄 Iterative Refinement Agent Analysis")
                    
                    for idx, img in enumerate(generated_images[:1]):  # Analyze first image
                        refinement_analysis = iterative_refinement_agent(img, prompt)
                        
                        col_ref1, col_ref2 = st.columns([1, 2])
                        
                        with col_ref1:
                            st.metric("Quality Score", f"{refinement_analysis['quality_score']:.1f}%")
                            if refinement_analysis['should_refine']:
                                st.warning("⚠️ Refinement Recommended")
                            else:
                                st.success("✅ Quality Acceptable")
                        
                        with col_ref2:
                            if refinement_analysis['issues_detected']:
                                st.markdown("**Issues Detected:**")
                                for issue in refinement_analysis['issues_detected']:
                                    st.warning(f"• {issue}")
                            
                            st.markdown("**Suggestions:**")
                            for suggestion in refinement_analysis['refinement_suggestions']:
                                st.info(f"• {suggestion}")
                
                # Show AI Agent Activity Summary
                agents_used = []
                if use_prompt_agent: agents_used.append("Prompt Enhancement")
                if use_negative_agent: agents_used.append("Smart Negative Prompt")
                if use_quality_agent: agents_used.append("Quality Control")
                if use_param_agent: agents_used.append("Parameter Optimization")
                if use_style_agent: agents_used.append("Style Detection")
                if use_composition_agent: agents_used.append("Composition Analysis")
                if use_color_agent: agents_used.append("Color Harmony")
                if use_safety_agent: agents_used.append("Content Safety")
                if use_refinement_agent: agents_used.append("Iterative Refinement")
                
                if agents_used:
                    st.info(f"🤖 **{len(agents_used)} AI Agents Active:** {', '.join(agents_used)}")
                
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

# AI Agent Information Panel
st.markdown("---")
st.markdown("### 🤖 About the AI Agents (9 Intelligent Agents)")

col_agent1, col_agent2, col_agent3 = st.columns(3)

with col_agent1:
    st.markdown("""
    **🤖 Core Agents:**
    
    **Prompt Enhancement**
    - Analyzes prompt completeness
    - Adds quality keywords
    - Suggests lighting & style
    
    **Smart Negative Prompt**
    - Context-aware generation
    - Category-specific negatives
    - Prevents quality issues
    
    **Quality Control**
    - Detects image issues
    - Fixes brightness/contrast
    - Applies smart sharpening
    """)

with col_agent2:
    st.markdown("""
    **🎨 Creative Agents:**
    
    **Style Detection**
    - Analyzes content type
    - Suggests artistic style
    - Recommends lighting
    
    **Composition Analysis**
    - Optimal framing
    - Aspect ratio suggestions
    - Composition rules
    
    **Color Harmony**
    - Color palette recommendations
    - Temperature analysis
    - Saturation optimization
    """)

with col_agent3:
    st.markdown("""
    **🔧 Technical Agents:**
    
    **Parameter Optimization**
    - Analyzes complexity
    - Device optimization
    - Speed/quality balance
    
    **Content Safety**
    - Appropriateness check
    - Suggests alternatives
    - Ensures safe content
    
    **Iterative Refinement**
    - Quality scoring
    - Issue detection
    - Improvement suggestions
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 2rem;">
    <p><strong>AI Image Studio with Multi-Agent System</strong></p>
    <p>Powered by LCM + 9 Intelligent AI Agents</p>
    <p style="font-size: 0.9rem;">🤖 9 AI Agents • ⚡ 10x Faster • 🚀 GPU Optimized • 💻 CPU Friendly • 🎨 Smart Generation</p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">
        Core Agents: Prompt Enhancement • Smart Negative • Quality Control • Parameter Optimization<br/>
        Creative Agents: Style Detection • Composition Analysis • Color Harmony<br/>
        Technical Agents: Content Safety • Iterative Refinement
    </p>
</div>
""", unsafe_allow_html=True)
