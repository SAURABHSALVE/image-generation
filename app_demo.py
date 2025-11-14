import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import time
from PIL import ImageEnhance

st.set_page_config(page_title="AI Image Demo", page_icon="🎨", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
h1 {color: white; text-align: center; font-size: 3rem; margin: 2rem 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_tiny_model():
    """Smallest SD model - 500MB only!"""
    pipe = StableDiffusionPipeline.from_pretrained(
        "nota-ai/bk-sdm-small",
        torch_dtype=torch.float32,
        safety_checker=None
    ).to("cpu")
    pipe.enable_attention_slicing(1)
    return pipe

st.markdown("<h1>🎨 AI Image Demo</h1>", unsafe_allow_html=True)
st.info("⚡ Tiny model (500MB) - Fast download & generation!")

with st.spinner("Loading... (first time: 1-2 min)"):
    model = load_tiny_model()

st.success("✅ Ready!")

col1, col2 = st.columns(2)

with col1:
    prompt = st.text_area("What to generate:", "a beautiful sunset", height=100)
    steps = st.slider("Quality (steps)", 10, 25, 15)
    
    if st.button("🎨 Generate", use_container_width=True):
        if prompt:
            with st.spinner(f"Generating (~{steps * 2}s)..."):
                start = time.time()
                img = model(prompt, num_inference_steps=steps, guidance_scale=7.5).images[0]
                img = ImageEnhance.Sharpness(img).enhance(1.2)
                elapsed = time.time() - start
                
                st.session_state.image = img
                st.session_state.time = elapsed

with col2:
    if "image" in st.session_state:
        st.image(st.session_state.image, use_container_width=True)
        st.success(f"✅ Generated in {st.session_state.time:.1f}s")
        
        import io, base64
        buf = io.BytesIO()
        st.session_state.image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(f'<a href="data:image/png;base64,{b64}" download="image.png">📥 Download</a>', unsafe_allow_html=True)
    else:
        st.info("👈 Enter prompt and generate")
