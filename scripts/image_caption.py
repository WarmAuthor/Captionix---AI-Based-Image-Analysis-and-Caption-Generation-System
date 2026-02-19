"""
scripts/image_caption.py
Generates caption for an image using BLIP (Salesforce/blip-image-captioning-base)
Usage:
  python image_caption.py --image sample.jpg
"""
import argparse
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

try:
    import streamlit as st
    @st.cache_resource(show_spinner=False)
    def load_model(device='cpu'):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        return processor, model
except ImportError:
    # Running outside Streamlit (e.g. CLI) — no caching
    def load_model(device='cpu'):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        return processor, model

def caption_image(image_path, device='cpu', max_length=50):
    processor, model = load_model(device)
    img = Image.open(image_path).convert('RGB')
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=max_length, num_beams=5)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    caption = caption_image(args.image, device=args.device)
    print("Caption:", caption)
