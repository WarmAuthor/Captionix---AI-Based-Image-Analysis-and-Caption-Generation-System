"""
app.py
Streamlit demo to upload an image and:
- Generate caption via BLIP
- Run YOLOv8 object detection (annotated image)
- Show predicted class (if classifier weights exist)
Run: streamlit run app.py
"""
import streamlit as st
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from scripts.image_caption import caption_image
from scripts.classifier import predict_image

# ── Cached model loaders ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_yolo(model_name='yolov8n.pt'):
    from ultralytics import YOLO
    return YOLO(model_name)

# ── Helper: run YOLO and return (annotated PIL image, num detections) ─────────

def run_detection_pil(pil_img, model_name='yolov8n.pt'):
    tmp_path = 'tmp_input.jpg'
    pil_img.save(tmp_path)
    model = load_yolo(model_name)
    results = model.predict(source=tmp_path, conf=0.25, save=False, verbose=False)
    num_objects = len(results[0].boxes)
    annotated_bgr = results[0].plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    return Image.fromarray(annotated_rgb), num_objects

# ── Transform for classifier ──────────────────────────────────────────────────

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── App UI ────────────────────────────────────────────────────────────────────

st.title("AI-Powered Smart Image Analyzer & Caption Generator")
st.markdown("Covers **Data Science** (EDA), **ML** (classifier), **Computer Vision** (YOLO detection), and **Generative AI** (BLIP captioning).")

uploaded = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("---")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 1. Image Captioning ───────────────────────────────────────────────────
    st.header("🤖 Generative AI — Image Captioning")
    with st.spinner("Generating caption with BLIP..."):
        tmp_caption = 'tmp_caption.jpg'
        img.save(tmp_caption)
        caption = caption_image(tmp_caption, device=device)
    st.success(f"**Caption:** {caption}")

    st.write("---")

    # ── 2. Object Detection ───────────────────────────────────────────────────
    st.header("👁️ Computer Vision — Object Detection (YOLOv8)")
    with st.spinner("Running YOLOv8 detection..."):
        det_img, num_objects = run_detection_pil(img)

    if num_objects == 0:
        st.info("No objects detected in this image (YOLOv8 is trained on COCO classes: people, cars, animals, etc.).")
    else:
        st.success(f"**{num_objects} object(s) detected.**")
    st.image(det_img, caption="YOLOv8 Detections", use_container_width=True)

    st.write("---")

    # ── 3. Classification ─────────────────────────────────────────────────────
    st.header("🧠 ML — Image Classification (CIFAR-10)")
    weights_path = 'models/cnn_classifier.pth'
    if os.path.exists(weights_path):
        t = test_transform(img)
        try:
            predicted = predict_image(weights_path, t, device=device, arch='resnet50')
            st.success(f"**Predicted class:** {predicted}")
        except Exception as e:
            st.error(f"Error running classifier: {e}")
    else:
        st.warning("No classifier weights found. Train the model first:\n```\npython scripts/classifier.py --train\n```")

    st.write("---")

    # ── 4. Data Science Notes ─────────────────────────────────────────────────
    st.header("📊 Data Science — Quick Notes")
    st.info("Run `python scripts/data_analysis.py` to visualize class distribution & samples from the `data/` folder.")

else:
    st.info("⬆️ Upload an image to start. Supported formats: JPG, JPEG, PNG.")
