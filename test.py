"""
Streamlit app: Object detection + spatial relationships + scene captioning
--------------------------------------------------------------------------
• Detects objects with YOLOv8‑n
• Assigns unique labels ("chair 1", "chair 2", …) and draws bounding boxes
• Computes simple spatial relationships (above/below, left/right) between every
  detected pair of objects
• Generates a BLIP caption of the scene
• Shows annotated image, relationships list and caption in the UI
• Lets user download the annotated image

Requirements (install with pip):
    streamlit transformers torch pillow ultralytics

Run:
    streamlit run streamlit_yolo_blip_app.py
"""

import io
import time
from itertools import combinations

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipForConditionalGeneration, BlipProcessor
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Streamlit page configuration — MUST be the first Streamlit command!
# -----------------------------------------------------------------------------

st.set_page_config(page_title="YOLO‑BLIP Scene Analyzer", layout="centered")

# -----------------------------------------------------------------------------
# Model loading (cached so it only happens once per running session)
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=True)
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)
    yolo = YOLO("yolov8n.pt")  # nano model for speed – swap with yolov8s/m/l if needed
    return processor, model, yolo, device


processor, blip_model, yolo_model, DEVICE = load_models()

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def draw_detections(img: Image.Image, boxes, labels):
    """Draw red boxes + label text on a copy of the image and return it."""
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    # Try Arial; fallback to default pillow font if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for box, name in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text_y = y1 - 12 if y1 - 12 > 0 else y1 + 4
        draw.text((x1, text_y), name, fill="red", font=font)

    return draw_img


def compute_relationships(boxes, unique_labels):
    """Return list of spatial relationships strings for every pair of boxes."""
    rels = []
    for (i, b1), (j, b2) in combinations(enumerate(boxes), 2):
        l1, l2 = unique_labels[i], unique_labels[j]
        x11, y11, x12, y12 = b1
        x21, y21, x22, y22 = b2
        cx1, cy1 = (x11 + x12) / 2, (y11 + y12) / 2
        cx2, cy2 = (x21 + x22) / 2, (y21 + y22) / 2

        overlap_x = max(0, min(x12, x22) - max(x11, x21))
        overlap_y = max(0, min(y12, y22) - max(y11, y21))

        # Vertical relations (above/below) if x‑overlap is significant
        if overlap_x > 0.1 * min(x12 - x11, x22 - x21):
            if cy1 < cy2:
                rels.append(f"{l1} is above {l2}")
            elif cy1 > cy2:
                rels.append(f"{l1} is below {l2}")

        # Horizontal relations (left/right) if y‑overlap is significant
        if overlap_y > 0.1 * min(y12 - y11, y22 - y21):
            if cx1 < cx2:
                rels.append(f"{l1} is to the left of {l2}")
            elif cx1 > cx2:
                rels.append(f"{l1} is to the right of {l2}")

    return rels


def generate_caption(img: Image.Image):
    """Generate a BLIP caption for the image and return (caption, inference_time)."""
    inputs = processor(img, return_tensors="pt").to(DEVICE)
    start = time.time()
    generated_ids = blip_model.generate(
        **inputs, max_new_tokens=100, num_beams=5, repetition_penalty=1.5
    )
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    inference_time = time.time() - start
    return caption, inference_time


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.title("📸 YOLO‑BLIP Scene Analyzer")
st.markdown(
    "Upload an image to detect objects, find spatial relationships, and generate a scene caption."
)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image bytes and convert to PIL
    img = Image.open(uploaded_file).convert("RGB")

    # ---------------------------------------------------------------------
    # 1. Object detection
    # ---------------------------------------------------------------------
    with st.spinner("Running YOLOv8…"):
        results = yolo_model(img)  # ultralytics accepts PIL Image directly
        boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
        raw_labels = [yolo_model.names[int(c)] for c in results[0].boxes.cls]

    # Give each detection a unique name (chair 1, chair 2 …)
    name_counts = {}
    unique_labels = []
    for lbl in raw_labels:
        name_counts[lbl] = name_counts.get(lbl, 0) + 1
        unique_labels.append(f"{lbl} {name_counts[lbl]}")

    # ---------------------------------------------------------------------
    # 2. Draw detections & compute relationships
    # ---------------------------------------------------------------------
    annotated_img = draw_detections(img, boxes, unique_labels)
    relationships = compute_relationships(boxes, unique_labels)

    # ---------------------------------------------------------------------
    # 3. Generate BLIP caption
    # ---------------------------------------------------------------------
    caption, cap_time = generate_caption(img)

    # ---------------------------------------------------------------------
    # 4. UI Outputs
    # ---------------------------------------------------------------------
    st.subheader("Annotated Image")
    st.image(annotated_img, use_column_width=True)

    # Download button
    buf = io.BytesIO()
    annotated_img.save(buf, format="JPEG")
    st.download_button(
        label="Download annotated image",
        data=buf.getvalue(),
        file_name="labeled_output.jpg",
        mime="image/jpeg",
    )

    if relationships:
        st.subheader("Spatial Relationships 🧭")
        for r in relationships:
            st.write("- ", r)
    else:
        st.info("No spatial relationships detected.")

    st.subheader("Scene Description 📝")
    st.write(caption)
    st.caption(f"Inference time: {cap_time:.2f} s on {DEVICE}")

    st.success("Done!")
else:
    st.info("👈 Upload an image to get started.")
