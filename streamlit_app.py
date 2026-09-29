import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

REPO_ID = "mohitbharti1530/deepfake-detector-model"
FILENAME = "deepfake_model.pth"

st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️")

# Same preprocessing as training (no Normalize)
tfm = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
])


@st.cache_resource(show_spinner="Model load ho raha hai...")
def load_model():
    torch.set_num_threads(2)
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)
    path = hf_hub_download(REPO_ID, FILENAME)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def p_fake(model, pil_img):
    x = tfm(pil_img.convert("RGB")).unsqueeze(0)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

    st.write("FAKE:", probs[0].item())
    st.write("REAL:", probs[1].item())

    return probs[0].item()


def show_result(pf):
    label = "FAKE" if pf > 0.5 else "REAL"
    conf = pf if pf > 0.5 else 1 - pf
    if label == "FAKE":
        st.error(f"Result: {label} ({conf:.1%} confidence)")
    else:
        st.success(f"Result: {label} ({conf:.1%} confidence)")
    st.progress(float(pf), text=f"Fake probability: {pf:.1%}")


st.title("Deepfake Detector")
st.write("EfficientNet-B4 based. Face image ya chhota video upload karo.")
st.caption(
    "Demo only: ~94% accuracy on the 140K Real and Fake Faces test set. "
    "Results may be less reliable on new generators or real-world videos."
)

model = load_model()
tab_img, tab_vid = st.tabs(["Image", "Video"])

with tab_img:
    up = st.file_uploader("Face image", type=["jpg", "jpeg", "png"], key="img")
    if up is not None:
        img = Image.open(up)
        st.image(img, width=300)
        with st.spinner("Check ho raha hai..."):
            pf = p_fake(model, img)
        show_result(pf)

with tab_vid:
    upv = st.file_uploader("Video (short, 5-15 sec)", type=["mp4", "mov", "avi"], key="vid")
    if upv is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(upv.read())
            tmp_path = tmp.name
        st.video(tmp_path)

        every_n, max_frames = 10, 20
        cap = cv2.VideoCapture(tmp_path)
        scores, i = [], 0
        bar = st.progress(0.0, text="Frames check ho rahe hain...")
        while cap.isOpened() and len(scores) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every_n == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                scores.append(p_fake(model, Image.fromarray(rgb)))
                bar.progress(len(scores) / max_frames, text="Frames check ho rahe hain...")
            i += 1
        cap.release()
        os.remove(tmp_path)
        bar.empty()

        if scores:
            show_result(float(np.mean(scores)))
            st.caption(f"{len(scores)} frames ka average liya gaya.")
        else:
            st.warning("Video se frames nahi mil paye.")
