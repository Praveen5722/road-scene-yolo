"""Streamlit interface for the medical-image research prototype."""
from pathlib import Path
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from medical_cnn.model import RegularizedCNN

CHECKPOINT=Path("artifacts/medical_cnn.pt")
PREPROCESS=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])

@st.cache_resource
def load_model(path):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload=torch.load(path,map_location=device,weights_only=False)
    classes=payload["classes"]
    model=RegularizedCNN(len(classes),float(payload.get("dropout",.4)))
    model.load_state_dict(payload["model_state_dict"]); model.to(device).eval()
    return model,classes,device

st.set_page_config(page_title="Medical Image Research Demo",page_icon="🧬",layout="centered")
st.title("Computer-Aided Medical Image Diagnosis")
st.caption("Regularized Convolutional Neural Networks • Educational research prototype")
st.warning("Not for clinical use. This prototype is not medically validated and cannot diagnose or replace a qualified clinician.")
if not CHECKPOINT.exists():
    st.info("No trained model found. Prepare a labeled dataset and train the model.")
    st.code("python scripts/train_medical.py --data data/medical_images --epochs 20")
    st.stop()
uploaded=st.file_uploader("Upload a medical image",type=["png","jpg","jpeg","bmp"])
if uploaded:
    try: image=Image.open(uploaded).convert("RGB")
    except Exception: st.error("Could not read this image file."); st.stop()
    st.image(image,caption="Uploaded image",use_container_width=True)
    if st.button("Run research classification",type="primary"):
        model,classes,device=load_model(str(CHECKPOINT))
        with torch.no_grad(): probs=torch.softmax(model(PREPROCESS(image).unsqueeze(0).to(device)),dim=1)[0].cpu()
        order=torch.argsort(probs,descending=True); idx=int(order[0])
        st.subheader("Model output — not a diagnosis")
        st.metric("Top predicted dataset label",classes[idx],f"{probs[idx].item():.1%} model score")
        st.caption("Scores are model outputs, not calibrated clinical probabilities.")
        st.bar_chart({classes[int(i)]:float(probs[int(i)]) for i in order})
        st.caption(f"Inference device: {device}")
