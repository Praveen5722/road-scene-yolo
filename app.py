"""Streamlit UI for road scene understanding. Run: streamlit run app.py"""
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from road_scene.detector import RoadDetector
from road_scene.scene import analyze_scene, draw_zones
from road_scene.video import process_video, count_unique

ROOT = Path(__file__).parent
SAMPLE = ROOT / "assets" / "sample_street.jpg"
st.set_page_config(page_title="Road Scene Understanding", page_icon="🚗", layout="wide")

@st.cache_resource(show_spinner="Loading YOLO model...")
def get_detector(weights: str, imgsz: int) -> RoadDetector:
    return RoadDetector(weights, imgsz=imgsz)

def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def show_alerts(report):
    if not report.alerts:
        st.success("No hazards detected")
    for alert in report.alerts:
        {"danger": st.error, "warning": st.warning, "info": st.info}[alert.level](alert.message)

st.sidebar.title("⚙️ Settings")
weights = st.sidebar.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt"])
conf = st.sidebar.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
imgsz = st.sidebar.select_slider("Inference size", options=[320, 416, 640], value=640)
st.sidebar.caption("Detects road-relevant COCO/KITTI classes.")

st.title("🚗 Intelligent Road Scene Understanding")
st.caption("YOLOv8 object detection + rule-based scene analysis")
tab_img, tab_vid, tab_about = st.tabs(["📷 Image", "🎞️ Video", "ℹ️ About"])

with tab_img:
    source = st.radio("Image source", ["Sample image", "Upload"], horizontal=True)
    img = cv2.imread(str(SAMPLE)) if source == "Sample image" else None
    if source == "Upload":
        up = st.file_uploader("Upload a road photo", type=["jpg", "jpeg", "png"])
        if up is not None:
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.info("Choose the sample image or upload a photo to begin.")
    else:
        detector = get_detector(weights, imgsz)
        detector.conf = conf
        res = detector.detect(img)
        report = analyze_scene(res.detections, res.shape)
        show_zones = st.checkbox("Show zones and near line")
        shown = draw_zones(res.annotated) if show_zones else res.annotated
        left, right = st.columns([3, 2])
        with left:
            st.image(rgb(shown), caption="Detections")
            ok, buf = cv2.imencode(".png", shown)
            if ok:
                st.download_button("⬇️ Download annotated image", buf.tobytes(), "annotated.png", "image/png")
        with right:
            m1, m2 = st.columns(2)
            m1.metric("Objects found", len(res.detections))
            m2.metric("Inference time", f"{res.inference_ms:.0f} ms")
            st.subheader("Scene summary")
            st.write(report.summary)
            show_alerts(report)
            if report.objects:
                st.subheader("Objects")
                st.dataframe(pd.DataFrame([{"object": o.name, "zone": o.zone, "distance": o.distance, "confidence": round(o.conf, 2)} for o in report.objects]), hide_index=True)

with tab_vid:
    st.write("Upload a short dashcam/street clip. CPU inference is faster with a larger frame stride.")
    vid = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
    c1, c2 = st.columns(2)
    stride = c1.slider("Process every Nth frame", 1, 10, 3)
    max_frames = c2.slider("Max frames to read", 60, 900, 300, 30)
    unique = st.checkbox("Also count unique objects (tracking, slower)")
    if vid is not None and st.button("▶️ Run analysis", type="primary"):
        suffix = Path(vid.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(vid.read())
            tmp_path = tmp.name
        out_path = tmp_path + "_annotated.mp4"
        detector = get_detector(weights, imgsz)
        detector.conf = conf
        bar = st.progress(0.0, text="Analysing video...")
        out = process_video(detector, tmp_path, out_path, stride, max_frames, progress=lambda p: bar.progress(p, text="Analysing video..."))
        bar.empty()
        if not out["counts"]:
            st.error("No frames could be read from this video.")
        else:
            st.success(f"Processed {len(out['counts'])} frames")
            df = pd.DataFrame(out["counts"]).fillna(0)
            if not df.empty:
                st.subheader("Objects per frame over time")
                st.line_chart(df)
            if out["samples"]:
                st.subheader("Sample frames")
                cols = st.columns(len(out["samples"]))
                for col, frame in zip(cols, out["samples"]):
                    col.image(rgb(frame))
            if unique:
                with st.spinner("Tracking objects..."):
                    st.subheader("Unique objects in the clip")
                    st.json(count_unique(detector, tmp_path, max_frames))
            with open(out["out_path"], "rb") as f:
                st.download_button("⬇️ Download annotated video", f.read(), "annotated.mp4", "video/mp4")

with tab_about:
    st.markdown("""
**Pipeline:** image/video → YOLOv8 → road-class filtering → scene analysis → annotated output + alerts.

**Scene rules:** horizontal thirds define left/ahead/right; image geometry estimates near/far. Vulnerable road users ahead and near trigger a BRAKE alert; vehicles ahead and near trigger KEEP DISTANCE.
""")
    arch = ROOT / "assets" / "architecture.png"
    if arch.exists():
        st.image(str(arch), caption="System architecture")
    st.caption("Distance is estimated from image geometry only. Educational prototype, not a safety system.")
