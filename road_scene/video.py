"""Video helpers: annotated output, per-frame counts, and unique-object counting."""
import cv2

def process_video(detector, path, out_path="annotated.mp4", stride=3, max_frames=300, progress=None, n_samples=4):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    limit = min(total, max_frames)
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (W, H))
    sample_every = max(1, (limit // stride) // max(1, n_samples))
    counts, samples, i = [], [], 0
    while i < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            res = detector.detect(frame)
            out.write(res.annotated)
            c = {}
            for d in res.detections:
                c[d.name] = c.get(d.name, 0) + 1
            counts.append(c)
            if (len(counts) - 1) % sample_every == 0 and len(samples) < n_samples:
                samples.append(res.annotated)
        i += 1
        if progress:
            progress(min(i / limit, 1.0))
    cap.release(); out.release()
    return {"counts": counts, "out_path": str(out_path), "samples": samples}

def count_unique(detector, path, max_frames=300):
    seen = {}
    stream = detector.model.track(source=str(path), conf=detector.conf, imgsz=detector.imgsz, classes=detector.class_ids, persist=True, stream=True, verbose=False)
    for k, r in enumerate(stream):
        if r.boxes.id is not None:
            for tid, c in zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
                seen.setdefault(r.names[c], set()).add(tid)
        if k >= max_frames:
            break
    return {n: len(s) for n, s in seen.items()}
