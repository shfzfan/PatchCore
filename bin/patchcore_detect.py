"""
PatchCore ultra-min inference script (outputs only: heat + overlay + CSV)
- Decision: z = (score - mean_img) / std_img; z >= Z_THRESHOLD => DEFECT
- Visualization: OpenCV-only via cv2.applyColorMap (no matplotlib branch)
- No CPU/GPU auto switching; no FAISS GPU fallback (set DEVICE_STR/FAISS_ON_GPU manually)
"""

import os
import sys
import json
import time

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2

# ====== Manual settings: no CPU/GPU "auto handling" ======
DEVICE_STR = "cpu"          # set to "cuda:0" to use GPU
FAISS_ON_GPU = False        # set True to use FAISS GPU (if your environment supports it)
FAISS_WORKERS = 8

# ====== Main knobs you will change ======
MODEL_DIR    = "/home/ubuntu/cummins_project/patchcore-inspection/pc_results/project/metal_nut/models/mvtec_metal_nut"
INPUT_PATH   = "/home/ubuntu/cummins_project/patchcore-inspection/mvtec/metal_nut/test/good"
OUT_DIR      = "/home/ubuntu/cummins_project/patchcore-inspection/pc_results/project/metal_nut/pred"
IMG_THR_JSON = os.path.join(OUT_DIR, "img_threshold.json")

RESIZE, IMAGESIZE = 366, 320
OVERLAY_ALPHA = 0.20
Z_THRESHOLD = 4.0

# Overlay text (smaller font)
FONT_SCALE = 0.55
FONT_THICKNESS = 1
TEXT_ORG = (12, 26)# top-left origin (x, y)

# ====== Paths: ensure patchcore is importable (repo layout: repo_root/src/patchcore) ======
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from patchcore import patchcore, common  # noqa: E402


def list_images(path: str):
    if os.path.isdir(path):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
        return sorted(
            os.path.join(path, fn)
            for fn in os.listdir(path)
            if fn.lower().endswith(exts)
        )
    return [path]


def load_mu_sigma(json_path: str):
    with open(json_path, "r") as f:
        d = json.load(f)
    mu = float(d["mean_img"])
    sd = float(d["std_img"])
    return mu, max(sd, 1e-12)


def main():
    # ---- 0) IO ----
    img_paths = list_images(INPUT_PATH)
    os.makedirs(os.path.join(OUT_DIR, "heat"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "overlay"), exist_ok=True)

    # ---- 1) Threshold params (4z) ----
    mu, sd = load_mu_sigma(IMG_THR_JSON)

    # ---- 2) PatchCore init (fixed device/FAISS config; no auto switching) ----
    device = torch.device(DEVICE_STR)
    nn_method = common.FaissNN(FAISS_ON_GPU, FAISS_WORKERS)
    pc = patchcore.PatchCore(device)
    pc.load_from_path(load_path=MODEL_DIR, device=device, nn_method=nn_method)
    pc.eval()

    # ---- 3) Preprocess (match training) ----
    tf = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(IMAGESIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # ---- 4) CSV ----
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUT_DIR, f"dataresult_{ts}.csv")
    lines = ["filename,score,z,decision,time_ms\n"]

    out_heat = os.path.join(OUT_DIR, "heat")
    out_overlay = os.path.join(OUT_DIR, "overlay")

    with torch.no_grad():
        for idx, p in enumerate(img_paths, 1):
            stem = os.path.splitext(os.path.basename(p))[0]

            # original image
            img_rgb = np.array(Image.open(p).convert("RGB"))
            H, W = img_rgb.shape[:2]

            # model input
            x = tf(Image.fromarray(img_rgb)).unsqueeze(0).to(device).float()

            t0 = time.time()
            scores, segs = pc._predict(x)
            dt_ms = (time.time() - t0) * 1000.0

            score = float(scores[0])
            seg = segs[0].detach().cpu().numpy() if torch.is_tensor(segs[0]) else np.asarray(segs[0])

            # per-image heatmap min-max normalization (minimal)
            seg = seg.astype(np.float32)
            seg01 = (seg - seg.min()) / (seg.max() - seg.min() + 1e-12)

            # resize back to original size -> grayscale heat
            h01_up = cv2.resize(seg01, (W, H), interpolation=cv2.INTER_LINEAR)
            heat_gray = (h01_up * 255.0).astype(np.uint8)

            # pseudocolor (OpenCV)
            heat_bgr = cv2.applyColorMap(heat_gray, cv2.COLORMAP_JET)
            heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

            # overlay
            over = (img_rgb.astype(np.float32) * (1.0 - OVERLAY_ALPHA) +
                    heat_rgb.astype(np.float32) * OVERLAY_ALPHA)
            over = np.clip(over, 0, 255).astype(np.uint8)

            # decision: 4z
            z = (score - mu) / sd
            decision = "DEFECT" if (z >= Z_THRESHOLD) else "OK"

            # annotation (cv2 uses BGR)
            over_bgr = cv2.cvtColor(over, cv2.COLOR_RGB2BGR)
            text = f"{decision} | score={score:.3f}  z={z:.2f}"
            color_bgr = (0, 0, 255) if decision == "DEFECT" else (0, 255, 0)

            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
            x0, y0 = TEXT_ORG[0] - 6, TEXT_ORG[1] - th - 6
            cv2.rectangle(over_bgr, (x0, y0), (x0 + tw + 12, y0 + th + base + 12), (0, 0, 0), thickness=-1)
            cv2.putText(over_bgr, text, TEXT_ORG, cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE, color_bgr, FONT_THICKNESS, cv2.LINE_AA)

            over_anno = cv2.cvtColor(over_bgr, cv2.COLOR_BGR2RGB)

            # save
            Image.fromarray(heat_gray).save(os.path.join(out_heat, f"{stem}_heat.png"))
            Image.fromarray(over_anno).save(os.path.join(out_overlay, f"{stem}_overlay.png"))

            # CSV
            lines.append(f"{os.path.basename(p)},{score:.6f},{z:.3f},{decision},{dt_ms:.1f}\n")

            print(f"[{idx}/{len(img_paths)}] {decision} score={score:.6f} z={z:.2f} time={dt_ms:.1f}ms -> {stem}")

    with open(csv_path, "w") as f:
        f.writelines(lines)

    print("\n✅ Done")
    print(f"  Heat:    {out_heat}")
    print(f"  Overlay: {out_overlay}")
    print(f"  CSV:     {csv_path}")


if __name__ == "__main__":
    main()