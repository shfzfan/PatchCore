import os
import sys
import json
import time
import base64
import io
import re
import math
import shutil

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2
import zxingcpp

# =========================
# Manual settings
# =========================
DEVICE_STR = "cuda:0"
FAISS_ON_GPU = False
FAISS_WORKERS = 8

RESIZE = 366
IMAGESIZE = 320
OVERLAY_ALPHA = 0.20
Z_THRESHOLD = 2.0

FONT_SCALE = 0.55
FONT_THICKNESS = 1
TEXT_ORG = (12, 26)

AUTO_SCALE_TO_IMAGE_SIZE = False
LINE_THICKNESS = 3
POINT_RADIUS = 4

# =========================
# Base paths
# =========================
BASE_DIR = "/home/ubuntu/cummins_project/patchcore-inspection/test_set"
PICTURE_DIR = os.path.join(BASE_DIR, "picture")
TEMPLATE_JSON_PATH = os.path.join(BASE_DIR, "result_template.json")
RESULT_JSON_PATH = os.path.join(BASE_DIR, "result.json")

# =========================
# PatchCore points
# preprocess.json is used here
# =========================
POINT_CFG = {
    "1": {
        "name": "P1",
        "json_key": "point 1",
        "model_dir": os.path.join(BASE_DIR, "P1", "models", "mvtec_P1"),
        "thr_json": os.path.join(BASE_DIR, "P1", "img_threshold.json"),
        "preprocess_json": os.path.join(BASE_DIR, "P1", "preprocess.json"),
    },
    "2": {
        "name": "P2",
        "json_key": "point 2",
        "model_dir": os.path.join(BASE_DIR, "P2", "models", "mvtec_P2"),
        "thr_json": os.path.join(BASE_DIR, "P2", "img_threshold.json"),
        "preprocess_json": os.path.join(BASE_DIR, "P2", "preprocess.json"),
    },
    "3": {
        "name": "P3",
        "json_key": "point 3",
        "model_dir": os.path.join(BASE_DIR, "P3", "models", "mvtec_P3"),
        "thr_json": os.path.join(BASE_DIR, "P3", "img_threshold.json"),
        "preprocess_json": os.path.join(BASE_DIR, "P3", "preprocess.json"),
    },
    "4": {
        "name": "P4",
        "json_key": "point 4",
        "model_dir": os.path.join(BASE_DIR, "P4", "models", "mvtec_P4"),
        "thr_json": os.path.join(BASE_DIR, "P4", "img_threshold.json"),
        "preprocess_json": os.path.join(BASE_DIR, "P4", "preprocess.json"),
    },
    "6": {
        "name": "P6",
        "json_key": "point 6",
        "model_dir": os.path.join(BASE_DIR, "P6", "models", "mvtec_P6"),
        "thr_json": os.path.join(BASE_DIR, "P6", "img_threshold.json"),
        "preprocess_json": os.path.join(BASE_DIR, "P6", "preprocess.json"),
    },
}

# =========================
# 005 UR region
# =========================
UR_REGION_POINTS = [
    [919.1954022988506, 549.4252873563219],
    [1015.9770114942529, 638.1839080459771]
]
UR_PADDING = (0, 0)

# =========================
# Make patchcore importable
# =========================
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from patchcore import patchcore, common


def load_mu_sigma(json_path: str):
    with open(json_path, "r") as f:
        d = json.load(f)
    mu = float(d["mean_img"])
    sd = max(float(d["std_img"]), 1e-12)
    return mu, sd


def ensure_result_json():
    if not os.path.exists(TEMPLATE_JSON_PATH):
        raise FileNotFoundError(f"Template not found: {TEMPLATE_JSON_PATH}")

    if not os.path.exists(RESULT_JSON_PATH):
        shutil.copyfile(TEMPLATE_JSON_PATH, RESULT_JSON_PATH)


def load_result_json():
    ensure_result_json()
    with open(RESULT_JSON_PATH, "r") as f:
        return json.load(f)


def save_result_json(data: dict):
    with open(RESULT_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)


def find_image_by_token(token: str):
    stem = f"{int(token):03d}"
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]

    for ext in exts:
        path = os.path.join(PICTURE_DIR, stem + ext)
        if os.path.exists(path):
            return path
    return None


def rgb_array_to_base64_png(img_rgb: np.ndarray):
    buffer = io.BytesIO()
    Image.fromarray(img_rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def safe_filename(text: str):
    return re.sub(r'[^A-Za-z0-9._-]+', "_", text.strip())


def point_done(block):
    if not isinstance(block, dict):
        return False
    return (
        str(block.get("original_image", "")).strip() != "" and
        str(block.get("overlay_image", "")).strip() != "" and
        str(block.get("result", "")).strip() != ""
    )


def is_all_done(data: dict):
    if str(data.get("P/N", "")).strip() == "":
        return False
    if str(data.get("S/N", "")).strip() == "":
        return False

    required_points = ["point 1", "point 2", "point 3", "point 4", "point 6"]
    for key in required_points:
        if not point_done(data.get(key)):
            return False

    return True


def finalize_and_exit_if_done():
    data = load_result_json()
    if not is_all_done(data):
        return

    sn = str(data["S/N"]).strip()
    final_name = safe_filename(sn) + ".json"
    final_path = os.path.join(BASE_DIR, final_name)

    os.replace(RESULT_JSON_PATH, final_path)
    raise SystemExit(0)


# =========================
# preprocess.json
# 先 mask，再 crop
# =========================
def scale_points(points, sx, sy):
    return [[p[0] * sx, p[1] * sy] for p in points]


def draw_shape_on_mask(mask, shape, line_thickness=3, point_radius=4):
    shape_type = shape.get("shape_type", "polygon")
    points = shape.get("points", [])

    if not points:
        return

    if shape_type == "polygon":
        pts = np.array(points, dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)

    elif shape_type == "rectangle":
        if len(points) >= 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            x_min = int(round(min(x1, x2)))
            y_min = int(round(min(y1, y2)))
            x_max = int(round(max(x1, x2)))
            y_max = int(round(max(y1, y2)))
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, thickness=-1)

    elif shape_type == "circle":
        if len(points) >= 2:
            cx, cy = points[0]
            px, py = points[1]
            r = int(round(math.sqrt((px - cx) ** 2 + (py - cy) ** 2)))
            cv2.circle(mask, (int(round(cx)), int(round(cy))), r, 255, thickness=-1)

    elif shape_type == "linestrip":
        pts = np.array(points, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=line_thickness)

    elif shape_type == "line":
        if len(points) >= 2:
            p1 = tuple(np.array(points[0], dtype=np.int32))
            p2 = tuple(np.array(points[1], dtype=np.int32))
            cv2.line(mask, p1, p2, 255, thickness=line_thickness)

    elif shape_type == "point":
        x, y = points[0]
        cv2.circle(mask, (int(round(x)), int(round(y))), point_radius, 255, thickness=-1)


def apply_preprocess_rule_to_bgr(image_bgr, rule_json_path):
    if not os.path.exists(rule_json_path):
        return image_bgr

    with open(rule_json_path, "r", encoding="utf-8") as f:
        rule = json.load(f)

    img_h, img_w = image_bgr.shape[:2]
    src_w = rule["source_image"]["width"]
    src_h = rule["source_image"]["height"]

    if img_w != src_w or img_h != src_h:
        if not AUTO_SCALE_TO_IMAGE_SIZE:
            raise ValueError(
                f"Image size ({img_w}x{img_h}) does not match rule size ({src_w}x{src_h})"
            )
        sx = img_w / src_w
        sy = img_h / src_h
    else:
        sx = 1.0
        sy = 1.0

    # 1) 先在整图上画 mask
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for shape in rule["shapes"]:
        scaled_shape = {
            "shape_type": shape.get("shape_type", "polygon"),
            "points": scale_points(shape.get("points", []), sx, sy)
        }
        draw_shape_on_mask(
            mask,
            scaled_shape,
            line_thickness=LINE_THICKNESS,
            point_radius=POINT_RADIUS
        )

    # 2) 先 mask
    masked = np.zeros_like(image_bgr)
    masked[mask > 0] = image_bgr[mask > 0]

    # 3) 再 crop
    crop = rule["crop"]
    x = int(round(crop["x"] * sx))
    y = int(round(crop["y"] * sy))
    w = int(round(crop["width"] * sx))
    h = int(round(crop["height"] * sy))

    x = max(0, min(x, img_w))
    y = max(0, min(y, img_h))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    result = masked[y:y+h, x:x+w]
    return result


class PatchCoreManager:
    def __init__(self):
        self.device = torch.device(DEVICE_STR)
        self.models = {}

        self.tf = transforms.Compose([
            transforms.Resize(RESIZE),
            transforms.CenterCrop(IMAGESIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            ),
        ])

    def load_model(self, key: str):
        if key in self.models:
            return

        cfg = POINT_CFG[key]
        nn_method = common.FaissNN(FAISS_ON_GPU, FAISS_WORKERS)

        pc = patchcore.PatchCore(self.device)
        pc.load_from_path(
            load_path=cfg["model_dir"],
            device=self.device,
            nn_method=nn_method
        )
        pc.eval()

        mu, sd = load_mu_sigma(cfg["thr_json"])

        dummy = torch.zeros(1, 3, IMAGESIZE, IMAGESIZE, device=self.device)
        with torch.inference_mode():
            _ = pc._predict(dummy)

        self.models[key] = {
            "pc": pc,
            "mu": mu,
            "sd": sd,
            "name": cfg["name"],
            "json_key": cfg["json_key"],
            "preprocess_json": cfg["preprocess_json"],
        }

    def preload_all(self):
        for key in POINT_CFG:
            self.load_model(key)

    def infer_one(self, key: str, image_path: str):
        self.load_model(key)
        bundle = self.models[key]

        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # 先 preprocess: mask -> crop
        img_bgr_proc = apply_preprocess_rule_to_bgr(
            img_bgr,
            bundle["preprocess_json"]
        )

        img_rgb = cv2.cvtColor(img_bgr_proc, cv2.COLOR_BGR2RGB)

        x = self.tf(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device).float()

        t0 = time.time()
        with torch.inference_mode():
            scores, segs = bundle["pc"]._predict(x)
        dt_ms = (time.time() - t0) * 1000.0

        score = float(scores[0])
        seg = segs[0]
        if torch.is_tensor(seg):
            seg = seg.detach().cpu().numpy()
        else:
            seg = np.asarray(seg)

        z = (score - bundle["mu"]) / bundle["sd"]
        decision = "DEFECT" if z >= Z_THRESHOLD else "OK"

        return {
            "name": bundle["name"],
            "json_key": bundle["json_key"],
            "img_rgb": img_rgb,          # 这里已经是预处理后的图
            "seg": seg,
            "score": score,
            "z": z,
            "decision": decision,
            "time_ms": dt_ms,
        }


def build_overlay_rgb(result: dict):
    img_rgb = result["img_rgb"]
    seg = result["seg"]
    score = result["score"]
    z = result["z"]
    decision = result["decision"]
    point_name = result["name"]

    h, w = img_rgb.shape[:2]

    seg = seg.astype(np.float32)
    seg01 = (seg - seg.min()) / (seg.max() - seg.min() + 1e-12)
    seg_up = cv2.resize(seg01, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_gray = (seg_up * 255.0).astype(np.uint8)

    heat_bgr = cv2.applyColorMap(heat_gray, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    over = (
        img_rgb.astype(np.float32) * (1.0 - OVERLAY_ALPHA) +
        heat_rgb.astype(np.float32) * OVERLAY_ALPHA
    )
    over = np.clip(over, 0, 255).astype(np.uint8)

    over_bgr = cv2.cvtColor(over, cv2.COLOR_RGB2BGR)
    text = f"{point_name} | {decision} | score={score:.3f}  z={z:.2f}"
    color_bgr = (0, 0, 255) if decision == "DEFECT" else (0, 255, 0)

    (tw, th), base = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
    )
    x0, y0 = TEXT_ORG[0] - 6, TEXT_ORG[1] - th - 6
    cv2.rectangle(
        over_bgr,
        (x0, y0),
        (x0 + tw + 12, y0 + th + base + 12),
        (0, 0, 0),
        thickness=-1
    )
    cv2.putText(
        over_bgr,
        text,
        TEXT_ORG,
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        color_bgr,
        FONT_THICKNESS,
        cv2.LINE_AA
    )

    return cv2.cvtColor(over_bgr, cv2.COLOR_BGR2RGB)


def save_patchcore_result(result: dict):
    data = load_result_json()
    json_key = result["json_key"]

    overlay_rgb = build_overlay_rgb(result)
    final_result = "PASS" if result["decision"] == "OK" else "DEFECT"

    # original_image 现在存“预处理后的图片”
    data[json_key] = {
        "original_image": rgb_array_to_base64_png(result["img_rgb"]),
        "overlay_image": rgb_array_to_base64_png(overlay_rgb),
        "result": final_result
    }

    save_result_json(data)
    return final_result


# =========================
# 005 UR
# =========================
def points_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def expand_bbox(bbox, img_shape, pad_x, pad_y):
    x1, y1, x2, y2 = bbox
    h, w = img_shape[:2]
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return x1, y1, x2, y2


def crop_roi(img, bbox):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def detect_ur_from_crop(crop):
    qr = cv2.QRCodeDetector()
    text, _, _ = qr.detectAndDecode(crop)
    if text:
        return text

    variants = []

    variants.append(crop)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variants.append(gray)

    big = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(big)

    gray_big = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(binary)

    for img in variants:
        results = zxingcpp.read_barcodes(img)
        if results:
            return results[0].text

    return ""


def decode_ur_text_from_005(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return ""

    bbox = points_to_bbox(UR_REGION_POINTS)
    bbox = expand_bbox(bbox, img.shape, UR_PADDING[0], UR_PADDING[1])
    crop = crop_roi(img, bbox)
    if crop is None or crop.size == 0:
        return ""

    return detect_ur_from_crop(crop).strip()


def parse_pn_sn(decoded_text: str):
    parts = [x.strip() for x in decoded_text.split(";") if x.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def save_pn_sn(pn: str, sn: str):
    data = load_result_json()
    data["P/N"] = pn
    data["S/N"] = sn
    save_result_json(data)


def process_ur_command():
    image_path = find_image_by_token("5")
    if image_path is None:
        return

    decoded_text = decode_ur_text_from_005(image_path)
    if not decoded_text:
        return

    pn, sn = parse_pn_sn(decoded_text)
    if not pn or not sn:
        return

    save_pn_sn(pn, sn)
    finalize_and_exit_if_done()


def process_patchcore_command(manager: PatchCoreManager, token: str):
    image_path = find_image_by_token(token)
    if image_path is None:
        return

    result = manager.infer_one(token, image_path)
    save_patchcore_result(result)
    finalize_and_exit_if_done()


def process_command(manager: PatchCoreManager, token: str):
    token = token.strip()
    if not token:
        return

    if token == "5":
        process_ur_command()
        return

    if token in POINT_CFG:
        process_patchcore_command(manager, token)


def main():
    ensure_result_json()

    manager = PatchCoreManager()
    manager.preload_all()

    while True:
        s = input().strip()
        if not s:
            continue

        if s.lower() in ["q", "quit", "exit"]:
            break

        tokens = s.replace(",", " ").split()
        for token in tokens:
            process_command(manager, token)


if __name__ == "__main__":
    main()