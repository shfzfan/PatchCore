# -*- coding: utf-8 -*-
"""
PatchCore 阈值标定（无监督分位数阈值）：
- 输入：一批“正常图”文件夹
- 输出：img_threshold.json（包含 t_img、均值、方差、元信息）
- 要求：与 detect 时完全一致的 MODEL_DIR / 预处理尺寸 / backbone 等
"""

import os, sys, json, time
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# ========= 0) 基本路径 =========
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)   # 不改库源码也能 import patchcore.*

# ========= 1) 配置（按需修改） =========
# 模型目录：必须含 patchcore_params.pkl 和 nnscorer_search_index.faiss
MODEL_DIR = "/home/ubuntu/cummins_project/patchcore-inspection/pc_results/project/vanes_test/models/mvtec_vanes"

# 正常图文件夹（用于标定）
NORMAL_DIR_FOR_CALIB = "/home/ubuntu/cummins_project/patchcore-inspection/mvtec/vanes/train/good"

# 输出目录（会把阈值 JSON 保存到这里）
OUT_DIR = "/home/ubuntu/cummins_project/patchcore-inspection/pc_results/project/vanes_test/pred"
IMG_THR_JSON = os.path.join(OUT_DIR, "img_threshold.json")

# 预处理尺寸（务必与训练/推理一致）
RESIZE, IMAGESIZE = 366, 320

# DataLoader
BATCH_SIZE = 1
NUM_WORKERS = 0

# 设备/FAISS
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
FAISS_ON_GPU, FAISS_WORKERS = False, 8

# ========= 2) 依赖（库层不改） =========
from patchcore import patchcore, common

# ========= 3) 工具 =========
def list_images(path: str):
    assert os.path.isdir(path), f"正常图路径不是文件夹：{path}"
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    files = sorted(
        f for f in (os.path.join(path, x) for x in os.listdir(path))
        if os.path.isfile(f) and f.lower().endswith(exts)
    )
    if not files:
        raise RuntimeError(f"文件夹中没有找到图片: {path}")
    return files

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# ========= 4) 数据集与 collate =========
class GenericImageDataset(torch.utils.data.Dataset):
    IMNET_MEAN = (0.485, 0.456, 0.406)
    IMNET_STD  = (0.229, 0.224, 0.225)
    def __init__(self, paths, resize, imagesize):
        self.paths = paths
        self.imagesize = imagesize
        self.tf = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(self.IMNET_MEAN, self.IMNET_STD),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        x = self.tf(Image.open(p).convert("RGB"))
        # 与库中的 dataloader 对齐：返回 (image, label, mask)
        label = 0
        mask = torch.zeros((self.imagesize, self.imagesize), dtype=torch.uint8)
        return x, label, mask

def safe_collate(batch):
    xs, ys, ms = zip(*batch)
    xs = torch.stack([x if torch.is_tensor(x) else torch.as_tensor(x) for x in xs], dim=0)
    ys = torch.as_tensor(ys)
    ms = torch.stack(ms, dim=0) if torch.is_tensor(ms[0]) else torch.as_tensor(ms)
    return xs, ys, ms

# ========= 5) 核心：标定 & 保存 =========
def calibrate_image_threshold(pc, normal_dir, resize, imagesize, device, batch_size=1):
    paths = list_images(normal_dir)
    print(f"🔧 开始标定：正常图 {len(paths)} 张，来自 {normal_dir}")
    ds = GenericImageDataset(paths, resize, imagesize)
    ld = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=USE_CUDA, collate_fn=safe_collate
    )

    scores = []
    pc.eval()
    with torch.no_grad():
        for xb, _, _ in ld:
            if torch.is_tensor(xb) and xb.ndim == 3:
                xb = xb.unsqueeze(0)
            s, _ = pc._predict(xb.to(device).float())   # s: List[float]
            scores.extend([float(v) for v in s])

    scores = np.asarray(scores, dtype=np.float64)
    mu  = float(scores.mean())
    sd  = float(scores.std() + 1e-12)
    print(f"✅ 标定完成：μ={mu:.6f}  σ={sd:.6f}")
    return mu, sd, int(scores.size)

def save_img_threshold(json_path, mu, sd, meta: dict):
    meta = dict(meta or {})
    meta.update({
        "mean_img": float(mu),
        "std_img": float(sd),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    ensure_dir(os.path.dirname(json_path))
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"💾 已保存阈值 JSON → {json_path}")

# ========= 6) 主流程 =========
def main():
    # 6.1 模型文件自检
    pp = os.path.join(MODEL_DIR, "patchcore_params.pkl")
    fa = os.path.join(MODEL_DIR, "nnscorer_search_index.faiss")
    if not (os.path.exists(pp) and os.path.exists(fa)):
        raise RuntimeError(f"[模型目录不完整]\n  {pp}\n  {fa}")

    # 6.2 构建 NN 方法
    try:
        nn_method = common.FaissNN(FAISS_ON_GPU, FAISS_WORKERS)
    except Exception as e:
        print(f"⚠️ FAISS GPU 初始化失败（改用CPU）：{e}")
        nn_method = common.FaissNN(False, FAISS_WORKERS)

    # 6.3 加载 PatchCore
    pc = patchcore.PatchCore(DEVICE)
    pc.load_from_path(load_path=MODEL_DIR, device=DEVICE, nn_method=nn_method)

    # 6.4 标定
    mu, sd, n = calibrate_image_threshold(
        pc, NORMAL_DIR_FOR_CALIB, RESIZE, IMAGESIZE, DEVICE, batch_size=BATCH_SIZE
    )

    # 6.5 保存 JSON（包含元信息便于追踪）
    meta = {
        "method": "quantile",
        "num_normals": n,
        "model_dir": MODEL_DIR,
        "resize": RESIZE,
        "imagesize": IMAGESIZE,
        "use_cuda": bool(USE_CUDA),
    }
    save_img_threshold(IMG_THR_JSON, mu, sd, meta)

if __name__ == "__main__":
    if not USE_CUDA:
        print("ℹ️ 未检测到可用 GPU，将使用 CPU（FAISS 也走 CPU）。")
    main()
