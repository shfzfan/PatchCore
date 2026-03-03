## -*- coding: utf-8 -*-
"""
通用推理脚本（无需改库源码）：
- 加载已训练好的 PatchCore 模型（load_from_path）
- 输入：单张图片路径 或 图片文件夹
- 输出：灰度热力图、伪彩热力图、原图叠加、raw heatmap(.npy)、CSV（含分数与判定）
- 标定 JSON 仅包含 μ/σ；检测时通过 FPR 计算最终阈值 T_FINAL=μ+z(1−FPR)*σ
- 新增：
  * 打印每张图预测用时
  * Overlay 仅绘制原始热力图 Top 50% 像素（阈值在原始 seg 上，以中位数划分；不归一化参与挑点）
  * 在 Overlay 上标注判定与分数信息（decision/score/mean/z）
"""

import os, sys, json, time, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
import cv2 as cv

# ========= 0) BASE ROOT =========
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR) 

# ========= 1) DIRECTION =========
MODEL_DIR  = "/home/ubuntu/cummins_project/patchcore-inspection/pc_results/project/metal_nut/models/mvtec_metal_nut"
INPUT_PATH = "/home/ubuntu/cummins_project/patchcore-inspection/mvtec/metal_nut/test/good"
OUT_DIR    = "/home/ubuntu/cummins_project/patchcore-inspection/pc_results/project/metal_nut/pred"
IMG_THR_JSON = os.path.join(OUT_DIR, "img_threshold.json")

RESIZE, IMAGESIZE = 366, 320
NORM_MODE   = "dataset" 
OVERLAY_ALPHA = 0.20         # overlay
USE_PATCHCORE_COLORMAP = True
PC_COLORMAP = "jet"

BATCH_SIZE, NUM_WORKERS = 1, 0
USE_CUDA   = torch.cuda.is_available()
DEVICE     = torch.device("cuda:0" if USE_CUDA else "cpu")
FAISS_ON_GPU, FAISS_WORKERS = USE_CUDA, 8

SHOW_RAW = False
SHOW_MAX = 4

# —— confirm FPR in detect  —— #
FPR_OVERRIDE = 0.00002   

# ========= 2) supply =========
from patchcore import patchcore, common

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# ========= 3) 工具 =========
def list_images(path: str):
    if os.path.isdir(path):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
        files = sorted(
            f for f in (os.path.join(path, x) for x in os.listdir(path))
            if os.path.isfile(f) and f.lower().endswith(exts)
        )
        if not files:
            raise RuntimeError(f"文件夹中没有找到图片: {path}")
        return files
    elif os.path.isfile(path):
        return [path]
    else:
        raise RuntimeError(f"输入路径不存在: {path}")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def colorize_heatmap01(h01: np.ndarray) -> np.ndarray:
    """把[0,1]热力图转 RGB uint8；优先用 matplotlib 'jet' 对齐官方风格。"""
    h01 = np.clip(h01, 0.0, 1.0)
    if USE_PATCHCORE_COLORMAP and HAVE_MPL:
        cmap = plt.get_cmap(PC_COLORMAP)
        rgb01 = cmap(h01)[..., :3]
        return (rgb01 * 255.0).astype(np.uint8)
    h255 = (h01 * 255.0).astype(np.uint8)
    if HAVE_CV2:
        color = cv2.applyColorMap(h255, cv2.COLORMAP_JET)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        return color
    color = np.zeros((*h01.shape, 3), dtype=np.uint8); color[..., 0] = h255
    return color

def overlay(img_rgb_u8: np.ndarray, heat_rgb_u8: np.ndarray, alpha=0.45) -> np.ndarray:
    out = img_rgb_u8.astype(np.float32) * (1 - alpha) + heat_rgb_u8.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)

def draw_text_on_image(img_rgb_u8: np.ndarray, text: str, color=(0,255,0)) -> np.ndarray:
    """在图像左上角绘制文本。优先用 OpenCV；否则用 PIL。"""
    if HAVE_CV2:
        out = img_rgb_u8.copy()
        # 背景条
        cv2.rectangle(out, (10,10), (10+520, 10+60), (0,0,0), thickness=-1)
        cv2.putText(out, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        return out
    # PIL 路径
    pil = Image.fromarray(img_rgb_u8)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    # 背景框
    bbox = draw.textbbox((0,0), text, font=font)
    bg_w, bg_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.rectangle([10, 10, 10+bg_w+20, 10+bg_h+20], fill=(0,0,0))
    draw.text((20,20), text, fill=color, font=font)
    return np.array(pil)

def load_mu_sigma(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    mu = float(data["mean_img"])
    sd = float(data["std_img"])
    n  = int(data.get("n", 0))
    print(f"📥 read threashold: μ={mu:.6f}, σ={sd:.6f}, n={n}  From {json_path}")
    return mu, sd, n, data

def _norm_ppf(p: float) -> float:
    """标准正态分布的逆 CDF 近似（Acklam 近似），用于把 FPR 换算为 z 值。"""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

# ========= 4) 数据集与安全 collate =========
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
        x = self.tf(Image.open(p).convert("RGB"))  # -> Tensor [3,H,W]
        label = 0
        mask = torch.zeros((self.imagesize, self.imagesize), dtype=torch.uint8)  # 占位
        return x, label, mask

def safe_collate(batch):
    xs, ys, ms = zip(*batch)
    xs = [x if torch.is_tensor(x) else torch.as_tensor(x) for x in xs]
    xs = torch.stack(xs, dim=0)  # [B,3,H,W]
    ys = torch.as_tensor(ys)
    ms = torch.stack(ms, dim=0) if torch.is_tensor(ms[0]) else torch.as_tensor(ms)
    return xs, ys, ms

# ========= 5) 主流程 =========
def main():
    # 5.1 模型自检
    pp = os.path.join(MODEL_DIR, "patchcore_params.pkl")
    fa = os.path.join(MODEL_DIR, "nnscorer_search_index.faiss")
    if not (os.path.exists(pp) and os.path.exists(fa)):
        raise RuntimeError(f"[模型目录不完整]\n  {pp}\n  {fa}")

    # 5.2 输入与输出
    paths = list_images(INPUT_PATH)
    out_heat    = ensure_dir(os.path.join(OUT_DIR, "heat"))
    out_color   = ensure_dir(os.path.join(OUT_DIR, "heat_color"))
    out_overlay = ensure_dir(os.path.join(OUT_DIR, "overlay"))
    out_npy     = ensure_dir(os.path.join(OUT_DIR, "heat_npy"))
    out_csv     = ensure_dir(OUT_DIR)  # CSV 放 OUT_DIR 根目录

    # 5.3 DataLoader（含安全 collate）
    ds = GenericImageDataset(paths, RESIZE, IMAGESIZE)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=USE_CUDA,
        collate_fn=safe_collate
    )
    xb, _, _ = next(iter(loader))
    assert torch.is_tensor(xb), f"images 应为 Tensor，实际是 {type(xb)}"

    # 5.4 NN 检索（GPU 不可用自动退 CPU）
    try:
        nn_method = common.FaissNN(FAISS_ON_GPU, FAISS_WORKERS)
    except Exception as e:
        print(f"⚠️ FAISS GPU 初始化失败（改用CPU）：{e}")
        nn_method = common.FaissNN(False, FAISS_WORKERS)

    # 5.5 加载模型
    pc = patchcore.PatchCore(DEVICE)
    pc.load_from_path(load_path=MODEL_DIR, device=DEVICE, nn_method=nn_method)

    # 5.6 读取 μ/σ，并根据 FPR 计算最终阈值
    if not os.path.exists(IMG_THR_JSON):
        raise FileNotFoundError(f"未找到阈值文件：{IMG_THR_JSON}")
    MU_IMG, SD_IMG, N_CALIB, _thr_meta = load_mu_sigma(IMG_THR_JSON)

    if FPR_OVERRIDE is None or not (0.0 < FPR_OVERRIDE < 1.0):
        raise ValueError("当前标定 JSON 不含 t_img，仅含 μ/σ。请在脚本中设置 FPR_OVERRIDE∈(0,1)，例如 0.01 表示 1%。")
    z = _norm_ppf(1.0 - FPR_OVERRIDE)      # 上侧分位 z 值
    eps = 1e-12
    T_FINAL = MU_IMG + z * max(SD_IMG, eps)
    print(f"Using FPR={FPR_OVERRIDE:.4f} → z={z:.3f} ⇒ T_FINAL={T_FINAL:.6f}")

    # 5.7 推理（记录每张图耗时）  # <<<
    all_scores, all_segs, all_times = [], [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            elif isinstance(batch, dict):
                images = batch.get("image", batch.get("images"))
            else:
                images = batch
            if isinstance(images, list):
                images = torch.stack(
                    [x if torch.is_tensor(x) else torch.as_tensor(x) for x in images],
                    dim=0
                )
            if torch.is_tensor(images) and images.ndim == 3:
                images = images.unsqueeze(0)

            t0 = time.time()                                         # <<<
            scores, segs = pc._predict(images.to(DEVICE).float())
            dt = time.time() - t0                                    # <<<
            per_image_ms = (dt / len(scores)) * 1000.0               # <<<
            all_times.extend([per_image_ms] * len(scores))           # <<< 为每张记录相同的 per-image 时间

            batch_scores = [float(s) for s in scores]
            batch_segs_np = []
            for s in segs:
                s_np = s.detach().cpu().numpy() if torch.is_tensor(s) else np.asarray(s)
                batch_segs_np.append(s_np)

            all_scores.extend(batch_scores)
            all_segs.extend(batch_segs_np)

            if SHOW_RAW and HAVE_MPL:
                to_show = min(len(batch_segs_np), SHOW_MAX)
                for b in range(to_show):
                    raw = batch_segs_np[b]
                    plt.figure(figsize=(4.2, 4))
                    plt.imshow(raw, cmap="jet")
                    plt.title(f"raw seg (score={batch_scores[b]:.3f})")
                    plt.colorbar(shrink=0.8); plt.axis("off")
                    plt.show()

    # 5.8 list -> ndarray（便于统一处理/保存）
    def stack_to_numpy(seg_list):
        arrs = [np.asarray(s) for s in seg_list]
        try:
            return np.stack(arrs, axis=0)  # [N,H,W]
        except Exception as e:
            shapes = [np.asarray(s).shape for s in seg_list]
            raise RuntimeError(f"无法 np.stack，元素形状不一致: {shapes}") from e

    segs_raw = stack_to_numpy(all_segs)   # [N,H,W] 原始热图，不归一化  # <<<
    segs = segs_raw.copy()

    # 5.9 （仅用于可视化着色）归一化到 [0,1] —— 但“Top50% 挑点”使用 segs_raw  # <<<
    if NORM_MODE == "dataset":
        smin, smax = float(segs.min()), float(segs.max())
        segs01 = (segs - smin) / (smax - smin + 1e-12)
        per_image_norm = False
    else:
        segs01 = np.empty_like(segs, dtype=np.float32)
        for i in range(segs.shape[0]):
            h = segs[i]
            segs01[i] = (h - h.min()) / (h.max() - h.min() + 1e-12)
        per_image_norm = True

    # 5.10 保存结果 + 打印/CSV 判定
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_csv, f"detect_results_{timestamp}.csv")
    csv_cols = ["filename", "score", "z", "t_final", "decision", "time_ms"]  # <<<
    csv_lines = [",".join(csv_cols) + "\n"]

    for i, img_path in enumerate(paths):
        img0 = np.array(Image.open(img_path).convert("RGB"))
        H, W = img0.shape[:2]

        # ---- 1) 可视化着色图（用 segs01）----
        h01 = segs01[i]
        if HAVE_CV2:
            h01_up = cv2.resize(h01, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            h01_up = np.array(
                Image.fromarray((h01 * 255).astype(np.uint8)).resize((W, H))
            ) / 255.0
        heat_color = colorize_heatmap01(h01_up)
        over = overlay(img0, heat_color, alpha=OVERLAY_ALPHA)


        # # over 是 RGB np.uint8[H,W,3]
        # cv.imshow("overlay", cv.cvtColor(over, cv.COLOR_RGB2BGR))
        # cv.waitKey(0)                  # 按任意键关闭
        # cv.destroyAllWindows()

        
        # ---- 3) 判定/标注 ----
        s = all_scores[i]
        z_score  = (s - MU_IMG) / (SD_IMG if SD_IMG > 0 else 1e-12)
        decision = "DEFECT" if (s >= T_FINAL) else "OK"
        color = (255, 0, 0) if decision == "DEFECT" else (0, 255, 0)  # BGR? 我们用 RGB 顺序
        text = f"{decision} | score={s:.3f}  mean={MU_IMG:.3f}  std={SD_IMG:.3f} z={z_score:.2f}"
        over_anno = draw_text_on_image(over, text, color=color)        # <<<
        
        # ---- 4) 保存各类结果 ----
        stem = os.path.splitext(os.path.basename(img_path))[0]
        Image.fromarray((h01_up * 255).astype(np.uint8)).save(os.path.join(out_heat, f"{stem}_heat.png"))
        Image.fromarray(heat_color).save(os.path.join(out_color, f"{stem}_heat_color.png"))
        Image.fromarray(over_anno).save(os.path.join(out_overlay, f"{stem}_overlay.png"))
        np.save(os.path.join(out_npy, f"{stem}_heat.npy"), h01_up.astype(np.float32))

        # ---- 5) 打印与 CSV ----
        per_image_ms = all_times[i]
        print(f"[{i+1}/{len(paths)}] time={per_image_ms:.1f}ms  score={s:.6f}, z={z_score:.2f} "
              f"(T={T_FINAL:.6f}, norm={'per_image' if per_image_norm else 'dataset'}) "
              f"=> {decision} -> {stem}")
        csv_lines.append(f"{os.path.basename(img_path)},{s:.6f},{z_score:.3f},{T_FINAL:.6f},{decision},{per_image_ms:.1f}\n")

    with open(csv_path, "w") as f:
        f.writelines(csv_lines)

    print(f"\n✅ Done！Have Processed {len(paths)} image. Results have saved into：\n"
          f"  Heat (gray):   {out_heat}\n"
          f"  Heat (color):  {out_color}\n"
          f"  Overlay:       {out_overlay}\n"
          f"  Raw .npy:      {out_npy}\n"
          f"  CSV:           {csv_path}\n")

if __name__ == "__main__":
    if not USE_CUDA:
        print("ℹ️ 未检测到可用 GPU，将使用 CPU（FAISS 也走 CPU）。")
    main()
