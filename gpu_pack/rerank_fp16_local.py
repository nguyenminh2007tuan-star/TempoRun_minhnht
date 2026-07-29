"""Rerank Qwen3-VL-Reranker-8B ở bf16 FULL (không nén) trên 1 GPU >=24GB.

MỤC ĐÍCH: bản 78.2 dùng int8 (nén, 30k điểm dồn vào ~1.7k giá trị). bf16 giữ độ
phân giải điểm đầy đủ -> tie-break tốt hơn ở các task near-tie. Chạy 1 card lớn nên
KHÔNG trải 2 GPU -> hết bug meta-tensor của Kaggle T4x2.

CHUẨN BỊ (trên box GPU thuê, ví dụ RTX 4090 / A5000 / A6000 / A100):
  pip install -r requirements.txt
  # frames.tar đã kèm -> giải nén ra ./frames (28833 ảnh phẳng {video}__{ms}.jpg)
  tar xf frames.tar            # tạo thư mục frames/
  python rerank_fp16_local.py  # ~30-60 phút; ra scores_fp16_d100.json + submission_fp16_d100.zip

Tải scores_fp16_d100.json về máy -> gửi lại để blend/so với 78.2.
"""
import glob, json, os, sys, time, zipfile
import numpy as np
import torch

# ---- đường dẫn local (cùng thư mục) ----
HERE       = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(HERE, "frames")
POOL       = os.path.join(HERE, "candidates_SQ8_d100.json")  # pool fusion lại bằng QUERY fp16
                                                                 # (đổi 31/300 rank-1 so pool int8). Đổi về
                                                                 # candidates_SQ8_d100.json nếu muốn pool cũ.
TASKS      = os.path.join(HERE, "public_round_tasks.jsonl")
DEPTH      = 100
REPO       = "Qwen/Qwen3-VL-Reranker-8B"
LAB_COMMIT = "b212dc8c91a8164aef1ea2de9c1a867611e75c04"
OUT        = os.path.join(HERE, "submission_fp16_d100.json")
BATCH      = 8   # 24GB đủ; giảm xuống 4 nếu OOM
# ----------------------------------------

assert torch.cuda.is_available(), "không thấy GPU"
free, total = torch.cuda.mem_get_info()
print(f"[gpu] {torch.cuda.get_device_name(0)} | trống {free/1e9:.1f}/{total/1e9:.1f}G", flush=True)
if total/1e9 < 22:
    print("  !!! CẢNH BÁO: card < 24GB, bf16 8B (~16GB weight + activation) có thể OOM. "
          "Nếu OOM: giảm BATCH, hoặc thuê card to hơn.", flush=True)

# ---- index ảnh phẳng ----
IMG = {os.path.basename(p): p for p in glob.glob(os.path.join(FRAMES_DIR, "*.jpg"))}
print(f"[frames] {len(IMG)} ảnh trong {FRAMES_DIR}", flush=True)
assert len(IMG) > 1000, f"thiếu frames — đã 'tar xf frames.tar' chưa? (thấy {len(IMG)})"

# ---- tải model + vá modules.json (bug trong repo Qwen) ----
from huggingface_hub import snapshot_download
print(f"[model] tải {REPO} (~18GB, tuỳ mạng)...", flush=True)
_t = time.time()
local_dir = snapshot_download(REPO, revision=LAB_COMMIT, max_workers=8)
print(f"[model] xong {(time.time()-_t)/60:.1f} phút -> {local_dir}", flush=True)

mp = os.path.join(local_dir, "modules.json")
mods = json.load(open(mp)); fixed = False
for m in mods:
    p = m.get("path", "")
    if p and not os.path.isdir(os.path.join(local_dir, p)):
        real = [d for d in os.listdir(local_dir)
                if os.path.isdir(os.path.join(local_dir, d)) and d.endswith("LogitScore")]
        if real:
            print(f"  vá modules.json: '{p}' -> '{real[0]}'", flush=True)
            m["path"] = real[0]; fixed = True
if fixed:                       # file là symlink tới blob dùng chung -> xoá rồi ghi mới
    os.remove(mp); json.dump(mods, open(mp, "w"), indent=2)

# ---- nạp bf16 1 card (KHÔNG device_map='auto' -> không meta tensor) ----
from sentence_transformers import CrossEncoder
mk = {"dtype": torch.float16, "device_map": {"": 0}}
print("[model] nạp bf16 (không lượng tử hoá, nhanh hơn int8)...", flush=True)
_t = time.time()
model = CrossEncoder(local_dir, model_kwargs=mk)
print(f"[model] nạp xong {(time.time()-_t)/60:.1f} phút | VRAM {torch.cuda.memory_allocated()/1e9:.1f}G", flush=True)

# ---- rerank ----
cands = json.load(open(POOL))
queries = {t["task_id"]: t["description"]
           for t in (json.loads(l) for l in open(TASKS))}

from tqdm.auto import tqdm
preds, all_scores, t0 = [], {}, time.time()
for ti, (task_id, cand_list) in enumerate(tqdm(sorted(cands.items()), desc="rerank bf16", unit="task")):
    cand_list = cand_list[:DEPTH]
    cand_list = [c for c in cand_list
                 if f"{c['video_id']}__{int(c['frame_ms'])}.jpg" in IMG]
    if not cand_list:
        continue
    names = [f"{c['video_id']}__{int(c['frame_ms'])}.jpg" for c in cand_list]
    pairs = [(queries[task_id], {"image": IMG[n]}) for n in names]
    sc = np.asarray(model.predict(
        pairs, batch_size=BATCH,
        prompt="Retrieve images or text relevant to the user's query."),
        dtype=np.float64)
    if np.isnan(sc).any():
        raise SystemExit("NaN — thử giảm BATCH")
    order = np.argsort(-sc)[:10]
    preds.append({"task_id": task_id,
                  "results": [{"rank": r, "video_id": cand_list[i]["video_id"],
                               "frame_ms": int(cand_list[i]["frame_ms"])}
                              for r, i in enumerate(order, 1)]})
    all_scores[task_id] = [{"video_id": c["video_id"], "frame_ms": int(c["frame_ms"]),
                            "rerank": float(s), "fusion": float(c.get("score", 0.0))}
                           for c, s in zip(cand_list, sc)]

json.dump({"predictions": preds}, open(OUT, "w"))
json.dump(all_scores, open(os.path.join(HERE, "scores_fp16_d100.json"), "w"))
with zipfile.ZipFile(os.path.join(HERE, "submission_fp16_d100.zip"), "w") as z:
    z.write(OUT, "submission.json")
print(f"\nXONG {(time.time()-t0)/60:.1f} phút -> scores_fp16_d100.json + submission_fp16_d100.zip", flush=True)
print("Tải scores_fp16_d100.json về gửi lại để so với nền 78.2.", flush=True)
