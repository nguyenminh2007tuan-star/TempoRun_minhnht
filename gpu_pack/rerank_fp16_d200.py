"""Rerank fp16 DEPTH-200 — mở rộng pool từ 100 lên 200 candidate/task.

LÝ DO: nền fp16 d100 = 78.548 (R@10 0.96). 4% task còn lại nhiều khả năng frame đúng
nằm ở hạng 100-200 của fusion — rerank sâu hơn mới với tới được.

CHUẨN BỊ (trên pod, thư mục gpu_pack đã có sẵn frames d100):
  tar xf frames_d200_extra.tar     # thêm ~24.6k frame vào ./frames
  python rerank_fp16_d200.py       # ~60-70 phút (gấp đôi d100)

Ra: scores_fp16_d200.json + submission_fp16_d200.zip
"""
import glob, json, os, sys, time, zipfile
import numpy as np
import torch

HERE       = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(HERE, "frames")
POOL       = os.path.join(HERE, "candidates_SQ8_d200.json")
TASKS      = os.path.join(HERE, "public_round_tasks.jsonl")
DEPTH      = 200
REPO       = "Qwen/Qwen3-VL-Reranker-8B"
LAB_COMMIT = "b212dc8c91a8164aef1ea2de9c1a867611e75c04"
OUT        = os.path.join(HERE, "submission_fp16_d200.json")
BATCH      = 8

assert torch.cuda.is_available(), "không thấy GPU"
free, total = torch.cuda.mem_get_info()
print(f"[gpu] {torch.cuda.get_device_name(0)} | trống {free/1e9:.1f}/{total/1e9:.1f}G", flush=True)

IMG = {os.path.basename(p): p for p in glob.glob(os.path.join(FRAMES_DIR, "*.jpg"))}
print(f"[frames] {len(IMG)} ảnh trong {FRAMES_DIR}", flush=True)
assert len(IMG) > 50000, f"thiếu frames — đã 'tar xf frames_d200_extra.tar' chưa? (thấy {len(IMG)}, cần ~55k)"

from huggingface_hub import snapshot_download
print(f"[model] tải {REPO} (đã cache thì nhanh)...", flush=True)
local_dir = snapshot_download(REPO, revision=LAB_COMMIT, max_workers=8)

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
if fixed:
    os.remove(mp); json.dump(mods, open(mp, "w"), indent=2)

from sentence_transformers import CrossEncoder
mk = {"dtype": torch.float16, "device_map": {"": 0}}
print("[model] nạp fp16 1 card...", flush=True)
_t = time.time()
model = CrossEncoder(local_dir, model_kwargs=mk)
print(f"[model] nạp xong {(time.time()-_t)/60:.1f} phút | VRAM {torch.cuda.memory_allocated()/1e9:.1f}G", flush=True)

cands = json.load(open(POOL))
queries = {t["task_id"]: t["description"]
           for t in (json.loads(l) for l in open(TASKS))}

from tqdm.auto import tqdm
preds, all_scores, t0 = [], {}, time.time()
for ti, (task_id, cand_list) in enumerate(tqdm(sorted(cands.items()), desc="rerank fp16 d200", unit="task")):
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
json.dump(all_scores, open(os.path.join(HERE, "scores_fp16_d200.json"), "w"))
with zipfile.ZipFile(os.path.join(HERE, "submission_fp16_d200.zip"), "w") as z:
    z.write(OUT, "submission.json")
print(f"\nXONG {(time.time()-t0)/60:.1f} phút -> scores_fp16_d200.json + submission_fp16_d200.zip", flush=True)
