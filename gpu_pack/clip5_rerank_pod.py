"""CLIP-RERANK: chấm lại top-20 bằng Qwen3-VL-Reranker-8B fp16, mỗi candidate là
DẢI 3 FRAME (t-2s | t | t+2s) ghép ngang — reranker được thấy chuyển động.

Đây là nâng cấp ĐẦU VÀO của chính tín hiệu mạnh nhất, không phải trộn tín hiệu ngoài
(8 thí nghiệm đã chứng minh mọi tín hiệu ngoài đều yếu hơn reranker).

CHUẨN BỊ trên pod (4090/24GB):
  scp clip_strips5.tar clip_top20_manifest.json public_round_tasks.jsonl requirements.txt clip_rerank_pod.py lên
  tar xf clip_strips5.tar          # -> ./clip_strips/{video}__{ms}.jpg (5976 dải)
  pip như rerank cũ (torch 2.6.0 cu124 + requirements.txt)
  python clip_rerank_pod.py       # ~15-20 phút
Ra: scores_clip5_top20.json — TẢI VỀ để phân tích, chưa nộp vội.
"""
import glob, json, os, time
import numpy as np
import torch

HERE       = os.path.dirname(os.path.abspath(__file__))
STRIPS     = os.path.join(HERE, "clip_strips5")
MANIFEST   = os.path.join(HERE, "clip_top20_manifest.json")
TASKS      = os.path.join(HERE, "public_round_tasks.jsonl")
REPO       = "Qwen/Qwen3-VL-Reranker-8B"
LAB_COMMIT = "b212dc8c91a8164aef1ea2de9c1a867611e75c04"
OUT        = os.path.join(HERE, "scores_clip5_top20.json")
BATCH      = 2        # dải 3-frame rộng gấp 3 -> token gấp ~3, hạ batch so với bản 8

assert torch.cuda.is_available()
free, total = torch.cuda.mem_get_info()
print(f"[gpu] {torch.cuda.get_device_name(0)} | trống {free/1e9:.1f}/{total/1e9:.1f}G", flush=True)

IMG = {os.path.basename(p): p for p in glob.glob(os.path.join(STRIPS, "*.jpg"))}
print(f"[strips] {len(IMG)}", flush=True)
assert len(IMG) > 5000, "chưa tar xf clip_strips5.tar?"

from huggingface_hub import snapshot_download
print("[model] tải/lấy cache...", flush=True)
local_dir = snapshot_download(REPO, revision=LAB_COMMIT, max_workers=8)

mp = os.path.join(local_dir, "modules.json")
mods = json.load(open(mp)); fixed = False
for m in mods:
    p = m.get("path", "")
    if p and not os.path.isdir(os.path.join(local_dir, p)):
        real = [d for d in os.listdir(local_dir)
                if os.path.isdir(os.path.join(local_dir, d)) and d.endswith("LogitScore")]
        if real:
            m["path"] = real[0]; fixed = True
if fixed:
    os.remove(mp); json.dump(mods, open(mp, "w"), indent=2)

from sentence_transformers import CrossEncoder
model = CrossEncoder(local_dir, model_kwargs={"dtype": torch.float16, "device_map": {"": 0}})
print("[model] fp16 1 card OK", flush=True)

manifest = json.load(open(MANIFEST))
queries = {t["task_id"]: t["description"] for t in (json.loads(l) for l in open(TASKS))}

from tqdm.auto import tqdm
out, t0 = {}, time.time()
PROMPT = ("Retrieve images or text relevant to the user's query. Each image shows "
          "five consecutive moments of one video, 4 seconds apart, left to right. The MIDDLE panel is the moment in question; the others are its before/after context.")
for tid, cands in tqdm(sorted(manifest.items()), unit="task"):
    names = [f"{v}__{ms}.jpg" for v, ms in cands]
    ok = [(n, (v, ms)) for n, (v, ms) in zip(names, cands) if n in IMG]
    pairs = [(queries[tid], {"image": IMG[n]}) for n, _ in ok]
    sc = np.asarray(model.predict(pairs, batch_size=BATCH, prompt=PROMPT), dtype=np.float64)
    if np.isnan(sc).any():
        raise SystemExit("NaN — giảm BATCH")
    out[tid] = [{"video_id": v, "frame_ms": int(ms), "clip_score": float(s)}
                for (_, (v, ms)), s in zip(ok, sc)]

json.dump(out, open(OUT, "w"))
print(f"\nXONG {(time.time()-t0)/60:.1f} phút -> scores_clip5_top20.json — tải về phân tích.", flush=True)
