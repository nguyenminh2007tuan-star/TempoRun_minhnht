"""Embed SUB-QUERIES private bằng SigLIP2 (khớp CHÍNH XÁC ClipModel.encode_texts) — chạy Kaggle/GPU.

VÌ SAO: fusion mặc định encode SigLIP subs ON-THE-FLY. Nhưng trec26 đang kẹt đĩa/env,
nên precompute luôn trên Kaggle → fusion chỉ còn matmul (không nạp model). Qwen8B đã
precompute riêng (embed_qwen8b_private / kaggle_embed_queries). Cái này lo phần SigLIP.

KHỚP REGIME PUBLIC:
  - Model ViT-SO400M-14-SigLIP2-378, pretrained webli, precision fp16 (y enc_SQ8).
  - Subs LẤY TỪ decomposed_queries_private.json (LLM Qwen3.5-4B ~3.9 sub/task) — PHẢI
    có file này trước (decompose xong). Thứ tự subs y hệt fusion dựng: task theo file,
    sub theo list.
  - L2-normalize, encode_text -> fp32. (giống hệt ClipModel.encode_texts)

OUTPUT: siglip2_subs_private.npz gồm 3 mảng SONG SONG (căn theo KEY, không theo vị trí):
  emb (Nsub, D) fp16 | sub_task_id (Nsub,) | sub_pos (Nsub,)
Fusion (đã vá) nạp file này, tra (task_id, pos) -> vector, dựng Q theo đúng subq order.

CHUẨN BỊ (Kaggle): Add Input dataset có private_round_tasks.jsonl + decomposed_queries_private.json.
  pip install open_clip_torch  (Kaggle thường có sẵn)
"""
import json, os, sys, subprocess, time
import numpy as np

# open_clip + phiên bản open_clip_torch có ViT-SO400M-14-SigLIP2-378/webli (>=2.24)
try:
    import open_clip  # noqa: F401
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "open_clip_torch"], check=True)

TASKS   = "private_round_tasks.jsonl"
DECOMP  = "decomposed_queries_private.json"
MODEL   = "ViT-SO400M-14-SigLIP2-378"
PRETRAIN= "webli"
OUT     = "siglip2_subs_private.npz"

def find(fn):
    for root, _, files in os.walk("/kaggle/input"):
        if fn in files: return os.path.join(root, fn)
    if os.path.exists(fn): return fn
    raise FileNotFoundError(f"không thấy {fn} — Add Input dataset (Kaggle) hoặc để cạnh script")

tasks = [json.loads(l) for l in open(find(TASKS))]
deco  = json.load(open(find(DECOMP)))

# dựng subs THEO ĐÚNG THỨ TỰ fusion: task theo file, sub theo list (fallback split nếu thiếu)
# (Kaggle chạy dạng cell nên KHÔNG có __file__; retrieval_core cũng không có trên Kaggle
#  -> fallback split đơn giản. Thực tế mọi task đều có trong decompose nên nhánh này hiếm dùng.)
try:
    import re as _re
    def split_sentences(t):
        parts = [p.strip() for p in _re.split(r'(?<=[.!?])\s+', t.strip()) if p.strip()]
        return parts or [t.strip()]
except Exception:
    def split_sentences(t): return [t]

sub_text, sub_task_id, sub_pos = [], [], []
for t in tasks:
    entry = deco.get(t["task_id"])
    subs = entry["subs"] if entry else split_sentences(t["description"])
    for j, s in enumerate(subs):
        sub_text.append(s); sub_task_id.append(t["task_id"]); sub_pos.append(j)
print(f"[data] {len(tasks)} task -> {len(sub_text)} sub-queries ({len(sub_text)/len(tasks):.1f} avg)", flush=True)

import torch, open_clip
torch.backends.cudnn.enabled = False   # y ClipModel (né lỗi cuDNN GPU chung)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAIN)
model = model.to(dev, dtype=torch.float16).eval()
tok = open_clip.get_tokenizer(MODEL)
print(f"[model] {MODEL}/{PRETRAIN} fp16 on {dev}", flush=True)

t0 = time.time(); feats = []
B = 256
for i in range(0, len(sub_text), B):
    toks = tok(sub_text[i:i+B]).to(dev)
    with torch.no_grad():
        f = model.encode_text(toks)
        f = f / f.norm(dim=-1, keepdim=True)      # L2-normalize — GIỐNG HỆT ClipModel
        feats.append(f.float().cpu().numpy().astype(np.float32))
    if (i//B) % 4 == 0:
        print(f"  {min(i+B,len(sub_text))}/{len(sub_text)}", flush=True)
emb = np.concatenate(feats, 0)
print(f"[encode] {emb.shape} | norm TB {np.linalg.norm(emb,axis=1).mean():.4f} (phải ~1.0) | {time.time()-t0:.0f}s", flush=True)

np.savez(OUT,
         emb=emb.astype(np.float16),
         sub_task_id=np.array(sub_task_id),
         sub_pos=np.array(sub_pos, dtype=np.int32))
print(f"[done] {OUT} | {emb.shape[0]} sub-vectors x {emb.shape[1]} chiều -> tải về cho fusion", flush=True)
