"""Encode 300 query bang Qwen3-VL-Embedding-8B tren Kaggle (T4 x2) -> query_emb_qwen8b.npz.

VI SAO CAN CAI NAY:
  Ban da co index ANH Qwen3-VL-Embedding-8B (share qua Drive, khop 100% frame).
  No la bi-encoder nen de retrieve phai encode QUERY bang chinh model do -> ra
  vector 4096-chieu cung khong gian voi anh. File nay chi lam phan query (nhe).

KHOP RECIPE VOI PHIA ANH & voi index Qwen 2B cu trong retrieve_fusion.py:
  - SentenceTransformer, dtype fp16.
  - Query = FULL description (nguyen cau), KHONG phai sub-query.
  - normalize_embeddings=True. Khong them instruction thu cong (model tu xu ly).

VI SAO INT8 1 CARD (khong chia 2 GPU):
  8B fp16 ~16GB > 14.56GB mot T4. Da thu device_map="auto" chia 2 card nhung
  SentenceTransformer sau khi load lai goi .to(cuda:0) keo het ve GPU0 -> OOM
  (dinh 2 lan). int8 ~8GB VUA gon 1 T4, model luong tu hoa nen khong bi .to()
  keo di -> het duong hong. Sai so int8 ~0.5%, khong dang ke cho cosine retrieve.

CHUAN BI:
  1. Accelerator = GPU T4 x2 (hoac P100/T4 don deu duoc), Internet = ON
  2. Add Input: dataset co public_round_tasks.jsonl (dataset temporun-rerank cu).
  3. Chay ca cell. Xong tai /kaggle/working/query_emb_qwen8b.npz ve.
"""
import json, os, subprocess, sys, time

MODEL_ID   = "Qwen/Qwen3-VL-Embedding-8B"   # PHAI trung model phia anh da embed
TASKS_IN   = "private_round_tasks.jsonl"
OUT        = "/kaggle/working/query_emb_qwen8b_private.npz"
SKIP_INSTALL = False

WORK = "/kaggle/working"

def find(fname):
    for root, _, files in os.walk("/kaggle/input"):
        if fname in files:
            return os.path.join(root, fname)
    # fallback: co the nam ngay /kaggle/working
    if os.path.exists(os.path.join(WORK, fname)):
        return os.path.join(WORK, fname)
    raise FileNotFoundError(f"khong thay {fname} trong /kaggle/input — nho Add Input dataset")

if not SKIP_INSTALL:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "sentence-transformers", "transformers", "accelerate", "qwen-vl-utils",
                    "bitsandbytes"],
                   check=False)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # giam phan manh VRAM
import numpy as np
import torch

tasks = [json.loads(l) for l in open(find(TASKS_IN))]
task_ids = [t["task_id"] for t in tasks]
texts    = [t["description"] for t in tasks]          # FULL description, dung thu tu task
print(f"[data] {len(texts)} query | dai trung binh {int(np.mean([len(x) for x in texts]))} ky tu", flush=True)

import sentence_transformers as _stf
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig
# Chan cu .to() cua ST: model int8 da nam GPU0 qua device_map. Neu de ST goi .to()
# len model luong tu hoa se BAO LOI "cannot move 8-bit model". No-op la an toan vi
# khong can di chuyen gi. Cung la lop bao hiem cho moi phien ban ST tren Kaggle.
_stf.SentenceTransformer.to = lambda self, *a, **k: self
t0 = time.time()
# VI SAO 8-bit MOT CARD chu khong device_map 2 card:
#   8B fp16 = 16GB > 14.56GB mot T4 -> khong vua 1 card.
#   Nhung device_map='auto' 2 card thi SentenceTransformer sau khi load lai goi
#   .to(cuda:0) KEO HET model ve GPU0 -> nhoi day 14.5GB -> OOM (da dinh 2 lan).
#   8B int8 ~8GB VUA gon 1 T4 (con ~6GB cho activation), va vi model da luong tu hoa
#   nen transformers KHONG cho .to() -> ST bo qua buoc di chuyen -> het duong pha.
bnb = BitsAndBytesConfig(load_in_8bit=True)
st = SentenceTransformer(
    MODEL_ID,
    trust_remote_code=True,
    model_kwargs={"quantization_config": bnb, "device_map": {"": 0}},
)
try:
    print("[model] hf_device_map:", getattr(st[0].auto_model, "hf_device_map", "n/a"), flush=True)
except Exception:
    pass
print(f"[model] {MODEL_ID} nap xong {time.time()-t0:.0f}s", flush=True)

# batch nho -> activation nho, tranh OOM khi weights da chiem gan het card
Q = st.encode(texts, batch_size=2, convert_to_numpy=True,
              normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
print(f"[encode] {Q.shape} | norm trung binh {np.linalg.norm(Q,axis=1).mean():.4f} (phai ~1.0)", flush=True)

np.savez(OUT, task_ids=np.array(task_ids), emb=Q.astype(np.float16))
print(f"[done] ghi {OUT} | {Q.shape[0]} query x {Q.shape[1]} chieu | {time.time()-t0:.0f}s", flush=True)
