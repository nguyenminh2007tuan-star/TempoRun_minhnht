# Rerank bf16 + pool fusion-lại-bằng-query-fp16 — chạy trên GPU thuê

Gộp 2 cải thiện tầng nền:
1. POOL = `candidates_SQ8_fp16_d100.json` — fusion lại với QUERY Qwen8B fp16 (đầy đủ)
   thay query int8. RRF y hệt cũ (SigLIP w1 + Qwen8B w8, rrf_k 60). Đổi 31/300 rank-1.
2. Rerank ở **bf16** (không nén) thay int8.

## Yêu cầu: 1 GPU >=24GB (RTX 4090 / A5000 / A6000 / A100), ~40GB đĩa.

## Chạy
```bash
pip install -r requirements.txt          # cần torch+CUDA sẵn trên box
tar xf frames.tar                        # -> ./frames (28829 ảnh pool cũ)
tar xf frames_extra.tar -C frames        # -> thêm 1918 ảnh MỚI của pool fp16 vào ./frames
python rerank_fp16_local.py              # ~30-60 phút
```
OOM thì mở `rerank_fp16_local.py` giảm `BATCH = 8` -> 4.

## Kết quả
`scores_fp16_d100.json` + `submission_fp16_d100.zip`.
**Tải `scores_fp16_d100.json` về gửi lại** để blend/so với nền 78.2.
`submission_fp16_d100.zip` nộp thẳng được.

## Ghi chú
- Muốn TÁCH tác dụng (chỉ đo fp16-rerank, giữ pool cũ): sửa POOL trong runner về
  `candidates_SQ8_d100.json` (khỏi cần frames_extra).
- 4 frame video v3c2_09287 vắng (máy gốc không có) — runner tự bỏ.
