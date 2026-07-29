# TempoRun 2026 — Text→Keyframe Temporal Retrieval

Đội thi: **LongJustin** · Phương pháp: **SQ8-Fusion → VLM-Rerank → Centroid-Nudge → Consensus**

---

## 1. Giới thiệu phương pháp

Bài toán: cho một truy vấn tiếng Anh mô tả một khoảnh khắc, trả về tối đa 10 mốc
`(video_id, frame_ms)`; điểm tính theo frame có rơi vào interval đúng hay không.

Pipeline gồm bốn tầng:

1. **Fusion hai encoder (RRF).** Một encoder CLIP (SigLIP2 SO400M-378) và một encoder
   long-context (Qwen3-VL-Embedding-8B) tìm ứng viên độc lập, rồi hợp nhất bằng
   Reciprocal Rank Fusion. SigLIP2 chỉ đọc được ~64 token nên nó nhận **sub-query** do
   LLM viết lại; Qwen8B đọc **nguyên câu** — đây là góc nhìn mà không encoder CLIP nào
   trong nhóm thấy được. Trọng số 1 : 8 phản ánh chênh lệch chất lượng đã đo.

2. **Rerank bằng VLM cross-encoder** (Qwen3-VL-Reranker-8B, fp16) trên top-100 của pool.
   Đây là tín hiệu mạnh nhất và là thứ quyết định thứ hạng.

3. **Centroid-nudge rank-1.** Reranker hay chọn đúng cảnh nhưng đậu ở **mép** interval.
   Bước này dời `frame_ms` của rank-1 về **trọng tâm** của cụm điểm quanh nó (softmax
   trên logit, cửa sổ ±10 s, cùng video). Trường trọng số lấy từ hai pass "clip-strip":
   mỗi ứng viên được ghép thành dải 3 khung / 5 khung liên tiếp để reranker **nhìn thấy
   chuyển động** thay vì một khung tĩnh. Trên vòng public bước này là mức tăng lớn nhất.

4. **Consensus rank-2.** Khi argmax của trường 3-khung và 5-khung **cùng bầu** một mốc
   khác hẳn rank-1, mốc đó được chèn vào rank-2 và đẩy đuôi xuống. Về cấu trúc, R@1
   không thể giảm — chỉ hoà hoặc thắng.

Điểm chung của tầng 3 và 4: chúng **không trộn thêm tín hiệu ngoài** (các thí nghiệm
OCR / VLM-judge / ensemble đều làm điểm tệ đi) mà chỉ **đọc kỹ hơn chính reranker** —
trường điểm, cửa sổ chuyển động, sự đồng thuận giữa hai cửa sổ.

---

## 2. Cấu trúc repository

```
.
├── README.md
├── requirements.txt              # môi trường cho fusion (stage 0-4)
├── gpu_pack/requirements.txt     # môi trường cho rerank (stage 6) — pin chặt phiên bản
│
├── extract_keyframes.py          # stage 0a  video      -> keyframes + ts_ms.npy
├── extract_embed.py              # stage 0b  keyframes  -> index SigLIP2
├── extract_embed_qwen.py         # stage 0c  keyframes  -> index Qwen3-VL-Embedding-8B
│
├── kaggle_decompose_private.py   # stage 1a  query      -> sub-query (Qwen3.5-4B)
├── strip_queries.py              # stage 1b  query      -> query đã bỏ scaffolding
├── embed_qwen8b_private.py       # stage 2a  query      -> vector Qwen8B
├── embed_siglip2_private.py      # stage 2b  sub-query  -> vector SigLIP2
│
├── retrieve_fusion.py            # stage 4   RRF fusion -> candidate pool
├── build_pod_private.py          # stage 5   pool       -> frames_1f/ clip3/ clip5/
├── gpu_pack/private_rerank_all.py# stage 6   rerank 4 pass -> 4 file điểm
├── postprocess_private.py        # stage 7   centroid + consensus -> submission
│
├── configs/enc_SQ8_private.json  # cấu hình 2 encoder (đường dẫn TƯƠNG ĐỐI)
├── precomputed/                  # artifact nhỏ, đã nộp kèm (12 MB) — xem §5.3
├── data/                         # file task của hai vòng
└── submits_private/              # 5 submission đã nộp
```

Các script khác trong thư mục gốc là **mã nghiên cứu phụ** (thử OCR, VLM-judge, blend
nhiều tín hiệu, grounding…). Chúng **không tham gia** vào kết quả chính thức và không cần
chạy — mọi thứ cần thiết đã liệt kê ở trên.

---

## 3. Yêu cầu phần cứng và phần mềm

**Đã kiểm thử thành công trên:**

| Thành phần | Cấu hình |
|---|---|
| OS | Ubuntu 22.04 / 20.04 |
| Python | 3.10 (fusion chạy được cả 3.8) |
| GPU (stage 6, rerank) | **NVIDIA RTX PRO 4000 Blackwell 24 GB** — và trước đó RTX 4090 24 GB |
| GPU (stage 4, fusion) | GPU CUDA bất kỳ ≥ 8 GB (chỉ nhân ma trận, không nạp model) |
| VRAM tối thiểu | **20 GB** cho rerank (trọng số fp16 8B ≈ 16 GB) |
| CUDA | 12.4 (Ada/Ampere) hoặc **12.8 (Blackwell — bắt buộc, xem §11)** |
| NVIDIA Driver | **≥ 570** cho Blackwell · ≥ 525 cho Ada/Ampere (đã chạy trên 580.82.09) |
| PyTorch | 2.6.0+cu124 (Ada) / 2.11.0+cu128 (Blackwell) |
| Đĩa trống | **≈ 50 GB** (index 6 GB + keyframes 17 GB + ảnh rerank 6 GB + trọng số 16 GB) |
| Phụ thuộc hệ thống | `ffmpeg` / `ffprobe` trên PATH (stage 0a) |

Stage 0 (trích keyframe + xây index) rất tốn thời gian — xem §5.2 để **tải index dựng sẵn**
thay vì xây lại.

---

## 4. Cài đặt môi trường

### 4.1 Conda (khuyến nghị)

```bash
conda create -n temporun python=3.10 -y
conda activate temporun

# PyTorch — CHỌN ĐÚNG dòng theo kiến trúc GPU (xem §11):
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124   # Ada/Ampere
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128        # Blackwell

pip install -r requirements.txt              # fusion
pip install -r gpu_pack/requirements.txt     # rerank (pin chặt phiên bản)
```

### 4.2 pip thuần

Giống hệt phần trên, bỏ hai dòng `conda`. Hai file requirements đều ghi rõ phiên bản cho
các thư viện quan trọng (`transformers==5.8.1`, `sentence-transformers==5.4.1`,
`huggingface_hub==1.14.0`, …).

---

## 5. Checkpoint và tài nguyên bổ sung

### 5.1 Trọng số model — tải tự động từ HuggingFace khi chạy

| Model | Dùng ở | Dung lượng | Ghi chú |
|---|---|---|---|
| `Qwen/Qwen3-VL-Reranker-8B` | stage 6 | ≈ 16 GB | **ghim revision `b212dc8c91a8164aef1ea2de9c1a867611e75c04`** (đã hard-code trong script) |
| `Qwen/Qwen3-VL-Embedding-8B` | stage 0c | ≈ 16 GB | chỉ cần nếu xây lại index |
| `Qwen/Qwen3.5-4B` | stage 1a | ≈ 8 GB | chỉ cần nếu sinh lại sub-query |
| `ViT-SO400M-14-SigLIP2-378` / `webli` | stage 0b, 2b | ≈ 1.7 GB | qua `open_clip` |

Máy chạy cần vào được `huggingface.co`. Đặt `HF_HOME` sang ổ đủ chỗ nếu ổ hệ thống nhỏ:

```bash
export HF_HOME=/duong/dan/o/dia/lon/hf
```

### 5.2 Index ảnh dựng sẵn — **cần tải**

Index là embedding của toàn bộ keyframe; xây lại mất khoảng 6 giờ GPU cho riêng Qwen8B.

| Tài nguyên | Dung lượng | Giải nén vào |
|---|---|---|
| `index_qwen8b.tar.part00/01/02` | 4.8 GB (3 phần) | `index_qwen8b/shards/*.npz` |
| `index_siglip2.tar` | 1.4 GB | `index_siglip2/shards/*.npz` |

`index_qwen8b.tar` vượt giới hạn 2 GB mỗi file của GitHub Release nên được chia thành ba
phần; ghép lại bằng `cat index_qwen8b.tar.part* > index_qwen8b.tar` (script bên dưới tự làm).

**Tải tại:** https://github.com/nguyenminh2007tuan-star/TempoRun_minhnht/releases/tag/v1.0
(link trực tiếp: `.../releases/download/v1.0/<tên-file>`) — công khai, không cần đăng nhập.

Checksum SHA-256 (cũng có trong `index.sha256` của Release):

```
35e99e4f8193c3faee4946ba6f91e851f32ae6acdf120edccd876660accd96fa  index_siglip2.tar
6073119c54989e780e3e9338dace16dc44d4cf7344682d0973cb294f7a7da0cf  index_qwen8b.tar.part00
79c6939a9c57619263e9621deeb5b3906453e17f4fcd4ba491963366a1c5637e  index_qwen8b.tar.part01
e72f576692a767854252fb24a59cf9924daea8f3e2b4e97ddf302c5a7a5a0e09  index_qwen8b.tar.part02
```

Sau khi ghép, `index_qwen8b.tar` phải có SHA-256
`bf3f03a29c358ac07fc7d9deabda3d2d5754daa87dedc88763766b8747f694b1`.

Cách tự động — tải, kiểm checksum, giải nén, đếm shard trong một lệnh:

```bash
bash scripts/download_index.sh
```

Hoặc thủ công:

```bash
sha256sum -c index.sha256                          # kiểm file tải về
cat index_qwen8b.tar.part* > index_qwen8b.tar      # ghép 3 phần
tar xf index_qwen8b.tar && tar xf index_siglip2.tar
```

Mỗi shard là `<video_id>.npz` chứa `emb[K, D] float16` và `ts_ms[K] int32`.
Số shard kỳ vọng: **index_qwen8b 5006**, **index_siglip2 5004** (xem §11).

### 5.3 Artifact phía truy vấn — **đã nộp kèm trong repo** (`precomputed/`, 12 MB)

| File | Sinh bởi | Vì sao nộp kèm |
|---|---|---|
| `decomposed_queries_private.json` | stage 1a | LLM sinh; giữ đúng bản đã dùng cho kết quả chính thức |
| `private_queries_stripped.json` | stage 1b | tái tạo được bằng `strip_queries.py` (đã kiểm: **khớp bit-for-bit**) |
| `query_emb_qwen8b_private.npz` | stage 2a | tránh chạy lại encoder 8B chỉ để mã hoá 700 câu |
| `siglip2_subs_private.npz` | stage 2b | như trên |

Đây chính là **tài nguyên dùng để tạo kết quả chính thức**. Nhờ có chúng, người chấm có
thể bắt đầu thẳng từ **stage 4** mà vẫn tái lập đúng bài nộp.

---

## 6. Dữ liệu đầu vào

```
dataset/
├── V3C1/videos/<id>/<id>.mp4
└── V3C2/videos/<id>/<id>.mp4
```

File task (`data/private_round_tasks.jsonl`, `data/public_round_tasks.jsonl`) — mỗi dòng:

```json
{"task_id": "T0001", "description": "...", "submission_type": "temporal_video_retrieval", "max_predictions": 10}
```

Mọi đường dẫn đều truyền qua tham số dòng lệnh hoặc file cấu hình; **không có đường dẫn
tuyệt đối nào bị hard-code** trong các script của pipeline chính.

---

## 7. Kết quả đầu ra

`submits_private/*.json` (kèm `.zip` chứa `submission.json`):

```json
{"predictions": [
  {"task_id": "T0001",
   "results": [{"rank": 1, "video_id": "v3c1_07125", "frame_ms": 90883}]}
]}
```

Đã kiểm tự động trên cả 5 file: đủ **700 task**, đúng **10 kết quả/task**, `rank` chạy
1→10 không trùng, `frame_ms` là số nguyên, **không có cặp `(video_id, frame_ms)` trùng
trong cùng một task**.

**Bản chính thức: `submits_private/slot1_ANCHOR_raw_clip3.zip`** — dùng đúng công thức
đạt điểm cao nhất ở vòng public. Bốn file còn lại là các biến thể đã nộp trong cùng vòng
(luật lấy điểm cao nhất trong các lượt nộp).

---

## 8 & 9. Hướng dẫn chạy — từng script và toàn bộ pipeline

Các stage **phải chạy đúng thứ tự** dưới đây; mỗi stage tiêu thụ đầu ra của stage trước.

Đặt biến cho gọn:

```bash
export DATASET=/duong/dan/dataset          # chứa V3C1/ V3C2/
export TASKS=data/private_round_tasks.jsonl
mkdir -p outputs
```

### Đường ngắn (khuyến nghị) — dùng index tải sẵn + `precomputed/`

```bash
# Stage 0a — trích keyframe (CPU, ffmpeg; nhiều giờ, chạy song song được, resumable)
python extract_keyframes.py --dataset-root $DATASET --out keyframes
#   chạy song song nhiều tiến trình: thêm --shard-index i --shard-count N

# Stage 4 — fusion RRF -> candidate pool
python retrieve_fusion.py \
    --encoders configs/enc_SQ8_private.json \
    --tasks $TASKS \
    --decomposed precomputed/decomposed_queries_private.json \
    --out outputs/submission_fusion.json \
    --candidates-out outputs/candidates_SQ8_private_d200.json \
    --cand-keyframes 400 --rrf-k 60 --top-k 10 \
    --index-root . --emb-root .

# Stage 5 — dựng ảnh cho reranker (CPU, ~15 phút với 32 tiến trình)
python build_pod_private.py \
    --keyframes keyframes \
    --pool outputs/candidates_SQ8_private_d200.json \
    --out rerank_inputs

# Stage 6 — rerank 4 pass (GPU ≥ 20 GB; ~5 giờ trên RTX PRO 4000)
python gpu_pack/private_rerank_all.py \
    --images rerank_inputs \
    --pool outputs/candidates_SQ8_private_d200.json \
    --tasks $TASKS \
    --strip precomputed/private_queries_stripped.json \
    --manifest rerank_inputs/clip_top20_private.json \
    --out-dir outputs

# Stage 7 — centroid + consensus -> 5 submission
python postprocess_private.py --scores-dir outputs --out-dir submits_private
```

### Đường đầy đủ — sinh lại cả artifact phía truy vấn

Thêm bốn bước trước stage 4 (cần GPU; hai script embed vốn viết cho notebook Kaggle, xem §11):

```bash
python strip_queries.py --tasks $TASKS --out precomputed/private_queries_stripped.json
python kaggle_decompose_private.py     # -> decomposed_queries_private.json
python embed_qwen8b_private.py         # -> query_emb_qwen8b_private.npz
python embed_siglip2_private.py        # -> siglip2_subs_private.npz
```

### Xây lại index từ đầu (không bắt buộc, ~8 giờ GPU)

```bash
python extract_embed.py      --keyframes keyframes --out index_siglip2 \
                             --model ViT-SO400M-14-SigLIP2-378 --pretrained webli --precision fp16
python extract_embed_qwen.py --keyframes keyframes --out index_qwen8b
```

### Kiểm tra tính trung thực của bước hậu xử lý (tuỳ chọn)

```bash
python postprocess_private.py --validate --scores-dir outputs
```

Cờ này dựng lại các submission của **vòng public** từ điểm số đã lưu và khẳng định chúng
khớp **bit-for-bit** với bản đã nộp. Cần các file điểm của vòng public; **không bắt buộc**
để tạo submission vòng private.

---

## 10. Tham số mặc định

Mọi giá trị dưới đây **đã là mặc định trong mã nguồn** — chạy đúng các lệnh ở §9 sẽ tái
lập bài nộp, không cần chỉnh gì thêm.

| Tham số | Giá trị | Ở đâu |
|---|---|---|
| Trọng số encoder | SigLIP2 = 1, Qwen8B = 8 | `configs/enc_SQ8_private.json` |
| Hằng số RRF `k` | 60 | `retrieve_fusion.py --rrf-k` |
| Ứng viên mỗi encoder | 400 | `--cand-keyframes` |
| Độ sâu pool ghi ra | 200 | `retrieve_fusion.py` |
| **Độ sâu rerank** | **100** | `gpu_pack/private_rerank_all.py --depth` |
| Batch rerank | 1-khung 8 · 3-khung 4 · 5-khung 2 | `--batch-1f/--batch-c3/--batch-c5` |
| Hình học clip-strip | 3-khung `[-2, 0, +2] s` · 5-khung `[-8, -4, 0, +4, +8] s` | `build_pod_private.py` |
| Cửa sổ centroid | ±10 000 ms, softmax nhiệt độ 1, **chỉ rank-1** | `postprocess_private.py` |
| Cổng consensus | 5 000 ms | `postprocess_private.py` |
| Revision reranker | `b212dc8c…75c04` | `gpu_pack/private_rerank_all.py` |

Lưu ý: pool ghi ra ở độ sâu 200 nhưng **chỉ rerank 100** — độ sâu 100 là cấu hình đã
kiểm chứng ở vòng public; 100 ứng viên phía sau gần như không bao giờ được reranker đẩy
lên mà lại tốn gấp đôi thời gian GPU.

---

## 11. Lỗi và giới hạn đã biết

**GPU Blackwell (RTX PRO 4000/5000/6000, RTX 50xx) — bắt buộc dùng CUDA 12.8.**
Cài `torch==2.6.0+cu124` trên các GPU này vẫn chạy được tới lúc nạp model rồi lỗi:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Khắc phục:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; x=torch.randn(2000,2000,device='cuda'); print('OK', float((x@x).sum()))"
```

Cảnh báo `torchaudio ... requires torch==2.4.1` xuất hiện sau khi cài là vô hại —
pipeline không dùng `torchaudio`.

**Thiếu VRAM ở stage 6.** Trọng số fp16 chiếm ~16 GB; card 24 GB vừa đủ. Nếu OOM hoặc
script báo `NaN scores`, hạ batch của **đúng pass đó**: `--batch-1f 4`, `--batch-c5 1`.
Mỗi pass ghi file ngay khi xong nên lỗi ở pass sau không làm mất kết quả pass trước.

**Thiếu đĩa.** Cần ~50 GB. Trên máy thuê GPU, nên đặt `HF_HOME` sang volume lớn và giải
nén ảnh rerank vào **ổ cục bộ** — giải hơn 100 000 file nhỏ lên ổ mạng chậm hơn nhiều lần.

**Keyframe do người chấm tự trích có thể lệch so với index đã nộp.** Index được xây bằng
**ffmpeg 4.2.7** (`extract_keyframes.py` lấy I-frame). Bản ffmpeg khác có thể chọn tập
I-frame hơi khác, khiến `ts_ms` không trùng tuyệt đối. Pipeline **không crash** —
`build_pod_private.py` luôn lấy keyframe **gần nhất** — nhưng dải clip có thể lệch vài
trăm ms và điểm chênh chút ít. Số kỳ vọng để đối chiếu: **5006 clip**, **≈ 579 000
keyframe**, trung bình ~116 keyframe/clip. Muốn khớp tuyệt đối thì dùng ffmpeg 4.2.x.

**Hai clip không có keyframe.** `v3c2_08047` và `v3c2_09287` không trích được keyframe nên
`index_siglip2` chỉ có 5004 shard (Qwen8B vẫn đủ 5006). Ảnh hưởng đã đo trên vòng private:
**0 task** có rank-1 hoặc top-20 rơi vào hai clip này; chỉ 10 trên khoảng 140 000 ứng viên
ở phần đuôi pool bị bỏ qua. Đây là hành vi đã lường trước, không phải lỗi.

**Hai script embed viết cho Kaggle.** `embed_qwen8b_private.py` và
`embed_siglip2_private.py` dò input trong `/kaggle/input` và ghi ra `/kaggle/working`
(chúng vốn chạy trong notebook). Vì **kết quả của chúng đã nộp kèm trong `precomputed/`**,
người chấm **không cần chạy lại**; nếu muốn chạy, hãy sửa hai hằng số đường dẫn ở đầu file.

**Tính tất định và random seed.** Pipeline **không có nguồn ngẫu nhiên nào**, nên không
cần đặt seed: không có bước huấn luyện, toàn bộ là suy luận; LLM sinh sub-query chạy
**greedy** (`do_sample=False`, không temperature/top-p); fusion, rerank và hậu xử lý đều
là phép toán tất định. Chạy lại `postprocess_private.py` cho ra file **giống hệt theo
md5** (đã kiểm). Chạy lại `retrieve_fusion.py` trên cùng máy tái lập candidate pool
**bit-for-bit** (đã kiểm: lệch 0/700 task).

Sai khác **có thể** xảy ra khi đổi máy: thứ tự cộng dồn dấu chấm động trên GPU khác kiến
trúc có thể xê dịch vài ứng viên **gần như đồng điểm** ở phần đuôi bảng xếp hạng. Mức ảnh
hưởng dự kiến rất nhỏ và hầu như không chạm rank-1.

**Không có thao tác thủ công.** Toàn bộ prediction sinh ra tự động từ mã nguồn trong repo:
không gán nhãn tay, không sửa kết quả theo từng task, không gọi API trả phí, không gửi dữ
liệu test ra dịch vụ ngoài. Model duy nhất tải từ mạng là trọng số mở trên HuggingFace
(§5.1), tải một lần rồi chạy cục bộ.

---

## 12. Thông tin nộp bài

| Mục | Nội dung |
|---|---|
| Tên đội | **LongJustin** |
| Tên phương pháp | SQ8-Fusion → VLM-Rerank → Centroid-Nudge → Consensus |
| Repository | https://github.com/nguyenminh2007tuan-star/TempoRun_minhnht |
| Commit / tag | release tag **`v1.0`** |
| Cài đặt môi trường | Conda hoặc pip — xem §4 |
| Phần cứng đã kiểm thử | RTX PRO 4000 Blackwell 24 GB (CUDA 12.8) và RTX 4090 24 GB (CUDA 12.4) |
| Lệnh chạy chính | xem §9 |
| Liên hệ | _(điền)_ |

> `README_baseline_BTC.md` là README gốc của baseline do BTC cung cấp, giữ lại để tham khảo.
