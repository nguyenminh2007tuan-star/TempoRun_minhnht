"""Post-process 81.944: centroid-nudge rank-1 + consensus rank-2.
Code moi từ transcript (sub_clip5centroid + sub_consensus_r2). Validate tái hiện
public bit-for-bit rồi mới build 5 bản private.
"""
import json, zipfile, os
import numpy as np

W = 10000

def top10_by_rerank(scores):
    return {t: [{"video_id": c["video_id"], "frame_ms": int(c["frame_ms"])}
                for c in sorted(l, key=lambda c: -c["rerank"])[:10]]
            for t, l in scores.items()}

def centroid_nudge(base, clip, rr, W=W):
    """Dời rank-1 về trọng tâm softmax(T=1); field=clip nếu >=3 frame cùng video ±W, else rr nếu >=2."""
    out = {}
    for tid, res0 in base.items():
        res = [dict(r) for r in res0]; r1 = res[0]; r1ms = int(r1["frame_ms"])
        nearc = [(int(r["frame_ms"]), r["clip_score"]) for r in clip.get(tid, [])
                 if r["video_id"] == r1["video_id"] and abs(int(r["frame_ms"]) - r1ms) <= W]
        nearb = [(int(r["frame_ms"]), r["rerank"]) for r in rr.get(tid, [])
                 if r["video_id"] == r1["video_id"] and abs(int(r["frame_ms"]) - r1ms) <= W]
        near = nearc if len(nearc) >= 3 else (nearb if len(nearb) >= 2 else None)
        if near:
            a = np.array([m for m, _ in near], float); s = np.array([x for _, x in near], float)
            w = np.exp(s - s.max()); c = int(round((a * w).sum() / w.sum()))
            if c != r1ms: res[0]["frame_ms"] = c
        out[tid] = res
    return out

def consensus_r2(base, c3, c5, gate=5000, W=W):
    """Chèn rank-2 nơi argmax clip3 & clip5 đồng thuận (cùng video, |a3-a5|<=gate), khác rank-1."""
    def centroid5(tid, vid, ms):
        near = [(int(r["frame_ms"]), r["clip_score"]) for r in c5[tid]
                if r["video_id"] == vid and abs(int(r["frame_ms"]) - ms) <= W]
        if len(near) < 2: return ms
        a = np.array([m for m, _ in near], float); s = np.array([x for _, x in near], float)
        w = np.exp(s - s.max()); return int(round((a * w).sum() / w.sum()))
    out = {}; n_ins = 0
    for tid, res0 in base.items():
        res = [dict(r) for r in res0]; r1 = res[0]
        a3 = max(c3[tid], key=lambda r: r["clip_score"])
        a5 = max(c5[tid], key=lambda r: r["clip_score"])
        cons = (a3["video_id"] == a5["video_id"]
                and abs(int(a3["frame_ms"]) - int(a5["frame_ms"])) <= gate)
        diff = (a5["video_id"] != r1["video_id"]
                or abs(int(a5["frame_ms"]) - int(r1["frame_ms"])) > 5000)
        if cons and diff:
            res = [r1, {"video_id": a5["video_id"], "frame_ms": centroid5(tid, a5["video_id"], int(a5["frame_ms"]))}] + res[1:9]
            n_ins += 1
        out[tid] = res
    return out, n_ins

def dedup_backfill(res, pool):
    """Bỏ trùng (video,ms) giữ hạng cao nhất; lấp thêm từ pool rerank cho đủ 10 distinct. Rank-1 giữ nguyên."""
    seen, clean = set(), []
    for r in res:
        k = (r["video_id"], int(r["frame_ms"]))
        if k not in seen:
            seen.add(k); clean.append({"video_id": r["video_id"], "frame_ms": int(r["frame_ms"])})
    for c in sorted(pool, key=lambda c: -c["rerank"]):
        if len(clean) >= 10: break
        k = (c["video_id"], int(c["frame_ms"]))
        if k not in seen:
            seen.add(k); clean.append({"video_id": c["video_id"], "frame_ms": int(c["frame_ms"])})
    return clean[:10]

def to_preds(sub):
    return {"predictions": [{"task_id": t, "results": [
        {"rank": i, "video_id": r["video_id"], "frame_ms": int(r["frame_ms"])}
        for i, r in enumerate(res, 1)]} for t, res in sorted(sub.items())]}

def r1_map(pred):
    return {p["task_id"]: (p["results"][0]["video_id"], p["results"][0]["frame_ms"]) for p in pred["predictions"]}

def load_sub(fn):
    return {p["task_id"]: p["results"] for p in json.load(open(fn))["predictions"]}

# =================== VALIDATE trên PUBLIC (bit-for-bit) ===================
def validate():
    print("===== VALIDATE tái hiện public =====")
    fp16 = load_sub("submission_fp16_top10.json")
    bf   = json.load(open("scores_bf16_oldpool_d100.json"))
    c3p  = json.load(open("scores_clip_top20.json"))
    c5p  = json.load(open("scores_clip5_top20.json"))

    def cmp(mine, saved_fn, name):
        saved = json.load(open(saved_fn))
        a = r1_map(to_preds(mine)); b = r1_map(saved)
        # so KHỚP toàn bộ top-10, không chỉ rank-1
        ma = {p["task_id"]: [(r["video_id"], int(r["frame_ms"])) for r in p["results"]] for p in to_preds(mine)["predictions"]}
        mb = {p["task_id"]: [(r["video_id"], int(r["frame_ms"])) for r in p["results"]] for p in saved["predictions"]}
        diff = sum(1 for k in ma if ma.get(k) != mb.get(k))
        print(f"  {name}: lệch {diff}/{len(ma)} task  ->  {'KHỚP BIT-FOR-BIT ✅' if diff==0 else 'LỆCH ❌'}")
        return diff == 0

    ok = True
    # clip5-centroid (field clip5, fallback bf rerank)
    m5 = centroid_nudge(fp16, c5p, bf)
    ok &= cmp(m5, "sub_clip5centroid.json", "clip5centroid")
    # clip3-centroid (field clip3=clip_top20, fallback bf)
    m3 = centroid_nudge(fp16, c3p, bf)
    ok &= cmp(m3, "sub_clipcentroid.json", "clipcentroid(clip3)")
    # consensus_r2 trên sub_clipcentroid đã lưu, gate 5s
    cc = load_sub("sub_clipcentroid.json")
    mc, n = consensus_r2(cc, c3p, c5p, gate=5000)
    ok &= cmp(mc, "sub_consensus_r2.json", "consensus_r2(gate5s)")
    print(f"  (consensus chèn {n} task)")
    return ok

def build_private(scores_dir=".", out_dir="submits_private"):
    os.makedirs(out_dir, exist_ok=True)
    j = lambda fn: os.path.join(scores_dir, fn)
    raw   = json.load(open(j("scores_1f_raw.json")))
    strip = json.load(open(j("scores_1f_strip.json")))
    c3    = json.load(open(j("scores_clip3.json")))
    c5    = json.load(open(j("scores_clip5.json")))
    base_raw, base_strip = top10_by_rerank(raw), top10_by_rerank(strip)
    n_tasks = len(raw)

    # 5 slot ĐA DẠNG RANK-1 (gate chỉ đổi rank-2 nên bỏ). Trục thật đổi rank-1:
    # query {raw,strip} × centroid-field {clip3,clip5,OFF}. Slot1 = y-chang-public. Slot5 = hedge tắt centroid.
    slots = [
        ("slot1_ANCHOR_raw_clip3",  base_raw,   raw,   c3, "cent"),   # = phương pháp 81.944
        ("slot2_strip_clip3",       base_strip, strip, c3, "cent"),   # strip + phương pháp thắng (best judgment)
        ("slot3_strip_clip5",       base_strip, strip, c5, "cent"),   # đổi field clip5
        ("slot4_raw_clip5",         base_raw,   raw,   c5, "cent"),   # raw + clip5
        ("slot5_strip_NOcent",      base_strip, strip, c3, "off"),    # HEDGE: tắt centroid (phòng centroid lệch private)
    ]
    r1s = {}
    print("\n===== BUILD 5 bản PRIVATE =====")
    for name, base, rr, cfield, mode in slots:
        if mode == "cent":
            moved = sum(1 for t in base
                        if centroid_nudge({t: base[t]}, cfield, rr)[t][0]["frame_ms"] != base[t][0]["frame_ms"])
            nud = centroid_nudge(base, cfield, rr)
        else:
            moved = 0; nud = {t: [dict(r) for r in base[t]] for t in base}
        fin, n = consensus_r2(nud, c3, c5, gate=5000)
        fin = {t: dedup_backfill(fin[t], rr.get(t, [])) for t in fin}   # bỏ trùng + lấp đủ 10
        pred = to_preds(fin)
        assert len(pred["predictions"]) == n_tasks, f"thiếu task: {len(pred['predictions'])}/{n_tasks}"
        assert all(1 <= len(p["results"]) <= 10 for p in pred["predictions"])
        p = os.path.join(out_dir, f"{name}.json")
        json.dump(pred, open(p, "w"))
        with zipfile.ZipFile(p.replace(".json", ".zip"), "w") as z: z.write(p, "submission.json")
        r1s[name] = r1_map(pred)
        print(f"  {name}: nudge rank-1 {moved}/{n_tasks} | consensus chèn {n}/{n_tasks} -> {name}.zip")

    # đa dạng chéo: ma trận khác rank-1 giữa mọi cặp slot
    print("\n  Ma trận khác rank-1 (số task) giữa các slot:")
    names = list(r1s)
    print("      " + "  ".join(f"s{i+1}" for i in range(len(names))))
    for i, a in enumerate(names):
        row = [str(sum(1 for k in r1s[a] if r1s[a][k] != r1s[b][k])).rjust(3) for b in names]
        print(f"    s{i+1} " + " ".join(row) + f"   {a}")

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Centroid-nudge + consensus post-process -> submissions")
    ap.add_argument("--scores-dir", default=".",
                    help="directory holding the four reranker score files "
                         "(scores_1f_raw / scores_1f_strip / scores_clip3 / scores_clip5)")
    ap.add_argument("--out-dir", default="submits_private", help="where the submission json/zip go")
    ap.add_argument("--validate", action="store_true",
                    help="OPTIONAL self-check: re-derive the saved public-round submissions and "
                         "assert they match bit-for-bit. Needs the public-round score files, which "
                         "are NOT required to build the private submissions.")
    args = ap.parse_args()

    if args.validate:
        if not validate():
            print("\n❌ CHƯA khớp — DỪNG.")
            sys.exit(1)
        print("\n✅ Reproduce KHỚP — hàm trung thực.")

    build_private(args.scores_dir, args.out_dir)
    print(f"\n✅ XONG — submissions trong {args.out_dir}/")
