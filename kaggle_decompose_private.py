"""Decompose 700 query private -> sub-queries (Qwen3.5-4B) trên Kaggle T4. ~30 phút.

VÌ SAO: SigLIP2 (CLIP tower) cắt ở 64 token; query private là câu-đơn-dài (93% không tách
được bằng dấu chấm) -> nếu không decompose thì SigLIP nuốt 1 query dài, 9% bị truncate,
fusion lệch cán cân w1/w8. LLM viết lại thành ~4 sub visual ngắn -> khớp regime public
(3.9 sub/task) -> GIỮ NGUYÊN weight 8 (không phải đoán weight khi mù).

CHUẨN BỊ: Add Input dataset có private_round_tasks.jsonl. Accelerator T4, Internet ON.
Chạy xong tải /kaggle/working/decomposed_queries_private.json về.
"""
import json, os, re, subprocess, sys, time

MODEL = "Qwen/Qwen3.5-4B"
OUT   = "/kaggle/working/decomposed_queries_private.json"

def find(fn):
    for r,_,fs in os.walk("/kaggle/input"):
        if fn in fs: return os.path.join(r,fn)
    raise FileNotFoundError(fn)

subprocess.run([sys.executable,"-m","pip","install","-q",
    "transformers==5.8.1","accelerate","tokenizers","safetensors"], check=False)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = """You rewrite a long video-search query as short sub-queries for a CLIP retrieval system.

Rules:
- Output 2 to 4 sub-queries, one per line, no numbering, no extra text.
- Each sub-query must be under 20 words and independently meaningful: repeat the subject in each (e.g. "a man in a red jacket ..."), never leave a dangling clause.
- Together they must cover every distinct visual detail of the original (people, clothing, objects, actions, scene, time of day, camera motion).
- Do not invent details that are not in the original.

Query: {query}"""

def parse_subs(text):
    lines=[re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*","",l).strip() for l in text.strip().splitlines()]
    subs=[l for l in lines if 3<=len(l.split())<=25]
    return subs[:4]

tasks=[json.loads(l) for l in open(find("private_round_tasks.jsonl"))]
done=json.load(open(OUT)) if os.path.exists(OUT) else {}
todo=[t for t in tasks if t["task_id"] not in done]
print(f"{len(todo)}/{len(tasks)} to do", flush=True)

tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda:0").eval()
print("loaded fp16", flush=True)

t0=time.time()
for i,t in enumerate(todo):
    m=[{"role":"user","content":PROMPT.format(query=t["description"])}]
    try: enc=tok.apply_chat_template(m,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors="pt")
    except TypeError: enc=tok.apply_chat_template(m,add_generation_prompt=True,return_dict=True,return_tensors="pt")
    enc={k:v.to("cuda:0") for k,v in enc.items()}
    with torch.no_grad():
        out=model.generate(**enc,max_new_tokens=140,do_sample=False)
    txt=tok.decode(out[0][enc["input_ids"].shape[1]:],skip_special_tokens=True)
    subs=parse_subs(txt) or [t["description"]]
    done[t["task_id"]]={"full":t["description"],"subs":subs}
    if (i+1)%25==0:
        json.dump(done,open(OUT,"w"),ensure_ascii=False,indent=1)
        el=time.time()-t0; print(f"{i+1}/{len(todo)} | {el/(i+1):.1f}s/q | còn ~{(len(todo)-i-1)*el/(i+1)/60:.0f}p", flush=True)

json.dump(done,open(OUT,"w"),ensure_ascii=False,indent=1)
nsub=sum(len(v["subs"]) for v in done.values())
print(f"DONE {len(done)} query -> {nsub} sub ({nsub/len(done):.1f}/task, public 3.9) -> tải {OUT} về", flush=True)
