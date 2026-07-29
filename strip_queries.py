"""Stage 1b — strip question scaffolding from a round's queries.

The reranker is a text->image matcher. Words like "identify the exact moment when"
or "at what timestamp does" describe the ASK, not the picture, so they add tokens
that match nothing visual. Removing them leaves the scene description the reranker
can actually ground, while the ORIGINAL query is kept for the other rerank pass —
the submitted portfolio hedges between the two rather than betting on either.

Only the scaffolding goes: every clause that describes what is on screen is kept,
and a rewrite that would leave too little text is rejected (the length guards below).

Example
-------
python strip_queries.py --tasks data/private_round_tasks.jsonl \
                        --out precomputed/private_queries_stripped.json
"""
import argparse, json, re

LEAD_CORE = (r'(please\s+)?'
    r'(identify|locate|determine|find|pinpoint|specify|capture|show|reveal)\s+'
    r'(the\s+)?(exact\s+|specific\s+|precise\s+)?(moment|instant|frame|point|time)\s+'
    r'(in\s+(the\s+)?(video|clip|time-?lapse|sequence|scene)\s+)?'
    r'(just\s+)?(that\s+|when\s+|where\s+|at\s+which\s+|in\s+which\s+|the\s+)?')
LEAD     = re.compile(r'^\s*' + LEAD_CORE, re.I)
MIDLEAD  = re.compile(r'\s*[;,.]\s+' + LEAD_CORE, re.I)
WHATLEAD = re.compile(r'^\s*what\s+(is|are)\s+(the\s+)?(exact\s+|specific\s+|precise\s+)?', re.I)
MIDQ     = re.compile(r',?\s*(at\s+)?(what|which)\s+(exact\s+|specific\s+|precise\s+)?'
    r'(moment|instant|point|timestamp|time|frame)\s+(does\s+|do\s+|is\s+|are\s+|will\s+|has\s+|have\s+|when\s+)?', re.I)

def strip_q(d):
    s = d.strip()
    for pat in (LEAD, WHATLEAD):
        s2 = pat.sub('', s, count=1)
        if s2 != s and len(s2) > 20:            # keep the rewrite only if enough text survives
            s = s2.strip(); break
    for _ in range(2):                          # a query can carry more than one mid-sentence ask
        m = MIDLEAD.search(s)
        if m and len(s[m.end():]) > 15:
            sep = '. ' if s[m.start():m.start()+2].startswith('.') else ', '
            s = s[:m.start()].rstrip(' ,;.') + sep + s[m.end():].lstrip()
        else:
            break
    m = MIDQ.search(s)
    if m and len(s) - (m.end() - m.start()) > 20:
        pre = s[:m.start()].rstrip(' ,'); post = s[m.end():].lstrip()
        s = (pre + ', ' + post) if pre else post
    s = s.rstrip('?').strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="round task file (.jsonl)")
    ap.add_argument("--out", required=True, help="output {task_id: stripped_query} json")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.tasks)]
    out = {t["task_id"]: strip_q(t["description"]) for t in tasks}
    json.dump(out, open(args.out, "w"), ensure_ascii=False)

    changed = sum(1 for t in tasks if out[t["task_id"]] != t["description"])
    # anything still carrying an ask means the patterns missed a phrasing
    leftover = re.compile(
        r'\b(identify|locate|pinpoint|determine)\s+the\s+(exact\s+|specific\s+)?(moment|instant|point)'
        r'|(at\s+)?(what|which)\s+(exact\s+)?(moment|instant|timestamp|point)\s+(does|do|is|are)', re.I)
    left = [t["task_id"] for t in tasks if leftover.search(out[t["task_id"]])]
    short = [t["task_id"] for t in tasks if len(out[t["task_id"]]) < 25]
    print(f"[strip] {len(tasks)} queries | rewritten {changed} | scaffolding left {len(left)} | very short {len(short)}")
    print(f"-> {args.out}")

if __name__ == "__main__":
    main()
