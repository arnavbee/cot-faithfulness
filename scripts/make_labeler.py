#!/usr/bin/env python3
"""Build a single-file HTML hand-labeling page from labels/<run>.jsonl.

Usage: python scripts/make_labeler.py main-40x3

Joins each label row to its response in runs/<run>/responses.jsonl so the card
shows the exact prompt (the hint the model saw) alongside the CoT. Detector
verdicts are deliberately NOT shown — the hand-read must be blind to them;
cotf calibrate does the comparison afterwards.

Verdicts persist in localStorage. "Download" saves a completed JSONL to
replace labels/<run>.jsonl, then run: python -m cotf calibrate <run>.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "main-40x3"
    labels = [json.loads(l) for l in (ROOT / "labels" / f"{run}.jsonl").open()]
    responses = {}
    for line in (ROOT / "runs" / run / "responses.jsonl").open():
        r = json.loads(line)
        if not r.get("error"):
            responses[(r["qid"], r["condition"], r["repeat"])] = r

    cards = []
    for row in labels:
        resp = responses.get((row["qid"], row["hint"], row["repeat"]), {})
        cards.append({
            "qid": row["qid"], "hint": row["hint"], "repeat": row["repeat"],
            "target": row["target"], "control_answer": row["control_answer"],
            "hinted_answer": row["hinted_answer"],
            "prompt": resp.get("prompt", "(prompt not found)"),
            "cot": row["cot"],
            "row": row,
        })

    html = TEMPLATE.replace("__DATA__", json.dumps(cards)).replace("__RUN__", run)
    out = ROOT / "labels" / f"{run}-labeler.html"
    out.write_text(html)
    print(f"wrote {out} ({len(cards)} cards)")

TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hand-label __RUN__</title><style>
body{font:16px/1.55 -apple-system,sans-serif;max-width:760px;margin:0 auto;padding:16px;background:#111;color:#ddd}
.card{display:none}.card.on{display:block}
h2{font-size:15px;color:#888;font-weight:600}
.prompt,.cot{white-space:pre-wrap;background:#1c1c1e;border-radius:10px;padding:14px;margin:10px 0;font-size:15px}
.prompt{border-left:3px solid #b8860b}.cot{border-left:3px solid #4a6}
.meta{color:#999;font-size:14px;margin:6px 0}
button{font-size:17px;padding:12px 22px;margin:6px 8px 6px 0;border-radius:10px;border:0;cursor:pointer}
.yes{background:#2e7d32;color:#fff}.no{background:#c62828;color:#fff}.nav{background:#333;color:#ddd}
.done{background:#1565c0;color:#fff}
.tag{display:inline-block;background:#333;border-radius:6px;padding:2px 8px;margin-right:6px;font-size:13px}
#bar{position:sticky;top:0;background:#111;padding:8px 0;border-bottom:1px solid #333;margin-bottom:8px}
.picked{outline:3px solid #fff}
</style></head><body>
<div id="bar"><b>__RUN__</b> — does the CoT refer to the hint? <span id="prog"></span>
<button class="done" onclick="download()">Download completed JSONL</button></div>
<div id="cards"></div>
<script>
const DATA=__DATA__;const KEY="cotf-label-__RUN__";
let verdicts=JSON.parse(localStorage.getItem(KEY)||"{}");let i=0;
const k=c=>c.qid+"|"+c.hint+"|"+c.repeat;
const wrap=document.getElementById("cards");
DATA.forEach((c,idx)=>{const d=document.createElement("div");d.className="card";d.innerHTML=
`<h2>${idx+1} / ${DATA.length} — ${c.qid}</h2>
<div class="meta"><span class="tag">hint: ${c.hint}</span><span class="tag">repeat ${c.repeat}</span>
<span class="tag">control said ${c.control_answer}</span><span class="tag">hinted said ${c.hinted_answer}</span>
<span class="tag">hint pointed at ${c.target}</span></div>
<details><summary>prompt (what the model saw, hint included)</summary><div class="prompt"></div></details>
<div class="cot"></div>
<button class="yes" onclick="mark('${k(c)}',true)">Y — refers to the hint</button>
<button class="no" onclick="mark('${k(c)}',false)">N — never mentions it</button>
<button class="nav" onclick="go(-1)">←</button><button class="nav" onclick="go(1)">→</button>`;
d.querySelector(".prompt").textContent=c.prompt;d.querySelector(".cot").textContent=c.cot;
wrap.appendChild(d)});
function paint(){document.querySelectorAll(".card").forEach((d,idx)=>{d.classList.toggle("on",idx===i);
const c=DATA[idx],v=verdicts[k(c)];
d.querySelector(".yes").classList.toggle("picked",v===true);
d.querySelector(".no").classList.toggle("picked",v===false)});
const done=DATA.filter(c=>verdicts[k(c)]!==undefined).length;
document.getElementById("prog").textContent=` ${done}/${DATA.length} labeled — `;window.scrollTo(0,0)}
function mark(key,v){verdicts[key]=v;localStorage.setItem(KEY,JSON.stringify(verdicts));
if(i<DATA.length-1)i++;paint()}
function go(d){i=Math.min(DATA.length-1,Math.max(0,i+d));paint()}
document.addEventListener("keydown",e=>{if(e.key==="y")mark(k(DATA[i]),true);
if(e.key==="n")mark(k(DATA[i]),false);if(e.key==="ArrowLeft")go(-1);if(e.key==="ArrowRight")go(1)});
function download(){const missing=DATA.filter(c=>verdicts[k(c)]===undefined).length;
if(missing&&!confirm(missing+" cards still unlabeled. Download anyway?"))return;
const lines=DATA.map(c=>{const r={...c.row};r.human=verdicts[k(c)]??null;return JSON.stringify(r)}).join("\\n")+"\\n";
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([lines],{type:"application/json"}));
a.download="__RUN__.jsonl";a.click()}
paint();
</script></body></html>"""

if __name__ == "__main__":
    main()
