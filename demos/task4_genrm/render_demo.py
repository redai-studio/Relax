# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Render the Task 4 contract trace as a self-contained interactive page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relax · GenRM scaling contract</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#0b1421;color:#eaf1f6}
*{box-sizing:border-box}body{margin:0}main{max-width:1220px;margin:auto;padding:38px 24px 70px}
header{display:flex;align-items:start;justify-content:space-between;gap:22px;border-bottom:1px solid #283d51;padding-bottom:25px}
.eyebrow{color:#8fb8ff;font-size:12px;letter-spacing:.16em;font-weight:700;text-transform:uppercase}h1{font-size:clamp(30px,4vw,48px);letter-spacing:-.04em;margin:7px 0}h2{font-size:19px;margin:5px 0 15px}p{color:#a7b8c9;line-height:1.55;margin:0}
.pill{border:1px solid #536884;border-radius:99px;padding:8px 12px;color:#a7caff;white-space:nowrap;font-size:12px}
.grid{display:grid;grid-template-columns:1.25fr .75fr;gap:17px}.panel,.stat{background:#111f30;border:1px solid #30465c;border-radius:16px}.panel{padding:23px;margin-top:18px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:23px}.stat{padding:17px}.stat span{display:block;color:#9eb4c5;font-size:12px}.stat strong{display:block;margin-top:8px;font-size:26px;font-variant-numeric:tabular-nums}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.tabs button{background:#1b3147;border:1px solid #38536e;color:#d7e6f4;border-radius:9px;padding:9px 13px;font:inherit;font-size:13px;cursor:pointer}.tabs button.active{background:#315c9b;border-color:#82aeff}
input[type=range]{width:100%;accent-color:#7faafa;margin:14px 0}.event{font-size:14px;padding:14px 16px;background:#1b3044;border-left:3px solid #82aeff;border-radius:0 8px 8px 0;min-height:77px;line-height:1.5}.event.fail{border-color:#ef9a6c;background:#342b31}
.rail{display:flex;flex-wrap:wrap;gap:7px;margin-top:18px}.rail span{height:10px;min-width:12px;flex:1;border-radius:3px;background:#37546b}.rail span.past{background:#6c9ded}.rail span.fail.past{background:#e69b70}
.stack{display:grid;gap:10px}.replica{border:1px solid #395875;background:#182b40;border-radius:11px;padding:15px}.replica h3{margin:0 0 7px;font-size:16px}.replica p{font-size:12px}.badge{display:inline-block;border-radius:5px;font-size:11px;padding:4px 6px;margin-right:6px;background:#214e69;color:#9fe4f0}.badge.closed{background:#63463e;color:#f7bd9f}
.resource{display:flex;justify-content:space-between;gap:10px;border-bottom:1px solid #2c4257;padding:10px 0;font-size:13px}.resource:last-child{border:0}.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px}.muted{color:#8ca5b7;font-size:12px}.check{display:flex;gap:10px;padding:9px 0;border-bottom:1px solid #294155;font-size:13px}.check:last-child{border:0}.ok{color:#8de4c1;font-weight:700}.footer{border-top:1px solid #284057;margin-top:25px;padding-top:18px;color:#91a9ba;font-size:12px;line-height:1.6}
@media(max-width:760px){header{display:block}.pill{display:inline-block;margin-top:15px}.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}
</style><main>
<header><div><div class="eyebrow">Relax / Task 4 / contract simulation</div><h1>Scale without losing a score.</h1><p>Replay the proposed GenRM 1→2→1 lifecycle. Each frame shows routing, admitted calls and PG ownership.</p></div><div class="pill">Mock engine · no Ray or GPU</div></header>
<section class="stats"><div class="stat"><span>Initial protected</span><strong>1</strong></div><div class="stat"><span>Current replicas</span><strong id="current"></strong></div><div class="stat"><span>Ready to route</span><strong id="ready"></strong></div><div class="stat"><span>Manager PGs live</span><strong id="pgs"></strong></div></section>
<section class="grid"><div class="panel"><div class="eyebrow">01 / Lifecycle</div><h2>Request and routing trace</h2><div class="tabs" id="tabs"></div><p class="muted" id="position"></p><input type="range" id="step" min="0" value="0"><div id="event" class="event"></div><div id="rail" class="rail"></div>
<div class="stack" id="replicas" style="margin-top:22px"></div></div>
<div class="panel"><div class="eyebrow">02 / Ownership</div><h2>Resources at this step</h2><div id="resources"></div><div class="eyebrow" style="margin-top:29px">03 / Contract checks</div><h2>Invariants exercised</h2><div id="checks"></div></div></section>
<div class="footer">This is a deterministic in-memory simulator of RFC #351. Calls, health checks, backend idle checks, workers and placement groups are mock objects. It does not prove Task 3 integration, real drain behavior, SGLang scoring consistency, Autoscaler decisions or text-training E2E.</div>
</main><script id="payload" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('payload').textContent),names=Object.keys(D.scenarios);let selected=names[0],index=0;
const $=id=>document.getElementById(id);function n(tag,text,cls){const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e}
function render(){const events=D.scenarios[selected],e=events[index];$('current').textContent=e.current;$('ready').textContent=e.ready;$('pgs').textContent=e.live_pgs.filter(x=>x.startsWith('manager')).length;
 $('position').textContent=`${selected} · event ${index+1}/${events.length}`;$('step').max=events.length-1;$('step').value=index;
 $('event').className='event'+(/FAILED|PENDING|REJECT/.test(e.kind)?' fail':'');$('event').textContent=`${e.kind} — ${e.message}`;
 $('rail').replaceChildren(...events.map((v,i)=>n('span','',''+(i<=index?'past ':'')+(/FAILED|PENDING|REJECT/.test(v.kind)?'fail':''))));
 const replicas=$('replicas');replicas.replaceChildren();const known=[{id:'initial-0',pg:'training-pg',owner:'initial'}];
 if(e.live_pgs.includes('manager-pg-1')||e.published.includes('genrm-1')||e.inflight['genrm-1'])known.push({id:'genrm-1',pg:'manager-pg-1',owner:'manager'});
 known.forEach(r=>{const box=n('div','','replica'),title=n('h3',r.id),route=e.published.includes(r.id),badge=n('span',route?'ROUTABLE':'CLOSED','badge'+(route?'':' closed'));
 box.append(title,badge,n('span',`${r.owner} owned · ${r.pg}`,'muted'),n('p',`${(e.inflight[r.id]||[]).length} accepted scoring call(s) in flight`));replicas.append(box)});
 const resources=$('resources');resources.replaceChildren(...e.live_pgs.map(pg=>{const row=n('div','','resource');row.append(n('span',pg,'mono'),n('span',pg==='training-pg'?'original owner':'manager owned'));return row}));
 $('checks').replaceChildren(...Object.entries(D.checks).map(([key,value])=>{const row=n('div','','check');row.append(n('span',value===true?'✓':Array.isArray(value)?'→':'×',value===true?'ok':''),n('span',`${key.replaceAll('_',' ')}${Array.isArray(value)?': '+value.join('/') :''}`));return row}));
}
names.forEach((name,i)=>{const button=n('button',name);button.onclick=()=>{selected=name;index=0;document.querySelectorAll('#tabs button').forEach((b,j)=>b.classList.toggle('active',j===i));render()};$('tabs').append(button)});
$('step').oninput=e=>{index=Number(e.target.value);render()};document.querySelector('#tabs button').click();
</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace("__DATA__", serialized), encoding="utf-8")


if __name__ == "__main__":
    main()
