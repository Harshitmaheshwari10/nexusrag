"""
NexusRAG Utilities
- Source card formatting
- Knowledge graph HTML (D3.js force graph, self-contained)
"""

import json
import html
from typing import List, Dict, Any


def format_sources(
    doc_sources: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
) -> str:
    """Return HTML for source cards below an AI message."""
    if not doc_sources and not web_results:
        return ""

    parts = ['<div style="margin-top:0.8rem; border-top:1px solid #1e1e3a; padding-top:0.6rem;">']
    parts.append('<div style="font-size:0.7rem; color:#6b7280; font-family:\'JetBrains Mono\',monospace; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem;">Sources</div>')

    for s in doc_sources[:3]:
        score_pct = min(100, int(s["score"] * 100)) if s["score"] <= 1 else min(100, int(s["score"] * 10))
        safe_text = html.escape(s["text"][:180])
        safe_name = html.escape(s["doc_name"])
        parts.append(f"""
        <div class="source-card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="color:#c084fc; font-weight:500;">📄 {safe_name}</span>
            <span style="color:#6b7280; font-size:0.68rem;">chunk #{s['chunk_idx']}</span>
          </div>
          <div style="color:#9999bb; font-size:0.78rem; line-height:1.5;">{safe_text}...</div>
          <div class="score-bar" style="width:{score_pct}%;"></div>
        </div>
        """)

    for r in web_results[:2]:
        safe_title = html.escape(r.get("title", "Web Result"))
        safe_snippet = html.escape(r.get("snippet", "")[:160])
        safe_url = html.escape(r.get("url", "#"))
        parts.append(f"""
        <div class="web-result">
          <div style="font-weight:500; color:#4ade80; margin-bottom:3px;">🌐 {safe_title}</div>
          <div style="color:#9999bb; font-size:0.78rem; line-height:1.5;">{safe_snippet}...</div>
          {'<div style="margin-top:4px;"><a href="' + safe_url + '" target="_blank" style="font-size:0.68rem; color:#4ade80;">↗ ' + safe_url[:60] + '</a></div>' if safe_url and safe_url != "#" else ""}
        </div>
        """)

    parts.append("</div>")
    return "".join(parts)


def render_knowledge_graph_html(entities: Dict[str, List[str]]) -> str:
    """
    Renders a D3.js force-directed knowledge graph.
    Returns self-contained HTML string.
    """
    nodes = []
    links = []

    color_map = {
        "persons": "#c084fc",
        "orgs": "#60a5fa",
        "concepts": "#4ade80",
        "dates": "#fbbf24",
    }

    node_idx = {}
    idx = 0

    for cat, items in entities.items():
        for item in items[:8]:  # Limit per category
            if item not in node_idx:
                nodes.append({
                    "id": idx,
                    "label": item[:16] + ("…" if len(item) > 16 else ""),
                    "full": item,
                    "group": cat,
                    "color": color_map.get(cat, "#888"),
                })
                node_idx[item] = idx
                idx += 1

    # Connect nodes in the same category (ring topology)
    for cat, items in entities.items():
        cat_items = [it for it in items[:8] if it in node_idx]
        for i in range(len(cat_items) - 1):
            links.append({
                "source": node_idx[cat_items[i]],
                "target": node_idx[cat_items[i + 1]],
                "group": cat,
            })

    # Cross-link some nodes for visual interest
    all_groups = list(entities.keys())
    for i in range(len(all_groups) - 1):
        g1_items = [it for it in entities[all_groups[i]][:3] if it in node_idx]
        g2_items = [it for it in entities[all_groups[i+1]][:3] if it in node_idx]
        if g1_items and g2_items:
            links.append({
                "source": node_idx[g1_items[0]],
                "target": node_idx[g2_items[0]],
                "group": "cross",
            })

    nodes_json = json.dumps(nodes)
    links_json = json.dumps(links)

    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin:0; background:#07070f; overflow:hidden; }}
  canvas {{ display:block; }}
  .tooltip {{
    position:absolute; background:#1a1a2e; border:1px solid #2d2d4e;
    color:#e8e8f0; padding:4px 10px; border-radius:6px; font-size:11px;
    font-family:'JetBrains Mono',monospace; pointer-events:none; display:none;
  }}
</style>
</head>
<body>
<div class="tooltip" id="tooltip"></div>
<canvas id="c"></canvas>
<script>
const nodes = {nodes_json};
const links = {links_json};

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const W = canvas.width = window.innerWidth || 300;
const H = canvas.height = window.innerHeight || 300;

// Initialize positions
nodes.forEach((n, i) => {{
  const angle = (i / nodes.length) * Math.PI * 2;
  n.x = W/2 + Math.cos(angle) * (W/3 - 20);
  n.y = H/2 + Math.sin(angle) * (H/3 - 20);
  n.vx = 0; n.vy = 0;
}});

// Force simulation
function tick() {{
  // Repulsion
  for(let i=0;i<nodes.length;i++) {{
    for(let j=i+1;j<nodes.length;j++) {{
      const dx = nodes[j].x - nodes[i].x;
      const dy = nodes[j].y - nodes[i].y;
      const d = Math.sqrt(dx*dx+dy*dy) || 1;
      const f = 800 / (d*d);
      nodes[i].vx -= dx*f/d;
      nodes[i].vy -= dy*f/d;
      nodes[j].vx += dx*f/d;
      nodes[j].vy += dy*f/d;
    }}
  }}
  // Attraction (links)
  links.forEach(l => {{
    const s = nodes[l.source], t = nodes[l.target];
    const dx = t.x - s.x, dy = t.y - s.y;
    const d = Math.sqrt(dx*dx+dy*dy) || 1;
    const f = (d - 80) * 0.03;
    s.vx += dx/d*f; s.vy += dy/d*f;
    t.vx -= dx/d*f; t.vy -= dy/d*f;
  }});
  // Center gravity
  nodes.forEach(n => {{
    n.vx += (W/2 - n.x) * 0.005;
    n.vy += (H/2 - n.y) * 0.005;
    n.vx *= 0.85; n.vy *= 0.85;
    n.x += n.vx; n.y += n.vy;
    n.x = Math.max(20,Math.min(W-20,n.x));
    n.y = Math.max(20,Math.min(H-20,n.y));
  }});
}}

function draw() {{
  ctx.clearRect(0,0,W,H);
  // Links
  links.forEach(l => {{
    const s=nodes[l.source], t=nodes[l.target];
    ctx.beginPath();
    ctx.moveTo(s.x,s.y);
    ctx.lineTo(t.x,t.y);
    ctx.strokeStyle = l.group==='cross' ? '#2d2d4e' : '#1e1e3a';
    ctx.lineWidth = l.group==='cross' ? 0.5 : 1;
    ctx.stroke();
  }});
  // Nodes
  nodes.forEach(n => {{
    ctx.beginPath();
    ctx.arc(n.x,n.y,5,0,Math.PI*2);
    ctx.fillStyle = n.color;
    ctx.fill();
    ctx.strokeStyle = n.color + '44';
    ctx.lineWidth = 2;
    ctx.stroke();
    // Label
    ctx.fillStyle = n.color;
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillText(n.label, n.x+8, n.y+3);
  }});
}}

let frame = 0;
function loop() {{
  if(frame < 200) tick();
  draw();
  frame++;
  requestAnimationFrame(loop);
}}
loop();
</script>
</body>
</html>
"""
