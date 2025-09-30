# docs/assets/app.py

import io
import json
import pandas as pd
import plotly.express as px

from js import document, console
from pyodide.ffi import create_proxy
from pyodide.http import pyfetch

# --------------------------
# Helpers
# --------------------------
def _el(id_): return document.getElementById(id_)

def set_status(msg: str):
    e = _el("status-text")
    if e: e.innerText = msg

def set_progress(pct: int):
    p = _el("csv-progress")
    t = _el("csv-progress-text")
    pct = int(max(0, min(100, pct)))
    if p: p.value = pct
    if t: t.innerText = f"{pct}%"

def _format_numeric(val):
    try:
        return f"{float(val):.3f}"
    except Exception:
        return val

def df_preview_html(df, max_rows=15, max_cols=12):
    sub = df.copy()
    if sub.shape[1] > max_cols:
        keep = list(sub.columns[: max_cols - 1]) + [sub.columns[-1]]
        sub = sub[keep]
    if sub.shape[0] > max_rows:
        sub = sub.head(max_rows)
    for c in sub.columns:
        if pd.api.types.is_numeric_dtype(sub[c]):
            sub[c] = sub[c].map(_format_numeric)
    return sub.to_html(index=False, classes="tbl tbl-compact grid", border=0, escape=False)

def preview_df(df: pd.DataFrame):
    box = _el("csv-preview")
    if box:
        box.innerHTML = df_preview_html(df)

def describe_df(df: pd.DataFrame) -> str:
    cols = [c for c in ("train_time_s", "test_acc_pct") if c in df.columns]
    if not cols:
        return "<em>No numeric summary columns (train_time_s, test_acc_pct) present.</em>"
    try:
        return df[cols].describe().to_html(classes="small tbl grid", border=0)
    except Exception as e:
        return f"<em>Could not compute summary: {e}</em>"

# --------------------------
# Figure builders
# --------------------------
def build_speed_acc_figs(df: pd.DataFrame):
    sub = df.dropna(subset=["train_time_s", "test_acc_pct"]).copy()
    for c in ("train_time_s", "test_acc_pct"):
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

    color_col = next((c for c in ["lr","momentum","decay","step","layers","units","batch","threads"]
                      if c in sub.columns), None)

    # fig 1: plain scatter
    fig1 = px.scatter(
        sub, x="train_time_s", y="test_acc_pct",
        color=color_col,
        hover_data=[c for c in ["model_key","layers","units","batch","threads","lr","momentum","decay","step"]
                    if c in sub.columns],
        title="Speed vs Test Accuracy"
    )
    fig1.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.6, color="white")))
    fig1.update_layout(margin=dict(l=60, r=20, t=50, b=50), template="plotly_white")
    fig1.update_yaxes(ticksuffix="%")

    # fig 2: with ±σ bands
    mean = sub["test_acc_pct"].mean()
    std  = sub["test_acc_pct"].std()
    x_min = sub["train_time_s"].min()
    x_max = sub["train_time_s"].max()

    fig2 = px.scatter(sub, x="train_time_s", y="test_acc_pct",
                      title="Speed vs Test Accuracy — with σ bands")
    # shaded rectangles under the points
    bands = [
        (mean-std,   mean+std,   "rgba(230,159,0,0.18)"),   # ±1σ
        (mean-2*std, mean+2*std, "rgba(86,180,233,0.12)"),  # ±2σ
        (mean-3*std, mean+3*std, "rgba(0,158,115,0.08)"),   # ±3σ
    ]
    shapes = []
    for lo, hi, color in bands:
        shapes.append(dict(
            type="rect", xref="x", yref="y",
            x0=x_min, x1=x_max, y0=lo, y1=hi,
            fillcolor=color, line=dict(width=0), layer="below"
        ))
    fig2.update_layout(shapes=shapes, margin=dict(l=60, r=20, t=50, b=50), template="plotly_white")
    fig2.add_hline(y=mean, line_color="red", line_width=1.5)
    fig2.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0.6, color="white")))
    fig2.update_yaxes(ticksuffix="%")

    return fig1, fig2

def render_all(df: pd.DataFrame):
    preview_df(df)
    s = _el("summary-log")
    if s:
        s.innerHTML = describe_df(df)

    plots_div = _el("plots")
    sigma_div = _el("plot-speed-acc-sigma")
    if not (plots_div and sigma_div):
        return

    fig1, fig2 = build_speed_acc_figs(df)

    # Embed as self-contained HTML snippets; avoids JS interop issues
    plots_div.innerHTML = fig1.to_html(include_plotlyjs="cdn", full_html=False)
    sigma_div.innerHTML = fig2.to_html(include_plotlyjs=False, full_html=False)

# --------------------------
# CSV loading + manifest
# --------------------------
async def fetch_text(url: str) -> str:
    resp = await pyfetch(url)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status} for {url}")
    return await resp.string()

def _normalize_manifest(data):
    # accept several shapes and normalize to [{'name', 'path'}]
    files = []
    if isinstance(data, dict) and "files" in data:
        files = data["files"]
    elif isinstance(data, dict):
        files = [{"name": k, "path": v} for k, v in data.items()]
    elif isinstance(data, list):
        files = data
    else:
        return []

    norm = []
    for it in files:
        if isinstance(it, str):
            path = it; name = it.rsplit("/", 1)[-1]
            norm.append({"name": name, "path": path})
        elif isinstance(it, dict):
            path = it.get("path") or it.get("file") or it.get("url")
            name = it.get("name") or it.get("label") or (path.rsplit("/", 1)[-1] if path else None)
            if path and name:
                norm.append({"name": name, "path": path})
    return norm

async def load_manifest():
    sel = _el("csv-select")
    if not sel:
        set_status("csv-select not found")
        return
    set_status("loading list…"); set_progress(5)
    try:
        txt = await fetch_text("./csv/index.json")
        data = json.loads(txt)
        items = _normalize_manifest(data)
        sel.innerHTML = ""
        if not items:
            set_status("no CSVs found in manifest"); return
        for item in items:
            opt = document.createElement("option")
            opt.value = item["path"]; opt.text = item["name"]
            sel.appendChild(opt)
        set_status("ready"); set_progress(0)
    except Exception as e:
        console.error(e)
        set_status(f"failed to load list: {e}")

async def load_selected_csv(_evt=None):
    sel = _el("csv-select")
    if not sel or not sel.value:
        set_status("no file selected"); return
    url = sel.value
    set_status(f"loading {url}…"); set_progress(15)
    try:
        txt = await fetch_text(url); set_progress(40)
        df = pd.read_csv(io.StringIO(txt))
        for c in ("train_time_s", "test_acc_pct"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        set_progress(85)
        render_all(df)
        set_progress(100); set_status("loaded")
    except Exception as e:
        console.error(e)
        set_status(f"load error: {e}")

# --------------------------
# DOM wiring
# --------------------------
def bind_dom():
    (_el("btn-reload-index")
        and _el("btn-reload-index").addEventListener("click", create_proxy(lambda e: load_manifest())))
    (_el("btn-load-selected")
        and _el("btn-load-selected").addEventListener("click", create_proxy(load_selected_csv)))
    set_progress(0)
    (_el("csv-preview") and setattr(_el("csv-preview"), "innerHTML", ""))
    (_el("summary-log") and setattr(_el("summary-log"), "innerHTML", "<em>Select a CSV, then Load Selected.</em>"))
    (_el("plots") and setattr(_el("plots"), "innerHTML", ""))
    (_el("plot-speed-acc-sigma") and setattr(_el("plot-speed-acc-sigma"), "innerHTML", ""))
    set_status("ready")

async def boot():
    bind_dom()
    await load_manifest()

if document.readyState == "loading":
    def _on_ready(_evt):
        import asyncio; asyncio.ensure_future(boot())
    document.addEventListener("DOMContentLoaded", create_proxy(_on_ready))
else:
    import asyncio; asyncio.ensure_future(boot())
