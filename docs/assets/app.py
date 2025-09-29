# docs/assets/app.py

import io
import json
import pandas as pd
import plotly.express as px

from js import document, console
from pyodide.ffi import create_proxy
from pyodide.http import pyfetch

# --------------------------
# Small DOM helpers
# --------------------------
def _el(id_): 
    return document.getElementById(id_)

def set_status(msg: str):
    e = _el("status-text")
    if e:
        e.innerText = msg

def set_progress(pct: int):
    p = _el("csv-progress")
    t = _el("csv-progress-text")
    if p:
        p.value = max(0, min(100, int(pct)))
    if t:
        t.innerText = f"{int(max(0, min(100, pct)))}%"

# --------------------------
# Pretty preview helpers
# --------------------------
def _format_numeric(val):
    try:
        f = float(val)
        return f"{f:.2f}"
    except Exception:
        return val

def df_preview_html(df: pd.DataFrame, max_rows=15, max_cols=12) -> str:
    """Compact HTML table with sticky header + zebra rows."""
    sub = df.copy()

    # limit columns (keep the last column so long paths remain visible)
    if sub.shape[1] > max_cols:
        keep = list(sub.columns[: max_cols - 1]) + [sub.columns[-1]]
        sub = sub[keep]

    # limit rows
    if sub.shape[0] > max_rows:
        sub = sub.head(max_rows)

    # light numeric formatting
    for c in sub.columns:
        if pd.api.types.is_numeric_dtype(sub[c]):
            sub[c] = sub[c].map(_format_numeric)

    return sub.to_html(
        index=False,
        classes="tbl tbl-compact",
        border=0,
        escape=False,
    )

def preview_df(df: pd.DataFrame):
    """Fill #csv-preview with a neat, scrollable table + a tiny shape note."""
    box = _el("csv-preview")
    if not box:
        return

    # Build table
    html = df_preview_html(df)

    # Tiny shape note
    shape_note = f'<div class="tiny light" style="margin:4px 8px 6px">rows: {len(df):,} • cols: {len(df.columns)}</div>'

    box.innerHTML = shape_note + html

# --------------------------
# Simple summary & plot
# --------------------------
def describe_df(df: pd.DataFrame) -> str:
    cols = [c for c in ("train_time_s", "test_acc_pct") if c in df.columns]
    if not cols:
        return "<em>No numeric summary columns (train_time_s, test_acc_pct) present.</em>"
    try:
        return df[cols].describe().to_html(classes="small")
    except Exception as e:
        return f"<em>Could not compute summary: {e}</em>"

def plot_speed_vs_acc(df: pd.DataFrame) -> str:
    target = {"train_time_s", "test_acc_pct"}
    if not target.issubset(df.columns):
        missing = target - set(df.columns)
        return f"<em>CSV missing required columns: {', '.join(sorted(missing))}</em>"

    color_col = next((c for c in ["lr","momentum","decay","step","layers","units","batch","threads"]
                      if c in df.columns), None)

    data = df.dropna(subset=["train_time_s", "test_acc_pct"]).copy()
    for c in ("train_time_s", "test_acc_pct"):
        data[c] = pd.to_numeric(data[c], errors="coerce")

    fig = px.scatter(
        data,
        x="train_time_s",
        y="test_acc_pct",
        color=color_col,
        hover_data=[c for c in ["model_key","layers","units","batch","threads","lr","momentum","decay","step"]
                    if c in data.columns],
        title="Speed vs Test Accuracy"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.6, color="white")))
    fig.update_layout(margin=dict(l=60, r=20, t=50, b=50), template="plotly_white")
    fig.update_yaxes(ticksuffix="%")
    return fig.to_html(include_plotlyjs="cdn", full_html=False)

def render_all(df: pd.DataFrame):
    preview_df(df)

    s = _el("summary-log")
    if s:
        s.innerHTML = describe_df(df)

    plot_div = _el("plots")
    if plot_div:
        plot_div.innerHTML = plot_speed_vs_acc(df)

# --------------------------
# CSV loading + manifest
# --------------------------
async def fetch_text(url: str) -> str:
    resp = await pyfetch(url)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status} for {url}")
    return await resp.string()

def _normalize_manifest(data):
    """
    Accepts:
      1) {"files": [{"name": "...", "path": "csv/.."}, ...]}
      2) ["csv/a.csv", "csv/b.csv", ...]
      3) [{"path": "csv/a.csv", "name": "A"}, {"file": "csv/b.csv", "label": "B"}]
      4) {"A": "csv/a.csv", "B": "csv/b.csv"}   # mapping name -> path
    Returns: list[{"name": str, "path": str}]
    """
    files = []
    # Case 1: wrapper object with "files"
    if isinstance(data, dict) and "files" in data:
        files = data["files"]
    # Case 4: mapping name -> path
    elif isinstance(data, dict):
        files = [{"name": k, "path": v} for k, v in data.items()]
    # Case 2 or 3: raw list
    elif isinstance(data, list):
        files = data
    else:
        return []

    norm = []
    for it in files:
        if isinstance(it, str):
            path = it
            name = it.rsplit("/", 1)[-1]
            norm.append({"name": name, "path": path})
        elif isinstance(it, dict):
            path = it.get("path") or it.get("file") or it.get("url")
            name = it.get("name") or it.get("label")
            if not name and path:
                name = path.rsplit("/", 1)[-1]
            if path and name:
                norm.append({"name": name, "path": path})
    return norm

async def load_manifest():
    """Load docs/csv/index.json and populate the <select>."""
    sel = _el("csv-select")
    if not sel:
        set_status("csv-select not found")
        return

    set_status("loading list…")
    set_progress(5)
    try:
        txt = await fetch_text("./csv/index.json")
        data = json.loads(txt)
        items = _normalize_manifest(data)

        # Clear options
        sel.innerHTML = ""
        if not items:
            set_status("no CSVs found in manifest")
            return

        for item in items:
            opt = document.createElement("option")
            opt.value = item["path"]  # e.g., "csv/all_results.csv"
            opt.text  = item["name"]
            sel.appendChild(opt)

        set_status("ready")
        set_progress(0)
    except Exception as e:
        console.error(e)
        try:
            console.log("manifest raw:", txt)
        except Exception:
            pass
        set_status(f"failed to load list: {e}")

async def load_selected_csv(_evt=None):
    sel = _el("csv-select")
    if not sel or sel.value == "":
        set_status("no file selected")
        return
    url = sel.value
    set_status(f"loading {url}…")
    set_progress(15)
    try:
        txt = await fetch_text(url)
        set_progress(40)
        df = pd.read_csv(io.StringIO(txt))

        # Ensure numeric for key columns
        for c in ("train_time_s", "test_acc_pct"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        set_progress(85)
        render_all(df)
        set_progress(100)
        set_status("loaded")
    except Exception as e:
        console.error(e)
        set_status(f"load error: {e}")

# --------------------------
# DOM wiring
# --------------------------
def bind_dom():
    btn_reload = _el("btn-reload-index")
    if btn_reload:
        btn_reload.addEventListener("click", create_proxy(lambda e: load_manifest()))
    btn_load = _el("btn-load-selected")
    if btn_load:
        btn_load.addEventListener("click", create_proxy(load_selected_csv))

    # initial states
    set_progress(0)
    prev = _el("csv-preview");  prev and setattr(prev, "innerText", "")
    summ = _el("summary-log");  summ and setattr(summ, "innerHTML", "<em>Select a CSV, then Load Selected.</em>")
    plot = _el("plots");        plot and setattr(plot, "innerHTML", "")
    set_status("ready")

async def boot():
    bind_dom()
    await load_manifest()

# If DOM already loaded, run immediately; else wait.
if document.readyState == "loading":
    def _on_ready(_evt):
        import asyncio
        asyncio.ensure_future(boot())
    document.addEventListener("DOMContentLoaded", create_proxy(_on_ready))
else:
    import asyncio
    asyncio.ensure_future(boot())
