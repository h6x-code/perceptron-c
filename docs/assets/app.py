# docs/assets/app.py

import io
import json
import asyncio
import pandas as pd
import plotly.express as px

from js import document, console, window
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
    if p:
        p.value = max(0, min(100, int(pct)))
    if t:
        t.innerText = f"{int(max(0, min(100, pct)))}%"

def preview_df(df: pd.DataFrame):
    box = _el("csv-preview")
    if not box: return
    box.innerHTML = df_preview_html(df)

def describe_df(df: pd.DataFrame) -> str:
    cols = [c for c in ("train_time_s", "test_acc_pct") if c in df.columns]
    if not cols:
        return "<em>No numeric summary columns (train_time_s, test_acc_pct) present.</em>"
    try:
        return df[cols].describe().to_html(classes="small tbl grid")
    except Exception as e:
        return f"<em>Could not compute summary: {e}</em>"

def _format_numeric(val):
    try:
        f = float(val)
        return f"{f:.3f}"
    except Exception:
        return val

def df_preview_html(df, max_rows=15, max_cols=12):
    """Compact HTML table for preview box, with grid-friendly classes."""
    sub = df.copy()

    # limit columns (keep last col so long paths show up)
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
        classes="tbl tbl-compact grid",
        border=0,
        escape=False,
    )

async def wait_for_plotly(timeout=8.0):
    """Wait (up to timeout seconds) for window.Plotly to be available."""
    waited = 0.0
    step = 0.1
    while not getattr(window, "Plotly", None):
        await asyncio.sleep(step)
        waited += step
        if waited >= timeout:
            return False
    return True

def make_figures(df: pd.DataFrame):
    target = {"train_time_s", "test_acc_pct"}
    if not target.issubset(df.columns):
        missing = target - set(df.columns)
        msg = f"<em>CSV missing required columns: {', '.join(sorted(missing))}</em>"
        return msg, None

    color_col = next((c for c in ["lr","momentum","decay","step","layers","units","batch","threads"]
                      if c in df.columns), None)
    data = df.dropna(subset=["train_time_s", "test_acc_pct"]).copy()
    for c in ("train_time_s", "test_acc_pct"):
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # Scatter (colored if a sensible column exists)
    fig1 = px.scatter(
        data,
        x="train_time_s", y="test_acc_pct",
        color=color_col,
        hover_data=[c for c in ["model_key","layers","units","batch","threads","lr","momentum","decay","step"]
                    if c in data.columns],
        title="Speed vs Test Accuracy",
    )
    fig1.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.6, color="white")))
    fig1.update_layout(margin=dict(l=60, r=20, t=50, b=50), template="plotly_white")
    fig1.update_yaxes(ticksuffix="%")

    # Mean ±σ bands (second figure)
    mean_acc = data["test_acc_pct"].mean()
    std_acc  = data["test_acc_pct"].std()
    fig2 = px.scatter(
        data, x="train_time_s", y="test_acc_pct",
        title="Speed vs Test Accuracy (±σ bands)",
    )
    fig2.update_traces(marker=dict(size=7, opacity=0.75))
    fig2.add_hline(y=mean_acc, line_color="red", line_width=1.5)

    # shaded bands using shapes
    y1 = mean_acc - std_acc
    y2 = mean_acc + std_acc
    y3 = mean_acc - 2*std_acc
    y4 = mean_acc + 2*std_acc
    y5 = mean_acc - 3*std_acc
    y6 = mean_acc + 3*std_acc

    fig2.add_hrect(y0=y1, y1=y2, fillcolor="orange", opacity=0.20, line_width=0)
    fig2.add_hrect(y0=y3, y1=y4, fillcolor="blue",   opacity=0.10, line_width=0)
    fig2.add_hrect(y0=y5, y1=y6, fillcolor="green",  opacity=0.05, line_width=0)
    fig2.update_layout(margin=dict(l=60, r=20, t=50, b=50), template="plotly_white")
    fig2.update_yaxes(ticksuffix="%")

    return None, (fig1, fig2)

async def render_all(df: pd.DataFrame):
    preview_df(df)
    s = _el("summary-log")
    if s:
        s.innerHTML = describe_df(df)

    # Make figures (or message if columns missing)
    msg, figs = make_figures(df)
    plots_div = _el("plots")
    sigma_div = _el("plot-speed-acc-sigma")
    if msg:
        if plots_div: plots_div.innerHTML = msg
        if sigma_div: sigma_div.innerHTML = ""
        return

    fig1, fig2 = figs

    # Try native Plotly if available; otherwise fallback to embedding
    have_plotly = await wait_for_plotly(timeout=8.0)
    
    if have_plotly and getattr(window, "Plotly", None):
        # Prepare JSON specs
        spec1 = fig1.to_plotly_json()
        spec2 = fig2.to_plotly_json()

        if plots_div:
            window.Plotly.newPlot(
                plots_div,
                spec1.get("data", []),
                spec1.get("layout", {}),
                {"responsive": True},
            )
        if sigma_div:
            window.Plotly.newPlot(
                sigma_div,
                spec2.get("data", []),
                spec2.get("layout", {}),
                {"responsive": True},
            )
    else:
        # Fallback: embed full HTML (includes plotly.js)
        if plots_div:
            plots_div.innerHTML = fig1.to_html(include_plotlyjs="cdn", full_html=False)
        if sigma_div:
            sigma_div.innerHTML = fig2.to_html(include_plotlyjs="cdn", full_html=False)

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
    sel = document.getElementById("csv-select")
    if not sel:
        set_status("csv-select not found")
        return

    set_status("loading list…")
    set_progress(5)
    try:
        txt = await fetch_text("./csv/index.json")
        data = json.loads(txt)
        items = _normalize_manifest(data)

        sel.innerHTML = ""
        if not items:
            set_status("no CSVs found in manifest")
            return

        for item in items:
            opt = document.createElement("option")
            opt.value = item["path"]
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
        for c in ("train_time_s", "test_acc_pct"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        set_progress(85)
        await render_all(df)
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
    set_progress(0)
    prev = _el("csv-preview");  prev and setattr(prev, "innerHTML", "")
    summ = _el("summary-log");  summ and setattr(summ, "innerHTML", "<em>Select a CSV, then Load Selected.</em>")
    plot = _el("plots");        plot and setattr(plot, "innerHTML", "")
    sigma = _el("plot-speed-acc-sigma"); sigma and setattr(sigma, "innerHTML", "")
    set_status("ready")

async def boot():
    bind_dom()
    await load_manifest()

if document.readyState == "loading":
    def _on_ready(_evt):
        import asyncio
        asyncio.ensure_future(boot())
    document.addEventListener("DOMContentLoaded", create_proxy(_on_ready))
else:
    import asyncio
    asyncio.ensure_future(boot())
