import json
import pandas as pd
import plotly.express as px

from pyodide.http import open_url
from js import document

# -----------------------------
# Small DOM helpers
# -----------------------------
def set_status(text: str):
    el = document.getElementById("status-text")
    if el:
        el.innerText = text

def set_progress(pct: int):
    pct = max(0, min(100, int(pct)))
    bar = document.getElementById("csv-progress")
    txt = document.getElementById("csv-progress-text")
    if bar: bar.value = pct
    if txt: txt.innerText = f"{pct}%"

def set_html(id_: str, html: str):
    el = document.getElementById(id_)
    if el:
        el.innerHTML = html

def get_value(id_: str) -> str:
    el = document.getElementById(id_)
    return el.value if el else ""

# -----------------------------
# Data loading
# -----------------------------
CSV_INDEX_URL = "csv/index.json"   # served from /docs/csv/index.json on GH Pages
CSV_BASE_URL  = "csv/"             # files are under /docs/csv/

def load_index() -> list[str]:
    """Return a list of CSV filenames from csv/index.json."""
    set_status("loading index…")
    try:
        with open_url(CSV_INDEX_URL) as f:
            data = json.load(f)
        files = list(data.get("files", []))
        set_status("index loaded")
        return files
    except Exception as e:
        set_status("index load failed")
        set_html("csv-preview", f"<div class='error'>Failed to load index.json: {e}</div>")
        return []

def populate_select(files: list[str]):
    sel = document.getElementById("csv-select")
    if not sel:
        return
    # Clear
    while sel.firstChild:
        sel.removeChild(sel.firstChild)
    # Add options
    for name in files:
        opt = document.createElement("option")
        opt.value = name
        opt.text  = name
        sel.appendChild(opt)

def load_csv_to_df(filename: str) -> pd.DataFrame:
    """Read CSV into pandas using pyodide.open_url (same-origin)."""
    url = CSV_BASE_URL + filename
    set_status(f"loading {filename}…")
    set_progress(10)
    df = pd.read_csv(open_url(url))
    set_progress(100)
    set_status("ready")
    return df

# -----------------------------
# Visualization
# -----------------------------
def render_overview(df: pd.DataFrame):
    # Minimal sanity
    if not {"train_time_s", "test_acc_pct"}.issubset(df.columns):
        set_html("csv-preview", "<div class='error'>CSV missing required columns: train_time_s, test_acc_pct</div>")
        return

    # Show shape + head preview
    preview = (
        f"<div><b>Rows:</b> {len(df)}, "
        f"<b>Cols:</b> {len(df.columns)}</div>"
        + df.head(8).to_html(index=False)
    )
    set_html("csv-preview", preview)

    # Summary stats
    desc = df[["train_time_s", "test_acc_pct"]].describe().to_html()
    set_html("summary", desc)

    # Scatter plot
    color_col = "lr" if "lr" in df.columns else None
    fig = px.scatter(
        df.dropna(subset=["train_time_s","test_acc_pct"]),
        x="train_time_s", y="test_acc_pct",
        color=color_col,
        title="Speed vs Test Accuracy",
        hover_data=[c for c in ["model_key","layers","units","batch","threads","lr","momentum","decay","step"] if c in df.columns]
    )
    fig.update_traces(marker=dict(size=8, opacity=0.9))
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40))
    set_html("plots", fig.to_html(include_plotlyjs="cdn", full_html=False))

# -----------------------------
# Wire up controls
# -----------------------------
def reload_index(_=None):
    files = load_index()
    populate_select(files)
    # Optional: auto-load first file if present
    if files:
        document.getElementById("csv-select").value = files[0]

def load_selected(_=None):
    fname = get_value("csv-select")
    if not fname:
        set_html("csv-preview", "<div class='warn'>No CSV selected.</div>")
        return
    try:
        df = load_csv_to_df(fname)
        render_overview(df)
    except Exception as e:
        set_html("csv-preview", f"<div class='error'>Failed to load CSV: {e}</div>")

# Attach listeners
btn_reload = document.getElementById("btn-reload-index")
if btn_reload:
    btn_reload.addEventListener("click", reload_index)

btn_load = document.getElementById("btn-load-selected")
if btn_load:
    btn_load.addEventListener("click", load_selected)

# Initial boot
reload_index()
set_status("ready")
