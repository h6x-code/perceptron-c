# docs/assets/app.py

import io
import math
import pandas as pd
import plotly.express as px

from js import document, FileReader, console
from pyodide.ffi import create_proxy

STATUS = document.getElementById("status-text")
PROG   = document.getElementById("csv-progress")
PROGT  = document.getElementById("csv-progress-text")
PREV   = document.getElementById("csv-preview")
PLOTS  = document.getElementById("plot-speed-acc")      # charts tab, left tile
SUMMARY_HTML = document.getElementById("summary-log")    # summary tab (log-like area)

def set_status(msg: str):
    STATUS.innerText = msg

def set_progress(pct: int):
    PROG.value = max(0, min(100, int(pct)))
    PROGT.innerText = f"{int(PROG.value)}%"

def _describe(df: pd.DataFrame) -> str:
    cols = [c for c in ("train_time_s", "test_acc_pct") if c in df.columns]
    if not cols:
        return "<em>No numeric summary columns found.</em>"
    try:
        return df[cols].describe().to_html(classes="small")
    except Exception as e:
        return f"<em>Could not compute summary: {e}</em>"

def _plot_speed_vs_acc(df: pd.DataFrame) -> str:
    if not {"train_time_s", "test_acc_pct"} <= set(df.columns):
        return "<em>CSV missing required columns: train_time_s, test_acc_pct</em>"

    # Pick a reasonable default color column if available
    color_col = next((c for c in ["lr","momentum","decay","step","layers","units","batch","threads"]
                      if c in df.columns), None)

    fig = px.scatter(
        df.dropna(subset=["train_time_s", "test_acc_pct"]),
        x="train_time_s",
        y="test_acc_pct",
        color=color_col,
        hover_data=[c for c in ["model_key","layers","units","batch","threads","lr","momentum","decay","step"]
                    if c in df.columns],
        title="Speed vs Test Accuracy",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.6, color="white")))
    fig.update_layout(margin=dict(l=60, r=20, t=50, b=50), template="plotly_white")
    fig.update_yaxes(ticksuffix="%")
    # Return a self-contained snippet (Plotly JS will be loaded from CDN once)
    return fig.to_html(include_plotlyjs="cdn", full_html=False)

def _render_overview(df: pd.DataFrame):
    # Preview head()
    with io.StringIO() as s:
        df.head(8).to_string(buf=s, index=False, justify="left", max_cols=0)
        PREV.innerText = s.getvalue()

    # Summary
    SUMMARY_HTML.innerHTML = _describe(df)

    # Chart
    PLOTS.innerHTML = _plot_speed_vs_acc(df)

def _handle_text_csv(text: str):
    # Robust CSV read (Pyodide pandas supports this)
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        set_status(f"parse error: {e}")
        console.error(e)
        return

    # Minimal required columns
    required = {"train_time_s", "test_acc_pct"}
    missing = required - set(df.columns)
    if missing:
        set_status(f"CSV missing columns: {', '.join(sorted(missing))}")
        return

    # Coerce numeric columns we rely on
    for c in ("train_time_s", "test_acc_pct"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    set_progress(100)
    set_status("loaded")
    _render_overview(df)

def _on_file_change(evt):
    try:
        files = evt.target.files
        if not files or files.length == 0:
            set_status("no file selected")
            return
        set_status("reading…")
        set_progress(5)

        reader = FileReader()

        def onload(e):
            try:
                set_progress(50)
                text = e.target.result
                _handle_text_csv(text)
            except Exception as ex:
                set_status(f"error: {ex}")
                console.error(ex)

        reader.onload = create_proxy(onload)
        reader.readAsText(files[0])
    except Exception as ex:
        set_status(f"error: {ex}")
        console.error(ex)

def _on_demo_click(_evt):
    """
    Optional: if you add a demo CSV (e.g. docs/csv/demo.csv),
    this will fetch and render it. Otherwise, harmlessly no-ops.
    """
    set_status("loading demo…")
    set_progress(10)

    async def _fetch_demo():
        from pyodide.http import pyfetch
        try:
            # Adjust path if you store demo CSV differently
            resp = await pyfetch("./csv/demo.csv")
            if not resp.ok:
                set_status(f"demo fetch failed: {resp.status}")
                return
            set_progress(40)
            text = await resp.string()
            _handle_text_csv(text)
        except Exception as ex:
            set_status(f"demo error: {ex}")
            console.error(ex)

    # schedule (PyScript runs this coroutine)
    import asyncio
    asyncio.ensure_future(_fetch_demo())

def py_main():
    # Bind handlers once DOM is available (index.html loads PyScript after the DOM)
    file_input = document.getElementById("file-input")   # NOTE: matches index.html id exactly
    if file_input is None:
        # If you move the script tag above the input in HTML, guard for that:
        set_status("file input not found in DOM")
        return
    file_input.addEventListener("change", create_proxy(_on_file_change))

    demo_btn = document.getElementById("btn-demo")
    if demo_btn:
        demo_btn.addEventListener("click", create_proxy(_on_demo_click))

    # reset UI
    set_status("idle")
    set_progress(0)
    PREV.innerText = ""
    PLOTS.innerHTML = ""
    SUMMARY_HTML.innerHTML = "<em>Load a CSV to see summary and charts.</em>"

# Run on import
py_main()
