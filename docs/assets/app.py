# app.py (drop-in)

import io
import pandas as pd
import plotly.express as px

from js import document, FileReader, setTimeout
from pyodide.ffi import create_proxy

# ---------- small DOM helpers ----------
def S (id_):
    return document.getElementById(id_)

def set_html(id_, html):
    el = S(id_)
    if el is not None:
        el.innerHTML = html

def set_text(id_, text):
    el = S(id_)
    if el is not None:
        el.innerText = text

def set_value(id_, value):
    el = S(id_)
    if el is not None:
        el.value = value

# ---------- event handlers ----------
def handle_file(event):
    files = event.target.files
    if not files or files.length == 0:
        print("[app] no file selected")
        return

    file = files[0]
    reader = FileReader()

    def onload(e):
        try:
            text = e.target.result
            df = pd.read_csv(io.StringIO(text))

            # Progress (optional elements)
            set_value("csv-progress", 100)
            set_text("csv-progress-text", "100%")

            # Summary (if the columns exist)
            cols = [c for c in ["train_time_s", "test_acc_pct"] if c in df.columns]
            if cols:
                summary_html = df[cols].describe().to_html()
            else:
                summary_html = "<em>Columns 'train_time_s' and/or 'test_acc_pct' not found.</em>"
            set_html("summary", summary_html)

            # Plot (only if required columns exist)
            if "train_time_s" in df.columns and "test_acc_pct" in df.columns:
                color_col = None
                for cand in ["lr", "momentum", "decay", "step", "layers", "units", "batch", "threads"]:
                    if cand in df.columns:
                        color_col = cand
                        break

                fig = px.scatter(
                    df,
                    x="train_time_s",
                    y="test_acc_pct",
                    color=color_col,
                    title="Speed vs. Test Accuracy",
                    labels={"train_time_s": "Train time (s)", "test_acc_pct": "Test Accuracy (%)"},
                )
                fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.6, color="white")))
                fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40))
                fig.update_yaxes(ticksuffix="%")

                fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
                set_html("plots", fig_html)
            else:
                set_html("plots", "<em>Missing required columns to plot.</em>")

        except Exception as ex:
            set_html("summary", f"<pre style='color:#b00'>Error parsing CSV: {ex}</pre>")
            set_html("plots", "")
            raise

    reader.onload = create_proxy(onload)
    reader.readAsText(file)

# Bind after DOM is ready; retry if the element isn't yet present
def bind_input():
    inp = S("csv-input")
    if inp is not None:
        inp.addEventListener("change", create_proxy(handle_file))
        # optional: clear any prior summary/plots
        # set_html("summary", "")
        # set_html("plots", "")
        print("[app] bound #csv-input")
    else:
        # DOM not ready or element not in page; try again shortly
        setTimeout(create_proxy(lambda *_: bind_input()), 50)

bind_input()
