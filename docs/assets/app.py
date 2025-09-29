# docs/assets/app.py

import io
import math
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from js import document, FileReader, console
from pyodide.ffi import create_proxy

OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

# ---------- helpers ----------
def _set_status(txt: str):
    document.getElementById("status-text").innerText = txt

def _numeric_like(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s): return True
    try:
        pd.to_numeric(s)
        return True
    except Exception:
        return False

def _auto_discrete_labels(series: pd.Series, max_bins: int = 8, strategy: str = "quantile") -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series.astype("string")

    nunique = s.nunique(dropna=True)
    if nunique <= max_bins:
        return series.astype("string")

    if _numeric_like(series):
        s_num = pd.to_numeric(series, errors="coerce")
        if strategy == "uniform":
            cats = pd.cut(s_num, bins=max_bins, include_lowest=True)
        else:
            cats = pd.qcut(s_num, q=max_bins, duplicates="drop")

        intervals = list(cats.cat.categories)
        labels = []
        for i, iv in enumerate(intervals):
            lo, hi = iv.left, iv.right
            labels.append(f"[{lo:.4g}, {hi:.4g})" if i < len(intervals)-1 else f"[{lo:.4g}, {hi:.4g}]")
        out = pd.Series(cats, index=series.index).astype(str)
        mapping = {str(intervals[i]): labels[i] for i in range(len(intervals))}
        return out.map(lambda v: mapping.get(v, v))

    # high-cardinality categorical -> top-K + "other"
    counts = s.astype("string").value_counts()
    keep = set(counts.index[: max_bins - 1])
    return series.astype("string").map(lambda v: v if v in keep else "other")

def _df_summary_html(df: pd.DataFrame) -> str:
    cols = [c for c in ["train_time_s","test_acc_pct"] if c in df.columns]
    if not cols:
        return "<em>No train_time_s / test_acc_pct columns found.</em>"
    return df[cols].describe().to_html(classes="tiny-table", border=0)

def _scatter_speed_acc(df: pd.DataFrame, color_by: str=None, mode="auto", bins=8, bin_strategy="quantile",
                       logx=False, logy=False, title="Speed vs Test Accuracy"):
    data = df.dropna(subset=["train_time_s","test_acc_pct"]).copy()
    color_kwargs = {}
    color_title = color_by or ""

    if color_by and color_by in data.columns:
        is_num = _numeric_like(data[color_by])
        chosen = mode
        if chosen == "auto":
            chosen = "continuous" if is_num else "discrete"

        if chosen == "continuous" and is_num:
            data[color_by] = pd.to_numeric(data[color_by], errors="coerce")
            color_kwargs.update(dict(color=data[color_by], color_continuous_scale="Cividis"))
        else:
            cats = _auto_discrete_labels(data[color_by], max_bins=bins, strategy=bin_strategy)
            data[color_by] = cats
            data = data.dropna(subset=[color_by])
            k = max(2, min(len(OKABE_ITO), data[color_by].nunique()))
            color_kwargs.update(dict(color=data[color_by], color_discrete_sequence=OKABE_ITO[:k]))
            if cats.nunique(dropna=True) > bins:
                color_title = f"{color_by} (binned)"

    fig = px.scatter(
        data,
        x="train_time_s", y="test_acc_pct",
        hover_data=[c for c in ["model_key","layers","units","batch","threads","lr","momentum","decay","step"] if c in data.columns],
        labels={"train_time_s":"Train time (s)","test_acc_pct":"Test Accuracy (%)", color_by or "":" "},
        title=title,
        **color_kwargs
    )
    fig.update_traces(marker=dict(size=8, opacity=0.9, line=dict(width=0.6, color="white")))
    fig.update_xaxes(type="log" if logx else "linear", showgrid=True)
    fig.update_yaxes(type="log" if logy else "linear", showgrid=True, ticksuffix="%")
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=50, b=40), legend_title=color_title)
    return fig

def _scatter_with_sigma(df: pd.DataFrame, title="Speed vs Test Accuracy (σ bands)"):
    data = df.dropna(subset=["train_time_s","test_acc_pct"]).copy()
    if data.empty:
        fig = go.Figure()
        fig.update_layout(title="No data", template="plotly_white")
        return fig

    mean_acc = data["test_acc_pct"].mean()
    std_acc  = data["test_acc_pct"].std()
    fig = px.scatter(
        data, x="train_time_s", y="test_acc_pct",
        title=title,
        hover_data=[c for c in ["model_key","layers","units","batch","threads","lr","momentum","decay","step"] if c in data.columns],
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0.6, color="white")))
    xmin, xmax = data["train_time_s"].min(), data["train_time_s"].max()
    # extend to axes edges (Plotly updates it dynamically; we use data range which is fine)
    bands = [
        (mean_acc-std_acc,  mean_acc+std_acc,  "rgba(255,165,0,0.20)"),  # orange
        (mean_acc-2*std_acc,mean_acc+2*std_acc,"rgba(0,116,217,0.12)"),  # blue-ish
        (mean_acc-3*std_acc,mean_acc+3*std_acc,"rgba(0,128,0,0.08)"),    # green
    ]
    for lo, hi, rgba in bands:
        fig.add_shape(type="rect", x0=xmin, x1=xmax, y0=lo, y1=hi,
                      line=dict(width=0), fillcolor=rgba, layer="below")
    fig.add_hline(y=mean_acc, line_color="red", line_width=1.5)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True, ticksuffix="%")
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=50, b=40))
    return fig

def _render_plot(div_id: str, fig):
    html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    document.getElementById(div_id).innerHTML = html

# ---------- state ----------
STATE = {
    "df": None
}

# ---------- UI wiring ----------
def _populate_color_by(df: pd.DataFrame):
    sel = document.getElementById("color-by")
    sel.innerHTML = ""
    candidates = [c for c in ["lr","momentum","decay","step","layers","units","batch","threads"] if c in df.columns]
    for c in candidates:
        opt = document.createElement("option")
        opt.value = c
        opt.textContent = c
        sel.appendChild(opt)

def _read_controls():
    color_by = document.getElementById("color-by").value or None
    mode     = document.getElementById("color-mode").value
    bins     = int(document.getElementById("bins-count").value or "8")
    strat    = document.getElementById("bin-strategy").value
    logx     = document.getElementById("log-x").checked
    logy     = document.getElementById("log-y").checked
    return color_by, mode, bins, strat, logx, logy

def _toggle_bin_controls():
    df = STATE["df"]
    if df is None:
        document.getElementById("bin-controls").style.display = "none"
        return
    color_by, mode, bins, strat, *_ = _read_controls()
    if (mode != "discrete") or (color_by is None) or (color_by not in df.columns):
        document.getElementById("bin-controls").style.display = "none"
        return
    s = df[color_by].dropna()
    show = _numeric_like(s) and s.nunique(dropna=True) > bins
    document.getElementById("bin-controls").style.display = ("inline-flex" if show else "none")

def _redraw_plots():
    df = STATE["df"]
    if df is None: return
    color_by, mode, bins, strat, logx, logy = _read_controls()
    fig1 = _scatter_speed_acc(df, color_by=color_by, mode=mode, bins=bins, bin_strategy=strat, logx=logx, logy=logy)
    _render_plot("plot-speed-acc", fig1)
    fig2 = _scatter_with_sigma(df)
    _render_plot("plot-speed-acc-sigma", fig2)

# ---------- file handling ----------
def _handle_file_change(evt):
    files = evt.target.files
    if not files or files.length == 0:
        return
    f = files[0]
    reader = FileReader.new()

    def onload(e):
        try:
            text = e.target.result
            df = pd.read_csv(io.StringIO(text))
        except Exception as ex:
            console.error("CSV parse failed:", str(ex))
            _set_status("CSV parse failed")
            return

        STATE["df"] = df
        document.getElementById("csv-progress").value = 100
        document.getElementById("csv-progress-text").innerText = "100%"

        # brief preview
        document.getElementById("csv-preview").innerHTML = _df_summary_html(df)

        _populate_color_by(df)
        _toggle_bin_controls()
        _redraw_plots()
        document.querySelector('.tab[data-panel="charts"]').click()
        _set_status("ready")

    reader.onload = create_proxy(onload)
    reader.readAsText(f)
    _set_status("loading…")

def _load_demo(_evt=None):
    # Tiny demo inlined; replace with fetch if you host a sample CSV
    sample = "train_time_s,test_acc_pct,lr,momentum,decay,step,layers,units,batch,threads\n" \
             "10.5,97.4,0.05,0.9,0.95,3,2,\"128,64\",128,8\n" \
             "7.2,96.8,0.05,0.9,0.95,3,1,64,64,8\n" \
             "5.9,95.9,0.03,0.9,0.95,3,1,64,32,8\n"
    df = pd.read_csv(io.StringIO(sample))
    STATE["df"] = df
    document.getElementById("csv-progress").value = 100
    document.getElementById("csv-progress-text").innerText = "100%"
    document.getElementById("csv-preview").innerHTML = _df_summary_html(df)
    _populate_color_by(df)
    _toggle_bin_controls()
    _redraw_plots()
    document.querySelector('.tab[data-panel="charts"]').click()
    _set_status("demo loaded")

# ---------- boot ----------
def _boot():
    file_input = document.getElementById("file-input")
    if file_input is not None:
        file_input.addEventListener("change", create_proxy(_handle_file_change))
    btn_demo = document.getElementById("btn-demo")
    if btn_demo is not None:
        btn_demo.addEventListener("click", create_proxy(_load_demo))

    # controls that trigger redraw
    for id_ in ["color-by","color-mode","bin-strategy","bins-count","log-x","log-y"]:
        el = document.getElementById(id_)
        if el is not None:
            el.addEventListener("change", create_proxy(lambda e: (_toggle_bin_controls(), _redraw_plots())))

    _set_status("idle")

document.addEventListener("DOMContentLoaded", create_proxy(lambda e: _boot()))
