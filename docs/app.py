import pandas as pd
import plotly.express as px
from js import document

def handle_file(event):
    file = event.target.files[0]
    reader = __new__(FileReader())

    def onload(e):
        text = e.target.result
        df = pd.read_csv(pd.compat.StringIO(text))

        # Update progress
        progress = document.getElementById("csv-progress")
        progress.value = 100
        document.getElementById("csv-progress-text").innerText = "100%"

        # Display summary
        summary = df[["train_time_s", "test_acc_pct"]].describe().to_html()
        document.getElementById("summary").innerHTML = summary

        # Plot
        fig = px.scatter(df, x="train_time_s", y="test_acc_pct",
                         color="lr" if "lr" in df.columns else None,
                         title="Speed vs Test Accuracy")
        fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
        document.getElementById("plots").innerHTML = fig_html

    reader.onload = onload
    reader.readAsText(file)

# Wire up file input
file_input = document.getElementById("csv-input")
file_input.addEventListener("change", handle_file)
