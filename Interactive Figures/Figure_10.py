import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


file = open("EvalN100_out.txt", "r")
lines = file.readlines()
file.close()
data = {0: 0}
for line in lines:
    if "Training progress: sample" in line:
        line = line.strip("\n")
        sample_idx_sub1 = "sample ("
        sample_idx_sub2 = " / "
        idx1 = line.index(sample_idx_sub1)
        idx2 = line.index(sample_idx_sub2)
        sample_idx = line[idx1 + len(sample_idx_sub1) : idx2]
        sample_idx_sub1 = "Running accuracy: "
        sample_idx_sub2 = "%"
        idx1 = line.index(sample_idx_sub1)
        idx2 = line.index(sample_idx_sub2)
        accuracy = float(line[idx1 + len(sample_idx_sub1) : idx2])
        data[sample_idx] = accuracy


fig = make_subplots(rows=2, cols=1, subplot_titles=("<b>(a)</b>", "<b>(b)</b>"))
fig.add_trace(
    go.Scatter(
        x=np.array(list(data.keys()), dtype=float), #* 1e-4,
        y=list(data.values()),
        line=dict(color="royalblue", width=6),
        marker=dict(size=10),
        mode="lines+markers",
    )
)

Num = [10, 30, 100, 300, 500]
Mean = [49.50, 63.04, 82.99, 83.72, 84.51]
Std = [2.19, 2.22, 1.14, 0.51, 0.92]
fig.add_trace(
    go.Scatter(
        x=Num,
        y=Mean,
        error_y=dict(type="data", array=Std, visible=True, thickness=6, width=6),
        mode="lines+markers",
        line=dict(color="royalblue", width=6),
        marker=dict(size=10),
    ),
    col=1,
    row=2,
)

fig.layout.annotations[0].update(y=1.01)
fig.layout.annotations[1].update(y=0.39)
fig.update_layout(
    xaxis_title="<b>Number of Images Presented</b>",
    yaxis_title="<b>Accuracy (%)</b>",
    xaxis2_title="<b>Number of Output Neurons</b>",
    yaxis2_title="<b>Accuracy (%)</b>",
    plot_bgcolor="white",
    showlegend=False,
    height=1000,
    width=500,
)
fig["layout"]["xaxis"].update(
    range=[0, 6e4],
    linewidth=4,
    linecolor="black",
    mirror=True,
    title_font=dict(size=22),
    tickfont=dict(size=16),
    # gridwidth=0.25,
    # gridcolor="grey",
    zerolinecolor="black",
)
fig["layout"]["yaxis"].update(
    range=[0, 90],
    tickmode="array",
    linewidth=4,
    linecolor="black",
    mirror=True,
    title_font=dict(size=22),
    tickfont=dict(size=16),
    # gridwidth=0.25,
    # gridcolor="grey",
    zerolinecolor="black",
)
fig["layout"]["xaxis2"].update(
    range=[0, 500],
    linewidth=4,
    linecolor="black",
    mirror=True,
    title_font=dict(size=22),
    tickfont=dict(size=16),
    # gridwidth=0.25,
    # gridcolor="grey",
    zerolinecolor="black",
)
fig["layout"]["yaxis2"].update(
    range=[45, 90],
    linewidth=4,
    linecolor="black",
    mirror=True,
    title_font=dict(size=22),
    tickvals=np.arange(45, 95, 5),
    tickfont=dict(size=16),
    # gridwidth=0.25,
    # gridcolor="grey",
    zerolinecolor="black",
)
fig.update_xaxes(tickfont_family="Arial Black")
fig.update_yaxes(tickfont_family="Arial Black")
fig['layout'].update(scene=dict(aspectmode="data"))
fig.write_html(
    "Figure_10.html", include_plotlyjs="cdn", include_mathjax="cdn", full_html=True
)
fig.write_image("Figure_10.svg")
fig.show()
