import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


QPaper = {10: 60, 50: 81, 300: 93.5}
DnC = {100: 82.9, 400: 87, 1600: 91.9, 6400: 95}
BioMim = {10: 62.4, 30: 76.65}
MultMem = {50: 77.2}
Brivio = {500: 85}
Resistive = {50: 76.8}
DoubBarr = {10: 65, 20: 70, 50: 77, 100: 82}
CNT = {10: 65, 20: 66, 40: 68, 80: 70}
ThisPaper = {
    10: [52.56, 2.19],
    30: [68.23, 2.22],
    100: [84.51, 1.02],
    300: [86, 0.52],
    500: [86.26, 0.78],
}

fig = go.Figure()

works = [ThisPaper, DnC, Brivio, QPaper, CNT, MultMem, Resistive, BioMim, DoubBarr]
labels = ["This Paper", "[25]", "[26]", "[27]", "[32]", "[63]", "[65]", "[66]", "[67]"]
colors = [
    "royalblue",
    "firebrick",
    "yellow",
    "darkgreen",
    "black",
    "magenta",
    "black",
    "cyan",
    "magenta",
]
marker_symbols = [
    "x",
    "x",
    "diamond",
    "circle",
    "triangle-up",
    "hexagram",
    "triangle-down",
    "cross",
    "square",
]
marker_symbols = [symbol + "-open" for symbol in marker_symbols]
for idx, work in enumerate(works):
    n_dim = np.array(list(work.values())).ndim
    assert n_dim in [1, 2]
    if n_dim == 1:
        if idx == 0:
            fig.add_trace(
                go.Scatter(
                    x=np.log10(list(work.keys())),
                    # x=list(work.keys()),
                    y=list(work.values()),
                    name=labels[idx],
                    line=dict(width=2, color=colors[idx]),
                    mode="lines+markers",
                    marker_line_width=2,
                    marker_symbol=marker_symbols[idx],
                    marker=dict(size=15),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=np.log10(list(work.keys())),
                    # x=list(work.keys()),
                    y=list(work.values()),
                    name=labels[idx],
                    line=dict(width=2, dash="dash", color=colors[idx]),
                    mode="lines+markers",
                    marker_line_width=2,
                    marker_symbol=marker_symbols[idx],
                    marker=dict(size=15),
                )
            )
    else:
        mean = list(x[0] for x in np.array(list(work.values())))
        std = list(x[1] for x in np.array(list(work.values())))
        if idx == 0:
            fig.add_trace(
                go.Scatter(
                    x=np.log10(list(work.keys())),
                    # x=list(work.keys()),
                    y=mean,
                    error_y=dict(
                        type="data", array=std, visible=True, thickness=4, width=12
                    ),
                    name=labels[idx],
                    line=dict(width=4, color=colors[idx]),
                    mode="lines+markers",
                    marker_line_width=3,
                    marker_symbol=marker_symbols[idx],
                    marker=dict(size=15),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=np.log10(list(work.keys())),
                    # x=list(work.keys()),
                    y=mean,
                    error_y=dict(
                        type="data", array=std, visible=True, thickness=6, width=12
                    ),
                    name=labels[idx],
                    line=dict(width=2, dash="dash", color=colors[idx]),
                    mode="lines+markers",
                    marker_line_width=2,
                    marker_symbol=marker_symbols[idx],
                    marker=dict(size=15),
                )
            )

fig.update_layout(
    xaxis_title="<b>log<sub>10</sub> (Number of Output Neurons)</b>",
    yaxis_title="<b>Classification Accuracy (%)</b>",
    plot_bgcolor="white",
    legend=dict(
        # orientation="h",
        yanchor="bottom",
        xanchor="right",
        x=0.95,
        y=0.05,
        font=dict(
            size=18,
        ),
        bordercolor="Black",
        borderwidth=4,
    ),
    height=750,
    width=750,
)
fig.update_xaxes(
    range=[1, 4],
    linewidth=4,
    linecolor="black",
    mirror=True,
    title_font=dict(size=22),
    tickfont=dict(size=16),
    tickvals=np.arange(1, 5, 1),
    # ticktext=[r"$\textbf{\text{10}^{\text{%d}}}$" % tick for tick in np.arange(1, 5, 1)],
    # gridwidth=0.05,
    # gridcolor="grey",
    zerolinecolor="black",
)
fig.update_yaxes(
    range=[50, 96],
    linewidth=4,
    linecolor="black",
    mirror=True,
    title_font=dict(size=22),
    tickfont=dict(size=16),
    tick0=50,
    dtick=5,
    gridwidth=0.01,
    gridcolor="lightgrey",
    zerolinecolor="black",
)
# fig.update(layout_showlegend=False)
fig.update_xaxes(tickfont_family="Arial Black")
fig.update_yaxes(tickfont_family="Arial Black")
fig.write_html(
    "Figure_12.html", include_plotlyjs="cdn", include_mathjax="cdn", full_html=True
)
fig.write_image("Figure_12.svg")
fig.show()
