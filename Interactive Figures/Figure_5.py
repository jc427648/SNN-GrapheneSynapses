import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.io import loadmat

data = loadmat("STDP_window.mat")["out"]

x = data[0, :]
y = data[1, :] * 1e3

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        name="<b>STDP Window</b>",
        line=dict(color="darkgreen", width=6),
        mode="lines+markers",
        marker=dict(size=10),
    )
)
fig.update_layout(
    xaxis_title="<b>ΔT (ms)</b>",
    yaxis_title="<b>ΔI (mA)</b>",
    plot_bgcolor="white",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        font=dict(
            size=18,
        ),
    ),
    width=500,
    height=500,
)
fig.update_xaxes(
    range=[-80, 80],
    tickmode="array",
    linewidth=4,
    linecolor="black",
    mirror=True,
    title_font=dict(size=22),
    tickfont=dict(size=16),
    # gridwidth=0.25,
    # gridcolor="grey",
    # zerolinecolor="black",
)
fig.update_yaxes(
    range=[-2, 2],
    linewidth=4,
    linecolor="black",
    mirror=True,
    title_font=dict(size=22),
    tickfont=dict(size=16),
    tick0=-4,
    dtick=1,
    # gridwidth=0.25,
    # gridcolor="grey",
    # zerolinecolor="black",
)
fig.update_xaxes(tickfont_family="Arial Black")
fig.update_yaxes(tickfont_family="Arial Black")
fig.write_html(
    "Figure_5.html", include_plotlyjs="cdn", include_mathjax=False, full_html=True
)
fig.write_image("Figure_5.svg")
fig.show()
