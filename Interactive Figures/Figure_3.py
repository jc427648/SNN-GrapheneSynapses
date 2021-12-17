import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.io import loadmat


data_1 = pd.read_csv("pentacenesingle200_slow_k2400.txt", sep="\t")
data_2 = pd.read_csv("pentacenesingle200_2_slow_k2400.txt", sep="\t")
data_3 = pd.read_csv("pentacenesingle200_3_slow_k2400.txt", sep="\t")
data_4 = pd.read_csv("pentacenesingle200_4_slow_k2400.txt", sep="\t")

t = np.linspace(0, data_4["Time[sec]"].values[-1], len(data_4["Time[sec]"].values))
V_ref = data_4["VOLT[V]"].values
I_ref = data_4["CURR[A]"].values
I_VTEAM = loadmat("VTEAM_I.mat")["I"][0]
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=V_ref,
        y=I_ref * 1e3,
        name="<b>Experimental Data</b>",
        line=dict(color="royalblue", width=6),
    )
)
fig.add_trace(
    go.Scatter(
        x=V_ref,
        y=I_VTEAM * 1e3,
        name="<b>VTEAM Model (Fitted)</b>",
        line=dict(color="firebrick", width=6, dash="dash"),
    )
)
fig.update_layout(
    xaxis_title="<b>Voltage (V)</b>",
    yaxis_title="<b>Current (mA)</b>",
    plot_bgcolor="white",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        font=dict(
            size=18,
        ),
        bordercolor="black",
        borderwidth=4,
    ),
    width=500,
    height=500,
)
fig.update_xaxes(
    range=[-5, 5],
    tickmode="array",
    tickvals=np.arange(-5, 6, 1),
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
    range=[-4, 4],
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
    "Figure_3.html", include_plotlyjs="cdn", include_mathjax=False, full_html=True
)
fig.write_image("Figure_3.svg")
fig.show()
