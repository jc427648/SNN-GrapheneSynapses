import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.io import loadmat


def f_actP(x, params):
    n = x.size
    x1 = params[0]
    x2 = params[1]
    amp_p = -params[3]
    amp_n = -params[2]
    xop = params[4]
    xon = params[5]
    out = np.zeros((n))
    i1 = np.where(x[x < 0] > x1)
    out[i1] = -amp_n * (np.exp(x[i1] / xon) - np.exp(x1 / xon)) / (1 - np.exp(x1 / xon))
    i2 = np.where(x < x2)
    i2 = np.clip(i2, np.where(x == 0)[0][0] + 1, n - 1)
    out[i2] = (
        amp_p * (np.exp(-x[i2] / xop) - np.exp(-x2 / xop)) / (1 - np.exp(-x2 / xop))
    )
    return out


def det_post_synapse(delta_t_value):
    delta_t_idx = np.where(delta_t == delta_t_value)[0][0]
    return init_params[7] * f_actP(x - delta_t[delta_t_idx], init_params)


delta_t_value = 20  # ms
init_params = [-28.0, 45, 2.2, 1.1, 30, 30, 1.0, 0.4]
delta_t = np.linspace(-80, 80, 321)
x = np.linspace(-80, 80, 4001)
pre_synapse = -init_params[6] * f_actP(x, init_params)
post_synapse = det_post_synapse(delta_t_value)

fig = make_subplots(rows=2, cols=1, subplot_titles=("<b>(a)</b><br>", "<b>(b)<br></b>"))
fig.add_trace(
    go.Scatter(
        x=x,
        y=pre_synapse,
        name="<b>Pre-synaptic Voltage (V)</b>",
        line=dict(color="darkgreen", width=6),
    ),
    row=1,
    col=1,
)
for step in np.arange(-80, 90, 10):
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="royalblue", width=6),
            name="<b>Post-synaptic Voltage (V)</b>",
            x=x,
            y=det_post_synapse(step),
        ),
        row=1,
        col=1,
    )

for step in np.arange(-80, 90, 10):
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="firebrick", width=6, dash="dot"),
            name="<b>Overall Memristive Voltage (V)</b>",
            x=x,
            y=det_post_synapse(step) - pre_synapse,
        ),
        row=1,
        col=1,
    )

wave_current = loadmat("WaveCurr.mat")["WaveCurr"]
for step in np.arange(-80, 90, 10):
    delta_t_idx = np.where(delta_t == step)[0][0]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=wave_current[delta_t_idx, :] * 1e3,
            visible=False,
            name="<b>Current Waveform (mA)</b>",
            fill="tozeroy",
            fillcolor="royalblue",
            line=dict(color="black"),
        ),
        row=2,
        col=1,
    )

fig.data[10].visible = True
fig.data[27].visible = True
fig.data[44].visible = True
steps = []
for i in range(17):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)}],
        label=str(list(np.arange(-80, 90, 10))[i]),  # layout attribute
    )
    step["args"][0]["visible"][i + 1] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i + 18] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i + 18 + 17] = True
    step["args"][0]["visible"][0] = True
    steps.append(step)

sliders = [
    dict(active=9, currentvalue={"prefix": "ΔT (ms): "}, pad={"t": 50}, steps=steps)
]

fig.layout.annotations[0].update(y=1.01)
fig.layout.annotations[1].update(y=0.39)
fig.update_layout(
    sliders=sliders,
    xaxis_title="<b>Time (ms)</b>",
    yaxis_title="<b>Voltage (V)</b>",
    xaxis2_title="<b>Time (ms)</b>",
    yaxis2_title="<b>Current (mA)</b>",
    plot_bgcolor="white",
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.2,
        font=dict(
            size=18,
        ),
    ),
    height=1000,
    width=500,
)
fig["layout"]["xaxis"].update(
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
    # range=[-1.5, 3],
    tickmode="array",
    tickvals=np.arange(-1.5, 3.5, 0.5),
    linewidth=4,
    linecolor="black",
    mirror=True,
    title_font=dict(size=22),
    tickfont=dict(size=16),
    tick0=-4,
    dtick=1,
    # gridwidth=0.25,
    # gridcolor="grey",
    zerolinecolor="black",
)
fig["layout"]["xaxis2"].update(
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
    linewidth=4,
    linecolor="black",
    mirror=True,
    title_font=dict(size=22),
    tickfont=dict(size=16),
    # gridwidth=0.25,
    # gridcolor="grey",
    zerolinecolor="black",
)
fig.update_xaxes(tickfont_family="Arial Black")
fig.update_yaxes(tickfont_family="Arial Black")
fig.update_layout(xaxis2=dict(autorange=True), yaxis2=dict(autorange=True))
fig.write_html(
    "Figure_4.html", include_plotlyjs="cdn", include_mathjax=False, full_html=True
)
fig.update_layout(xaxis2=dict(autorange=False), yaxis2=dict(autorange=False))
fig["layout"]["xaxis2"].update(range=[-4, 2])
fig["layout"]["yaxis2"].update(range=[0, 0.8])
fig.write_image("Figure_4.svg")
fig.show()
