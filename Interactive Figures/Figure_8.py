import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


fig = go.Figure()
for step in np.arange(1, 41):
    data = pd.read_csv("SNN/epoch_%d.csv" % step)
    t = data["0"].values
    for i in range(0, 10):
        n_data = data["0.%d" % (i + 1)].values
        fig.add_trace(
            go.Scatter(
                visible=False,
                marker=dict(color="royalblue"),
                x=t[n_data == (i + 1)],
                y=n_data[n_data == (i + 1)],
                mode="markers",
                name="Neuron %d" % (i + 1),
                marker_symbol='circle-open',
                marker_line_width=2,
            )
        )

for i in range(0, 10):
    fig.data[i].visible = True

steps = []
for i in range(1, 41):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)}],
        label=str(i),
    )
    for j in range(0, 10):
        step["args"][0]["visible"][(i - 1) * 10 + j] = True

    steps.append(step)

sliders = [
    dict(active=0, currentvalue={"prefix": "Epoch: "}, pad={"t": 50}, steps=steps)
]


fig.update_layout(
    sliders=sliders,
    xaxis_title="<b>Time (s)</b>",
    yaxis_title="<b>Neuron</b>",
    plot_bgcolor="white",
    # legend=dict(
    #     orientation="h",
    #     yanchor="top",
    #     y=1.5,
    #     font=dict(
    #         size=18,
    #     ),
    # ),
    width=500,
    height=550,
)

fig["layout"]["xaxis"].update(
    range=[0, 1],
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
    range=[0, 11],
    tickmode="array",
    tickvals=np.arange(1, 11, 1),
    linewidth=4,
    linecolor="black",
    mirror=True,
    title_font=dict(size=22),
    tickfont=dict(size=16),
    # gridwidth=0.25,
    # gridcolor="grey",
    zerolinecolor="black",
)
fig.update_traces(marker_size=10)
fig.update(layout_showlegend=False)
fig.update_xaxes(tickfont_family="Arial Black")
fig.update_yaxes(tickfont_family="Arial Black")
fig.write_html(
    "Figure_8.html", include_plotlyjs="cdn", include_mathjax=False, full_html=True
)

for i in range(0, 10):
    fig.data[i].visible = False

i = 1
for j in range(0, 10):
        fig.data[10*(i - 1) + j].visible = True

fig.write_image("Figure_8_0.svg")
for j in range(0, 10):
        fig.data[10*(i - 1) + j].visible = False


i = 20
for j in range(0, 10):
        fig.data[10*(i - 1) + j].visible = True

fig.write_image("Figure_8_20.svg")
for j in range(0, 10):
        fig.data[10*(i - 1) + j].visible = False
i = 40
for j in range(0, 10):
        fig.data[10*(i - 1) + j].visible = True

fig.write_image("Figure_8_40.svg")
for j in range(0, 10):
        fig.data[10*(i - 1) + j].visible = False

for i in range(0, 10):
    fig.data[i].visible = True

fig.show()
