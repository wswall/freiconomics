import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from sympy import Symbol, log

from models.new_classical import (
    ModelFunctions,
    NewClassicalModel,
    stochastics_to_sequences,
    c,
    eta,
    labor,
    t,
)


BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css"


def generate_slider_div(variable, start=0.1, stop=1, step=0.1, value=0.5, label=None):
    component = dcc.RangeSlider if isinstance(value, list) else dcc.Slider
    slider = component(
        start,
        stop,
        step,
        value=value,
        id=f"{variable}-slider",
        included=True,
        marks=None,
        updatemode="drag",
    )
    label = label or f"${variable}$"
    md = dcc.Markdown(label, mathjax=True)
    return html.Div(
        children=[
            html.Div(md, className="col-2"),
            html.Div(slider, className="col-10"),
        ],
        id=f"{variable}-slider-div",
        className="row",
    )


def generate_slider_group(class_name, slider_dict):
    children = []
    for variable in slider_dict:
        slider = generate_slider_div(variable, **slider_dict[variable])
        children.append(slider)
    return html.Div(children=children, className=class_name)


def generate_plot(plot_id):
    return html.Div(
        dcc.Graph(id=plot_id, mathjax=True, style={"font-size": "20px"}),
        className="row",
    )


slider_definitions = {
    "alpha": {
        "start": 0.1,
        "stop": 0.9,
        "step": 0.1,
        "value": 0.4,
        "label": "$\\alpha$",
    },
    "theta_0": {
        "start": 1000,
        "stop": 20000,
        "step": 1000,
        "value": 17154,
        "label": "$\\theta_0$",
    },
    "theta_g": {
        "start": 0.001,
        "stop": 0.01,
        "step": 0.002,
        "value": 0.005,
        "label": "$\\theta_g$",
    },
    "theta_g_range": {
        "start": -0.01,
        "stop": 0.01,
        "step": 0.001,
        "value": [-0.001, 0.003],
        "label": "$\\theta_g$",
    },
    "phi": {"value": 0.37, "label": "$\\phi$"},
    "delta": {"stop": 2, "value": 1, "label": "$\\delta$"},
    "eta": {"value": 0.35, "label": "$\\eta$"},
}
sliders = generate_slider_group("row", slider_definitions)
switch = html.Div(dcc.Checklist(["Stochastic"], [], id="switch"), className="row")
components = html.Div(children=[switch, sliders], className="col-md-4")
plots = html.Div(children=[generate_plot("model")], className="col-md-8")
content = html.Div(
    children=[plots, components],
    className="row",
)

body = html.Div(children=content, className="row", style={"padding-top": "50px"})
app = dash.Dash(__name__, external_stylesheets=[BOOTSTRAP])
app.layout = html.Div(children=[body], className="container")

alpha = Symbol(r"\alpha")
beta = Symbol(r"\beta")
delta = Symbol(r"\delta")
phi = Symbol(r"\phi")
theta_growth = Symbol(r"\theta_growth")
theta_0 = Symbol(r"\theta_0")


@app.callback(
    [
        Output("model", "figure"),
        Output("theta_g-slider-div", "hidden"),
        Output("theta_g_range-slider-div", "hidden"),
    ],
    [
        Input("switch", "value"),
        Input("alpha-slider", "value"),
        Input("theta_0-slider", "value"),
        Input("theta_g-slider", "value"),
        Input("theta_g_range-slider", "value"),
        Input("phi-slider", "value"),
        Input("delta-slider", "value"),
        Input("eta-slider", "value"),
    ],
)
def update_outputs(
    stochastic, alpha_, theta_0_, theta_growth_, theta_growth_range, phi_, delta_, eta_
):
    params = {
        alpha: alpha_,
        delta: delta_,
        phi: phi_,
        theta_0: theta_0_,
        theta_growth: theta_growth_,
        eta: eta_,
    }
    labor_function = labor ** (1 - alpha)
    tfp = (
        (1 + theta_growth) * theta_0
        if stochastic
        else (1 + theta_growth) ** t * theta_0
    )
    consumption_utility = log(c)
    labor_disutility = -(delta / (1 + phi)) * labor ** (1 + phi)
    functions = ModelFunctions(
        labor_function, tfp, consumption_utility, labor_disutility
    )
    model = NewClassicalModel(functions)

    nu = 0.5635
    growth_transition_weights = {0: [nu, 1 - nu], 1: [1 - nu, nu]}
    if stochastic:
        stochastic_params = {
            theta_growth: {
                "values": theta_growth_range,
                "weights": growth_transition_weights,
            }
        }
        stochastic_sequences = stochastics_to_sequences(stochastic_params, 310)
        stochastic_sequences.insert(0, {theta_growth: 0, theta_0: theta_0_})
        for i in range(1, len(stochastic_sequences)):
            theta = tfp.subs(stochastic_sequences[i - 1])
            stochastic_sequences[i][theta_0] = theta
        history = model.simulate(310, params, param_sequences=stochastic_sequences)
    else:
        history = model.simulate(310, params)

    mod = go.Figure(
        data=[
            go.Scatter(
                x=history.index,
                y=history.y.astype(float),
                mode="lines",
                name="y",
            ),
            go.Scatter(
                x=history.index,
                y=history.c.astype(float),
                mode="lines",
                name="c",
            ),
            go.Scatter(
                x=history.index,
                y=history.g.astype(float),
                mode="lines",
                name="g",
            ),
        ]
    )
    mod.update_layout(
        title="Basic New Classical Model",
        yaxis_title="Value at time t",
        xaxis_title="Time t",
        xaxis_range=[0, len(history.index)],
        yaxis_range=[0, 50000],
        height=400,
        margin=dict(l=20, r=20, t=30, b=30),
    )

    return mod, bool(stochastic), not bool(stochastic)


if __name__ == "__main__":
    app.run()
