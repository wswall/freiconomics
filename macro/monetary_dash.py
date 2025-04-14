import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from models.monetary import MonetaryDynamics, Params


BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css"


def generate_slider(variable, start=0.1, stop=1, step=0.1, value=0.5, label=None):
    slider = dcc.Slider(
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
            className="row",
    )


def generate_slider_group(class_name, slider_dict):
    children = []
    for variable in slider_dict:
        slider = generate_slider(variable, **slider_dict[variable])
        children.append(slider)
    return html.Div(children=children, className=class_name)


def generate_plot(plot_id):
    return html.Div(
    dcc.Graph(id=plot_id, mathjax=True, style={"font-size": "20px"}), className="row"
    )


def generate_scatter(table, column):
    return go.Scatter(
        x=table.index,
        y=[float(val) for val in table[column]],
        mode="lines",
        name=column,
    )


nominal_plot = generate_plot("nominal")
real_plot = generate_plot("real")
plots = html.Div(children=[nominal_plot, real_plot], className="col-md-8")
slider_definitions = {
    "eta": {"value": 0.35, "start":0.3498, "stop": 0.3504, "step":0.00001, "label":"$\\eta$"},
    "tau": {"value": 0.3502, "start":0.3498, "stop":0.3504, "step":0.00001, "label":"$\\tau$"},
    "pi_target": {"value": 0.02, "start":0,"stop":0.1, "step":0.005, "label":"$\\pi_{target}$"},
    "phi_pi": {"value": 0.9, "start": 0.7, "stop": 0.99, "step": .05, "label":"$\\phi_\\pi$"},
    "phi_b": {"value": .001, "start": 0,"stop": 0.05, "step": .001, "label":"$\\phi_b$"},
    "g": {"value": 0.02, "start": 0,"stop":0.2, "step": 0.01, "label":"$g$"},
    "beta": {"value": 0.9988, "start": 0.998, "stop": 1, "step": 0.0001, "label":"$\\beta$"}
}
sliders = generate_slider_group('row', slider_definitions)
policy_div = html.Div(id="policy", className="row")
info = html.Div(children=[sliders, policy_div], className="col-md-4")
content = html.Div(children=[plots, info],className="row")

body = html.Div(children=content, className="row", style={"padding-top": "50px"})
app = dash.Dash(__name__, external_stylesheets=[BOOTSTRAP])
app.layout = html.Div(children=[body], className="container")


@app.callback(
    [
        Output("nominal", "figure"),
        Output("real", "figure"),
        Output("policy", "children")
    ],
    [
        Input("eta-slider", "value"),
        Input("tau-slider", "value"),
        Input("pi_target-slider", "value"),
        Input("phi_pi-slider", "value"),
        Input("phi_b-slider", "value"),
        Input("g-slider", "value"),
        Input("beta-slider", "value")
    ],
)
def update_outputs(eta, tau, pi_target, phi_pi, phi_b, g, beta):
    try:
        model = MonetaryDynamics(Params(eta, tau, pi_target, phi_pi, phi_b, g, beta))
    except ValueError as e:
        return go.Figure(), go.Figure(), [html.P(f"Error: {e.args[0]}")]
    history = model.simulate(101)
    nominal = go.Figure(
        data=[
            generate_scatter(history, "Y"),
            generate_scatter(history, "B"),
            generate_scatter(history, "G"),
            generate_scatter(history, "T"),
        ]
    )
    nominal.update_layout(
        title="Dynamics of Nominal Variables",
        xaxis_title="Time t",
        yaxis_title="Value at time t",
        xaxis_range=[0, 100],
        yaxis_range=[0, 500000],
        height=400,
        margin=dict(l=20, r=20, t=30, b=30),
    )

    real = go.Figure(data=[
        generate_scatter(history, "pi"),
        generate_scatter(history, "i"),
        generate_scatter(history, "b"),
    ])
    real.update_layout(
        title="Dynamics of Real Variables",
        xaxis_title="Time t",
        yaxis_title="Value at time t",
        height=250,
        margin=dict(l=40, r=40, t=30, b=30),
        xaxis_range=[0, 100],
        yaxis_range=[0, 1],
    )
    policies = {
        "monetary":"active" if model.monetary_active() else "passive", 
        "fiscal": "Non-Ricardian" if model.fiscal_active() else "Ricardian"
    }
    policy = [
        html.P(f"Monetary Policy: {policies['monetary']}"),
        html.P(f"Fiscal Policy: {policies['fiscal']}")
    ]
    return nominal, real, policy


if __name__ == "__main__":
    app.run()
