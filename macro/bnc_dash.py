import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from sympy import Symbol, log

from models.bnc import BncFunctions, Bnc, c, eta, labor, t


BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css"


def generate_slider(variable, start=0.1, stop=1, step=0.1, value=0.5):
    return dcc.Slider(
        start,
        stop,
        step,
        value=value,
        id=f"{variable}-slider",
        included=True,
        marks=None,
        updatemode="drag",
    )


def generate_slider_div(symbol, slider):
    label = dcc.Markdown(f"$\\{symbol}$", mathjax=True)
    return html.Div(
        children=[
            html.Div(label, className="col-2"),
            html.Div(slider, className="col-10"),
        ],
        className="row",
    )


alpha_slider = generate_slider("alpha", start=0.1, stop=0.9, step=0.1, value=0.4)
theta_slider = generate_slider("theta_0", start=1000, stop=20000, step=1000, value=17154)
theta_growth_slider = generate_slider(
    "theta_g", start=0.001, stop=0.01, step=0.002, value=0.005
)
phi_slider = generate_slider("phi", value=0.37)
delta_slider = generate_slider("delta", stop=2, value=1)
eta_slider = generate_slider("eta", value=0.35)

slider_group_1 = html.Div(
    children=[
        generate_slider_div("alpha", alpha_slider),
        generate_slider_div("theta_0", theta_slider),
        generate_slider_div("theta_g", theta_growth_slider),
        generate_slider_div("phi", phi_slider),
    ],
    className="col-xs-6",
)
slider_group_2 = html.Div(
    children=[
        generate_slider_div("delta", delta_slider),
        generate_slider_div("eta", eta_slider),
    ],
    className="col-xs-6",
)
sliders = html.Div(children=[slider_group_1, slider_group_2], className="row")
model_plot = html.Div(
    dcc.Graph(id="model", mathjax=True, style={"font-size": "20px"}), className="row"
)
production_plot = html.Div(
    dcc.Graph(id="profit", mathjax=True, style={"font-size": "20px"}), className="row"
)
log_utility_plot = html.Div(
    dcc.Graph(id="log_utility", mathjax=True, style={"font-size": "20px"}), className="row"
)
content = html.Div(
    children=[
        html.Div(children=[model_plot, production_plot, log_utility_plot], className="col-md-8"),
        html.Div(
            sliders,
            className="col-md-4",
        ),
    ],
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

labor_function = labor ** (1 - alpha)
tfp = (1 + theta_growth) ** t * theta_0
consumption_utility = log(c)
labor_disutility = -(delta / (1 + phi)) * labor ** (1 + phi)

functions = BncFunctions(labor_function, tfp, consumption_utility, labor_disutility)
model = Bnc(functions)


@app.callback(
    [
        Output("model", "figure"),
        Output("profit", "figure"),
        Output("log_utility", "figure"),
    ],
    [
        Input("alpha-slider", "value"),
        Input("theta_0-slider", "value"),
        Input("theta_g-slider", "value"),
        Input("phi-slider", "value"),
        Input("delta-slider", "value"),
        Input("eta-slider", "value"),
    ],
)
def update_outputs(alpha_, theta_0_, theta_growth_, phi_, delta_, eta_):
    parameters = {
        alpha: alpha_,
        delta: delta_,
        phi: phi_,
        theta_growth: theta_growth_,
        theta_0: theta_0_,
        eta: eta_,
    }
    history = model.simulate(101, parameters)

    mod = go.Figure(
        data=[
            go.Scatter(
                x=history.index,
                y=[float(val) for val in history.y],
                mode="lines",
                name="y",
            ),
            go.Scatter(
                x=history.index,
                y=[float(val) for val in history.c],
                mode="lines",
                name="c",
            ),
            go.Scatter(
                x=history.index,
                y=[float(val) for val in history.g],
                mode="lines",
                name="g",
            ),
        ]
    )
    mod.update_layout(
        title="Basic New Classical Model",
        yaxis_title="Value at time t",
        xaxis_title="Time t",
        xaxis_range=[0, 100],
        yaxis_range=[0, 50000],
        height=400,
        margin=dict(l=20, r=20, t=30, b=30),
    )

    l_star = float(model.eq_labor.subs(parameters))
    firm_profit = history.y - history.w * l_star
    profit = go.Scatter(
        x=history.index, y=[float(val) for val in firm_profit], mode="lines"
    )
    prof = go.Figure(data=[profit])
    prof.update_layout(
        title="Firm Profit",
        xaxis_title="Period t ($t$)",
        yaxis_title="Firm profit at time t ($d_t$)",
        height=250,
        margin=dict(l=40, r=40, t=30, b=30),
        xaxis_range=[0, 100],
        yaxis_range=[0, 50000],
    )

    utility_from_consumption = [
        float(consumption_utility.subs(c, val)) for val in history.c.values
    ]
    utility = go.Scatter(x=history.index, y=utility_from_consumption, mode="lines")
    util = go.Figure(data=[utility])
    util.update_layout(
        title=r"Utility from Consumption (sigma=1)",
        xaxis_title="Period t $(t)$)",
        yaxis_title="Utility at time t ($u(c_t)$)",
        height=250,
        margin=dict(l=40, r=40, t=30, b=30),
        xaxis_range=[0, 100],
        yaxis_range=[6, 12],
    )

    return mod, prof, util


if __name__ == "__main__":
    app.run()
