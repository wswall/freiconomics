from itertools import product

from dash import dcc, html, Dash
from dash.dependencies import Input, Output
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from sympy import Eq, solve, lambdify, IndexedBase, Expr
from sympy.abc import u

from consumer import UtilityFunction


BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css"


def _generate_utility_matrix(utility_function: Expr) -> npt.NDArray:
    # Generate a matrix of utility values for a two-good utility function.
    utility_matrix = np.empty((20, 20))
    for i, n in product(range(20), repeat=2):
        utility_matrix[i, n] = utility_function([0, i, n])
    return utility_matrix


def generate_utility_graph(utility_function: Expr) -> dcc.Graph:
    utility_matrix = _generate_utility_matrix(utility_function)
    surface = go.Surface(
        z=utility_matrix,
        opacity=0.5,
        colorbar_title="Utility",
        colorscale="thermal",
        showscale=True,
    )
    fig = go.Figure(data=[surface])
    fig.update_layout(
        title="Utility Curve",
        yaxis_title="$x_2$",
        xaxis_title="$x_1$",
        coloraxis_colorbar_x=-0.5,
        scene={
            "xaxis": {"nticks": 4},
            "yaxis": {"nticks": 4},
            "zaxis": {"nticks": 4, "range": [0, 20]},
            "xaxis_title": "x1",
            "yaxis_title": "x2",
            "zaxis_title": "u(x1, x2)",
        },
        margin={"r": 5, "l": 5, "b": 5, "t": 5}
    )
    return dcc.Graph(id="utility", figure=fig, mathjax=True, style={"font-size": "20px"})


def generate_indifference_graph() -> dcc.Graph:
    return dcc.Graph(id="indifference", mathjax=True, style={"font-size": "20px"})



def generate_body(utility_function: Expr) -> html.Div:
    utility_col = html.Div(generate_utility_graph(utility_function), className="col-xl-6")
    indifference_row = html.Div(generate_indifference_graph(), className="row")
    indifference_col = html.Div(indifference_row, className="col-xl-6")
    body = html.Div(children=[utility_col, indifference_col], className="row")
    return html.Div(body, className="container")


A, alpha = 1, 0.5
x = IndexedBase('x', positive=True)
ut = UtilityFunction(A * x[1] ** alpha * x[2] ** (1 - alpha))
indifference_x2 = solve(Eq(ut.expression, u), x[1])[0]
indifference_x2_solver = lambdify(["u", "x"], indifference_x2)
app = Dash(__name__, external_stylesheets=[BOOTSTRAP])
app.layout = generate_body(ut)


@app.callback(Output("indifference", "figure"), Input("utility", "hoverData"))
def update_indifference(hoverData):
    """
    Update the indifference curve plot based on the provided hover data.

    Parameters:
    - hoverData: A dictionary containing the hover data from the plot.

    Returns:
    - fig: A plotly Figure object representing the updated indifference curve plot.
    """
    z = hoverData["points"][0]["z"]
    xrange = np.linspace(0.001, 20.001, 100)
    y_sol = [indifference_x2_solver(z, [0, 0, x]) for x in xrange]
    fig = go.Figure(data=[go.Scatter(x=xrange, y=y_sol, mode="lines")])
    fig.update_layout(
        title="Indifference Curve u(x1, x2)",
        yaxis_title="$x_2$",
        xaxis_title="$x_1$",
        xaxis_range=[0, 20],
        yaxis_range=[0, 20],
    )
    return fig


if __name__ == "__main__":
    app.run()
