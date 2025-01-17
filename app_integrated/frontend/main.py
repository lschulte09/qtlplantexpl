from dash import Dash, html, dcc, Input, Output, State
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from search_kg import query_kg

KG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gwas_kg.json"))

app = Dash(__name__)
app.layout = html.Div([
    html.H1("Hello, here you can use the tool!", className="header"),
    html.Div([
        html.Div([
            dcc.Textarea(
                id="input-text",
                placeholder="Type your query here...",
                className="text-area"
            ),
        ], className="text-area-container"),

        html.Div([
            html.Div([
                dcc.Dropdown(
                    id="node-count-dropdown",
                    options=[
                        {"label": f"{i} Node{'s' if i > 1 else ''}", "value": i} for i in range(1, 11)
                    ],
                    value=5,
                    placeholder="Select number",
                    searchable=False,
                    clearable=False,
                    className="dropdown-menu"
                ),
            ], className="dropdown-wrapper"),

            html.Div([
                html.Button(
                    id="submit-button",
                    n_clicks=0,
                    className="submit-button"
                ),
            ], className="button-wrapper"),
        ], className="button-dropdown-container"),
    ], className="input-container"),

    html.Div(id="output-display", className="output-display"),
])

@app.callback(
    [Output("output-display", "children"), 
     Output("input-text", "value")],
    Input("submit-button", "n_clicks"),
    State("input-text", "value"),
    State("node-count-dropdown", "value"),
)
def gen_response(n_clicks, input_text, node_count):
    if n_clicks == 0 or not input_text:
        return "Here your response will appear!", ""
    try:
        results = query_kg(KG_PATH, input_text, top_k=node_count)
        response = "Results:\n" + "\n".join(
            [f"{idx + 1}. {res['label']}\n {res['text']}" for idx, res in enumerate(results)]
        )
        
        return response, ""
    except Exception as e:
        return f"Error: {e}", ""

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)