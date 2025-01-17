from dash import Dash, html, dcc, Input, Output, State

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Hello, enter your question", className="header"),
    html.Div([
        html.Div([
            dcc.Textarea(
                id="input-text",
                placeholder="Type your query here...",
                className="text-area"
            ),
        ], className="chat-wrapper"),

        html.Div([
            html.Button(
                id="submit-button",
                n_clicks=0,
                className="submit-button"
            ),
        ], className="button-wrapper"),
        
    ], className="container"),

    html.Div(id="output-display", className="output-display"),
])

@app.callback(
    [Output("output-display", "children"), 
     Output("input-text", "value")],
    Input("submit-button", "n_clicks"),
    State("input-text", "value"),
)
def gen_response(n_clicks, input_text):
    if n_clicks == 0 or not input_text:
        return "Here your response will appear", ""
    try:
        response = f"You asked: {input_text}"
        return response, ""
    except Exception as e:
        return f"Error: {e}", ""

if __name__ == "__main__":
    app.run_server(debug=True)