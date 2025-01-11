import dash
from dash import Dash, html, dcc, Input, Output, State



app = Dash(__name__)

app.layout = html.Div([
    html.H1("LLM Interface", style={'textAlign': 'center'}),
    dcc.Textarea(
        id='input-text',
        placeholder='Enter your prompt here...',
        style={'width': '80%', 'height': '100px', 'margin': '10px auto', 'display': 'block', 'fontSize': '16px'}
    ),
    html.Button('Submit', id='submit-button', n_clicks=0, style={'margin': '10px'}),
    html.Div(id='output-display', style={
        'marginTop': '20px',
        'padding': '10px',
        'border': '1px solid #ddd',
        'backgroundColor': '#f9f9f9',
        'width': '80%',
        'margin': '10px auto',
        'textAlign': 'left'
    })
])


def gen_response(num_clicks, input_text):
    if num_clicks == 0 or not input_text:
        return 'please enter a prompt'
    try:
        #this is where we should hook up the backend
        return 0
    except Exception as e:
        return f'error: {e}'



if __name__ == '__main__':
    app.run(debug=True)
