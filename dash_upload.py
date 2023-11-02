import base64
import json
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import datetime

import dash_bootstrap_components as dbc
from gamezmania import Gamezmania
import json

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ',
                           html.A('Select Files')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True),
    html.Div(id='output-data-upload'),
])


def parse_contents(contents):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    return json.loads(decoded.decode('utf-8'))


@callback(Output('output-data-upload', 'children'),
          Input('upload-data', 'contents'), State('upload-data', 'filename'),
          State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        out = [
            Gamezmania(raw_data=parse_contents(x)) for x in list_of_contents
        ]
    else:
        out = []
    res = []
    for g in out:
        res.extend([html.P(x) for x in g.upload_to_sql()])

    return res


if __name__ == '__main__':
    app.run(debug=True)
