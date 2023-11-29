import base64
import json
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
from dash.exceptions import PreventUpdate

import sqlite3

import dash_bootstrap_components as dbc
from gamezmania import Gamezmania
from predictions import PredictBid
import json

from pathlib import Path

parent_dir = Path().absolute().stem

app = Dash(__name__,
           url_base_pathname=f"/dash/{parent_dir}/",
           external_stylesheets=[dbc.themes.YETI],
           title="Uploader",
           meta_tags=[
               {
                   "name": "viewport",
                   "content": "width=device-width, initial-scale=1"
               },
           ])

server = app.server

tab1 = dbc.Tab(
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div(
                ['Drag and Drop or ',
                 html.A('Select Json Files')]),
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
        dbc.Spinner(html.Div(id='output-data-upload')),
    ]),
    label="Upload JSON")

tab2 = dbc.Tab(label="Add Player Name Mapping",
               children=[
                   dbc.InputGroup([
                       dbc.InputGroupText('Add New Player ID Mapping', ),
                       dbc.Input(id='player-id', placeholder="ID String"),
                       dbc.Input(id='player-name', placeholder="Player Name"),
                       dbc.Button(id='player-button', children="Submit"),
                   ]),
                   html.Div(id='player-map-response')
               ])

tab3 = dbc.Tab([
    dcc.Markdown("""### How to get game JSON DATA

Use the web inspector in your browser when looking at a game history to find the hidden XHR request that has the needed JSON data. Be sure to save this with a file name as a date such as 2023-01-01.json, etc.

There are two XHR requests, one is short with metadata, the other has the needed game data and starts with `{"g":{"i}` .... 
                             

                             """),
    html.Img(src=app.get_asset_url('chrome.png'), style={'width': '80%'})
],
               label='How to save game data JSON')

app.layout = dbc.Container(dbc.Tabs([tab1, tab2, tab3]))


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
            Gamezmania(raw_data=parse_contents(x), filename=_filename)
            for x, _filename in zip(list_of_contents, list_of_names)
        ]
    else:
        out = []
    res = []
    for g in out:
        res.extend([html.P(x) for x in g.upload_to_sql()])

    print(res)

    for g in out:
        p = PredictBid()
        res.append(p.train(g.unique_hash))

    return res


@callback(Output('player-map-response', 'children'),
          Input('player-button', 'n_clicks'), State('player-id', 'value'),
          State('player-name', 'value'))
def process_button(_, player_id, player_name):
    if not (player_id and player_name):
        raise PreventUpdate
    with sqlite3.connect("oh_hell.db") as con:
        cursor = con.cursor()
        res = cursor.execute(
            "insert into player_names(player_id,player_name) VALUES (?,?);",
            (player_id, player_name))
    return str(res)


if __name__ == '__main__':
    app.run(debug=True)
