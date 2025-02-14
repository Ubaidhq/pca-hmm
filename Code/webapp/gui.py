import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from data.simulate_data import create_futures_data, get_forward_curves
from datetime import datetime
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__)

# Define some consistent styling
CONTENT_STYLE = {
    'margin-left': 'auto',
    'margin-right': 'auto',
    'max-width': '1200px',
    'padding': '20px'
}

CONTROL_STYLE = {
    'padding': '20px',
    'margin-bottom': '20px',
    'border-radius': '5px',
    'background-color': '#f8f9fa',
    'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
}

# Layout
app.layout = html.Div([
    # Main container
    html.Div([
        # Title
        html.H1("CBOT Futures Contracts Visualization", 
                style={'textAlign': 'center', 'margin-bottom': '30px'}),
        
        # Commodity Selection
        html.Div([
            html.Label("Select Commodity:", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='commodity-select',
                options=[
                    {'label': 'Corn', 'value': 'CORN'},
                    {'label': 'Wheat', 'value': 'WHEAT'},
                    {'label': 'Soybean', 'value': 'SOYBEAN'}
                ],
                value='CORN',
                style={'margin-bottom': '20px'}
            ),
        ], style=CONTROL_STYLE),
        
        # Time Series Controls and Graph
        html.Div([
            html.Label("Time Series Date Range:", style={'font-weight': 'bold'}),
            dcc.RangeSlider(
                id='date-range',
                min=2010,
                max=2025,
                value=[2010, 2015],
                marks={str(year): str(year) for year in range(2010, 2026, 2)},
                step=None
            ),
        ], style=CONTROL_STYLE),
        
        html.Div([
            dcc.Graph(id='futures-graph')
        ], style={'margin-bottom': '30px'}),
        
        # Forward Curve Controls and Graph
        html.Div([
            html.Label("Forward Curve Date Range:", style={'font-weight': 'bold'}),
            dcc.DatePickerRange(
                id='forward-curve-dates',
                min_date_allowed=datetime(2010, 1, 1),
                max_date_allowed=datetime(2025, 12, 31),
                initial_visible_month=datetime(2010, 1, 1),
                start_date=datetime(2010, 1, 1),
                end_date=datetime(2010, 1, 5),
                style={'width': '100%'}
            ),
        ], style=CONTROL_STYLE),
        
        html.Div([
            dcc.Graph(id='forward-curve-graph')
        ]),
        
        # Hidden div for storing the data
        html.Div(id='stored-data', style={'display': 'none'})
    ], style=CONTENT_STYLE)
])

# Callback to store data when commodity changes
@app.callback(
    Output('stored-data', 'children'),
    Input('commodity-select', 'value')
)
def update_stored_data(commodity):
    df = create_futures_data(commodity=commodity)
    return df.to_json(date_format='iso')

# Callback for time series graph
@app.callback(
    Output('futures-graph', 'figure'),
    [Input('stored-data', 'children'),
     Input('date-range', 'value')]
)
def update_timeseries(stored_data, date_range):
    df = pd.read_json(stored_data)
    df.index = pd.to_datetime(df.index)
    
    mask = (df.index.year >= date_range[0]) & (df.index.year <= date_range[1])
    filtered_df = df[mask]
    
    fig = go.Figure()
    for contract in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_df[contract],
            name=contract,
            mode='lines'
        ))
    
    fig.update_layout(
        title="CBOT Futures Contracts Time Series",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=600,
        margin={'t': 50, 'b': 50}
    )
    
    return fig

# Callback for forward curve graph
@app.callback(
    Output('forward-curve-graph', 'figure'),
    [Input('stored-data', 'children'),
     Input('forward-curve-dates', 'start_date'),
     Input('forward-curve-dates', 'end_date')]
)
def update_forward_curves(stored_data, start_date, end_date):
    df = pd.read_json(stored_data)
    df.index = pd.to_datetime(df.index)
    
    fig = go.Figure()
    if start_date and end_date:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        curves = get_forward_curves(df, dates)
        for date, curve in curves.items():
            curve = curve.iloc[:10] if len(curve) > 10 else curve
            contract_numbers = list(range(1, len(curve) + 1))
            fig.add_trace(go.Scatter(
                x=contract_numbers,
                y=curve.values,
                name=date.strftime('%Y-%m-%d'),
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title="CBOT Forward Curves",
        xaxis_title="Contract Number (Front to Back)",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=600,
        margin={'t': 50, 'b': 50},
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            tickformat='d',
            range=[0.5, 10.5]
        )
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
