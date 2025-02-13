import dash
from dash import dcc, html
from dash.dependencies import Input, Output
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
        
        # Time Series Controls
        html.Div([
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
                
                html.Label("Time Series Date Range:", style={'font-weight': 'bold'}),
                dcc.RangeSlider(
                    id='date-range',
                    min=2010,
                    max=2025,
                    value=[2010, 2015],
                    marks={str(year): str(year) for year in range(2010, 2026, 2)},
                    step=None
                ),
            ], style={'width': '100%'})
        ], style=CONTROL_STYLE),
        
        # Time Series Graph
        html.Div([
            dcc.Graph(id='futures-graph')
        ], style={'margin-bottom': '30px'}),
        
        # Forward Curve Controls
        html.Div([
            html.Label("Forward Curve Date Range:", style={'font-weight': 'bold'}),
            dcc.DatePickerRange(
                id='forward-curve-dates',
                min_date_allowed=datetime(2010, 1, 1),
                max_date_allowed=datetime(2025, 12, 31),
                initial_visible_month=datetime(2010, 1, 1),
                start_date=datetime(2010, 1, 1),
                end_date=datetime(2010, 1, 5),  # Default to 5-day range
                style={'width': '100%'}
            ),
        ], style=CONTROL_STYLE),
        
        # Forward Curve Graph
        html.Div([
            dcc.Graph(id='forward-curve-graph')
        ])
    ], style=CONTENT_STYLE)
])

# Callback to update graphs
@app.callback(
    [Output('futures-graph', 'figure'),
     Output('forward-curve-graph', 'figure')],
    [Input('commodity-select', 'value'),
     Input('date-range', 'value'),
     Input('forward-curve-dates', 'start_date'),
     Input('forward-curve-dates', 'end_date')]
)
def update_graphs(commodity, date_range, start_date, end_date):
    # Create dataset
    df = create_futures_data(commodity=commodity)
    
    # Time series graph - only depends on commodity and date_range
    mask = (df.index.year >= date_range[0]) & (df.index.year <= date_range[1])
    filtered_df = df[mask]
    
    fig1 = go.Figure()
    for contract in filtered_df.columns:
        fig1.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_df[contract],
            name=contract,
            mode='lines'
        ))
    
    fig1.update_layout(
        title=f"CBOT {commodity} Futures Contracts Time Series",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=600,
        margin={'t': 50, 'b': 50}
    )
    
    # Forward curve graph - depends on all inputs
    fig2 = go.Figure()
    if start_date and end_date:
        # Use daily frequency instead of monthly
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        curves = get_forward_curves(df, dates)
        for date, curve in curves.items():
            # Ensure we only take the first 10 contracts
            curve = curve.iloc[:10] if len(curve) > 10 else curve
            contract_numbers = list(range(1, len(curve) + 1))
            fig2.add_trace(go.Scatter(
                x=contract_numbers,
                y=curve.values,
                name=date.strftime('%Y-%m-%d'),
                mode='lines+markers'
            ))
    
    fig2.update_layout(
        title=f"CBOT {commodity} Forward Curves",
        xaxis_title="Contract Number",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=600,
        margin={'t': 50, 'b': 50},
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            tickformat='d',
            range=[0.5, 10.5]  # Fix x-axis range to show all 10 contracts
        )
    )
    
    return fig1, fig2

if __name__ == '__main__':
    app.run_server(debug=True)
