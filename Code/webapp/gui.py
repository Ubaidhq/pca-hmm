import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from data.simulate_data import create_futures_data, get_forward_curves
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

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

TAB_STYLE = {
    'padding': '10px',
    'borderBottom': '1px solid #d6d6d6',
}

TAB_SELECTED_STYLE = {
    'borderTop': '3px solid #119DFF',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '10px'
}

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("CBOT Futures Analysis Dashboard", 
                style={'textAlign': 'center', 'margin-bottom': '30px'}),
        
        # Commodity Selection (global control)
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
    ]),
    
    # Main Tabs
    dcc.Tabs(id='main-tabs', value='data-tab', children=[
        # Data Visualization Tab
        dcc.Tab(label='Data Visualization', value='data-tab', 
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE,
                children=[
                    html.Div([
                        # Time Series Controls
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
                        
                        dcc.Graph(id='futures-graph'),
                        
                        # Forward Curve Controls
                        html.Div([
                            html.Label("Forward Curve Date Range:", style={'font-weight': 'bold'}),
                            dcc.DatePickerRange(
                                id='forward-curve-dates',
                                min_date_allowed=pd.Timestamp('2010-01-01'),
                                max_date_allowed=pd.Timestamp('2025-12-31'),
                                initial_visible_month=pd.Timestamp('2010-01-01'),
                                start_date=pd.Timestamp('2010-01-01'),
                                end_date=pd.Timestamp('2010-01-05')
                            ),
                        ], style=CONTROL_STYLE),
                        
                        dcc.Graph(id='forward-curve-graph'),
                    ])
                ]),
        
        # HMM Analysis Tab
        dcc.Tab(label='HMM Analysis', value='hmm-tab',
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE,
                children=[
                    dcc.Tabs(id='hmm-subtabs', value='model-selection', children=[
                        # Model Selection Tab
                        dcc.Tab(label='Model Selection', value='model-selection',
                               style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE,
                               children=[
                                   html.Div([
                                       html.H3("HMM Model Selection"),
                                       html.Div([
                                           html.Label("Number of States Range:"),
                                           dcc.RangeSlider(
                                               id='n-states-range',
                                               min=2,
                                               max=10,
                                               value=[2, 6],
                                               marks={i: str(i) for i in range(2, 11)},
                                               step=1
                                           ),
                                       ], style=CONTROL_STYLE),
                                       dcc.Graph(id='aic-bic-plot')
                                   ])
                               ]),
                        
                        # Regime Identification Tab
                        dcc.Tab(label='Regime Identification', value='regime-id',
                               style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE,
                               children=[
                                   html.Div([
                                       html.H3("Regime Identification"),
                                       html.Div([
                                           html.Label("Number of States:"),
                                           dcc.Slider(
                                               id='n-states-regime',
                                               min=2,
                                               max=6,
                                               value=3,
                                               marks={i: str(i) for i in range(2, 7)},
                                               step=1
                                           ),
                                       ], style=CONTROL_STYLE),
                                       dcc.Graph(id='regime-plot')
                                   ])
                               ]),
                        
                        # State Distribution Tab
                        dcc.Tab(label='State Distributions', value='state-dist',
                               style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE,
                               children=[
                                   html.Div([
                                       html.H3("State-Dependent Distributions"),
                                       dcc.Graph(id='state-dist-plot')
                                   ])
                               ]),
                    ]),
                ]),
        
        # Additional tabs can be added here
    ]),
    
    # Hidden div for storing the data
    html.Div(id='stored-data', style={'display': 'none'})
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
        # Convert string dates to timestamps if needed
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        # Get forward curves
        forward_curves = get_forward_curves(df, (start_date, end_date))
        
        # Plot each date's forward curve
        for date, curve in forward_curves.items():
            fig.add_trace(go.Scatter(
                x=list(range(1, len(curve) + 1)),  # Contract numbers
                y=curve.values,
                name=pd.Timestamp(date).strftime('%Y-%m-%d'),
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

# Add new callbacks for HMM analysis

@app.callback(
    Output('aic-bic-plot', 'figure'),
    [Input('stored-data', 'children'),
     Input('n-states-range', 'value')]
)
def update_aic_bic_plot(stored_data, n_states_range):
    df = pd.read_json(stored_data)
    df.index = pd.to_datetime(df.index)
    
    # Calculate returns
    returns = df.pct_change().dropna()
    
    # Prepare data for HMM
    X = StandardScaler().fit_transform(returns)
    X = X.reshape(-1, 1)
    
    # Calculate AIC and BIC for different numbers of states
    n_states_range = range(n_states_range[0], n_states_range[1] + 1)
    aic_scores = []
    bic_scores = []
    
    for n_states in n_states_range:
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
        model.fit(X)
        
        # Calculate scores
        n_params = n_states * n_states + 2 * n_states  # transition probs + means + variances
        aic = -2 * model.score(X) + 2 * n_params
        bic = -2 * model.score(X) + np.log(len(X)) * n_params
        
        aic_scores.append(aic)
        bic_scores.append(bic)
    
    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(n_states_range), y=aic_scores, name='AIC',
                            mode='lines+markers'))
    fig.add_trace(go.Scatter(x=list(n_states_range), y=bic_scores, name='BIC',
                            mode='lines+markers'))
    
    fig.update_layout(
        title="AIC and BIC vs. Number of Hidden States",
        xaxis_title="Number of States",
        yaxis_title="Score",
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    Output('regime-plot', 'figure'),
    [Input('stored-data', 'children'),
     Input('n-states-regime', 'value')]
)
def update_regime_plot(stored_data, n_states):
    df = pd.read_json(stored_data)
    df.index = pd.to_datetime(df.index)
    
    # Create continuous front month series
    continuous_series = pd.Series(index=df.index, dtype=float)
    
    for date in df.index:
        # Get non-NaN values for this date
        available_values = df.loc[date].dropna()
        if len(available_values) > 0:
            # Use the first available contract (front month)
            continuous_series[date] = available_values.iloc[0]
    
    # Remove any NaN values
    continuous_series = continuous_series.dropna()
    
    # Calculate log returns
    log_returns = np.log(continuous_series / continuous_series.shift(1)).dropna()
    
    # Prepare data for HMM
    scaler = StandardScaler()
    X = scaler.fit_transform(log_returns.values.reshape(-1, 1))
    
    # Fit HMM and predict states
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    model.fit(X)
    states = model.predict(X)
    
    # Create figure
    fig = go.Figure()
    
    # Plot the continuous price series
    fig.add_trace(go.Scatter(
        x=continuous_series.index,
        y=continuous_series.values,
        name='Price',
        line=dict(color='lightgray', width=1),
        showlegend=True
    ))
    
    # Add colored markers for different regimes
    colors = px.colors.qualitative.Set1[:n_states]
    for state in range(n_states):
        state_mask = states == state
        
        # Get the dates and prices for this state
        state_dates = log_returns.index[state_mask]
        state_prices = continuous_series.loc[state_dates]
        
        fig.add_trace(go.Scatter(
            x=state_dates,
            y=state_prices,
            name=f'State {state}',
            mode='markers',
            marker=dict(
                size=6,
                color=colors[state],
                symbol='circle'
            ),
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title="Price Series with HMM Regimes",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

@app.callback(
    Output('state-dist-plot', 'figure'),
    [Input('stored-data', 'children'),
     Input('n-states-regime', 'value')]
)
def update_state_dist_plot(stored_data, n_states):
    df = pd.read_json(stored_data)
    df.index = pd.to_datetime(df.index)
    
    # Create continuous front month series (same as in regime plot)
    continuous_series = pd.Series(index=df.index, dtype=float)
    for date in df.index:
        available_values = df.loc[date].dropna()
        if len(available_values) > 0:
            continuous_series[date] = available_values.iloc[0]
    
    continuous_series = continuous_series.dropna()
    
    # Calculate log returns (same as in regime plot)
    log_returns = np.log(continuous_series / continuous_series.shift(1)).dropna()
    
    # Prepare data for HMM - still standardize for fitting but keep original values for plotting
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(log_returns.values.reshape(-1, 1))
    
    # Fit HMM and predict states
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    model.fit(X_scaled)
    states = model.predict(X_scaled)
    
    # Create figure
    fig = go.Figure()
    
    # Add KDE for each state using the original (non-standardized) log returns
    for state in range(n_states):
        state_data = log_returns[states == state]
        
        # Create bins for better visualization
        bins = np.linspace(log_returns.min(), log_returns.max(), 100)
        hist, bin_edges = np.histogram(state_data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=hist,
            name=f'State {state}',
            mode='lines',
            fill='tozeroy',
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Distribution of Log-Returns by State",
        xaxis_title="Log-Returns",
        yaxis_title="Density",
        hovermode='x unified',
        height=600,
        showlegend=True,
        # Set x-axis range to focus on the main part of the distributions
        xaxis=dict(
            range=[log_returns.quantile(0.001), log_returns.quantile(0.999)],
            tickformat='.3f'
        ),
        # Add grid for better readability
        xaxis_gridcolor='lightgrey',
        yaxis_gridcolor='lightgrey',
        plot_bgcolor='white'
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
