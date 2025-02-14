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
from sklearn.decomposition import PCA

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
        
        # PCA Analysis Tab
        dcc.Tab(label='PCA Analysis', value='pca-tab',
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE,
                children=[
                    html.Div([
                        html.H3("Principal Component Analysis"),
                        
                        # Overall PCA
                        html.Div([
                            html.H4("Overall PCA"),
                            dcc.Graph(id='pca-components-plot'),
                        ], style={'margin-bottom': '40px'}),
                        
                        # PCA by Regime
                        html.Div([
                            html.H4("PCA by Regime"),
                            
                            # Number of states selector
                            html.Div([
                                html.Label("Number of States:", style={'font-weight': 'bold'}),
                                dcc.Slider(
                                    id='pca-regime-states',
                                    min=2,
                                    max=4,
                                    value=3,
                                    marks={i: str(i) for i in range(2, 5)},
                                    step=1
                                ),
                            ], style=CONTROL_STYLE),
                            
                            # Graphs for regime-specific PCA
                            dcc.Graph(id='pca-by-regime-plot'),
                            dcc.Graph(id='pca-loadings-plot')
                        ])
                    ])
                ]),
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
        available_values = df.loc[date].dropna()
        if len(available_values) > 0:
            continuous_series[date] = available_values.iloc[0]
    
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
    
    # Calculate state statistics
    total_points = len(states)
    state_stats = []
    
    for state in range(n_states):
        state_mask = states == state
        
        # Get the dates and prices for this state
        state_dates = log_returns.index[state_mask]
        state_prices = continuous_series.loc[state_dates]
        
        # Calculate percentage of points in this state
        state_count = np.sum(state_mask)
        state_percentage = (state_count / total_points) * 100
        
        # Store statistics
        state_stats.append(f"State {state}: {state_count} points ({state_percentage:.1f}%)")
        
        fig.add_trace(go.Scatter(
            x=state_dates,
            y=state_prices,
            name=f'State {state} ({state_count} pts, {state_percentage:.1f}%)',
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
        title="Price Series with HMM Regimes",  # Simplified title
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
        margin=dict(l=50, r=50, t=50, b=50)  # Reduced top margin back to original
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

@app.callback(
    Output('pca-components-plot', 'figure'),
    Input('stored-data', 'children')
)
def update_pca_plot(stored_data):
    df = pd.read_json(stored_data)
    df.index = pd.to_datetime(df.index)
    
    # Prepare data for PCA
    continuous_series = []
    all_dates = df.index.unique()
    
    for date in all_dates:
        row = df.loc[date].dropna()
        if len(row) >= 5:  # Ensure we have at least 5 contracts
            # Sort contracts by their expiry
            contracts = []
            for contract in row.index:
                month = contract[0]
                year = int('20' + contract[1:]) if len(contract) > 1 else 2000
                month_num = {'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12}[month]
                expiry = pd.Timestamp(f'{year}-{month_num:02d}-15')
                contracts.append((contract, expiry))
            
            # Sort by expiry and take first 5 contracts
            sorted_contracts = sorted(contracts, key=lambda x: x[1])[:5]
            contract_values = [row[contract[0]] for contract in sorted_contracts]
            
            if len(contract_values) == 5:  # Ensure we have exactly 5 contracts
                continuous_series.append(contract_values)
    
    # Convert to numpy array
    X = np.array(continuous_series)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    # Calculate variances
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100
    individual_variance = pca.explained_variance_ratio_ * 100
    
    # Create figure
    fig = go.Figure()
    
    # Add bar plot for individual variance
    fig.add_trace(go.Bar(
        x=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'],
        y=individual_variance,
        name='Individual Explained Variance',
        hovertemplate='%{x}<br>Variance: %{y:.1f}%'
    ))
    
    # Add cumulative line on secondary y-axis
    fig.add_trace(go.Scatter(
        x=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'],
        y=cumulative_variance,
        name='Cumulative Variance',
        yaxis='y2',
        line=dict(color='red'),
        hovertemplate='%{x}<br>Cumulative: %{y:.1f}%'
    ))
    
    fig.update_layout(
        title="Explained Variance per Principal Component",
        xaxis_title="Principal Component",
        yaxis_title="Individual Explained Variance (%)",
        yaxis2=dict(
            title="Cumulative Explained Variance (%)",
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        height=600,  # Increased height since it's the only plot
        showlegend=True,
        yaxis=dict(range=[0, max(individual_variance) * 1.1]),
        hovermode='x unified',
        bargap=0.3,
        plot_bgcolor='white',
        xaxis_gridcolor='lightgrey',
        yaxis_gridcolor='lightgrey'
    )
    
    return fig

@app.callback(
    [Output('pca-by-regime-plot', 'figure'),
     Output('pca-loadings-plot', 'figure')],
    [Input('stored-data', 'children'),
     Input('pca-regime-states', 'value')]
)
def update_regime_pca_plots(stored_data, n_states):
    df = pd.read_json(stored_data)
    df.index = pd.to_datetime(df.index)
    
    # First, identify regimes using HMM
    continuous_series = pd.Series(index=df.index, dtype=float)
    for date in df.index:
        available_values = df.loc[date].dropna()
        if len(available_values) > 0:
            continuous_series[date] = available_values.iloc[0]
    
    continuous_series = continuous_series.dropna()
    log_returns = np.log(continuous_series / continuous_series.shift(1)).dropna()
    
    # Fit HMM
    X_returns = StandardScaler().fit_transform(log_returns.values.reshape(-1, 1))
    hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", 
                               n_iter=1000, random_state=42)
    hmm_model.fit(X_returns)
    states = hmm_model.predict(X_returns)
    
    # Create state-date mapping
    state_dates = pd.Series(states, index=log_returns.index)
    
    # Prepare data for PCA by regime
    regime_pca_results = {}
    regime_loadings = {}
    
    for state in range(n_states):
        # Get dates for this regime
        regime_dates = state_dates[state_dates == state].index
        
        # Prepare term structure data for these dates
        term_structures = []
        for date in regime_dates:
            if date in df.index:
                row = df.loc[date].dropna()
                if len(row) >= 5:
                    contracts = []
                    for contract in row.index:
                        month = contract[0]
                        year = int('20' + contract[1:]) if len(contract) > 1 else 2000
                        month_num = {'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12}[month]
                        expiry = pd.Timestamp(f'{year}-{month_num:02d}-15')
                        contracts.append((contract, expiry))
                    
                    sorted_contracts = sorted(contracts, key=lambda x: x[1])[:5]
                    contract_values = [row[contract[0]] for contract in sorted_contracts]
                    
                    if len(contract_values) == 5:
                        term_structures.append(contract_values)
        
        if term_structures:
            # Perform PCA for this regime
            X = StandardScaler().fit_transform(np.array(term_structures))
            pca = PCA()
            pca.fit(X)
            
            regime_pca_results[state] = pca.explained_variance_ratio_ * 100
            regime_loadings[state] = pca.components_
    
    # Create figure for explained variance by regime
    variance_fig = go.Figure()
    
    for state in regime_pca_results:
        variance_fig.add_trace(go.Bar(
            x=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'],
            y=regime_pca_results[state],
            name=f'State {state}',
            hovertemplate='%{x}<br>Variance: %{y:.1f}%'
        ))
    
    variance_fig.update_layout(
        title="Explained Variance by Regime",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance (%)",
        barmode='group',
        height=500,
        showlegend=True,
        plot_bgcolor='white',
        xaxis_gridcolor='lightgrey',
        yaxis_gridcolor='lightgrey'
    )
    
    # Create figure for loadings comparison
    loadings_fig = go.Figure()
    
    contract_labels = ['M1', 'M2', 'M3', 'M4', 'M5']
    for state in regime_loadings:
        loadings_fig.add_trace(go.Scatter(
            x=contract_labels,
            y=regime_loadings[state][0],  # First principal component
            name=f'State {state}',
            mode='lines+markers'
        ))
    
    loadings_fig.update_layout(
        title="First Principal Component Loadings by Regime",
        xaxis_title="Contract Month",
        yaxis_title="Loading",
        height=500,
        showlegend=True,
        plot_bgcolor='white',
        xaxis_gridcolor='lightgrey',
        yaxis_gridcolor='lightgrey'
    )
    
    return variance_fig, loadings_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
