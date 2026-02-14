import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def plot_posterior_distributions(bayesian_test, n_samples=100000):
    samples_A = bayesian_test.get_posterior_samples('A', n_samples)
    samples_B = bayesian_test.get_posterior_samples('B', n_samples)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Posterior Distributions', 'Density Comparison'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    fig.add_trace(
        go.Histogram(x=samples_A, name='Group A', opacity=0.7, nbinsx=50),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=samples_B, name='Group B', opacity=0.7, nbinsx=50),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=np.linspace(0, 1, 1000),
                  y=np.exp(bayesian_test.results['A']['alpha']-1) * 
                    (1-np.linspace(0, 1, 1000))**(bayesian_test.results['A']['beta']-1),
                  name='Group A Density',
                  line=dict(color='blue', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=np.linspace(0, 1, 1000),
                  y=np.exp(bayesian_test.results['B']['alpha']-1) * 
                    (1-np.linspace(0, 1, 1000))**(bayesian_test.results['B']['beta']-1),
                  name='Group B Density',
                  line=dict(color='red', width=2)),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text="Bayesian Posterior Distributions"
    )
    fig.update_xaxes(title_text="Conversion Rate", row=1, col=1)
    fig.update_xaxes(title_text="Conversion Rate", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=2)
    
    return fig

def plot_uplift_distribution(bayesian_test, n_samples=100000):
    uplift_stats = bayesian_test.uplift_distribution(n_samples)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Absolute Uplift Distribution', 'Relative Uplift (%)'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}]]
    )
    
    fig.add_trace(
        go.Histogram(x=uplift_stats['absolute_uplift'], 
                     name='Absolute Uplift',
                     nbinsx=50,
                     marker_color='green'),
        row=1, col=1
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
    
    fig.add_trace(
        go.Histogram(x=uplift_stats['relative_uplift'], 
                     name='Relative Uplift (%)',
                     nbinsx=50,
                     marker_color='orange'),
        row=1, col=2
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
    
    ci_abs = uplift_stats['credible_interval_absolute']
    ci_rel = uplift_stats['credible_interval_relative']
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Uplift Analysis"
    )
    
    fig.update_xaxes(title_text="Absolute Uplift", row=1, col=1)
    fig.update_xaxes(title_text="Relative Uplift (%)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    return fig

def plot_sequential_history(sequential_test):
    df = sequential_test.get_history_df()
    
    fig = go.Figure()
    
    for group in ['A', 'B']:
        group_df = df[df['group'] == group]
        fig.add_trace(go.Scatter(
            x=group_df['cumulative_trials'],
            y=group_df['posterior_mean'],
            mode='lines+markers',
            name=f'Group {group}',
            line=dict(width=3)
        ))
    
    current_prob = sequential_test.get_current_probability()
    
    fig.update_layout(
        title=f'Sequential Bayesian Updating (P(B > A) = {current_prob:.3f})',
        xaxis_title='Cumulative Sample Size',
        yaxis_title='Posterior Mean Conversion Rate',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_dashboard(bayesian_test, frequentist_results, risk_metrics):
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Posterior Distributions', 'Risk Analysis',
                       'Uplift Distribution', 'Credible Intervals',
                       'Sequential Monitoring', 'Frequentist Comparison'),
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "table"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    return fig