import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import warnings
warnings.filterwarnings('ignore')

# Import our modules (same as before)
from bayesian_models import BayesianABTest, FrequentistABTest, SequentialBayesianTest
from visualizations import (
    plot_posterior_distributions, 
    plot_uplift_distribution,
    plot_sequential_history
)
from utils import (
    generate_simulated_data,
    format_results_for_display,
    calculate_required_sample_size,
    calculate_bayes_factor
)

st.set_page_config(
    page_title="Bayesian A/B Testing",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide all Streamlit default UI elements
hide_streamlit_style = """
<style>
    /* Hide all Streamlit default UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stStatusWidget"] {display: none !important;}
    .stApp > header {display: none;}
    
    /* Remove all default padding and set max-width */
    .stApp {
        max-width: 1000px !important;
        padding: 0 !important;
        margin: 0 auto !important;
    }
    
    .stApp > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    div[class*="stAppViewBlockContainer"] {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    
    /* Hide sidebar completely */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Main content container - perfectly centered */
    .main {
        max-width: 900px !important;
        margin: 0 auto !important;
        padding: 2rem 1.5rem !important;
    }
    
    .block-container {
        max-width: 900px !important;
        padding: 2rem 1.5rem !important;
        margin: 0 auto !important;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Custom CSS for black and white minimal design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global reset */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Main container */
    .main-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Headers - clean, bold, minimal */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #000000;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #666666;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Section headers */
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #000000;
        margin: 1.75rem 0 1rem 0;
        letter-spacing: -0.01em;
        border-bottom: 1px solid #e5e5e5;
        padding-bottom: 0.75rem;
    }
    
    .subsection-title {
        font-size: 1rem;
        font-weight: 600;
        color: #333333;
        margin: 1.25rem 0 0.75rem 0;
    }
    
    /* Input styling - clean, minimal borders */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div > div {
        border: 1px solid #e5e5e5 !important;
        border-radius: 6px !important;
        background: #ffffff !important;
        color: #000000 !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.15s ease !important;
    }
    
    .stTextInput > div > div > input:hover,
    .stTextArea > div > div > textarea:hover,
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {
        border-color: #000000 !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #000000 !important;
        box-shadow: 0 0 0 1px #000000 !important;
        outline: none !important;
    }
    
    /* Button styling - pure black and white */
    .stButton > button {
        background: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        border: 1px solid #000000 !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    }
    
    /* Secondary button */
    .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #e5e5e5 !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #f5f5f5 !important;
        border-color: #000000 !important;
    }
    
    /* Metrics grid - clean cards */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: #fafafa;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        padding: 1.25rem 1rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #000000;
        transform: translateY(-1px);
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #666666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #000000;
        line-height: 1;
    }
    
    .metric-unit {
        font-size: 0.85rem;
        color: #999999;
        margin-left: 2px;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        background: #f5f5f5;
        color: #000000;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid #e5e5e5;
        margin: 0.5rem 0;
    }
    
    .status-badge-success {
        background: #f5f5f5;
        border-left: 3px solid #22c55e;
    }
    
    .status-badge-warning {
        background: #fef9e7;
        border-left: 3px solid #f59e0b;
    }
    
    .status-badge-info {
        background: #f0f7ff;
        border-left: 3px solid #3b82f6;
    }
    
    /* Insight box - minimal, black accent border */
    .insight-box {
        background: #fafafa;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
        border-left: 3px solid #000000;
    }
    
    .insight-title {
        font-weight: 600;
        margin-bottom: 0.75rem;
        color: #000000;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Info message */
    .info-message {
        background: #fafafa;
        border: 1px solid #e5e5e5;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        color: #000000;
        font-size: 0.9rem;
        margin: 1rem 0;
        border-left: 3px solid #000000;
    }
    
    /* Tab styling - centered, clean, no colors */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        justify-content: center;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #666666 !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.5rem !important;
        border: none !important;
        transition: all 0.15s ease !important;
        font-size: 0.95rem !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #000000 !important;
        background: #f5f5f5 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #000000 !important;
        border-bottom: 2px solid #000000 !important;
        background: transparent !important;
    }
    
    /* Radio buttons - pill style */
    .stRadio > div {
        gap: 0.75rem !important;
        flex-wrap: wrap !important;
    }
    
    .stRadio [role="radiogroup"] {
        gap: 0.75rem !important;
    }
    
    .stRadio [role="radio"] {
        background: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 20px !important;
        padding: 0.4rem 1.25rem !important;
        cursor: pointer !important;
        font-size: 0.9rem !important;
    }
    
    .stRadio [role="radio"][aria-checked="true"] {
        background: #000000 !important;
        border-color: #000000 !important;
        color: white !important;
    }
    
    .stRadio [role="radio"] > div:first-child {
        display: none !important;
    }
    
    /* Checkbox */
    .stCheckbox {
        margin: 0.5rem 0 !important;
    }
    
    .stCheckbox > div > div > div {
        border-color: #e5e5e5 !important;
    }
    
    .stCheckbox [aria-checked="true"] > div > div > div {
        background-color: #000000 !important;
        border-color: #000000 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #e5e5e5 !important;
        height: 4px !important;
    }
    
    .stSlider [role="slider"] {
        background: #000000 !important;
        border: 2px solid white !important;
        width: 16px !important;
        height: 16px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        margin-top: -6px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 500 !important;
        color: #000000 !important;
        background: #fafafa !important;
        border-radius: 6px !important;
        border: 1px solid #e5e5e5 !important;
        padding: 0.75rem 1rem !important;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e5e5e5 !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
        padding: 1.5rem !important;
    }
    
    /* Data table */
    .dataframe-container {
        background: white;
        border-radius: 6px;
        border: 1px solid #e5e5e5;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .stDataFrame {
        border: none !important;
    }
    
    .stDataFrame [data-testid="StyledDataFrameDataCell"] {
        font-size: 0.9rem !important;
    }
    
    /* Plotly charts - minimal, no backgrounds */
    .js-plotly-plot {
        border-radius: 6px;
        overflow: hidden;
    }
    
    /* Divider */
    .divider {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: #e5e5e5;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999999;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e5e5e5;
    }
    
    /* Hide default labels */
    label[data-testid="stWidgetLabel"] {
        display: none;
    }
    
    /* Metric columns layout */
    .metric-columns {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #e5e5e5 !important;
    }
    
    .stProgress > div > div > div {
        background-color: #000000 !important;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .metrics-row {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .metric-columns {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1rem !important;
            font-size: 0.85rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def format_number(num):
    """Format large numbers with commas"""
    if pd.isna(num) or num is None:
        return "—"
    if isinstance(num, (int, float)):
        if abs(num) >= 1e6:
            return f"{num/1e6:.1f}M"
        elif abs(num) >= 1e3:
            return f"{num/1e3:.1f}K"
        return f"{int(num):,}"
    return str(num)

def format_float(num, decimals=2):
    """Format float with specified decimals"""
    if pd.isna(num) or num is None:
        return "—"
    return f"{num:.{decimals}f}"

def format_percent(num, decimals=1):
    """Format as percentage"""
    if pd.isna(num) or num is None:
        return "—"
    return f"{num:.{decimals}%}"

def render_metric_card(label, value, unit=""):
    """Render a consistent metric card"""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
    </div>
    """

def render_status_badge(text, type="info"):
    """Render a status badge"""
    badge_class = "status-badge"
    if type == "success":
        badge_class += " status-badge-success"
    elif type == "warning":
        badge_class += " status-badge-warning"
    elif type == "info":
        badge_class += " status-badge-info"
    
    return f'<span class="{badge_class}">{text}</span>'

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'bayes_test' not in st.session_state:
    st.session_state.bayes_test = None
if 'sequential_test' not in st.session_state:
    st.session_state.sequential_test = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'analyze'

def main():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">Bayesian A/B Testing</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Posterior probabilities · Expected uplift · Sequential analysis · Decision theory</p>', 
                unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        analyze_page = st.button("Analyze", use_container_width=True,
                               type="primary" if st.session_state.current_page == 'analyze' else "secondary")
        if analyze_page:
            st.session_state.current_page = 'analyze'
    
    with col2:
        sequential_page = st.button("Sequential", use_container_width=True,
                                   type="primary" if st.session_state.current_page == 'sequential' else "secondary")
        if sequential_page:
            st.session_state.current_page = 'sequential'
    
    with col3:
        compare_page = st.button("Compare", use_container_width=True,
                                type="primary" if st.session_state.current_page == 'compare' else "secondary")
        if compare_page:
            st.session_state.current_page = 'compare'
    
    with col4:
        design_page = st.button("Design", use_container_width=True,
                               type="primary" if st.session_state.current_page == 'design' else "secondary")
        if design_page:
            st.session_state.current_page = 'design'
    
    with col5:
        learn_page = st.button("Learn", use_container_width=True,
                              type="primary" if st.session_state.current_page == 'learn' else "secondary")
        if learn_page:
            st.session_state.current_page = 'learn'
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # Render current page
    if st.session_state.current_page == 'analyze':
        show_analyze_page()
    elif st.session_state.current_page == 'sequential':
        show_sequential_page()
    elif st.session_state.current_page == 'compare':
        show_compare_page()
    elif st.session_state.current_page == 'design':
        show_design_page()
    elif st.session_state.current_page == 'learn':
        show_learn_page()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <span style="font-weight: 500;">Bayesian A/B Testing</span> · 
        Beta-Binomial · Posterior · Expected Loss · Sequential Analysis · 
        Built with Streamlit · Black & White Edition
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_analyze_page():
    st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)
    
    # Data input section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_method = "Simulated"
        st.button("Simulated", use_container_width=True,
                 type="primary" if st.session_state.get('input_method') == 'simulated' else "secondary")
        st.session_state.input_method = 'simulated'
    
    with col2:
        st.button("Manual Entry", use_container_width=True,
                 type="primary" if st.session_state.get('input_method') == 'manual' else "secondary")
    
    with col3:
        st.button("Upload CSV", use_container_width=True,
                 type="primary" if st.session_state.get('input_method') == 'upload' else "secondary")
    
    st.session_state.input_method = 'simulated'
    
    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    
    # Data configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-title">Group A (Control)</div>', unsafe_allow_html=True)
        
        if st.session_state.input_method == 'simulated':
            conversion_a = st.slider("True conversion rate", 0.01, 0.50, 0.10, 0.01, key="conv_a", format="%.2f")
            sample_a = st.slider("Sample size", 100, 10000, 1000, 100, key="sample_a")
        else:
            successes_a = st.number_input("Successes", min_value=0, value=100, key="succ_a")
            trials_a = st.number_input("Trials", min_value=1, value=1000, key="trials_a")
    
    with col2:
        st.markdown('<div class="subsection-title">Group B (Treatment)</div>', unsafe_allow_html=True)
        
        if st.session_state.input_method == 'simulated':
            conversion_b = st.slider("True conversion rate", 0.01, 0.50, 0.12, 0.01, key="conv_b", format="%.2f")
            sample_b = st.slider("Sample size", 100, 10000, 1000, 100, key="sample_b")
        else:
            successes_b = st.number_input("Successes", min_value=0, value=120, key="succ_b")
            trials_b = st.number_input("Trials", min_value=1, value=1000, key="trials_b")
    
    # Prior parameters
    with st.expander("Prior parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            alpha_prior = st.number_input("Alpha (successes)", min_value=0.1, value=1.0, step=0.1)
        with col2:
            beta_prior = st.number_input("Beta (failures)", min_value=0.1, value=1.0, step=0.1)
        
        st.markdown(f"""
        <div class="info-message">
            <strong>Prior represents:</strong> {alpha_prior-1:.1f} prior successes, {beta_prior-1:.1f} prior failures · 
            Prior mean: {alpha_prior/(alpha_prior+beta_prior):.1%}
        </div>
        """, unsafe_allow_html=True)
    
    # Run analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button("Run Bayesian Analysis", use_container_width=True)
    
    if run_button:
        with st.spinner("Computing posterior distributions..."):
            # Generate or use data
            if st.session_state.input_method == 'simulated':
                data = generate_simulated_data(conversion_a, conversion_b, sample_a, sample_b)
                successes_a = data['A']['successes']
                successes_b = data['B']['successes']
                trials_a = data['A']['trials']
                trials_b = data['B']['trials']
            else:
                trials_a = trials_a
                trials_b = trials_b
            
            # Run Bayesian test
            bayes_test = BayesianABTest(alpha_prior=alpha_prior, beta_prior=beta_prior)
            bayes_test.update_posterior(successes_a, trials_a, 'A')
            bayes_test.update_posterior(successes_b, trials_b, 'B')
            
            # Calculate results
            risk_metrics = bayes_test.calculate_risk()
            bayes_factor = calculate_bayes_factor(bayes_test)
            
            # Store in session state
            st.session_state.data = {
                'A': {'successes': successes_a, 'trials': trials_a},
                'B': {'successes': successes_b, 'trials': trials_b}
            }
            st.session_state.bayes_test = bayes_test
            st.session_state.results = {
                'risk_metrics': risk_metrics,
                'bayes_factor': bayes_factor
            }
            
            st.markdown(render_status_badge(f"✓ Analysis complete · Group B beats A with {risk_metrics['probability_B_beats_A']:.1%} probability", "success"), 
                       unsafe_allow_html=True)
    
    # Display results if available
    if st.session_state.bayes_test is not None and st.session_state.results is not None:
        bayes_test = st.session_state.bayes_test
        risk_metrics = st.session_state.results['risk_metrics']
        bayes_factor = st.session_state.results['bayes_factor']
        
        # Summary metrics
        st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rate_a = bayes_test.results['A']['conversion_rate']
            st.markdown(render_metric_card("Group A", format_percent(rate_a)), unsafe_allow_html=True)
        
        with col2:
            rate_b = bayes_test.results['B']['conversion_rate']
            st.markdown(render_metric_card("Group B", format_percent(rate_b)), unsafe_allow_html=True)
        
        with col3:
            st.markdown(render_metric_card("P(B > A)", format_percent(risk_metrics['probability_B_beats_A'])), 
                       unsafe_allow_html=True)
        
        with col4:
            st.markdown(render_metric_card("Expected Uplift", f"{risk_metrics['expected_uplift']:.1f}%"), 
                       unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Posterior distributions
        st.markdown('<div class="subsection-title">Posterior Distributions</div>', unsafe_allow_html=True)
        fig_posterior = plot_posterior_distributions(bayes_test)
        st.plotly_chart(fig_posterior, use_container_width=True)
        
        # Uplift analysis
        st.markdown('<div class="subsection-title">Uplift Analysis</div>', unsafe_allow_html=True)
        fig_uplift = plot_uplift_distribution(bayes_test)
        st.plotly_chart(fig_uplift, use_container_width=True)
        
        # Decision metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="subsection-title">Decision Analysis</div>', unsafe_allow_html=True)
            
            # Loss comparison
            loss_fig = go.Figure(data=[
                go.Bar(
                    x=['Choose A', 'Choose B'],
                    y=[risk_metrics['expected_loss_choose_A'], risk_metrics['expected_loss_choose_B']],
                    marker_color=['#222222', '#666666'],
                    text=[f"{risk_metrics['expected_loss_choose_A']:.3f}", f"{risk_metrics['expected_loss_choose_B']:.3f}"],
                    textposition='outside'
                )
            ])
            loss_fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Inter', size=11),
                yaxis=dict(
                    title="Expected Loss",
                    gridcolor='#f0f0f0',
                    zeroline=True,
                    zerolinecolor='#e5e5e5'
                ),
                showlegend=False
            )
            st.plotly_chart(loss_fig, use_container_width=True)
            
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">Recommended choice</div>
                <strong>Group {risk_metrics['recommended_choice']}</strong> · 
                {risk_metrics['expected_loss_choose_A']:.4f} vs {risk_metrics['expected_loss_choose_B']:.4f} expected loss
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="subsection-title">Evidence Strength</div>', unsafe_allow_html=True)
            
            # Bayes Factor gauge
            bf = bayes_factor['bayes_factor']
            
            # Determine evidence category
            if bf > 100:
                evidence = "Decisive"
                color = "#000000"
            elif bf > 30:
                evidence = "Very Strong"
                color = "#333333"
            elif bf > 10:
                evidence = "Strong"
                color = "#666666"
            elif bf > 3:
                evidence = "Substantial"
                color = "#999999"
            else:
                evidence = "Weak"
                color = "#cccccc"
            
            st.markdown(f"""
            <div style="background: #fafafa; border: 1px solid #e5e5e5; border-radius: 8px; padding: 1.5rem; text-align: center;">
                <div style="font-size: 3rem; font-weight: 700; color: {color};">{bf:.1f}</div>
                <div style="font-size: 0.9rem; color: #666666; margin-top: 0.5rem;">Bayes Factor</div>
                <div style="margin-top: 1rem; padding: 0.5rem; background: white; border-radius: 4px; border: 1px solid #e5e5e5;">
                    <strong>{evidence} evidence</strong> · {bayes_factor['interpretation']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability of meaningful effect
            min_uplift = st.slider("Minimum meaningful uplift", 0.0, 20.0, 5.0, 1.0, key="min_uplift_analyze", format="%.0f%%")
            samples_a = bayes_test.get_posterior_samples('A')
            samples_b = bayes_test.get_posterior_samples('B')
            prob_meaningful = np.mean((samples_b - samples_a) / samples_a * 100 > min_uplift)
            
            st.markdown(f"""
            <div style="margin-top: 1rem; padding: 1rem; background: #fafafa; border: 1px solid #e5e5e5; border-radius: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #666666;">P(> {min_uplift:.0f}% uplift)</span>
                    <span style="font-weight: 600;">{prob_meaningful:.1%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

def show_sequential_page():
    st.markdown('<div class="section-title">Sequential Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-message">
        <strong>Real-time Bayesian updating.</strong> Watch how posterior probabilities evolve as data arrives sequentially.
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.bayes_test is None:
        st.info("Run analysis first to see sequential updates")
        return
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        n_batches = st.slider("Number of batches", 5, 50, 20, 5)
    
    with col2:
        update_speed = st.select_slider("Update speed", options=["Slow", "Medium", "Fast"], value="Medium")
    
    # Run sequential simulation
    if st.button("Run Sequential Simulation", use_container_width=True):
        with st.spinner("Simulating sequential updates..."):
            # Get data from session state
            data_a = st.session_state.data['A']
            data_b = st.session_state.data['B']
            
            # Initialize sequential test
            seq_test = SequentialBayesianTest(alpha_prior=1.0, beta_prior=1.0)
            
            # Simulate sequential observations
            np.random.seed(42)
            
            # Create batch data
            successes_a_remaining = data_a['successes']
            successes_b_remaining = data_b['successes']
            trials_a_remaining = data_a['trials']
            trials_b_remaining = data_b['trials']
            
            batch_size_a = trials_a_remaining // n_batches
            batch_size_b = trials_b_remaining // n_batches
            
            progress_bar = st.progress(0)
            
            for i in range(n_batches):
                # Group A batch
                batch_trials_a = min(batch_size_a, trials_a_remaining)
                if batch_trials_a > 0:
                    if successes_a_remaining > 0:
                        batch_successes_a = np.random.hypergeometric(
                            successes_a_remaining, 
                            trials_a_remaining - successes_a_remaining,
                            batch_trials_a
                        )
                    else:
                        batch_successes_a = 0
                    
                    seq_test.add_observation('A', batch_successes_a, batch_trials_a)
                    successes_a_remaining -= batch_successes_a
                    trials_a_remaining -= batch_trials_a
                
                # Group B batch
                batch_trials_b = min(batch_size_b, trials_b_remaining)
                if batch_trials_b > 0:
                    if successes_b_remaining > 0:
                        batch_successes_b = np.random.hypergeometric(
                            successes_b_remaining,
                            trials_b_remaining - successes_b_remaining,
                            batch_trials_b
                        )
                    else:
                        batch_successes_b = 0
                    
                    seq_test.add_observation('B', batch_successes_b, batch_trials_b)
                    successes_b_remaining -= batch_successes_b
                    trials_b_remaining -= batch_trials_b
                
                progress_bar.progress((i + 1) / n_batches)
            
            progress_bar.empty()
            st.session_state.sequential_test = seq_test
            
            st.markdown(render_status_badge("✓ Sequential simulation complete", "success"), unsafe_allow_html=True)
    
    # Display sequential results
    if st.session_state.sequential_test is not None:
        seq_test = st.session_state.sequential_test
        
        # Posterior evolution
        st.markdown('<div class="subsection-title">Posterior Mean Evolution</div>', unsafe_allow_html=True)
        fig_seq = plot_sequential_history(seq_test)
        st.plotly_chart(fig_seq, use_container_width=True)
        
        # Probability evolution
        st.markdown('<div class="subsection-title">Probability B > A Over Time</div>', unsafe_allow_html=True)
        
        # Calculate probability at each step
        df_history = seq_test.get_history_df()
        prob_history = []
        
        steps = range(1, min(len(df_history[df_history['group'] == 'A']), 
                            len(df_history[df_history['group'] == 'B'])) + 1)
        
        temp_test = SequentialBayesianTest(alpha_prior=1.0, beta_prior=1.0)
        
        for step in steps:
            a_obs = df_history[(df_history['group'] == 'A') & (df_history['step'] <= step)]
            b_obs = df_history[(df_history['group'] == 'B') & (df_history['step'] <= step)]
            
            if not a_obs.empty and not b_obs.empty:
                cum_a = a_obs.iloc[-1]
                cum_b = b_obs.iloc[-1]
                
                temp_test = SequentialBayesianTest(alpha_prior=1.0, beta_prior=1.0)
                temp_test.add_observation('A', cum_a['cumulative_successes'], cum_a['cumulative_trials'])
                temp_test.add_observation('B', cum_b['cumulative_successes'], cum_b['cumulative_trials'])
                
                prob = temp_test.get_current_probability()
                prob_history.append({'step': step, 'probability': prob})
        
        if prob_history:
            prob_df = pd.DataFrame(prob_history)
            
            fig_prob = go.Figure()
            
            # Add probability line
            fig_prob.add_trace(go.Scatter(
                x=prob_df['step'],
                y=prob_df['probability'],
                mode='lines+markers',
                line=dict(color='#000000', width=2),
                marker=dict(size=6),
                name='P(B > A)'
            ))
            
            # Add decision threshold lines
            fig_prob.add_hline(
                y=0.95, 
                line_dash="dash", 
                line_color="#666666",
                annotation_text="Strong evidence (95%)",
                annotation_position="top left"
            )
            
            fig_prob.add_hline(
                y=0.5, 
                line_dash="dot", 
                line_color="#999999",
                annotation_text="No evidence",
                annotation_position="bottom left"
            )
            
            fig_prob.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Inter', size=11),
                xaxis=dict(
                    title="Batch Number",
                    gridcolor='#f0f0f0',
                    dtick=2
                ),
                yaxis=dict(
                    title="Probability",
                    gridcolor='#f0f0f0',
                    range=[0, 1],
                    tickformat='.0%'
                ),
                showlegend=False
            )
            
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # Decision timing
            final_prob = prob_history[-1]['probability']
            steps_to_95 = next((p['step'] for p in prob_history if p['probability'] >= 0.95), None)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Final Probability", f"{final_prob:.1%}")
            with col2:
                st.metric("Batches to 95%", steps_to_95 if steps_to_95 else "—")
            with col3:
                st.metric("Total Batches", len(prob_history))

def show_compare_page():
    st.markdown('<div class="section-title">Comparison</div>', unsafe_allow_html=True)
    
    if st.session_state.bayes_test is None or st.session_state.data is None:
        st.info("Run analysis first to compare methods")
        return
    
    bayes_test = st.session_state.bayes_test
    data = st.session_state.data
    
    # Run frequentist tests
    freq_test = FrequentistABTest()
    chi2_results = freq_test.chi_squared_test(
        data['A']['successes'], data['A']['trials'],
        data['B']['successes'], data['B']['trials']
    )
    prop_results = freq_test.proportion_test(
        data['A']['successes'], data['A']['trials'],
        data['B']['successes'], data['B']['trials']
    )
    
    # Comparison metrics
    st.markdown('<div class="subsection-title">Side by Side Comparison</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Bayesian Approach**")
        
        metrics_df = pd.DataFrame({
            'Metric': [
                'P(B > A)',
                'Expected Uplift',
                '95% Credible Interval',
                'Bayes Factor',
                'Recommended Choice'
            ],
            'Value': [
                f"{st.session_state.results['risk_metrics']['probability_B_beats_A']:.1%}",
                f"{st.session_state.results['risk_metrics']['expected_uplift']:.1f}%",
                f"[{st.session_state.results['risk_metrics']['uplift_ci'][0]:.1f}%, {st.session_state.results['risk_metrics']['uplift_ci'][1]:.1f}%]",
                f"{st.session_state.results['bayes_factor']['bayes_factor']:.1f}",
                st.session_state.results['risk_metrics']['recommended_choice']
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Frequentist Approach**")
        
        metrics_df = pd.DataFrame({
            'Metric': [
                'p-value (χ²)',
                'p-value (Z-test)',
                '95% Confidence Interval',
                'Significant at α=0.05',
                'Test Statistic'
            ],
            'Value': [
                f"{chi2_results['p_value']:.4f}",
                f"{prop_results['p_value']:.4f}",
                f"[{prop_results['ci_difference'][0]:.3f}, {prop_results['ci_difference'][1]:.3f}]",
                "✓ Yes" if prop_results['significant_at_0.05'] else "✗ No",
                f"Z = {prop_results['z_statistic']:.2f}"
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Visual comparison
    st.markdown('<div class="subsection-title">ATE Comparison</div>', unsafe_allow_html=True)
    
    # Calculate difference
    diff = data['B']['successes']/data['B']['trials'] - data['A']['successes']/data['A']['trials']
    
    fig = go.Figure()
    
    # Add Bayesian credible interval
    ci_low, ci_high = st.session_state.results['risk_metrics']['uplift_ci']
    ci_low_abs = ci_low / 100 * (data['A']['successes']/data['A']['trials'])
    ci_high_abs = ci_high / 100 * (data['A']['successes']/data['A']['trials'])
    
    fig.add_trace(go.Scatter(
        x=[ci_low_abs, ci_high_abs],
        y=[0.2, 0.2],
        mode='lines',
        line=dict(color='#000000', width=4),
        name='Bayesian CrI',
        showlegend=True
    ))
    
    # Add frequentist confidence interval
    fig.add_trace(go.Scatter(
        x=[prop_results['ci_difference'][0], prop_results['ci_difference'][1]],
        y=[0.1, 0.1],
        mode='lines',
        line=dict(color='#666666', width=4, dash='dash'),
        name='Frequentist CI',
        showlegend=True
    ))
    
    # Add point estimate
    fig.add_trace(go.Scatter(
        x=[diff],
        y=[0.15],
        mode='markers',
        marker=dict(color='#000000', size=12, symbol='diamond'),
        name='Observed Difference',
        showlegend=True
    ))
    
    # Add zero line
    fig.add_vline(x=0, line_dash="dot", line_color="#999999")
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', size=11),
        xaxis=dict(
            title="Absolute Difference in Conversion Rate",
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinecolor='#e5e5e5'
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
            gridcolor='#f0f0f0'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown('<div class="subsection-title">Interpretation</div>', unsafe_allow_html=True)
    
    bayes_prob = st.session_state.results['risk_metrics']['probability_B_beats_A']
    freq_p = prop_results['p_value']
    
    if bayes_prob > 0.95 and freq_p < 0.05:
        interpretation = "Both methods agree: Strong evidence that B is different from A."
    elif bayes_prob > 0.95 and freq_p >= 0.05:
        interpretation = "Bayesian finds strong evidence while frequentist does not. This can happen with small samples or when prior information is informative."
    elif bayes_prob <= 0.95 and freq_p < 0.05:
        interpretation = "Frequentist finds significance while Bayesian does not. This might indicate a small effect size or issues with multiple testing."
    else:
        interpretation = "Both methods agree: Insufficient evidence to conclude B is different from A."
    
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">Key insight</div>
        {interpretation}<br><br>
        <strong>Remember:</strong> p-values ask "How likely is this data if H₀ is true?"<br>
        Bayesian probabilities ask "How likely is H₁ given this data and prior?"
    </div>
    """, unsafe_allow_html=True)

def show_design_page():
    st.markdown('<div class="section-title">Experiment Design</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-message">
        <strong>Design your A/B test.</strong> Calculate required sample sizes and simulate different scenarios.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-title">Design Parameters</div>', unsafe_allow_html=True)
        
        baseline = st.slider("Baseline conversion rate", 0.01, 0.50, 0.10, 0.01, format="%.2f")
        mde = st.slider("Minimum detectable effect (relative)", 0.05, 0.50, 0.10, 0.01, format="%.0f%%")
        alpha = st.select_slider("Significance level (α)", options=[0.01, 0.05, 0.10], value=0.05)
        power = st.select_slider("Statistical power (1-β)", options=[0.7, 0.8, 0.9, 0.95], value=0.8)
    
    with col2:
        st.markdown('<div class="subsection-title">Sample Size</div>', unsafe_allow_html=True)
        
        # Calculate required sample size
        required_n = calculate_required_sample_size(
            mde=mde,
            baseline_rate=baseline,
            alpha=alpha,
            power=power
        )
        
        st.markdown(f"""
        <div style="background: #fafafa; border: 1px solid #e5e5e5; border-radius: 8px; padding: 2rem; text-align: center;">
            <div style="font-size: 3rem; font-weight: 700;">{required_n:,}</div>
            <div style="font-size: 0.9rem; color: #666666; margin-top: 0.5rem;">Required sample size per group</div>
            <div style="margin-top: 1.5rem; font-size: 0.9rem; color: #333333;">
                Total: {(required_n * 2):,} observations
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Power curve
    st.markdown('<div class="subsection-title">Power Analysis</div>', unsafe_allow_html=True)
    
    sample_sizes = np.arange(100, 5000, 100)
    powers = []
    
    for n in sample_sizes:
        effect_size = (baseline * (1 + mde) - baseline) / np.sqrt(baseline * (1 - baseline))
        from statsmodels.stats.power import NormalIndPower
        power_calc = NormalIndPower().solve_power(
            effect_size=effect_size,
            nobs1=n,
            alpha=alpha,
            ratio=1.0,
            alternative='two-sided'
        )
        powers.append(power_calc)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sample_sizes,
        y=powers,
        mode='lines',
        line=dict(color='#000000', width=3),
        fill='tozeroy',
        fillcolor='rgba(0,0,0,0.05)'
    ))
    
    # Add target power line
    fig.add_hline(
        y=power, 
        line_dash="dash", 
        line_color="#666666",
        annotation_text=f"Target power = {power:.0%}",
        annotation_position="top left"
    )
    
    # Add current sample size
    fig.add_vline(
        x=required_n, 
        line_dash="dot", 
        line_color="#999999",
        annotation_text=f"Required n = {required_n}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', size=11),
        xaxis=dict(
            title="Sample Size per Group",
            gridcolor='#f0f0f0'
        ),
        yaxis=dict(
            title="Statistical Power",
            gridcolor='#f0f0f0',
            range=[0, 1],
            tickformat='.0%'
        ),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scenario simulation
    st.markdown('<div class="subsection-title">Scenario Simulation</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sim_baseline = st.number_input("True A rate", min_value=0.01, max_value=0.50, value=0.10, step=0.01, format="%.2f")
    with col2:
        sim_treatment = st.number_input("True B rate", min_value=0.01, max_value=0.50, value=0.12, step=0.01, format="%.2f")
    with col3:
        sim_n = st.number_input("Sample size", min_value=100, max_value=10000, value=1000, step=100)
    
    if st.button("Simulate Scenario", use_container_width=True):
        # Run multiple simulations
        n_sims = 100
        bayesian_correct = 0
        frequentist_correct = 0
        
        with st.spinner(f"Running {n_sims} simulations..."):
            for i in range(n_sims):
                # Generate data
                data = generate_simulated_data(sim_baseline, sim_treatment, sim_n, sim_n, seed=i)
                
                # Bayesian
                bayes_test = BayesianABTest(alpha_prior=1.0, beta_prior=1.0)
                bayes_test.update_posterior(data['A']['successes'], data['A']['trials'], 'A')
                bayes_test.update_posterior(data['B']['successes'], data['B']['trials'], 'B')
                prob = bayes_test.probability_B_beats_A()
                if (sim_treatment > sim_baseline and prob > 0.95) or (sim_treatment < sim_baseline and prob < 0.05):
                    bayesian_correct += 1
                
                # Frequentist
                prop_test = FrequentistABTest().proportion_test(
                    data['A']['successes'], data['A']['trials'],
                    data['B']['successes'], data['B']['trials']
                )
                if (sim_treatment > sim_baseline and prop_test['p_value'] < 0.05) or \
                   (sim_treatment < sim_baseline and prop_test['p_value'] < 0.05):
                    frequentist_correct += 1
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bayesian correct decisions", f"{bayesian_correct}/{n_sims}", f"{bayesian_correct/n_sims:.1%}")
        with col2:
            st.metric("Frequentist correct decisions", f"{frequentist_correct}/{n_sims}", f"{frequentist_correct/n_sims:.1%}")

def show_learn_page():
    st.markdown('<div class="section-title">Learning Resources</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">The Bayesian Advantage</div>
        Bayesian A/B testing gives you:<br>
        • <strong>Probabilistic interpretation</strong> — P(B > A) = 95% is intuitive<br>
        • <strong>Sequential analysis</strong> — Update beliefs as data arrives<br>
        • <strong>Decision theory</strong> — Expected loss for business decisions<br>
        • <strong>Prior knowledge</strong> — Incorporate historical information
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Key Concepts**")
        st.markdown("""
        • **Prior** — Beliefs before seeing data  
        • **Likelihood** — Probability of observed data  
        • **Posterior** — Updated beliefs after data  
        • **Credible Interval** — Range containing true value with 95% probability  
        • **Expected Loss** — Cost of making wrong decision  
        • **Bayes Factor** — Evidence strength for H₁ vs H₀
        """)
    
    with col2:
        st.markdown("**Interpretation Guide**")
        st.markdown("""
        **P(B > A)**  
        >95% — Strong evidence B is better  
        80–95% — Moderate evidence  
        50–80% — Weak evidence  
        <5% — Strong evidence A is better  
        
        **Bayes Factor**  
        >100 — Decisive evidence  
        30–100 — Very strong  
        10–30 — Strong  
        3–10 — Substantial  
        1–3 — Anecdotal  
        <1 — Supports H₀
        """)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #fafafa; border: 1px solid #e5e5e5; border-radius: 8px; padding: 1.5rem;">
        <strong>When to use each method:</strong><br><br>
        
        <strong>Bayesian (Beta-Binomial):</strong> Conversion rates, proportions, any binary outcome — especially when you want intuitive probabilities or have prior information<br><br>
        
        <strong>Frequentist (p-values):</strong> When you need regulatory approval, have pre-registered analysis, or are comparing with published literature<br><br>
        
        <strong>Sequential Analysis:</strong> When monitoring experiments in real-time, want to stop early, or need to make ongoing decisions<br><br>
        
        <strong>Decision Theory:</strong> When business impact matters more than statistical significance, or you need to choose between multiple options
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()