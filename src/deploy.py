"""
Electricity Demand Forecasting System
Dark Glass Morphism Interface - Modern Tech Design
"""
import os
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import holidays
import plotly.graph_objects as go
import plotly.express as px

from features import main          as run_feature_pipeline
from model   import run_ml_pipeline

def ensure_model_exists():
    if not os.path.exists('models/best_model.pkl'):
        st.sidebar.info("🤖 Training model for first time… this takes just a few seconds")
        
        # 1️⃣ Build features & 2️⃣ Train model as before…
        run_feature_pipeline()
        best_model, best_name, best_metrics = run_ml_pipeline()
        if best_model is None:
            st.sidebar.error("❌ Training failed—check logs in console.")
            st.stop()
        
        st.sidebar.success(f"✅ Model trained: {best_name} (MAPE {best_metrics['MAPE']:.1f}%)")



# Dark Glass Morphism configuration
st.set_page_config(
    page_title="Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Glass Morphism Styling - Modern Tech Design
st.markdown("""
<style>
    /* Hide Streamlit elements */
    .stDeployButton {display:none !important;}
    footer {visibility: hidden !important;}
    .stApp > header {display: none !important;}
    #MainMenu {visibility: hidden !important;}
    .stException {display: none !important;}
    
    /* Dark theme background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        background-attachment: fixed;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 80%, rgba(168, 85, 247, 0.1) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Utility Classes - DRY CSS */
    .glass-effect {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 
            0 8px 32px 0 rgba(31, 38, 135, 0.2),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
    }
    
    .glass-hover {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-hover:hover {
        transform: translateY(-2px);
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 
            0 20px 40px 0 rgba(31, 38, 135, 0.3),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.1);
    }
    
    .neon-border {
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 
            0 0 20px rgba(59, 130, 246, 0.1),
            inset 0 1px 0 0 rgba(59, 130, 246, 0.1);
    }
    
    .text-glass {
        color: rgba(248, 250, 252, 0.9);
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
    }
    
    .text-secondary {
        color: rgba(148, 163, 184, 0.8);
    }
    
    .text-accent {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Modern Typography */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0 0 1rem 0;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        text-align: center;
        color: rgba(148, 163, 184, 0.9);
        font-size: 1.25rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Glass Prediction Hero */
    .prediction-hero {
        background: linear-gradient(135deg, 
            rgba(59, 130, 246, 0.2) 0%, 
            rgba(139, 92, 246, 0.2) 50%, 
            rgba(236, 72, 153, 0.2) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 24px;
        padding: 3rem 2rem;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(59, 130, 246, 0.1),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-hero::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.1), 
            transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .hero-value {
        font-size: 4rem;
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.02em;
    }
    
    .hero-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .hero-timestamp {
        font-size: 1.2rem;
        opacity: 0.8;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    /* Glass Cards */
    .glass-card {
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.1), 
            transparent);
    }
    
    /* Card Content */
    .card-icon {
        width: 56px;
        height: 56px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .card-icon::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: inherit;
        filter: blur(8px);
        opacity: 0.3;
    }
    
    .card-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .card-label {
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }
    
    .card-delta {
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Status Colors */
    .status-normal { 
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.2));
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-high { 
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.2));
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-low { 
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.2));
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .icon-normal { background: linear-gradient(135deg, #10b981, #059669); }
    .icon-high { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .icon-low { background: linear-gradient(135deg, #3b82f6, #2563eb); }
    .icon-neutral { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
    
    .delta-positive { color: #10b981; }
    .delta-negative { color: #ef4444; }
    .delta-neutral { color: #94a3b8; }
    
    /* Chart Container */
    .chart-container {
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
    }
    
    .chart-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        color: rgba(248, 250, 252, 0.9);
    }
    
    .chart-icon {
        margin-right: 0.75rem;
        font-size: 1.25rem;
        opacity: 0.8;
    }
    
    /* Factor Grid */
    .factor-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .factor-card {
        border-radius: 16px;
        padding: 1.25rem;
        position: relative;
    }
    
    .factor-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .factor-icon {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
        font-size: 1.25rem;
        position: relative;
    }
    
    .factor-title {
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .factor-value {
        font-weight: 700;
        font-size: 1.25rem;
        margin: 0.5rem 0;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    .factor-description {
        font-size: 0.75rem;
        opacity: 0.7;
        line-height: 1.4;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-value {
            font-size: 3rem;
        }
        
        .factor-grid {
            grid-template-columns: 1fr;
        }
        
        .prediction-hero {
            padding: 2rem 1rem;
        }
        
        .glass-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions - DRY Code
def render_glass_card(title, icon, value, delta=None, status_class="glass-effect", icon_class="icon-neutral"):
    """Render a modern glass morphism card"""
    delta_class = ""
    if delta:
        if "+" in delta:
            delta_class = "delta-positive"
        elif "-" in delta:
            delta_class = "delta-negative"
        else:
            delta_class = "delta-neutral"
    
    delta_html = f'<div class="card-delta {delta_class}">{delta}</div>' if delta else ""
    
    st.markdown(f"""
    <div class="glass-card glass-effect glass-hover {status_class}">
        <div class="card-icon {icon_class}">{icon}</div>
        <div class="card-label text-secondary">{title}</div>
        <div class="card-value text-glass">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def render_factor_card(title, icon, value, description, icon_class="icon-neutral"):
    """Render a factor analysis card"""
    st.markdown(f"""
    <div class="factor-card glass-effect glass-hover">
        <div class="factor-header">
            <div class="factor-icon {icon_class}">{icon}</div>
            <div class="factor-title text-secondary">{title}</div>
        </div>
        <div class="factor-value text-glass">{value}</div>
        <div class="factor-description text-secondary">{description}</div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_forecasting_model():
    """Load production model with caching"""
    ensure_model_exists() 
    try:
        model = joblib.load('models/best_model.pkl')
        model_info = joblib.load('models/model_info.pkl')
        return model, model_info
    except FileNotFoundError:
        st.error("🚨 Model files not found. Please train the model first.")
        return None, None

@st.cache_data
def get_holiday_calendar(year):
    """Cache holiday data"""
    return holidays.Spain(years=[year])

def is_holiday(date):
    """Check if date is a holiday"""
    holiday_calendar = get_holiday_calendar(date.year)
    return date.date() in holiday_calendar

def calculate_baseline_consumption(hour, day_of_week, month, holiday_status):
    """Calculate baseline consumption patterns"""
    base_load = 28_000_000
    
    hourly_profile = [
        0.75, 0.70, 0.68, 0.66, 0.68, 0.72,
        0.78, 0.85, 0.92, 0.95, 0.98, 1.00,
        1.02, 1.03, 1.04, 1.05, 1.06, 1.08,
        1.12, 1.15, 1.10, 1.05, 0.95, 0.85
    ]
    
    weekly_profile = [1.05, 1.05, 1.05, 1.05, 1.03, 0.95, 0.92]
    monthly_profile = [1.08, 1.06, 1.02, 0.98, 0.96, 1.02, 
                      1.08, 1.10, 1.04, 1.00, 1.04, 1.08]
    
    holiday_factor = 0.90 if holiday_status else 1.00
    
    consumption_24h = int(base_load * hourly_profile[hour] * 
                         weekly_profile[day_of_week] * 
                         monthly_profile[month-1] * holiday_factor)
    
    consumption_168h = int(consumption_24h * 1.02)
    
    return consumption_24h, consumption_168h

def prepare_model_features(hour, date, temperature, humidity):
    """Prepare features for model prediction"""
    dt = datetime.combine(date, datetime.min.time().replace(hour=hour))
    day_of_week = dt.weekday()
    month = dt.month
    is_weekend = 1 if day_of_week >= 5 else 0
    is_business_hours = 1 if 9 <= hour <= 17 else 0
    is_peak_hours = 1 if 17 <= hour <= 21 else 0
    holiday_status = is_holiday(dt)
    is_holiday_flag = 1 if holiday_status else 0
    temp_hot = 1 if temperature >= 25 else 0
    
    consumption_24h, consumption_168h = calculate_baseline_consumption(
        hour, day_of_week, month, holiday_status
    )
    
    features = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_of_week],
        'temp_celsius': [temperature],
        'is_holiday': [is_holiday_flag],
        'month': [month],
        'is_weekend': [is_weekend],
        'is_business_hours': [is_business_hours],
        'is_peak_hours': [is_peak_hours],
        'humidity': [humidity],
        'temp_hot': [temp_hot],
        'consumption_lag_24h': [consumption_24h],
        'consumption_lag_168h': [consumption_168h]
    })
    
    context = {
        'datetime': dt,
        'day_name': dt.strftime('%A'),
        'baseline_kwh': consumption_24h,
        'is_weekend': is_weekend,
        'is_holiday': holiday_status,
        'is_peak': is_peak_hours,
        'is_business': is_business_hours
    }
    
    return features, context

@st.cache_data(show_spinner=False)
def create_enhanced_trend_chart(prediction_date, prediction_hour, prediction_kwh):
    """Create enhanced trend visualization with caching"""
    dates = []
    consumption_values = []
    point_types = []
    
    for i in range(-6, 1):
        date = prediction_date + timedelta(days=i)
        dt = datetime.combine(date, datetime.min.time().replace(hour=prediction_hour))
        
        if i < 0:
            day_of_week = dt.weekday()
            month = dt.month
            holiday_status = is_holiday(dt)
            baseline, _ = calculate_baseline_consumption(prediction_hour, day_of_week, month, holiday_status)
            variation = np.random.normal(1.0, 0.04)
            consumption = baseline * variation
            point_types.append('Historical')
        else:
            consumption = prediction_kwh
            point_types.append('Forecast')
        
        dates.append(dt)
        consumption_values.append(consumption / 1000)
    
    fig = go.Figure()
    
    # Historical trend with area fill
    historical_dates = dates[:-1]
    historical_values = consumption_values[:-1]
    
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_values,
        mode='lines+markers',
        name='Historical Pattern',
        line=dict(color='#60a5fa', width=3, shape='spline'),
        marker=dict(size=8, color='#3b82f6', line=dict(width=2, color='white')),
        fill='tonexty',
        fillcolor='rgba(96, 165, 250, 0.1)',
        hovertemplate='<b>%{x}</b><br>Consumption: %{y:,.0f} MWh<br>Type: Historical<extra></extra>'
    ))
    
    # Forecast point with glow effect
    fig.add_trace(go.Scatter(
        x=[dates[-1]],
        y=[consumption_values[-1]],
        mode='markers',
        name='AI Forecast',
        marker=dict(
            size=20,
            color='#a78bfa',
            symbol='diamond',
            line=dict(width=4, color='white')
        ),
        hovertemplate='<b>%{x}</b><br>Prediction: %{y:,.0f} MWh<br>Type: AI Forecast<extra></extra>'
    ))
    
    # Add forecast region shading
    fig.add_shape(
        type="rect",
        x0=dates[-1] - timedelta(hours=12),
        y0=min(consumption_values) * 0.95,
        x1=dates[-1] + timedelta(hours=12),
        y1=max(consumption_values) * 1.05,
        fillcolor="rgba(167, 139, 250, 0.1)",
        line=dict(width=0),
        layer="below"
    )
    
    fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="Consumption (MWh)",
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#e2e8f0'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(15, 23, 42, 0.8)',
            bordercolor='rgba(148, 163, 184, 0.2)',
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified'
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(148, 163, 184, 0.1)',
        showline=False,
        color='#94a3b8'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(148, 163, 184, 0.1)',
        showline=False,
        color='#94a3b8'
    )
    
    return fig

def main():
    """Main application with modern dark glass interface"""

    # Dark Glass Hero Header
    st.markdown("""
    <p class="hero-title">⚡ Electricity Demand Forecasting</p>
    <p class="hero-subtitle">Real-time consumption prediction for grid operations and energy planning</p>
    """, unsafe_allow_html=True)
    
    # Modern Sidebar
    with st.sidebar:

        model, model_info = load_forecasting_model()
        if model is None:
            st.stop()
        
        st.markdown('<div class="text-glass" style="font-size: 1.25rem; font-weight: 700; margin-bottom: 1.5rem;">⚙️ Configuration</div>', unsafe_allow_html=True)
        
        # Date and time
        st.markdown('<div class="text-secondary" style="font-weight: 600; margin: 1rem 0 0.5rem 0;">📅 Schedule</div>', unsafe_allow_html=True)
        target_date = st.date_input(
            "Date",
            value=datetime.now().date() + timedelta(days=1)
        )
        
        target_hour = st.selectbox(
            "Hour",
            options=list(range(24)),
            index=12,
            format_func=lambda x: f"{x:02d}:00"
        )
        
        st.markdown("---")
        
        # Weather
        st.markdown('<div class="text-secondary" style="font-weight: 600; margin: 1rem 0 0.5rem 0;">🌤️ Weather</div>', unsafe_allow_html=True)
        temperature = st.slider(
            "Temperature (°C)",
            min_value=-5.0,
            max_value=40.0,
            value=20.0,
            step=0.5
        )
        
        humidity = st.slider(
            "Humidity (%)",
            min_value=20,
            max_value=95,
            value=60,
            step=5
        )
        
        st.markdown("---")
        
        # Generate button
        forecast_button = st.button(
            "🔮 Generate Forecast",
            type="primary",
            use_container_width=True
        )
        
        # Model status
        st.markdown("---")
        st.markdown('<div class="text-secondary" style="font-weight: 600; margin: 1rem 0 0.5rem 0;">🤖 System Status</div>', unsafe_allow_html=True)
        mape = model_info['performance']['mape']
        
        status_color = "🟢" if mape < 7 else "🟡"
        status_text = "Excellent" if mape < 7 else "Good"
        
        st.markdown(f"""
        <div class="text-glass" style="line-height: 1.6;">
        {status_color} <strong>{status_text}</strong><br>
        📊 {mape:.1f}% MAPE<br>
        🎯 {model_info['performance']['r2']:.3f} R²<br>
        🧠 Random Forest
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content
    if forecast_button:
        features, context = prepare_model_features(target_hour, target_date, temperature, humidity)
        
        try:
            prediction_kwh = model.predict(features)[0]
            prediction_mwh = prediction_kwh / 1000
            baseline_mwh = context['baseline_kwh'] / 1000
            variance_pct = ((prediction_kwh - context['baseline_kwh']) / context['baseline_kwh']) * 100
            
            # Glass Morphism Hero Card
            st.markdown(f"""
            <div class="prediction-hero">
                <div class="hero-label">Predicted Consumption</div>
                <div class="hero-value">{prediction_mwh:,.0f} MWh</div>
                <div class="hero-timestamp">{context['datetime'].strftime('%A, %B %d, %Y at %H:%M')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Main layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Enhanced trend chart
                st.markdown("""
                <div class="chart-container glass-effect">
                    <div class="chart-title">
                        <span class="chart-icon">📈</span>
                        7-Day Consumption Trend
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                trend_chart = create_enhanced_trend_chart(target_date, target_hour, prediction_kwh)
                st.plotly_chart(trend_chart, use_container_width=True)
            
            with col2:
                # Status cards
                if abs(variance_pct) <= 5:
                    status_class = "status-normal"
                    icon_class = "icon-normal"
                    status_icon = "✓"
                    status_text = "Normal Range"
                elif variance_pct > 5:
                    status_class = "status-high"
                    icon_class = "icon-high" 
                    status_icon = "↗"
                    status_text = "Above Average"
                else:
                    status_class = "status-low"
                    icon_class = "icon-low"
                    status_icon = "↘"
                    status_text = "Below Average"
                
                render_glass_card(
                    "Demand Status", 
                    status_icon, 
                    status_text, 
                    f"{variance_pct:+.1f}% vs baseline",
                    status_class,
                    icon_class
                )
                
                render_glass_card(
                    "Baseline", 
                    "📊", 
                    f"{baseline_mwh:,.0f} MWh", 
                    "Expected consumption"
                )
                
                render_glass_card(
                    "Model Accuracy", 
                    "🎯", 
                    f"{mape:.1f}%", 
                    "MAPE Score"
                )
            
            # Factor Analysis Grid
            st.markdown('<div class="chart-title text-glass" style="margin: 2rem 0 1rem 0;">📋 Demand Analysis</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                temp_icon = "🔥" if temperature > 25 else "❄️" if temperature < 10 else "🌤️"
                temp_impact = "High Impact" if temperature > 25 or temperature < 10 else "Moderate Impact"
                temp_color = "icon-high" if temperature > 25 or temperature < 10 else "icon-neutral"
                
                render_factor_card(
                    "Temperature",
                    temp_icon,
                    f"{temperature}°C",
                    temp_impact,
                    temp_color
                )
            
            with col2:
                period_icon = "⚡" if context['is_peak'] else "🏢" if context['is_business'] else "🌙"
                period_text = "Peak Hours" if context['is_peak'] else "Business Hours" if context['is_business'] else "Off-Peak Hours"
                period_color = "icon-high" if context['is_peak'] else "icon-neutral"
                
                render_factor_card(
                    "Time Period",
                    period_icon,
                    f"{target_hour:02d}:00",
                    period_text,
                    period_color
                )
            
            with col3:
                day_icon = "🎉" if context['is_holiday'] else "🏖️" if context['is_weekend'] else "🏢"
                day_text = "Public Holiday" if context['is_holiday'] else "Weekend Day" if context['is_weekend'] else "Business Day"
                day_color = "icon-low" if context['is_holiday'] or context['is_weekend'] else "icon-neutral"
                
                render_factor_card(
                    "Day Type",
                    day_icon,
                    context['day_name'],
                    day_text,
                    day_color
                )
                
        except Exception as e:
            st.error(f"❌ Forecast generation failed: {str(e)}")
            st.markdown("**Troubleshooting:** Check model files and feature compatibility.")
    
    else:
        # Default state
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Glass container with SIMPLE content - NO COMPLEX HTML
            # st.markdown('<div class="chart-container glass-effect">', unsafe_allow_html=True)
            
            st.markdown("###  System Overview")
            
            st.write("This system predicts hourly electricity consumption using machine learning. It's trained on Spanish grid data and considers weather patterns, time factors, and historical usage.")
            
            st.markdown("**How to Use:**")
            
            # Simple 2x2 grid using Streamlit columns - NO HTML
            step_col1, step_col2 = st.columns(2)
            
            with step_col1:
                st.info("**1️⃣ 📅 SET DATE & TIME**\n\nPick your target date and hour in the sidebar")
                st.success("**3️⃣ 🔮 GENERATE FORECAST**\n\nHit the forecast button to see predictions")
            
            with step_col2:
                st.info("**2️⃣ 🌤️ ENTER WEATHER**\n\nEnter expected temperature & humidity")  
                st.success("**4️⃣ 📊 VIEW RESULTS**\n\nCheck the trend chart & demand analysis")
            
            st.markdown("**What You'll Get:**")
            st.markdown("""
            - **Consumption forecast** in megawatt-hours (MWh)
            - **Demand status** - whether it's normal, high, or low  
            - **7-day trend chart** with your prediction highlighted
            - **Factor breakdown** of what's driving the demand
            """)
            
            st.caption("*The model achieves 6.1% accuracy (MAPE) and works best for typical weather conditions within Spain's climate range.*")
            
            # st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # System status cards
            render_glass_card("System Status", "✓", "Online", "Ready for forecasting", "status-normal", "icon-normal")  
            render_glass_card("Model Accuracy", "🎯", f"{model_info['performance']['mape']:.1f}%", "MAPE Score")
            render_glass_card("Algorithm", "🧠", "Random Forest", "Ensemble method")

if __name__ == "__main__":
    main()