"""
Disaster Finance Model - Streamlit Application

Interactive tool for exploring the market-based disaster financing framework
proposed in "Transforming Disaster Financing: An Alternative to FEMA Funding"

Author: Josh Curry et al.
Published: Domestic Preparedness, December 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, 'src')

from disaster_generator import (
    DisasterEventGenerator, 
    RegionalRiskProfile, 
    DisasterType,
    PRESET_PROFILES
)
from funding_waterfall import (
    FundingWaterfall, 
    TraditionalFEMAModel,
    FundingLayer,
    LayerConfiguration
)
from simulation_runner import SimulationRunner, generate_summary_report
from noaa_data import NOAADataCalibrator, NOAA_SUMMARY_STATS, STATE_COST_DATA, REGIONAL_GROUPINGS
from new_visualizations import (
    create_choropleth,
    create_sankey_comparison,
    create_disbursement_timeline,
    create_exceedance_curve,
    create_frequency_trend,
    EXCEEDANCE_PROFILES,
    format_cost as viz_format_cost,
)


def format_currency(value_millions: float, decimals: int = 1) -> str:
    """
    Format currency values with appropriate T/B/M/K notation.
    
    Args:
        value_millions: Value in millions USD
        decimals: Number of decimal places (default 1)
    
    Returns:
        Formatted string like "$1.2T", "$456B", or "$50M"
    """
    if value_millions >= 1_000_000:
        # Trillions
        trillions = value_millions / 1_000_000
        if trillions >= 100:
            return f"${trillions:,.0f}T"
        elif trillions >= 10:
            return f"${trillions:,.1f}T"
        else:
            return f"${trillions:,.{decimals}f}T"
    elif value_millions >= 1000:
        # Billions
        billions = value_millions / 1000
        if billions >= 100:
            return f"${billions:,.0f}B"
        elif billions >= 10:
            return f"${billions:,.1f}B"
        else:
            return f"${billions:,.{decimals}f}B"
    elif value_millions >= 1:
        # Millions
        if value_millions >= 100:
            return f"${value_millions:,.0f}M"
        elif value_millions >= 10:
            return f"${value_millions:,.1f}M"
        else:
            return f"${value_millions:,.{decimals}f}M"
    else:
        # Convert to thousands
        thousands = value_millions * 1000
        return f"${thousands:,.0f}K"


def format_number(value: float, decimals: int = 0) -> str:
    """Format large numbers with K/M/B suffixes."""
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:,.{decimals}f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:,.{decimals}f}M"
    elif value >= 1_000:
        return f"{value/1_000:,.{decimals}f}K"
    else:
        return f"{value:,.0f}"


# Page configuration
st.set_page_config(
    page_title="Disaster Finance Model",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5a6c7d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
    }
    .improvement-positive {
        color: #10b981;
        font-weight: 600;
    }
    .improvement-negative {
        color: #ef4444;
        font-weight: 600;
    }
    .layer-1 { background-color: #3b82f6; }
    .layer-2 { background-color: #8b5cf6; }
    .layer-3 { background-color: #06b6d4; }
    .layer-4 { background-color: #f59e0b; }
    .layer-5 { background-color: #ef4444; }
</style>
""", unsafe_allow_html=True)


def main():
    # Header with logo and competition branding
    logo_col, title_col, badge_col = st.columns([1, 4, 1.5])
    
    with logo_col:
        try:
            st.image("logo-dark.png", width=140)
        except Exception:
            st.markdown("**Pierce College**")
    
    with title_col:
        st.header("Disaster Finance Model")
        st.markdown(
            '<p class="sub-header">Monte Carlo simulation comparing market-based disaster financing '
            'vs. traditional FEMA funding</p>', 
            unsafe_allow_html=True
        )
    
    with badge_col:
        st.markdown("""
        <div style="text-align: right; padding-top: 0.5rem;">
            <div style="font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">
                IAEM 2026 · Undergraduate Division
            </div>
            <div style="font-size: 0.85rem; color: #aaa; margin-top: 2px;">
                Joshua Curry
            </div>
            <div style="font-size: 0.7rem; color: #666; margin-top: 2px;">
                Resilience Analytics Lab, LLC
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Reference to paper
    with st.expander("About This Model", expanded=False):
        st.markdown("""
        This tool implements the disaster financing framework proposed in:
        
        **"Transforming Disaster Financing: An Alternative to FEMA Funding"**  
        *Domestic Preparedness*, December 2025  
        *Authors: Josh Curry, Chandler Clough, Johnny Hicks, Andrew Jackson, Ryan Rockabrand*
        
        The model simulates a five-layer funding structure:
        
        | Layer | Source | Coverage Range | Key Benefit |
        |-------|--------|----------------|-------------|
        | 1 | Municipal Reserves | First $50M | Local tax base stabilization |
        | 2 | State Risk Pools | \\$50M–\\$250M | Regional diversification |
        | 3 | Catastrophe Bonds | \\$250M–\\$1B | Locked-in capital commitments |
        | 4 | Reinsurance Markets | \\$1B–\\$5B | Global risk distribution |
        | 5 | Federal Backstop | >\\$5B | Crisis-level market assurance |
        
        This contrasts with the traditional model where middle layers are largely absent, 
        leaving a "vast middle ground unfilled" between municipal reserves and federal appropriations.
        """)
    
    # Sidebar configuration
    st.sidebar.header("Simulation Parameters")
    
    # Profile selection
    profile_name = st.sidebar.selectbox(
        "Regional Risk Profile",
        options=list(PRESET_PROFILES.keys()),
        format_func=lambda x: PRESET_PROFILES[x].name,
        index=0
    )
    profile = PRESET_PROFILES[profile_name]
    
    # Display profile details
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Primary Hazard:** {profile.primary_hazard.value.title()}")
    st.sidebar.markdown(f"**Baseline Frequency:** {profile.baseline_frequency} events/year")
    st.sidebar.markdown(f"**Trend Multiplier:** {profile.trend_multiplier}x")
    
    st.sidebar.markdown("---")
    
    # Simulation controls
    n_years = st.sidebar.slider(
        "Years to Simulate",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Number of years per simulation run"
    )
    
    n_simulations = st.sidebar.slider(
        "Monte Carlo Iterations",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Number of simulation runs for statistical robustness"
    )
    
    # seed = st.sidebar.number_input(
    #     "Random Seed (for reproducibility)",
    #     min_value=0,
    #     max_value=99999,
    #     value=42,
    #     help="Set to same value to reproduce results"
    # )
    
    # Layer customization
    st.sidebar.markdown("---")
    st.sidebar.header("Layer Configuration")
    
    customize_layers = st.sidebar.checkbox("Customize Layer Thresholds", value=False)
    
    if customize_layers:
        layer_1_cap = st.sidebar.number_input("Layer 1 (Municipal) Cap ($M)", value=50, min_value=10, max_value=200)
        layer_2_cap = st.sidebar.number_input("Layer 2 (State Pool) Cap ($M)", value=250, min_value=100, max_value=500)
        layer_3_cap = st.sidebar.number_input("Layer 3 (Cat Bonds) Cap ($M)", value=1000, min_value=500, max_value=2000)
        layer_4_cap = st.sidebar.number_input("Layer 4 (Reinsurance) Cap ($M)", value=5000, min_value=2000, max_value=10000)
    else:
        layer_1_cap, layer_2_cap, layer_3_cap, layer_4_cap = 50, 250, 1000, 5000
    
    # Run simulation button
    run_button = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)
    
    # Main content area
    if run_button or 'results' in st.session_state:
        if run_button:
            with st.spinner("Running Monte Carlo simulation..."):
                runner = SimulationRunner(seed=seed)
                results = runner.run_monte_carlo(
                    profile, 
                    n_years=n_years, 
                    n_simulations=n_simulations
                )
                st.session_state['results'] = results
        
        results = st.session_state['results']
        
        # Key metrics row
        st.header("Key Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Events Simulated",
                format_number(results.total_events),
                help="Total disaster events across all simulation runs"
            )
        
        with col2:
            st.metric(
                "Avg Events/Year",
                f"{results.avg_events_per_year:.1f}",
                help="Average annual event frequency"
            )
        
        with col3:
            st.metric(
                "Total Losses Simulated",
                format_currency(results.total_losses),
                help="Cumulative economic losses across all simulations"
            )
        
        with col4:
            time_delta = results.time_improvement_days
            st.metric(
                "Time Improvement",
                f"{time_delta:.1f} days",
                delta=f"{time_delta:.1f} days faster" if time_delta > 0 else f"{abs(time_delta):.1f} days slower",
                delta_color="normal" if time_delta > 0 else "inverse"
            )
        
        st.markdown("---")
        
        # Model comparison section
        st.header("Model Comparison")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.subheader("Market-Based Model")
            st.metric("Coverage Ratio", f"{results.market_avg_coverage_ratio*100:.1f}%")
            st.metric("Avg Disbursement", f"{results.market_avg_disbursement_days:.1f} days")
            st.metric("Total Gaps", format_currency(results.market_total_gaps))
        
        with comp_col2:
            st.subheader("Traditional FEMA Model")
            st.metric("Coverage Ratio", f"{results.fema_avg_coverage_ratio*100:.1f}%")
            st.metric("Avg Disbursement", f"{results.fema_avg_disbursement_days:.1f} days")
            st.metric("Total Gaps", format_currency(results.fema_total_gaps))
        
        # Improvement summary
        improvement_pct = results.coverage_improvement_pct
        gap_reduction = results.fema_total_gaps - results.market_total_gaps
        
        if improvement_pct > 0:
            st.success(f"Market-based model shows **{improvement_pct:.1f}%** improvement in coverage ratio")
        else:
            st.warning(f"Market-based model shows **{abs(improvement_pct):.1f}%** lower coverage ratio")
        
        if gap_reduction > 0:
            st.success(f"Gap reduction: **{format_currency(gap_reduction)}** less in unfunded gaps")
        
        st.markdown("---")
        
        # Visualizations
        st.header("Detailed Analysis")
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Layer Utilization", 
            "Loss Distribution", 
            "Funding Flow",
            "Disbursement Timeline",
            "Geographic Risk Map",
            "Time Series",
            "NOAA Historical Data"
        ])
        
        with tab1:
            st.subheader("Funding Layer Utilization")
            
            # Layer utilization bar chart
            layer_data = []
            for layer_name, amount in results.layer_utilization.items():
                freq = results.layer_frequency.get(layer_name, 0)
                layer_data.append({
                    "Layer": layer_name,
                    "Total Disbursed ($M)": amount,
                    "Activations": freq
                })
            
            if layer_data:
                df_layers = pd.DataFrame(layer_data)
                
                fig_layers = px.bar(
                    df_layers,
                    x="Layer",
                    y="Total Disbursed ($M)",
                    color="Layer",
                    text="Activations",
                    title="Total Disbursements by Layer",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_layers.update_traces(texttemplate='%{text:,} activations', textposition='outside')
                fig_layers.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_layers, use_container_width=True)
                
                # Create formatted table for display
                display_data = []
                for layer_name, amount in results.layer_utilization.items():
                    freq = results.layer_frequency.get(layer_name, 0)
                    display_data.append({
                        "Layer": layer_name,
                        "Total Disbursed": format_currency(amount),
                        "Activations": f"{freq:,}"
                    })
                
                df_display = pd.DataFrame(display_data)
                st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        with tab2:
            st.subheader("Loss Distribution")
            
            # Loss percentiles
            loss_pct = results.loss_percentiles
            
            fig_dist = go.Figure()
            
            percentiles = ["p50", "p75", "p90", "p95", "p99", "max"]
            values = [loss_pct[p] for p in percentiles]
            labels = ["Median", "75th", "90th", "95th", "99th", "Maximum"]
            
            fig_dist.add_trace(go.Bar(
                x=labels,
                y=values,
                marker_color=['#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ef4444'],
                text=[format_currency(v) for v in values],
                textposition='outside'
            ))
            
            fig_dist.update_layout(
                title="Loss Distribution Percentiles (Single Event)",
                xaxis_title="Percentile",
                yaxis_title="Economic Loss ($M)",
                height=400
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Gap distribution
            st.subheader("Coverage Gap Distribution")
            gap_pct = results.gap_percentiles
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Median Gap", format_currency(gap_pct['p50']))
            col2.metric("90th Percentile Gap", format_currency(gap_pct['p90']))
            col3.metric("99th Percentile Gap", format_currency(gap_pct['p99']))
            
            # Loss exceedance curve
            st.markdown("---")
            st.subheader("Loss Exceedance Curve")
            st.caption(
                "Probability of a single event exceeding a given loss threshold, "
                "derived from log-normal severity distributions calibrated to NOAA NCEI data."
            )
            
            exc_profiles = st.multiselect(
                "Profiles to compare",
                options=list(EXCEEDANCE_PROFILES.keys()),
                default=[results.profile_name],
                key="exc_profiles",
            )
            
            if exc_profiles:
                fig_exc = create_exceedance_curve(
                    profiles_to_show=exc_profiles, seed=seed
                )
                st.plotly_chart(fig_exc, use_container_width=True)
        
        with tab3:
            st.subheader("Funding Flow Comparison")
            st.caption(
                "How disaster losses flow through each model's funding layers. "
                "Adjust the event size to see how the five-layer market model distributes risk "
                "versus the two-layer traditional model."
            )
            
            sankey_loss = st.slider(
                "Event Loss ($M)",
                min_value=100,
                max_value=15000,
                value=5000,
                step=100,
                format="$%dM",
                key="sankey_loss",
                help="Drag to see how different event sizes flow through each model"
            )
            
            fig_sankey = create_sankey_comparison(loss_millions=sankey_loss)
            st.plotly_chart(fig_sankey, use_container_width=True)
            
            # Also show the static five-layer structure diagram
            with st.expander("Five-Layer Protection Structure (static view)"):
                fig_waterfall = go.Figure()
                layers_static = [
                    ("Municipal Reserves", 0, 50, "#3b82f6"),
                    ("State Risk Pool", 50, 250, "#8b5cf6"),
                    ("Catastrophe Bonds", 250, 1000, "#06b6d4"),
                    ("Reinsurance Markets", 1000, 5000, "#f59e0b"),
                    ("Federal Backstop", 5000, 10000, "#ef4444"),
                ]
                for name, floor, ceiling, color in layers_static:
                    fig_waterfall.add_trace(go.Bar(
                        name=name,
                        x=[name],
                        y=[ceiling - floor],
                        base=[floor],
                        marker_color=color,
                        text=[f"${floor}M - ${ceiling}M"],
                        textposition="inside",
                        hovertemplate=f"<b>{name}</b><br>Range: ${floor}M - ${ceiling}M<extra></extra>"
                    ))
                fig_waterfall.update_layout(
                    title="Five-Layer Protection Structure",
                    xaxis_title="Layer",
                    yaxis_title="Coverage Range ($M)",
                    yaxis_type="log",
                    showlegend=False,
                    height=450
                )
                st.plotly_chart(fig_waterfall, use_container_width=True)
            
            st.info(
                "**Key insight:** The traditional model concentrates nearly all funding "
                "on a single federal instrument. The market-based model distributes risk "
                "across five contractually committed layers, filling the 'vast middle ground' "
                "between local reserves and federal appropriations."
            )
        
        with tab4:
            st.subheader("Disbursement Timeline Comparison")
            st.caption(
                "How quickly does money reach disaster-affected communities? "
                "The market model's parametric triggers and pre-committed capital "
                "close the 18-day gap between local reserves and federal appropriations."
            )
            
            timeline_loss = st.slider(
                "Event Loss ($M)",
                min_value=100,
                max_value=10000,
                value=2000,
                step=100,
                format="$%dM",
                key="timeline_loss",
                help="Drag to see timing for different event sizes"
            )
            
            fig_fema_tl, fig_market_tl, market_wavg, fema_wavg = create_disbursement_timeline(
                loss_millions=timeline_loss
            )
            
            st.plotly_chart(fig_fema_tl, use_container_width=True)
            st.plotly_chart(fig_market_tl, use_container_width=True)
            
            # Key timing metrics
            st.markdown("---")
            tcol1, tcol2, tcol3 = st.columns(3)
            time_saved = max(0, fema_wavg - market_wavg)
            tcol1.metric("Cat Bond Trigger", "72 hours", help="Parametric triggers disburse within 72 hours of threshold breach")
            tcol2.metric("FEMA PA Average", "21 days", help="Average FEMA Public Assistance disbursement timeline")
            tcol3.metric("Time Saved", f"~{time_saved:.0f} days", delta=f"{time_saved:.1f} days faster", help="Weighted-average improvement for this event size")
        
        with tab5:
            st.subheader("Geographic Risk Map")
            st.caption(
                "Cumulative billion-dollar disaster costs by state (1980–2024, CPI-adjusted). "
                "Dashed outlines show the regional profiles available for simulation."
            )
            
            # Hazard type filter
            hazard_filter = st.selectbox(
                "Color by hazard type",
                options=["total", "tropical_cyclone", "severe_storm", "drought", 
                         "flooding", "wildfire", "winter_storm", "freeze"],
                format_func=lambda x: x.replace("_", " ").title() if x != "total" else "All Hazards Combined",
                key="choropleth_hazard"
            )
            
            fig_choro = create_choropleth(
                highlight_profile=profile_name,
                color_by=hazard_filter,
            )
            st.plotly_chart(fig_choro, use_container_width=True)
            
            # Event frequency acceleration
            st.markdown("---")
            st.subheader("Event Frequency Acceleration")
            fig_freq = create_frequency_trend()
            st.plotly_chart(fig_freq, use_container_width=True)
            
            st.warning(
                "**2.7× increase** in average annual billion-dollar disaster frequency "
                "from the 1980s (3.3/year) to 2020–2024 (23.0/year). This acceleration "
                "is the fundamental driver for market-based financing alternatives."
            )
        
        with tab6:
            st.subheader("Simulation Time Series")
            
            # Create time series from year summaries
            if results.year_summaries:
                # Sample every nth summary to avoid overwhelming the chart
                sample_rate = max(1, len(results.year_summaries) // 500)
                sampled = results.year_summaries[::sample_rate]
                
                df_time = pd.DataFrame([{
                    "Year": s.year,
                    "Events": s.n_events,
                    "Total Losses ($M)": s.total_losses,
                    "Market Coverage (%)": s.market_coverage_ratio * 100,
                    "FEMA Coverage (%)": s.fema_coverage_ratio * 100,
                    "Market Disbursement (days)": s.market_avg_disbursement_days,
                    "FEMA Disbursement (days)": s.fema_avg_disbursement_days,
                } for s in sampled])
                
                # Coverage comparison
                fig_coverage = go.Figure()
                fig_coverage.add_trace(go.Scatter(
                    x=df_time["Year"],
                    y=df_time["Market Coverage (%)"],
                    name="Market-Based",
                    mode="lines",
                    line=dict(color="#3b82f6", width=2)
                ))
                fig_coverage.add_trace(go.Scatter(
                    x=df_time["Year"],
                    y=df_time["FEMA Coverage (%)"],
                    name="Traditional FEMA",
                    mode="lines",
                    line=dict(color="#ef4444", width=2)
                ))
                fig_coverage.update_layout(
                    title="Coverage Ratio Over Time",
                    xaxis_title="Year",
                    yaxis_title="Coverage (%)",
                    height=350
                )
                st.plotly_chart(fig_coverage, use_container_width=True)
                
                # Disbursement time comparison
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(
                    x=df_time["Year"],
                    y=df_time["Market Disbursement (days)"],
                    name="Market-Based",
                    mode="lines",
                    line=dict(color="#3b82f6", width=2)
                ))
                fig_time.add_trace(go.Scatter(
                    x=df_time["Year"],
                    y=df_time["FEMA Disbursement (days)"],
                    name="Traditional FEMA",
                    mode="lines",
                    line=dict(color="#ef4444", width=2)
                ))
                fig_time.update_layout(
                    title="Average Disbursement Time Over Time",
                    xaxis_title="Year",
                    yaxis_title="Days to Disburse",
                    height=350
                )
                st.plotly_chart(fig_time, use_container_width=True)
        
        with tab7:
            st.subheader("NOAA NCEI Historical Data (1980-2024)")
            
            st.markdown("""
            Model calibration is based on NOAA's National Centers for Environmental Information 
            (NCEI) Billion-Dollar Weather and Climate Disasters database. This authoritative dataset 
            tracks all U.S. disasters with costs exceeding $1 billion (CPI-adjusted).
            
            *Note: NOAA ceased support for this product in May 2025, but historical data remains archived.*
            """)
            
            # Key stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Events", f"{NOAA_SUMMARY_STATS['total_events']}")
            col2.metric("Cumulative Cost", f"${NOAA_SUMMARY_STATS['total_cost_billions']/1000:.2f}T")
            col3.metric("Avg Events/Year (Historical)", f"{NOAA_SUMMARY_STATS['average_events_per_year_1980_2023']:.1f}")
            col4.metric("Avg Events/Year (2020-24)", f"{NOAA_SUMMARY_STATS['average_events_per_year_2020_2024']:.1f}")
            
            st.markdown("---")
            
            # Events by type chart
            st.subheader("Events by Disaster Type")
            events_df = pd.DataFrame([
                {"Type": k.replace("_", " ").title(), "Events": v, "Cost ($B)": NOAA_SUMMARY_STATS['cost_by_type_billions'].get(k, 0)}
                for k, v in NOAA_SUMMARY_STATS['events_by_type'].items()
            ]).sort_values("Events", ascending=False)
            
            fig_events = px.bar(
                events_df,
                x="Type",
                y="Events",
                color="Cost ($B)",
                title="Billion-Dollar Disasters by Type (1980-2024)",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_events, use_container_width=True)
            
            # Cost distribution pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                cost_df = pd.DataFrame([
                    {"Type": k.replace("_", " ").title(), "Cost": v}
                    for k, v in NOAA_SUMMARY_STATS['cost_by_type_billions'].items()
                ])
                fig_pie = px.pie(
                    cost_df,
                    values="Cost",
                    names="Type",
                    title="Cost Distribution by Type ($2.9T Total)",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Trend by decade
                decade_df = pd.DataFrame([
                    {"Decade": k, "Events": v}
                    for k, v in NOAA_SUMMARY_STATS['events_by_decade'].items()
                ])
                fig_trend = px.bar(
                    decade_df,
                    x="Decade",
                    y="Events",
                    title="Event Frequency Trend by Decade",
                    color="Events",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            # State-level data
            st.subheader("State-Level Cost Data")
            
            # Top 10 states by total cost
            state_totals = []
            for state, costs in STATE_COST_DATA.items():
                if state != "US":
                    total = sum(costs.values())
                    state_totals.append({"State": state, "Total Cost ($M)": total})
            
            state_df = pd.DataFrame(state_totals).sort_values("Total Cost ($M)", ascending=False).head(15)
            
            fig_states = px.bar(
                state_df,
                x="State",
                y="Total Cost ($M)",
                title="Top 15 States by Billion-Dollar Disaster Costs (1980-2024)",
                color="Total Cost ($M)",
                color_continuous_scale="Oranges"
            )
            st.plotly_chart(fig_states, use_container_width=True)
            
            # Calibration info
            st.subheader("Model Calibration")
            calibrator = NOAADataCalibrator()
            st.code(calibrator.get_calibration_summary(), language=None)
        
        # Full report
        st.markdown("---")
        with st.expander("Full Simulation Report"):
            st.code(generate_summary_report(results), language=None)
        
        # Download results
        st.download_button(
            "Download Results (CSV)",
            data=pd.DataFrame([{
                "Profile": results.profile_name,
                "Years": results.n_years,
                "Simulations": results.n_simulations,
                "Total Events": results.total_events,
                "Total Losses ($M)": results.total_losses,
                "Market Coverage (%)": results.market_avg_coverage_ratio * 100,
                "FEMA Coverage (%)": results.fema_avg_coverage_ratio * 100,
                "Market Disbursement (days)": results.market_avg_disbursement_days,
                "FEMA Disbursement (days)": results.fema_avg_disbursement_days,
                "Market Gaps ($M)": results.market_total_gaps,
                "FEMA Gaps ($M)": results.fema_total_gaps,
            }]).to_csv(index=False),
            file_name="disaster_finance_results.csv",
            mime="text/csv"
        )
    
    else:
        # Initial state - show instructions
        st.info("Configure simulation parameters in the sidebar and click **Run Simulation** to begin.")
        
        # Show the layer structure
        st.header("The Five-Layer Model")
        
        layer_table = pd.DataFrame([
            {"Layer": 1, "Source": "Municipal Reserves", "Floor": "$0", "Ceiling": "$50M", 
             "Disbursement": "~3 days", "Buffer": "Local tax base stabilization"},
            {"Layer": 2, "Source": "State Risk Pool", "Floor": "$50M", "Ceiling": "$250M",
             "Disbursement": "~7 days", "Buffer": "Regional diversification"},
            {"Layer": 3, "Source": "Catastrophe Bonds", "Floor": "$250M", "Ceiling": "$1B",
             "Disbursement": "~3 days*", "Buffer": "Locked-in capital commitments"},
            {"Layer": 4, "Source": "Reinsurance Markets", "Floor": "$1B", "Ceiling": "$5B",
             "Disbursement": "~14 days", "Buffer": "Global risk distribution"},
            {"Layer": 5, "Source": "Federal Backstop", "Floor": "$5B", "Ceiling": "Unlimited",
             "Disbursement": "~21 days", "Buffer": "Crisis-level market assurance"},
        ])
        
        st.dataframe(layer_table, use_container_width=True, hide_index=True)
        
        st.caption("*Parametric triggers can achieve 72-hour disbursement vs. 21-day FEMA average")
        
        # Traditional model comparison
        st.header("Traditional vs. Market-Based")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Traditional FEMA Model")
            st.markdown("""
            - **Layer 1**: Municipal reserves (first $50M)
            - **Layers 2-4**: *Gap* - largely unfilled
            - **Layer 5**: Federal appropriations (everything else)
            
            **Challenges:**
            - Subject to annual appropriations
            - Post-disaster political negotiations
            - Average 21-day disbursement timeline
            - Vast middle ground unfilled
            """)
        
        with col2:
            st.subheader("Market-Based Alternative")
            st.markdown("""
            - **All five layers** actively engaged
            - State risk pools for regional coverage
            - Cat bonds with parametric triggers
            - Reinsurance for large-scale events
            - Federal backstop for catastrophic losses
            
            **Advantages:**
            - Contractually committed funding
            - 72-hour parametric payouts possible
            - Reduced political uncertainty
            - Risk-appropriate pricing incentives
            """)


if __name__ == "__main__":
    main()
