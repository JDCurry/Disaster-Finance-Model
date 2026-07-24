"""
New Visualization Module

Adds four publication-quality visualizations to the disaster finance dashboard:
1. US State Choropleth (geographic risk context)
2. Funding Flow Sankey (five-layer architecture argument)
3. Disbursement Timeline Comparison (speed-to-money argument)
4. Loss Exceedance Curve (Monte Carlo validation)

Author: Josh Curry
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple

from noaa_data import STATE_COST_DATA, REGIONAL_GROUPINGS


# ──────────────────────────────────────────────────────────────────────
# State abbreviation ↔ name mappings
# ──────────────────────────────────────────────────────────────────────
STATE_ABBREV_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia",
    "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "IA": "Iowa", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "PR": "Puerto Rico",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}

# Map region profile keys to the states they contain
PROFILE_TO_STATES = {
    "gulf_coast": ["TX", "LA", "MS", "AL", "FL"],
    "california": ["CA"],
    "midwest": ["IL", "IN", "OH", "MI", "WI", "MN", "IA", "MO"],
    "pacific_northwest": ["WA", "OR", "ID"],
    "northeast": ["NY", "NJ", "PA", "CT", "MA", "RI", "NH", "VT", "ME"],
    "plains": ["KS", "NE", "SD", "ND", "OK"],
    "texas": ["TX"],
}

# Colors for each profile region
PROFILE_COLORS = {
    "gulf_coast": "#E24B4A",
    "california": "#EF9F27",
    "midwest": "#378ADD",
    "pacific_northwest": "#1D9E75",
    "northeast": "#7F77DD",
    "plains": "#D85A30",
    "texas": "#BA7517",
}


def format_cost(value_millions: float) -> str:
    """Format currency with T/B/M notation."""
    if value_millions >= 1_000_000:
        return f"${value_millions / 1_000_000:,.1f}T"
    elif value_millions >= 1000:
        return f"${value_millions / 1000:,.0f}B"
    else:
        return f"${value_millions:,.0f}M"


# ──────────────────────────────────────────────────────────────────────
# 1. US STATE CHOROPLETH
# ──────────────────────────────────────────────────────────────────────

def create_choropleth(
    highlight_profile: Optional[str] = None,
    color_by: str = "total",
) -> go.Figure:
    """
    Create US state choropleth colored by cumulative disaster cost.

    Args:
        highlight_profile: Profile key to outline (e.g., 'gulf_coast')
        color_by: 'total' for all hazards, or a specific hazard type key

    Returns:
        Plotly Figure
    """
    states = []
    values = []
    hover_texts = []

    for abbrev, costs in STATE_COST_DATA.items():
        if abbrev == "US":
            continue
        name = STATE_ABBREV_TO_NAME.get(abbrev, abbrev)

        if color_by == "total":
            val = sum(costs.values())
        else:
            val = costs.get(color_by, 0)

        states.append(abbrev)
        values.append(val)

        # Build hover text with breakdown
        top_hazards = sorted(costs.items(), key=lambda x: -x[1])[:3]
        breakdown = "<br>".join(
            f"  {h.replace('_', ' ').title()}: {format_cost(c)}"
            for h, c in top_hazards if c > 0
        )
        # Determine which profile region this state belongs to
        region = "—"
        for prof_key, prof_states in PROFILE_TO_STATES.items():
            if abbrev in prof_states:
                region = prof_key.replace("_", " ").title()
                break

        hover_texts.append(
            f"<b>{name}</b><br>"
            f"Total: {format_cost(val)}<br>"
            f"Profile: {region}<br>"
            f"<br>Top hazards:<br>{breakdown}"
        )

    fig = go.Figure(go.Choropleth(
        locations=states,
        z=values,
        locationmode="USA-states",
        colorscale="YlOrRd",
        colorbar=dict(
            title=dict(text="Cost ($M)", font=dict(size=12)),
            tickprefix="$",
            ticksuffix="M",
            len=0.6,
            thickness=15,
        ),
        hovertext=hover_texts,
        hoverinfo="text",
        marker_line_color="white",
        marker_line_width=0.5,
    ))

    fig.update_layout(
        title=dict(
            text="Cumulative Billion-Dollar Disaster Costs by State (1980–2024)",
            font=dict(size=16),
        ),
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showlakes=True,
            lakecolor="rgb(200, 220, 240)",
            bgcolor="rgba(0,0,0,0)",
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


# ──────────────────────────────────────────────────────────────────────
# 2. FUNDING FLOW SANKEY
# ──────────────────────────────────────────────────────────────────────

# Layer definitions matching the five-layer model
MARKET_LAYERS = [
    {"name": "Municipal Reserves", "floor": 0, "ceil": 50, "color": "rgba(55, 138, 221, 0.7)"},
    {"name": "State Risk Pool", "floor": 50, "ceil": 250, "color": "rgba(127, 119, 221, 0.7)"},
    {"name": "Cat Bonds", "floor": 250, "ceil": 1000, "color": "rgba(29, 158, 117, 0.7)"},
    {"name": "Reinsurance", "floor": 1000, "ceil": 5000, "color": "rgba(239, 159, 39, 0.7)"},
    {"name": "Federal Backstop", "floor": 5000, "ceil": float("inf"), "color": "rgba(226, 75, 74, 0.7)"},
]

FEMA_LAYERS = [
    {"name": "Municipal Reserves", "floor": 0, "ceil": 50, "color": "rgba(55, 138, 221, 0.7)"},
    {"name": "FEMA / Federal", "floor": 50, "ceil": float("inf"), "color": "rgba(226, 75, 74, 0.7)"},
]


def _calc_layer_flows(loss: float, layers: list) -> List[dict]:
    """Calculate how much each layer absorbs for a given loss."""
    cumulative = 0
    flows = []
    for layer in layers:
        lo = max(layer["floor"], cumulative)
        hi = min(layer["ceil"], loss)
        amount = max(0, hi - lo)
        cumulative += amount
        flows.append({**layer, "amount": amount})
    return [f for f in flows if f["amount"] > 0]


def create_sankey_comparison(loss_millions: float = 5000) -> go.Figure:
    """
    Create side-by-side Sankey diagrams comparing market vs FEMA funding flow.

    Args:
        loss_millions: Total event loss in millions USD

    Returns:
        Plotly Figure with two Sankey subplots
    """
    from plotly.subplots import make_subplots

    market_flows = _calc_layer_flows(loss_millions, MARKET_LAYERS)
    fema_flows = _calc_layer_flows(loss_millions, FEMA_LAYERS)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Market-Based Model",
            f"Traditional FEMA Model",
        ),
        specs=[[{"type": "sankey"}, {"type": "sankey"}]],
        horizontal_spacing=0.08,
    )

    # Market model Sankey
    m_labels = [f"Event Loss\n{format_cost(loss_millions)}"] + [
        f"{f['name']}\n{format_cost(f['amount'])}" for f in market_flows
    ]
    m_colors_node = ["#888"] + [f["color"].replace("0.7", "0.9") for f in market_flows]
    m_source = [0] * len(market_flows)
    m_target = list(range(1, len(market_flows) + 1))
    m_values = [f["amount"] for f in market_flows]
    m_link_colors = [f["color"] for f in market_flows]

    fig.add_trace(
        go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                label=m_labels,
                color=m_colors_node,
                line=dict(width=0),
            ),
            link=dict(
                source=m_source,
                target=m_target,
                value=m_values,
                color=m_link_colors,
            ),
        ),
        row=1, col=1,
    )

    # FEMA model Sankey
    f_labels = [f"Event Loss\n{format_cost(loss_millions)}"] + [
        f"{f['name']}\n{format_cost(f['amount'])}" for f in fema_flows
    ]
    f_colors_node = ["#888"] + [f["color"].replace("0.7", "0.9") for f in fema_flows]
    f_source = [0] * len(fema_flows)
    f_target = list(range(1, len(fema_flows) + 1))
    f_values = [f["amount"] for f in fema_flows]
    f_link_colors = [f["color"] for f in fema_flows]

    fig.add_trace(
        go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                label=f_labels,
                color=f_colors_node,
                line=dict(width=0),
            ),
            link=dict(
                source=f_source,
                target=f_target,
                value=f_values,
                color=f_link_colors,
            ),
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=dict(
            text=f"Funding Flow: {format_cost(loss_millions)} Disaster Event",
            font=dict(size=16),
        ),
        height=450,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return fig


# ──────────────────────────────────────────────────────────────────────
# 3. DISBURSEMENT TIMELINE COMPARISON
# ──────────────────────────────────────────────────────────────────────

MARKET_TIMING = [
    {"name": "Municipal Reserves", "floor": 0, "ceil": 50, "days": 3, "color": "#378ADD"},
    {"name": "State Risk Pool", "floor": 50, "ceil": 250, "days": 7, "color": "#7F77DD"},
    {"name": "Cat Bonds (parametric)", "floor": 250, "ceil": 1000, "days": 3, "color": "#1D9E75"},
    {"name": "Reinsurance Markets", "floor": 1000, "ceil": 5000, "days": 14, "color": "#EF9F27"},
    {"name": "Federal Backstop", "floor": 5000, "ceil": float("inf"), "days": 21, "color": "#E24B4A"},
]

FEMA_TIMING = [
    {"name": "Municipal Reserves", "floor": 0, "ceil": 50, "days": 3, "color": "#378ADD"},
    {"name": "FEMA / Federal Appropriations", "floor": 50, "ceil": float("inf"), "days": 21, "color": "#E24B4A"},
]


def create_disbursement_timeline(loss_millions: float = 2000) -> tuple:
    """
    Create two separate Gantt-style timeline charts for market vs FEMA models.

    Args:
        loss_millions: Total event loss in millions USD

    Returns:
        Tuple of (market_figure, fema_figure, market_wavg, fema_wavg)
    """
    market_flows = _calc_layer_flows(loss_millions, MARKET_TIMING)
    fema_flows = _calc_layer_flows(loss_millions, FEMA_TIMING)

    # Calculate weighted averages
    m_total = sum(f["amount"] for f in market_flows)
    m_wavg = sum(f["amount"] * f["days"] for f in market_flows) / m_total if m_total > 0 else 0
    f_total = sum(f["amount"] for f in fema_flows)
    f_wavg = sum(f["amount"] * f["days"] for f in fema_flows) / f_total if f_total > 0 else 0

    max_day = 28

    # --- Traditional FEMA model chart ---
    fig_fema = go.Figure()
    for f in reversed(fema_flows):
        fig_fema.add_trace(go.Bar(
            y=[f["name"]],
            x=[f["days"]],
            orientation="h",
            marker_color=f["color"],
            text=[f"{format_cost(f['amount'])} — Day {f['days']}"],
            textposition="inside",
            textfont=dict(color="white", size=12),
            hovertemplate=f"<b>{f['name']}</b><br>Amount: {format_cost(f['amount'])}<br>Day {f['days']}<extra></extra>",
            showlegend=False,
        ))

    fig_fema.add_vline(
        x=f_wavg, line_dash="dash", line_color="#E24B4A", line_width=1.5,
        annotation_text=f"Weighted avg: {f_wavg:.1f} days",
        annotation_position="top",
        annotation_font=dict(size=11, color="#E24B4A"),
    )

    if loss_millions > 50:
        fig_fema.add_vrect(
            x0=3, x1=21,
            fillcolor="rgba(226, 75, 74, 0.08)", line_width=0,
            annotation_text="18-day gap: no middle-layer coverage",
            annotation_position="top",
            annotation_font=dict(size=11, color="#E24B4A"),
        )

    fig_fema.update_layout(
        title=dict(text="Traditional FEMA model", font=dict(size=14)),
        xaxis=dict(title="Days to disburse", range=[0, max_day], dtick=3,
                   gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(title=""),
        height=180,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=40),
        bargap=0.3,
    )

    # --- Market-based model chart ---
    fig_market = go.Figure()
    for f in reversed(market_flows):
        fig_market.add_trace(go.Bar(
            y=[f["name"]],
            x=[f["days"]],
            orientation="h",
            marker_color=f["color"],
            text=[f"{format_cost(f['amount'])} — Day {f['days']}"],
            textposition="inside",
            textfont=dict(color="white", size=12),
            hovertemplate=f"<b>{f['name']}</b><br>Amount: {format_cost(f['amount'])}<br>Day {f['days']}<extra></extra>",
            showlegend=False,
        ))

    fig_market.add_vline(
        x=m_wavg, line_dash="dash", line_color="#1D9E75", line_width=1.5,
        annotation_text=f"Weighted avg: {m_wavg:.1f} days",
        annotation_position="top",
        annotation_font=dict(size=11, color="#1D9E75"),
    )

    fig_market.update_layout(
        title=dict(text="Market-based model", font=dict(size=14)),
        xaxis=dict(title="Days to disburse", range=[0, max_day], dtick=3,
                   gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(title=""),
        height=260,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=40),
        bargap=0.3,
    )

    return fig_fema, fig_market, m_wavg, f_wavg


# ──────────────────────────────────────────────────────────────────────
# 4. LOSS EXCEEDANCE CURVE
# ──────────────────────────────────────────────────────────────────────

# Profile severity parameters (mu, sigma for log-normal in millions USD)
EXCEEDANCE_PROFILES = {
    "Gulf Coast (Hurricane Zone)": {"mu": 7.2, "sigma": 2.0, "color": "#E24B4A"},
    "California (Multi-Hazard)": {"mu": 6.8, "sigma": 2.0, "color": "#EF9F27"},
    "Midwest (Severe Weather Corridor)": {"mu": 5.8, "sigma": 1.2, "color": "#378ADD"},
    "Pacific Northwest": {"mu": 6.2, "sigma": 1.8, "color": "#1D9E75"},
    "Northeast Corridor": {"mu": 6.5, "sigma": 1.7, "color": "#7F77DD"},
    "Texas (Multi-Hazard)": {"mu": 7.0, "sigma": 1.9, "color": "#D85A30"},
    "Great Plains": {"mu": 6.0, "sigma": 1.5, "color": "#BA7517"},
}


def _generate_exceedance_data(
    mu: float,
    sigma: float,
    n_samples: int = 50000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate exceedance curve data from log-normal parameters.

    Returns:
        Tuple of (loss_thresholds, exceedance_probabilities)
        where exceedance_probabilities are in [0, 1].
    """
    rng = np.random.default_rng(seed)
    losses = rng.lognormal(mu, sigma, n_samples)
    losses.sort()

    # Generate log-spaced thresholds
    thresholds = np.logspace(1, 6.5, 200)  # $10M to ~$3T
    exceedance = np.array([
        np.sum(losses >= t) / n_samples for t in thresholds
    ])

    # Filter to non-zero exceedance
    mask = exceedance > 0
    return thresholds[mask], exceedance[mask]


def create_exceedance_curve(
    profiles_to_show: Optional[List[str]] = None,
    seed: int = 42,
) -> go.Figure:
    """
    Create loss exceedance probability curve.

    Args:
        profiles_to_show: List of profile names to plot (None = all)
        seed: Random seed for Monte Carlo sampling

    Returns:
        Plotly Figure with log-log exceedance curves
    """
    if profiles_to_show is None:
        profiles_to_show = list(EXCEEDANCE_PROFILES.keys())

    fig = go.Figure()

    for name in profiles_to_show:
        if name not in EXCEEDANCE_PROFILES:
            continue
        params = EXCEEDANCE_PROFILES[name]
        thresholds, exceedance = _generate_exceedance_data(
            params["mu"], params["sigma"], seed=seed
        )

        fig.add_trace(go.Scatter(
            x=thresholds,
            y=exceedance * 100,  # Convert to percentage
            mode="lines",
            name=name,
            line=dict(color=params["color"], width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba({int(params['color'][1:3], 16)}, {int(params['color'][3:5], 16)}, {int(params['color'][5:7], 16)}, 0.08)",
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Loss threshold: $%{x:,.0f}M<br>"
                "Exceedance: %{y:.2f}%<br>"
                "Return period: %{customdata:.0f} years"
                "<extra></extra>"
            ),
            customdata=100 / (exceedance * 100 + 0.001),
        ))

    # Add reference lines for key return periods
    for rp, label, dash in [(10, "1-in-10 year", "dot"), (100, "1-in-100 year", "dash")]:
        pct = 100 / rp
        fig.add_hline(
            y=pct, line_dash=dash, line_color="gray", line_width=0.8,
            annotation_text=label,
            annotation_position="right",
            annotation_font=dict(size=10, color="gray"),
        )

    fig.update_layout(
        title=dict(
            text="Loss Exceedance Curve (Single-Event)",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Loss Threshold ($M, log scale)",
            type="log",
            tickprefix="$",
            ticksuffix="M",
            gridcolor="rgba(128,128,128,0.15)",
            range=[1, 6],  # $10 to $1M (in millions = $10M to $1T)
        ),
        yaxis=dict(
            title="Annual Exceedance Probability (%)",
            type="log",
            gridcolor="rgba(128,128,128,0.15)",
            range=[-1.5, 2],  # 0.03% to 100%
        ),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=50, b=80),
    )

    return fig


# ──────────────────────────────────────────────────────────────────────
# 5. HELPER: Event frequency acceleration (bonus chart for poster)
# ──────────────────────────────────────────────────────────────────────

def create_frequency_trend() -> go.Figure:
    """
    Create event frequency acceleration chart showing the surge
    from ~3 events/year in the 1980s to 20.4/year in the recent
    five-year window (Curry et al., 2025).
    """
    decades = ["1980s", "1990s", "2000s", "2010s", "2020–24"]
    # Per-decade averages; recent five-year average is the paper's 20.4/yr.
    avg_per_year = [3.3, 5.7, 6.7, 13.1, 20.4]
    events = [33, 57, 67, 131, 102]  # implied counts (avg x years)
    years_in_period = [10, 10, 10, 10, 5]
    colors = ["#B5D4F4", "#85B7EB", "#378ADD", "#185FA5", "#E24B4A"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=decades,
        y=avg_per_year,
        marker_color=colors,
        text=[f"{v:.1f}/yr" for v in avg_per_year],
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Total events: %{customdata}<br>"
            "Average: %{y:.1f} events/year"
            "<extra></extra>"
        ),
        customdata=events,
    ))

    fig.update_layout(
        title=dict(
            text="Billion-Dollar Disaster Frequency Acceleration (1980–2024)",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=decades,
        ),
        yaxis=dict(
            title="Average Events per Year",
            gridcolor="rgba(128,128,128,0.15)",
        ),
        height=380,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=40),
    )

    return fig
