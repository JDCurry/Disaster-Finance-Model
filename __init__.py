"""
Disaster Finance Model

Monte Carlo simulation framework for comparing market-based disaster financing
against traditional FEMA-centric funding models.

Based on: "Transforming Disaster Financing: An Alternative to FEMA Funding"
Domestic Preparedness, December 2025
"""

from .disaster_generator import (
    DisasterEventGenerator,
    DisasterEvent,
    DisasterType,
    RegionalRiskProfile,
    PRESET_PROFILES,
)

from .funding_waterfall import (
    FundingWaterfall,
    TraditionalFEMAModel,
    FundingLayer,
    LayerConfiguration,
    WaterfallResult,
)

from .simulation_runner import (
    SimulationRunner,
    SimulationResults,
    YearSummary,
    generate_summary_report,
)

from .noaa_data import (
    NOAADataCalibrator,
    NOAA_SUMMARY_STATS,
    STATE_COST_DATA,
    REGIONAL_GROUPINGS,
    HistoricalEvent,
    HISTORICAL_EVENTS_2000_2021,
)

__version__ = "0.2.0"
__author__ = "Josh Curry et al."
