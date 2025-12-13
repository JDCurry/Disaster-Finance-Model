# Disaster Finance Model

Monte Carlo simulation tool for comparing market-based disaster financing against traditional FEMA-centric funding.

## Overview

This tool implements the five-layer disaster financing framework proposed in:

**"Transforming Disaster Financing: An Alternative to FEMA Funding"**  
*Domestic Preparedness*, December 2025  
*Authors: Josh Curry, Chandler Clough, Johnny Hicks, Andrew Jackson, Ryan Rockabrand*

## NOAA NCEI Data Calibration

The model is calibrated using authoritative data from NOAA's National Centers for Environmental Information (NCEI) Billion-Dollar Weather and Climate Disasters database (1980-2024):

| Metric | Value |
|--------|-------|
| Total Events | 403 billion-dollar disasters |
| Cumulative Cost | $2.915 trillion (CPI-adjusted) |
| Historical Avg (1980-2023) | 9.0 events/year |
| Recent Avg (2020-2024) | 23.0 events/year |
| Most Costly Type | Tropical cyclones ($1.54T, 53% of total) |
| Most Frequent Type | Severe storms (203 events, 50% of total) |

Data source: https://www.ncei.noaa.gov/access/billions/

## The Five-Layer Model

| Layer | Source | Coverage Range | Disbursement | Key Benefit |
|-------|--------|----------------|--------------|-------------|
| 1 | Municipal Reserves | First $50M | ~3 days | Local tax base stabilization |
| 2 | State Risk Pools | $50M-$250M | ~7 days | Regional diversification |
| 3 | Catastrophe Bonds | $250M-$1B | ~3 days* | Locked-in capital commitments |
| 4 | Reinsurance Markets | $1B-$5B | ~14 days | Global risk distribution |
| 5 | Federal Backstop | >$5B | ~21 days | Crisis-level market assurance |

*Parametric triggers can achieve 72-hour disbursement

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

### Use as a Python Library

```python
from src import (
    DisasterEventGenerator, 
    PRESET_PROFILES,
    FundingWaterfall,
    SimulationRunner,
    NOAADataCalibrator
)

# View NOAA calibration data
calibrator = NOAADataCalibrator()
print(calibrator.get_calibration_summary())

# Generate events with NOAA-calibrated parameters
gen = DisasterEventGenerator(seed=42)
profile = PRESET_PROFILES["gulf_coast"]
events = gen.generate_annual_events(profile, year=2025)

# Run simulation
runner = SimulationRunner(seed=42)
results = runner.run_monte_carlo(profile, n_years=50, n_simulations=100)

print(f"Market coverage: {results.market_avg_coverage_ratio*100:.1f}%")
print(f"Time improvement: {results.time_improvement_days:.1f} days faster")
```

## Regional Profiles

Pre-configured risk profiles calibrated to NOAA state-level cost data:

- **Gulf Coast**: Hurricane-dominant (TX, LA, MS, AL, FL)
- **California**: Wildfire primary with earthquake/drought secondary
- **Texas**: Multi-hazard (highest total state costs: $436B)
- **Midwest**: Severe storm corridor (IL, IN, OH, MI, WI, MN, IA, MO)
- **Plains**: Drought-dominant (KS, NE, SD, ND, OK)
- **Pacific Northwest**: Wildfire and earthquake (WA, OR, ID)
- **Northeast Corridor**: Hurricane and winter storms (NY, NJ, PA, CT, MA)

## Key Metrics Compared

The simulation compares two models:

### Market-Based Model (Proposed)
- All five layers actively engaged
- Parametric triggers for rapid disbursement (72 hours)
- Contractually committed funding
- Risk-appropriate pricing incentives

### Traditional FEMA Model
- Municipal reserves + federal appropriations only
- "Vast middle ground" unfilled
- Subject to annual appropriations
- 21-day average disbursement timeline (FEMA PA)

## Project Structure

```
disaster-finance-model/
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── src/
    ├── __init__.py
    ├── disaster_generator.py # Event generation (Monte Carlo)
    ├── funding_waterfall.py  # Five-layer funding model
    ├── simulation_runner.py  # Simulation orchestration
    └── noaa_data.py          # NOAA NCEI historical data
```

## License

Research and educational use. Based on publicly available policy proposals and government data.

## Citation

If using this tool for research, please cite:

> Curry, J., Clough, C., Hicks, J., Jackson, A., & Rockabrand, R. (2025). 
> Transforming Disaster Financing: An Alternative to FEMA Funding. 
> *Domestic Preparedness*.

Data citation:
> NOAA National Centers for Environmental Information (NCEI). 
> U.S. Billion-Dollar Weather and Climate Disasters (2024). 
> https://www.ncei.noaa.gov/access/billions/
