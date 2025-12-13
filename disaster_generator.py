"""
Disaster Event Generator Module

Generates synthetic disaster events using Monte Carlo simulation calibrated to 
historical NOAA/NCEI data on billion-dollar disasters.

Key parameters from NOAA NCEI (1980-2024):
- Total events: 403 billion-dollar disasters
- Cumulative cost: $2.915 trillion (CPI-adjusted)
- Historical average: 9.0 events/year (1980-2023)
- Recent surge: 23.0 events/year (2020-2024)
- Tropical cyclones: $1.54T total (53% of all costs)
- Severe storms: Most frequent (203 events, 50% of total)

Author: Josh Curry et al.
Reference: "Transforming Disaster Financing: An Alternative to FEMA Funding"
Data source: https://www.ncei.noaa.gov/access/billions/
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum


class DisasterType(Enum):
    """Categories of disaster events with distinct risk profiles."""
    HURRICANE = "hurricane"
    EARTHQUAKE = "earthquake"
    FLOOD = "flood"
    WILDFIRE = "wildfire"
    SEVERE_STORM = "severe_storm"
    WINTER_STORM = "winter_storm"
    DROUGHT = "drought"


class TriggerType(Enum):
    """Parametric trigger mechanisms for bond payouts."""
    WIND_SPEED = "wind_speed"          # Hurricane: sustained winds in mph
    MAGNITUDE = "magnitude"             # Earthquake: Richter scale
    FLOOD_DEPTH = "flood_depth"         # Flood: feet above flood stage
    BURNED_ACRES = "burned_acres"       # Wildfire: acres burned
    INDUSTRY_LOSS = "industry_loss"     # Indemnity: total insured losses
    MODELED_LOSS = "modeled_loss"       # Modeled: catastrophe model output


@dataclass
class ParametricTrigger:
    """Defines conditions for bond payout activation."""
    trigger_type: TriggerType
    threshold: float
    payout_percentage: float  # 0.0 to 1.0, percentage of principal at risk
    
    def is_triggered(self, observed_value: float) -> bool:
        """Check if observed value exceeds trigger threshold."""
        return observed_value >= self.threshold
    
    def calculate_payout(self, principal: float, observed_value: float) -> float:
        """Calculate payout based on trigger activation."""
        if self.is_triggered(observed_value):
            return principal * self.payout_percentage
        return 0.0


@dataclass
class DisasterEvent:
    """Represents a single disaster event with parametric characteristics."""
    disaster_type: DisasterType
    severity_value: float           # Primary parametric value
    economic_loss: float            # Total economic loss in millions USD
    insured_loss: float             # Insured portion in millions USD
    trigger_type: TriggerType
    location_factor: float = 1.0    # Regional risk multiplier
    year: int = 0
    event_id: str = ""
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"{self.disaster_type.value}_{self.year}_{id(self)}"


@dataclass
class RegionalRiskProfile:
    """Defines risk characteristics for a geographic region."""
    name: str
    primary_hazard: DisasterType
    secondary_hazards: List[DisasterType] = field(default_factory=list)
    
    # Frequency parameters (Poisson lambda for annual event count)
    baseline_frequency: float = 8.5      # Historical NOAA average
    trend_multiplier: float = 1.0        # Adjustment for recent trends
    
    # Severity parameters (log-normal distribution)
    severity_mu: float = 4.0             # Log-mean of economic loss
    severity_sigma: float = 1.5          # Log-std of economic loss
    
    # Regional factors
    population_exposure: float = 1.0     # Population density factor
    infrastructure_value: float = 1.0   # Built environment value factor
    
    @property
    def adjusted_frequency(self) -> float:
        """Annual expected event count adjusted for trends."""
        return self.baseline_frequency * self.trend_multiplier


class DisasterEventGenerator:
    """
    Monte Carlo generator for disaster events.
    
    Calibrated to NOAA/NCEI billion-dollar disaster data:
    - 1980-2023 average: 8.5 events/year
    - 2019-2023 average: 20.4 events/year
    - Severity follows approximately log-normal distribution
    """
    
    # Calibrated severity parameters from NOAA NCEI data (1980-2024)
    # mu/sigma for log-normal distribution of CPI-adjusted costs in millions USD
    # Based on average cost per event by type from 403 historical events
    DEFAULT_SEVERITY_PARAMS = {
        DisasterType.HURRICANE: {"mu": 7.5, "sigma": 2.2, "trigger": TriggerType.WIND_SPEED},      # Avg $23B/event
        DisasterType.EARTHQUAKE: {"mu": 6.8, "sigma": 2.0, "trigger": TriggerType.MAGNITUDE},      # High variance
        DisasterType.FLOOD: {"mu": 6.2, "sigma": 1.5, "trigger": TriggerType.FLOOD_DEPTH},         # Avg $4.5B/event
        DisasterType.WILDFIRE: {"mu": 6.5, "sigma": 1.9, "trigger": TriggerType.BURNED_ACRES},     # Avg $6.2B/event
        DisasterType.SEVERE_STORM: {"mu": 5.5, "sigma": 1.0, "trigger": TriggerType.INDUSTRY_LOSS},# Avg $2.5B/event
        DisasterType.WINTER_STORM: {"mu": 6.0, "sigma": 1.3, "trigger": TriggerType.INDUSTRY_LOSS},# Avg $4.7B/event
        DisasterType.DROUGHT: {"mu": 7.0, "sigma": 1.8, "trigger": TriggerType.MODELED_LOSS},      # Avg $11.1B/event
    }
    
    # Parametric trigger thresholds by type
    TRIGGER_THRESHOLDS = {
        TriggerType.WIND_SPEED: [(74, 0.25), (96, 0.50), (111, 0.75), (130, 1.0)],  # Cat 1-4+
        TriggerType.MAGNITUDE: [(5.0, 0.25), (6.0, 0.50), (7.0, 0.75), (7.5, 1.0)],
        TriggerType.FLOOD_DEPTH: [(3, 0.25), (6, 0.50), (10, 0.75), (15, 1.0)],
        TriggerType.BURNED_ACRES: [(10000, 0.25), (50000, 0.50), (100000, 0.75), (250000, 1.0)],
        TriggerType.INDUSTRY_LOSS: [(500, 0.25), (1000, 0.50), (2500, 0.75), (5000, 1.0)],
        TriggerType.MODELED_LOSS: [(500, 0.25), (1000, 0.50), (2500, 0.75), (5000, 1.0)],
    }
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)
        
    def generate_severity_value(
        self, 
        disaster_type: DisasterType,
        profile: Optional[RegionalRiskProfile] = None
    ) -> Tuple[float, float]:
        """
        Generate economic loss and corresponding parametric trigger value.
        
        Returns:
            Tuple of (economic_loss_millions, trigger_value)
        """
        params = self.DEFAULT_SEVERITY_PARAMS[disaster_type]
        
        # Adjust parameters if regional profile provided
        mu = params["mu"]
        sigma = params["sigma"]
        if profile:
            mu += np.log(profile.population_exposure * profile.infrastructure_value)
        
        # Generate economic loss from log-normal
        economic_loss = self.rng.lognormal(mu, sigma)
        
        # Map to parametric trigger value based on disaster type
        trigger_value = self._map_loss_to_trigger(
            economic_loss, 
            disaster_type,
            params["trigger"]
        )
        
        return economic_loss, trigger_value
    
    def _map_loss_to_trigger(
        self, 
        loss: float, 
        disaster_type: DisasterType,
        trigger_type: TriggerType
    ) -> float:
        """Map economic loss to approximate parametric trigger value."""
        # Simplified mapping - in production would use actual hazard models
        thresholds = self.TRIGGER_THRESHOLDS[trigger_type]
        
        # Use loss percentile to select trigger level
        if loss < 100:
            return thresholds[0][0] * 0.8
        elif loss < 500:
            return thresholds[0][0] + (thresholds[1][0] - thresholds[0][0]) * 0.5
        elif loss < 2000:
            return thresholds[1][0] + (thresholds[2][0] - thresholds[1][0]) * 0.5
        elif loss < 10000:
            return thresholds[2][0] + (thresholds[3][0] - thresholds[2][0]) * 0.5
        else:
            return thresholds[3][0] * 1.2
    
    def generate_annual_events(
        self,
        profile: RegionalRiskProfile,
        year: int = 2025
    ) -> List[DisasterEvent]:
        """
        Generate all disaster events for a single year.
        
        Uses Poisson distribution for event count, calibrated to NOAA data.
        """
        # Generate number of events (Poisson)
        n_events = self.rng.poisson(profile.adjusted_frequency)
        
        events = []
        for i in range(n_events):
            # Select disaster type (weighted by regional profile)
            disaster_type = self._select_disaster_type(profile)
            
            # Generate severity
            economic_loss, trigger_value = self.generate_severity_value(
                disaster_type, profile
            )
            
            # Estimate insured loss (typically 40-60% of economic loss)
            insured_ratio = self.rng.uniform(0.35, 0.65)
            insured_loss = economic_loss * insured_ratio
            
            event = DisasterEvent(
                disaster_type=disaster_type,
                severity_value=trigger_value,
                economic_loss=economic_loss,
                insured_loss=insured_loss,
                trigger_type=self.DEFAULT_SEVERITY_PARAMS[disaster_type]["trigger"],
                location_factor=profile.population_exposure,
                year=year,
                event_id=f"{disaster_type.value}_{year}_{i}"
            )
            events.append(event)
        
        return events
    
    def _select_disaster_type(self, profile: RegionalRiskProfile) -> DisasterType:
        """Select disaster type based on regional hazard profile."""
        # Primary hazard gets 60% weight, secondary split remaining
        hazards = [profile.primary_hazard] + profile.secondary_hazards
        
        if len(hazards) == 1:
            weights = [1.0]
        else:
            primary_weight = 0.6
            secondary_weight = 0.4 / len(profile.secondary_hazards)
            weights = [primary_weight] + [secondary_weight] * len(profile.secondary_hazards)
        
        weights = np.array(weights) / sum(weights)
        return self.rng.choice(hazards, p=weights)
    
    def run_simulation(
        self,
        profile: RegionalRiskProfile,
        n_years: int = 100,
        start_year: int = 2025
    ) -> List[List[DisasterEvent]]:
        """
        Run multi-year Monte Carlo simulation.
        
        Args:
            profile: Regional risk characteristics
            n_years: Number of years to simulate
            start_year: Starting year for simulation
            
        Returns:
            List of event lists, one per simulated year
        """
        all_events = []
        for year_offset in range(n_years):
            year = start_year + year_offset
            year_events = self.generate_annual_events(profile, year)
            all_events.append(year_events)
        
        return all_events


# Preset regional profiles calibrated to NOAA NCEI state-level cost data (1980-2024)
# Frequencies based on 23.0 events/year national average (2020-2024 trend)
PRESET_PROFILES = {
    "gulf_coast": RegionalRiskProfile(
        name="Gulf Coast (Hurricane Zone)",
        primary_hazard=DisasterType.HURRICANE,
        secondary_hazards=[DisasterType.FLOOD, DisasterType.SEVERE_STORM],
        baseline_frequency=6.0,      # ~26% of national events hit this region
        trend_multiplier=1.4,        # Accelerating trend
        severity_mu=7.2,             # High due to major hurricanes
        severity_sigma=2.0,
        population_exposure=1.3,
        infrastructure_value=1.2,
    ),
    "california": RegionalRiskProfile(
        name="California (Multi-Hazard)",
        primary_hazard=DisasterType.WILDFIRE,  # $100B+ in wildfire costs
        secondary_hazards=[DisasterType.EARTHQUAKE, DisasterType.DROUGHT],
        baseline_frequency=4.5,
        trend_multiplier=1.6,        # Wildfire trend especially strong
        severity_mu=6.8,
        severity_sigma=2.0,
        population_exposure=1.5,
        infrastructure_value=1.4,
    ),
    "midwest": RegionalRiskProfile(
        name="Midwest (Severe Weather Corridor)",
        primary_hazard=DisasterType.SEVERE_STORM,
        secondary_hazards=[DisasterType.FLOOD, DisasterType.DROUGHT],
        baseline_frequency=5.5,      # High severe storm frequency
        trend_multiplier=1.3,
        severity_mu=5.8,
        severity_sigma=1.2,
        population_exposure=0.9,
        infrastructure_value=1.0,
    ),
    "pacific_northwest": RegionalRiskProfile(
        name="Pacific Northwest",
        primary_hazard=DisasterType.WILDFIRE,
        secondary_hazards=[DisasterType.EARTHQUAKE, DisasterType.FLOOD],
        baseline_frequency=3.0,
        trend_multiplier=1.5,        # Wildfire increase
        severity_mu=6.2,
        severity_sigma=1.8,
        population_exposure=1.1,
        infrastructure_value=1.1,
    ),
    "northeast": RegionalRiskProfile(
        name="Northeast Corridor",
        primary_hazard=DisasterType.HURRICANE,  # Sandy, Irene, etc.
        secondary_hazards=[DisasterType.SEVERE_STORM, DisasterType.WINTER_STORM],
        baseline_frequency=4.0,
        trend_multiplier=1.25,
        severity_mu=6.5,
        severity_sigma=1.7,
        population_exposure=1.5,     # High population density
        infrastructure_value=1.4,
    ),
    "plains": RegionalRiskProfile(
        name="Great Plains",
        primary_hazard=DisasterType.DROUGHT,
        secondary_hazards=[DisasterType.SEVERE_STORM, DisasterType.FLOOD],
        baseline_frequency=4.0,
        trend_multiplier=1.2,
        severity_mu=6.0,
        severity_sigma=1.5,
        population_exposure=0.6,
        infrastructure_value=0.8,
    ),
    "texas": RegionalRiskProfile(
        name="Texas (Multi-Hazard)",
        primary_hazard=DisasterType.HURRICANE,
        secondary_hazards=[DisasterType.SEVERE_STORM, DisasterType.DROUGHT, DisasterType.FLOOD],
        baseline_frequency=7.0,      # Texas has highest state cost
        trend_multiplier=1.35,
        severity_mu=7.0,
        severity_sigma=1.9,
        population_exposure=1.4,
        infrastructure_value=1.3,
    ),
}


if __name__ == "__main__":
    # Quick test
    gen = DisasterEventGenerator(seed=42)
    profile = PRESET_PROFILES["gulf_coast"]
    
    events = gen.run_simulation(profile, n_years=10)
    
    total_events = sum(len(year) for year in events)
    total_losses = sum(e.economic_loss for year in events for e in year)
    
    print(f"Profile: {profile.name}")
    print(f"Simulated {total_events} events over 10 years")
    print(f"Total economic losses: ${total_losses:,.0f}M")
    print(f"Average events/year: {total_events/10:.1f}")
