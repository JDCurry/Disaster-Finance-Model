"""
Simulation Runner Module

Orchestrates Monte Carlo simulations comparing the proposed market-based
disaster financing model against traditional FEMA-centric funding.

Generates summary statistics and comparison metrics for analysis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

try:
    from disaster_generator import (
        DisasterEventGenerator, 
        RegionalRiskProfile, 
        DisasterEvent,
        PRESET_PROFILES
    )
    from funding_waterfall import (
        FundingWaterfall, 
        TraditionalFEMAModel,
        WaterfallResult,
        FundingLayer
    )
except ImportError:
    from .disaster_generator import (
        DisasterEventGenerator, 
        RegionalRiskProfile, 
        DisasterEvent,
        PRESET_PROFILES
    )
    from .funding_waterfall import (
        FundingWaterfall, 
        TraditionalFEMAModel,
        WaterfallResult,
        FundingLayer
    )


@dataclass
class YearSummary:
    """Summary statistics for a single simulated year."""
    year: int
    n_events: int
    total_losses: float
    
    # Market-based model metrics
    market_total_covered: float
    market_coverage_ratio: float
    market_avg_disbursement_days: float
    market_gaps: float
    
    # Traditional model metrics
    fema_total_covered: float
    fema_coverage_ratio: float
    fema_avg_disbursement_days: float
    fema_gaps: float
    
    # Comparison deltas
    coverage_improvement: float  # Market - FEMA
    time_improvement: float      # FEMA - Market (positive = faster)


@dataclass
class SimulationResults:
    """Complete results from a multi-year simulation run."""
    profile_name: str
    n_years: int
    n_simulations: int
    
    # Aggregate statistics
    total_events: int
    total_losses: float
    avg_events_per_year: float
    
    # Year-by-year summaries
    year_summaries: List[YearSummary]
    
    # Model comparison metrics
    market_avg_coverage_ratio: float
    market_avg_disbursement_days: float
    market_total_gaps: float
    
    fema_avg_coverage_ratio: float
    fema_avg_disbursement_days: float
    fema_total_gaps: float
    
    # Layer utilization (market model only)
    layer_utilization: Dict[str, float]
    layer_frequency: Dict[str, int]  # How often each layer was tapped
    
    # Distribution statistics
    loss_percentiles: Dict[str, float]
    gap_percentiles: Dict[str, float]
    
    @property
    def coverage_improvement_pct(self) -> float:
        """Percentage improvement in coverage (market vs FEMA)."""
        if self.fema_avg_coverage_ratio == 0:
            return 0.0
        return (self.market_avg_coverage_ratio - self.fema_avg_coverage_ratio) / self.fema_avg_coverage_ratio * 100
    
    @property
    def time_improvement_days(self) -> float:
        """Days faster for market-based model."""
        return self.fema_avg_disbursement_days - self.market_avg_disbursement_days


class SimulationRunner:
    """
    Runs Monte Carlo simulations comparing funding models.
    
    Generates probabilistic outcomes across many years to assess:
    - Coverage reliability under different scenarios
    - Disbursement timing differences
    - Layer utilization patterns
    - Funding gap risks
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        self.base_seed = seed
        
    def run_single_simulation(
        self,
        profile: RegionalRiskProfile,
        n_years: int = 50,
        start_year: int = 2025,
        sim_id: int = 0
    ) -> Tuple[List[List[WaterfallResult]], List[List[WaterfallResult]]]:
        """
        Run a single multi-year simulation.
        
        Returns:
            Tuple of (market_results_by_year, fema_results_by_year)
        """
        # Create generators with derived seed for reproducibility
        seed = self.base_seed + sim_id if self.base_seed else None
        
        event_gen = DisasterEventGenerator(seed=seed)
        market_model = FundingWaterfall(seed=seed)
        fema_model = TraditionalFEMAModel(seed=seed)
        
        market_results_by_year = []
        fema_results_by_year = []
        
        for year_offset in range(n_years):
            year = start_year + year_offset
            
            # Generate events for this year
            events = event_gen.generate_annual_events(profile, year)
            
            # Process through both models
            market_results = [market_model.process_event(e) for e in events]
            fema_results = [fema_model.process_event(e) for e in events]
            
            market_results_by_year.append(market_results)
            fema_results_by_year.append(fema_results)
        
        return market_results_by_year, fema_results_by_year
    
    def run_monte_carlo(
        self,
        profile: RegionalRiskProfile,
        n_years: int = 50,
        n_simulations: int = 100,
        start_year: int = 2025
    ) -> SimulationResults:
        """
        Run full Monte Carlo simulation with multiple iterations.
        
        Args:
            profile: Regional risk profile to simulate
            n_years: Years per simulation run
            n_simulations: Number of simulation iterations
            start_year: Starting year
            
        Returns:
            SimulationResults with comprehensive statistics
        """
        all_year_summaries = []
        all_losses = []
        all_market_gaps = []
        all_fema_gaps = []
        
        layer_totals = defaultdict(float)
        layer_counts = defaultdict(int)
        
        total_events = 0
        total_losses = 0.0
        
        market_coverage_sum = 0.0
        market_days_sum = 0.0
        market_gaps_sum = 0.0
        
        fema_coverage_sum = 0.0
        fema_days_sum = 0.0
        fema_gaps_sum = 0.0
        
        n_results = 0
        
        for sim_id in range(n_simulations):
            market_by_year, fema_by_year = self.run_single_simulation(
                profile, n_years, start_year, sim_id
            )
            
            for year_idx, (market_results, fema_results) in enumerate(
                zip(market_by_year, fema_by_year)
            ):
                year = start_year + year_idx
                
                if not market_results:
                    continue
                
                # Calculate year metrics
                n_events = len(market_results)
                year_losses = sum(r.total_loss for r in market_results)
                
                market_covered = sum(r.total_covered for r in market_results)
                market_ratio = market_covered / year_losses if year_losses > 0 else 1.0
                market_days = np.mean([r.weighted_avg_disbursement_days for r in market_results])
                market_gaps = sum(r.coverage_gap for r in market_results)
                
                fema_covered = sum(r.total_covered for r in fema_results)
                fema_ratio = fema_covered / year_losses if year_losses > 0 else 1.0
                fema_days = np.mean([r.weighted_avg_disbursement_days for r in fema_results])
                fema_gaps = sum(r.coverage_gap for r in fema_results)
                
                summary = YearSummary(
                    year=year,
                    n_events=n_events,
                    total_losses=year_losses,
                    market_total_covered=market_covered,
                    market_coverage_ratio=market_ratio,
                    market_avg_disbursement_days=market_days,
                    market_gaps=market_gaps,
                    fema_total_covered=fema_covered,
                    fema_coverage_ratio=fema_ratio,
                    fema_avg_disbursement_days=fema_days,
                    fema_gaps=fema_gaps,
                    coverage_improvement=market_ratio - fema_ratio,
                    time_improvement=fema_days - market_days
                )
                all_year_summaries.append(summary)
                
                # Accumulate totals
                total_events += n_events
                total_losses += year_losses
                
                market_coverage_sum += market_ratio
                market_days_sum += market_days
                market_gaps_sum += market_gaps
                
                fema_coverage_sum += fema_ratio
                fema_days_sum += fema_days
                fema_gaps_sum += fema_gaps
                
                n_results += 1
                
                # Track loss distribution
                for r in market_results:
                    all_losses.append(r.total_loss)
                    all_market_gaps.append(r.coverage_gap)
                    
                    # Track layer utilization
                    for layer, amount in r.layer_utilization.items():
                        # Use .value for display and dict keys
                        key = layer.value if hasattr(layer, 'value') else str(layer)
                        layer_totals[key] += amount
                        if amount > 0:
                            layer_counts[key] += 1
                
                for r in fema_results:
                    all_fema_gaps.append(r.coverage_gap)
        
        # Calculate percentiles
        all_losses = np.array(all_losses)
        all_market_gaps = np.array(all_market_gaps)
        
        loss_percentiles = {
            "p50": float(np.percentile(all_losses, 50)),
            "p75": float(np.percentile(all_losses, 75)),
            "p90": float(np.percentile(all_losses, 90)),
            "p95": float(np.percentile(all_losses, 95)),
            "p99": float(np.percentile(all_losses, 99)),
            "max": float(np.max(all_losses)),
        }
        
        gap_percentiles = {
            "p50": float(np.percentile(all_market_gaps, 50)),
            "p75": float(np.percentile(all_market_gaps, 75)),
            "p90": float(np.percentile(all_market_gaps, 90)),
            "p95": float(np.percentile(all_market_gaps, 95)),
            "p99": float(np.percentile(all_market_gaps, 99)),
        }
        
        return SimulationResults(
            profile_name=profile.name,
            n_years=n_years,
            n_simulations=n_simulations,
            total_events=total_events,
            total_losses=total_losses,
            avg_events_per_year=total_events / (n_years * n_simulations),
            year_summaries=all_year_summaries,
            market_avg_coverage_ratio=market_coverage_sum / n_results if n_results > 0 else 0,
            market_avg_disbursement_days=market_days_sum / n_results if n_results > 0 else 0,
            market_total_gaps=market_gaps_sum,
            fema_avg_coverage_ratio=fema_coverage_sum / n_results if n_results > 0 else 0,
            fema_avg_disbursement_days=fema_days_sum / n_results if n_results > 0 else 0,
            fema_total_gaps=fema_gaps_sum,
            layer_utilization=dict(layer_totals),
            layer_frequency=dict(layer_counts),
            loss_percentiles=loss_percentiles,
            gap_percentiles=gap_percentiles,
        )
    
    def run_scenario_comparison(
        self,
        profiles: List[RegionalRiskProfile],
        n_years: int = 50,
        n_simulations: int = 100
    ) -> Dict[str, SimulationResults]:
        """
        Run simulations across multiple regional profiles.
        
        Returns:
            Dictionary mapping profile names to results
        """
        results = {}
        for profile in profiles:
            results[profile.name] = self.run_monte_carlo(
                profile, n_years, n_simulations
            )
        return results


def _format_value(value_millions: float) -> str:
    """Format currency values with T/B/M notation for text reports."""
    if value_millions >= 1_000_000:
        # Trillions (value is already in millions, so /1M = trillions)
        trillions = value_millions / 1_000_000
        if trillions >= 100:
            return f"${trillions:,.0f}T"
        elif trillions >= 10:
            return f"${trillions:,.1f}T"
        else:
            return f"${trillions:,.2f}T"
    elif value_millions >= 1000:
        # Billions
        billions = value_millions / 1000
        if billions >= 100:
            return f"${billions:,.0f}B"
        elif billions >= 10:
            return f"${billions:,.1f}B"
        else:
            return f"${billions:,.2f}B"
    else:
        # Millions
        if value_millions >= 100:
            return f"${value_millions:,.0f}M"
        elif value_millions >= 10:
            return f"${value_millions:,.1f}M"
        else:
            return f"${value_millions:,.2f}M"


def generate_summary_report(results: SimulationResults) -> str:
    """Generate a text summary of simulation results."""
    lines = [
        f"=" * 60,
        f"SIMULATION RESULTS: {results.profile_name}",
        f"=" * 60,
        f"",
        f"Configuration:",
        f"  Years simulated: {results.n_years}",
        f"  Simulation runs: {results.n_simulations}",
        f"  Total events generated: {results.total_events:,}",
        f"  Average events/year: {results.avg_events_per_year:.1f}",
        f"  Total losses simulated: {_format_value(results.total_losses)}",
        f"",
        f"Loss Distribution:",
        f"  Median (P50): {_format_value(results.loss_percentiles['p50'])}",
        f"  P90: {_format_value(results.loss_percentiles['p90'])}",
        f"  P99: {_format_value(results.loss_percentiles['p99'])}",
        f"  Maximum: {_format_value(results.loss_percentiles['max'])}",
        f"",
        f"Market-Based Model Performance:",
        f"  Average coverage ratio: {results.market_avg_coverage_ratio*100:.1f}%",
        f"  Average disbursement time: {results.market_avg_disbursement_days:.1f} days",
        f"  Total funding gaps: {_format_value(results.market_total_gaps)}",
        f"",
        f"Traditional FEMA Model Performance:",
        f"  Average coverage ratio: {results.fema_avg_coverage_ratio*100:.1f}%",
        f"  Average disbursement time: {results.fema_avg_disbursement_days:.1f} days",
        f"  Total funding gaps: {_format_value(results.fema_total_gaps)}",
        f"",
        f"Comparative Improvement (Market vs FEMA):",
        f"  Coverage improvement: {results.coverage_improvement_pct:+.1f}%",
        f"  Time improvement: {results.time_improvement_days:+.1f} days faster",
        f"  Gap reduction: {_format_value(results.fema_total_gaps - results.market_total_gaps)}",
        f"",
        f"Layer Utilization (Market Model):",
    ]
    
    for layer_name, total in sorted(results.layer_utilization.items()):
        freq = results.layer_frequency.get(layer_name, 0)
        lines.append(f"  {layer_name}: {_format_value(total)} ({freq:,} activations)")
    
    lines.extend([
        f"",
        f"=" * 60,
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test run
    runner = SimulationRunner(seed=42)
    
    profile = PRESET_PROFILES["gulf_coast"]
    results = runner.run_monte_carlo(profile, n_years=20, n_simulations=50)
    
    print(generate_summary_report(results))
