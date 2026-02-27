"""
Funding Waterfall Module

Implements the five-layer disaster financing structure proposed in 
"Transforming Disaster Financing: An Alternative to FEMA Funding"

Layer Structure (from Table 1):
    Layer 1: Municipal Reserves     - First $50M losses
    Layer 2: State Risk Pools       - Next $200M ($50M-$250M)
    Layer 3: Cat Bonds (AAA)        - $200M-$1B range
    Layer 4: Reinsurance Markets    - $1B-$5B range
    Layer 5: Federal Backstop       - >$5B (crisis-level)

Author: Josh Curry et al.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

from disaster_generator import DisasterEvent, DisasterType


class FundingLayer(Enum):
    """The five layers of the proposed financing structure."""
    MunicipalReserves = "Municipal Reserves"
    StateRiskPool = "State Risk Pool"
    CatBonds = "Cat Bonds"
    Reinsurance = "Reinsurance"
    FederalBackstop = "Federal Backstop"


@dataclass
class LayerConfiguration:
    """Configuration for a single funding layer."""
    layer: FundingLayer
    name: str
    floor: float              # Minimum loss threshold (millions USD)
    ceiling: float            # Maximum coverage (millions USD)
    capacity: float           # Available funds (millions USD)
    
    # Timing characteristics
    disbursement_days: float  # Average days to disburse funds
    disbursement_std: float   # Std dev of disbursement time
    
    # Cost characteristics
    annual_premium_rate: float = 0.0  # As percentage of capacity
    admin_cost_rate: float = 0.0      # Administrative overhead
    
    # Reliability
    availability_probability: float = 1.0  # Probability funds available when needed
    
    def coverage_range(self) -> Tuple[float, float]:
        """Return the (floor, ceiling) coverage range."""
        return (self.floor, self.ceiling)
    
    def calculate_coverage(self, loss: float, cumulative_covered: float) -> float:
        """
        Calculate how much of a loss this layer covers.
        
        Args:
            loss: Total loss amount in millions
            cumulative_covered: Amount already covered by lower layers
            
        Returns:
            Amount covered by this layer in millions
        """
        remaining_loss = loss - cumulative_covered
        
        if remaining_loss <= 0:
            return 0.0
        
        # Loss must exceed floor to trigger this layer
        if cumulative_covered >= self.ceiling:
            return 0.0
        
        # Calculate coverage within this layer's range
        layer_floor = max(self.floor, cumulative_covered)
        layer_ceiling = min(self.ceiling, loss)
        
        if layer_ceiling <= layer_floor:
            return 0.0
        
        coverage = min(layer_ceiling - layer_floor, self.capacity)
        return max(0.0, coverage)


@dataclass 
class DisbursementEvent:
    """Tracks a single disbursement from the funding waterfall."""
    layer: FundingLayer
    amount: float               # Millions USD
    event_id: str
    days_to_disburse: float
    was_available: bool = True  # Whether funds were actually available
    gap_amount: float = 0.0     # Unfunded gap if unavailable


@dataclass
class WaterfallResult:
    """Complete results from processing an event through the waterfall."""
    event: DisasterEvent
    total_loss: float
    total_covered: float
    coverage_gap: float
    disbursements: List[DisbursementEvent]
    layer_utilization: Dict[FundingLayer, float]
    weighted_avg_disbursement_days: float
    
    @property
    def coverage_ratio(self) -> float:
        """Percentage of loss covered."""
        if self.total_loss == 0:
            return 1.0
        return self.total_covered / self.total_loss
    
    @property
    def has_gap(self) -> bool:
        """Whether there's an unfunded gap."""
        return self.coverage_gap > 0


class FundingWaterfall:
    """
    Implements the cascading funding structure for disaster financing.
    
    Based on the five-layer model from the Domestic Preparedness article:
    - Layer 1: Municipal Reserves (local tax base stabilization)
    - Layer 2: State Risk Pools (regional diversification)  
    - Layer 3: Catastrophe Bonds (locked-in capital commitments)
    - Layer 4: Reinsurance Markets (global risk distribution)
    - Layer 5: Federal Backstop (crisis-level market assurance)
    """
    
    # Default layer configurations based on Table 1 from paper
    DEFAULT_LAYERS = [
        LayerConfiguration(
            layer=FundingLayer.MunicipalReserves,
            name="Municipal Reserves",
            floor=0,
            ceiling=50,           # First $50M
            capacity=50,
            disbursement_days=3,   # Immediate local access
            disbursement_std=1,
            annual_premium_rate=0.0,  # Self-funded
            admin_cost_rate=0.02,
            availability_probability=0.95,  # May be depleted
        ),
        LayerConfiguration(
            layer=FundingLayer.StateRiskPool,
            name="State Risk Pool",
            floor=50,
            ceiling=250,          # Next $200M
            capacity=200,
            disbursement_days=7,   # State-level processing
            disbursement_std=3,
            annual_premium_rate=0.015,
            admin_cost_rate=0.03,
            availability_probability=0.92,
        ),
        LayerConfiguration(
            layer=FundingLayer.CatBonds,
            name="Cat Bonds",
            floor=250,
            ceiling=1000,         # $200M-$1B range
            capacity=750,
            disbursement_days=3,   # Parametric trigger = fast payout
            disbursement_std=1,    # 72-hour target from paper
            annual_premium_rate=0.045,  # ~4.5% coupon
            admin_cost_rate=0.02,
            availability_probability=0.98,  # Contractually committed
        ),
        LayerConfiguration(
            layer=FundingLayer.Reinsurance,
            name="Reinsurance",
            floor=1000,
            ceiling=5000,         # $1B-$5B range
            capacity=4000,
            disbursement_days=14,  # Reinsurance claim process
            disbursement_std=7,
            annual_premium_rate=0.06,
            admin_cost_rate=0.04,
            availability_probability=0.90,  # Market conditions vary
        ),
        LayerConfiguration(
            layer=FundingLayer.FederalBackstop,
            name="Federal Backstop",
            floor=5000,
            ceiling=float('inf'), # Unlimited (crisis-level)
            capacity=float('inf'),
            disbursement_days=21,  # FEMA PA average from paper
            disbursement_std=14,
            annual_premium_rate=0.0,  # Taxpayer funded
            admin_cost_rate=0.08,
            availability_probability=0.85,  # Subject to appropriations
        ),
    ]
    
    def __init__(
        self, 
        layers: Optional[List[LayerConfiguration]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the funding waterfall.
        
        Args:
            layers: Custom layer configurations, or use defaults
            seed: Random seed for stochastic elements
        """
        self.layers = layers if layers else self.DEFAULT_LAYERS.copy()
        self.layers.sort(key=lambda x: x.floor)  # Ensure proper order
        self.rng = np.random.default_rng(seed)
        
        # Track cumulative state across events
        self.total_disbursed = {layer.layer: 0.0 for layer in self.layers}
        self.events_processed = 0
        
    def process_event(self, event: DisasterEvent) -> WaterfallResult:
        """
        Process a disaster event through the funding waterfall.
        
        Args:
            event: The disaster event to process
            
        Returns:
            WaterfallResult with complete funding breakdown
        """
        loss = event.economic_loss
        disbursements = []
        layer_utilization = {}
        cumulative_covered = 0.0
        
        for layer_config in self.layers:
            # Calculate coverage from this layer
            coverage = layer_config.calculate_coverage(loss, cumulative_covered)
            
            if coverage > 0:
                # Check availability (stochastic)
                is_available = self.rng.random() < layer_config.availability_probability
                
                # Calculate disbursement timing
                days = max(1, self.rng.normal(
                    layer_config.disbursement_days,
                    layer_config.disbursement_std
                ))
                
                if is_available:
                    actual_coverage = coverage
                    gap = 0.0
                else:
                    actual_coverage = 0.0
                    gap = coverage
                
                disbursement = DisbursementEvent(
                    layer=layer_config.layer,
                    amount=actual_coverage,
                    event_id=event.event_id,
                    days_to_disburse=days,
                    was_available=is_available,
                    gap_amount=gap
                )
                disbursements.append(disbursement)
                
                cumulative_covered += actual_coverage
                layer_utilization[layer_config.layer] = coverage
                self.total_disbursed[layer_config.layer] += actual_coverage
        
        # Calculate weighted average disbursement time
        if disbursements:
            total_weighted_days = sum(
                d.amount * d.days_to_disburse 
                for d in disbursements if d.was_available
            )
            total_amount = sum(d.amount for d in disbursements if d.was_available)
            avg_days = total_weighted_days / total_amount if total_amount > 0 else 0
        else:
            avg_days = 0
        
        self.events_processed += 1
        
        return WaterfallResult(
            event=event,
            total_loss=loss,
            total_covered=cumulative_covered,
            coverage_gap=loss - cumulative_covered,
            disbursements=disbursements,
            layer_utilization=layer_utilization,
            weighted_avg_disbursement_days=avg_days
        )
    
    def process_year(
        self, 
        events: List[DisasterEvent]
    ) -> List[WaterfallResult]:
        """Process all events in a year through the waterfall."""
        return [self.process_event(event) for event in events]
    
    def calculate_annual_premiums(self) -> Dict[FundingLayer, float]:
        """Calculate total annual premium cost for each layer."""
        premiums = {}
        for layer in self.layers:
            if layer.capacity != float('inf'):
                premiums[layer.layer] = layer.capacity * layer.annual_premium_rate
            else:
                premiums[layer.layer] = 0.0  # Federal backstop not premium-based
        return premiums
    
    def get_layer_summary(self) -> Dict[str, Dict]:
        """Get summary statistics for each layer."""
        summary = {}
        for layer in self.layers:
            summary[layer.name] = {
                "floor": layer.floor,
                "ceiling": layer.ceiling if layer.ceiling != float('inf') else "Unlimited",
                "capacity": layer.capacity if layer.capacity != float('inf') else "Unlimited",
                "disbursement_days": layer.disbursement_days,
                "premium_rate": f"{layer.annual_premium_rate*100:.1f}%",
                "total_disbursed": self.total_disbursed[layer.layer],
            }
        return summary
    
    def reset_tracking(self):
        """Reset cumulative tracking statistics."""
        self.total_disbursed = {layer.layer: 0.0 for layer in self.layers}
        self.events_processed = 0


class TraditionalFEMAModel:
    """
    Comparison model representing current FEMA-centric funding.
    
    Key characteristics from the paper:
    - Public Assistance average: 21 days to disburse
    - Subject to annual appropriations
    - Post-disaster political negotiations
    - Limited to Layer 1 (municipal) and Layer 5 (federal) effectively
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        
        # Traditional model has gaps in middle layers
        self.layers = [
            LayerConfiguration(
                layer=FundingLayer.MunicipalReserves,
                name="Municipal Reserves",
                floor=0,
                ceiling=50,
                capacity=50,
                disbursement_days=3,
                disbursement_std=1,
                availability_probability=0.95,
            ),
            # Gap: No state pools, cat bonds, or reinsurance in traditional model
            LayerConfiguration(
                layer=FundingLayer.FederalBackstop,
                name="FEMA/Federal Appropriations",
                floor=50,  # Everything above municipal goes to federal
                ceiling=float('inf'),
                capacity=float('inf'),
                disbursement_days=21,  # FEMA PA average from paper
                disbursement_std=14,
                availability_probability=0.80,  # Subject to appropriations/politics
            ),
        ]
        
        self.total_disbursed = {layer.layer: 0.0 for layer in self.layers}
        self.events_processed = 0
    
    def process_event(self, event: DisasterEvent) -> WaterfallResult:
        """Process event through traditional FEMA model."""
        loss = event.economic_loss
        disbursements = []
        layer_utilization = {}
        cumulative_covered = 0.0
        
        for layer_config in self.layers:
            coverage = layer_config.calculate_coverage(loss, cumulative_covered)
            
            if coverage > 0:
                is_available = self.rng.random() < layer_config.availability_probability
                days = max(1, self.rng.normal(
                    layer_config.disbursement_days,
                    layer_config.disbursement_std
                ))
                
                actual_coverage = coverage if is_available else 0.0
                gap = 0.0 if is_available else coverage
                
                disbursement = DisbursementEvent(
                    layer=layer_config.layer,
                    amount=actual_coverage,
                    event_id=event.event_id,
                    days_to_disburse=days,
                    was_available=is_available,
                    gap_amount=gap
                )
                disbursements.append(disbursement)
                
                cumulative_covered += actual_coverage
                layer_utilization[layer_config.layer] = coverage
                self.total_disbursed[layer_config.layer] += actual_coverage
        
        # Calculate weighted average time
        if disbursements:
            total_weighted_days = sum(
                d.amount * d.days_to_disburse 
                for d in disbursements if d.was_available
            )
            total_amount = sum(d.amount for d in disbursements if d.was_available)
            avg_days = total_weighted_days / total_amount if total_amount > 0 else 0
        else:
            avg_days = 0
            
        self.events_processed += 1
        
        return WaterfallResult(
            event=event,
            total_loss=loss,
            total_covered=cumulative_covered,
            coverage_gap=loss - cumulative_covered,
            disbursements=disbursements,
            layer_utilization=layer_utilization,
            weighted_avg_disbursement_days=avg_days
        )


def compare_models(
    events: List[DisasterEvent],
    seed: Optional[int] = None
) -> Tuple[List[WaterfallResult], List[WaterfallResult]]:
    """
    Compare proposed market-based model vs traditional FEMA model.
    
    Returns:
        Tuple of (market_results, traditional_results)
    """
    market_model = FundingWaterfall(seed=seed)
    fema_model = TraditionalFEMAModel(seed=seed)
    
    market_results = [market_model.process_event(e) for e in events]
    traditional_results = [fema_model.process_event(e) for e in events]
    
    return market_results, traditional_results


if __name__ == "__main__":
    # Quick test
    from disaster_generator import DisasterEventGenerator, PRESET_PROFILES
    
    gen = DisasterEventGenerator(seed=42)
    profile = PRESET_PROFILES["gulf_coast"]
    
    # Generate events for one year
    events = gen.generate_annual_events(profile, year=2025)
    
    print(f"Generated {len(events)} events")
    
    # Process through both models
    market_results, fema_results = compare_models(events, seed=42)
    
    print("\n=== Market-Based Model ===")
    for r in market_results[:3]:
        print(f"Event {r.event.event_id}: ${r.total_loss:,.0f}M loss, "
              f"${r.total_covered:,.0f}M covered, {r.weighted_avg_disbursement_days:.1f} days")
    
    print("\n=== Traditional FEMA Model ===")
    for r in fema_results[:3]:
        print(f"Event {r.event.event_id}: ${r.total_loss:,.0f}M loss, "
              f"${r.total_covered:,.0f}M covered, {r.weighted_avg_disbursement_days:.1f} days")
