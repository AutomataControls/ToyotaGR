"""
OLYMPUS - Ensemble Racing Intelligence System for Toyota GR Cup Series
Real-Time Analytics for Race Engineering - Master Coordination System

Combines all 5 specialist AI models:
- MINERVA: Strategic decisions (pit strategy, tire management)
- ATLAS: Spatial intelligence (racing line, positioning)
- IRIS: Vehicle dynamics (throttle, brake, balance)
- CHRONOS: Timing analysis (lap times, pace)
- PROMETHEUS: Predictive analytics (forecasting, trends)

Loads pre-trained models from {model_name}_best.pt and coordinates their outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import os
import sys
from pathlib import Path

# Import all specialist models
try:
    # Add model paths for imports
    current_dir = Path(__file__).parent.parent
    sys.path.append(str(current_dir))
    
    from minerva.minerva import MinervaRacingModel, create_minerva_model
    from atlas.atlas import AtlasRacingModel, create_atlas_model
    from iris.iris import IrisRacingModel, create_iris_model
    from chronos.chronos import ChronosRacingModel, create_chronos_model
    from prometheus.prometheus import PrometheusRacingModel, create_prometheus_model
    
except ImportError as e:
    print(f"Warning: Could not import all specialist models: {e}")
    print("Ensure all model files are present in their respective directories")


class EnsembleCoordinator(nn.Module):
    """Coordinates outputs from all 5 specialist models"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Specialist model feature dimensions (all use 256)
        self.specialist_dim = hidden_dim
        
        # Cross-attention between specialists
        self.minerva_atlas_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.iris_chronos_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.prometheus_integration = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        
        # Ensemble fusion network
        self.ensemble_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),  # 5 specialists -> 2x hidden
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Decision coordination network
        self.decision_coordinator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Coordinated decision features
        )
        
        # Confidence weighting for each specialist
        self.specialist_weights = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # One weight per specialist
            nn.Softmax(dim=-1)
        )
        
        # Master recommendation system
        self.master_recommendations = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # [priority_action, confidence, urgency, risk_level, optimization_potential, strategic_advantage, immediate_action_required, long_term_benefit, coordination_quality, overall_performance]
        )
        
    def forward(self, specialist_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Coordinate outputs from all specialist models
        
        Args:
            specialist_outputs: Dictionary containing global states from each specialist
                - 'minerva': (batch, hidden_dim)
                - 'atlas': (batch, hidden_dim)
                - 'iris': (batch, hidden_dim)
                - 'chronos': (batch, hidden_dim)
                - 'prometheus': (batch, hidden_dim)
        
        Returns:
            coordinated_output: Master coordination results
        """
        # Extract specialist global states
        minerva_state = specialist_outputs['minerva']    # Strategic
        atlas_state = specialist_outputs['atlas']        # Spatial
        iris_state = specialist_outputs['iris']          # Dynamics
        chronos_state = specialist_outputs['chronos']    # Timing
        prometheus_state = specialist_outputs['prometheus']  # Predictive
        
        batch_size = minerva_state.shape[0]
        
        # Cross-attention coordination between related specialists
        # Strategic-Spatial coordination (pit strategy + track position)
        minerva_atlas_coord, _ = self.minerva_atlas_attention(
            minerva_state.unsqueeze(1),  # Query: strategy
            atlas_state.unsqueeze(1),    # Key/Value: spatial
            atlas_state.unsqueeze(1)
        )
        
        # Dynamics-Timing coordination (vehicle performance + lap times)
        iris_chronos_coord, _ = self.iris_chronos_attention(
            iris_state.unsqueeze(1),     # Query: dynamics
            chronos_state.unsqueeze(1),  # Key/Value: timing
            chronos_state.unsqueeze(1)
        )
        
        # Predictive integration (forecasts inform all decisions)
        prometheus_coord, _ = self.prometheus_integration(
            prometheus_state.unsqueeze(1),  # Query: predictions
            torch.stack([minerva_state, atlas_state, iris_state, chronos_state], dim=1),  # Key/Value: all others
            torch.stack([minerva_state, atlas_state, iris_state, chronos_state], dim=1)
        )
        
        # Prepare coordinated states
        coord_minerva = minerva_atlas_coord.squeeze(1)
        coord_atlas = atlas_state  # Atlas provides spatial foundation
        coord_iris = iris_chronos_coord.squeeze(1)
        coord_chronos = chronos_state  # Chronos provides timing foundation
        coord_prometheus = prometheus_coord.squeeze(1)
        
        # Ensemble fusion - combine all specialist insights
        ensemble_input = torch.cat([
            coord_minerva, coord_atlas, coord_iris, coord_chronos, coord_prometheus
        ], dim=-1)  # (batch, hidden_dim * 5)
        
        fused_representation = self.ensemble_fusion(ensemble_input)
        
        # Calculate specialist confidence weights
        specialist_confidences = self.specialist_weights(fused_representation)
        
        # Coordinated decision making
        coordinated_decisions = self.decision_coordinator(fused_representation)
        
        # Master recommendations
        master_recommendations = self.master_recommendations(fused_representation)
        
        return {
            'ensemble_state': fused_representation,
            'coordinated_decisions': coordinated_decisions,
            'specialist_weights': specialist_confidences,
            'master_recommendations': master_recommendations,
            'coordination_quality': torch.sigmoid(master_recommendations[:, 8]),  # Coordination quality score
            'cross_attention': {
                'minerva_atlas': coord_minerva,
                'iris_chronos': coord_iris,
                'prometheus_integration': coord_prometheus
            }
        }


class OlympusEnsembleModel(nn.Module):
    """
    OLYMPUS - Master Racing Intelligence Ensemble for Toyota GR Cup Series
    
    Loads and coordinates all 5 specialist models:
    - Loads pre-trained weights from {model_name}_best.pt files
    - Coordinates specialist outputs for unified decision making
    - Provides master race engineering recommendations
    """
    
    def __init__(
        self,
        models_dir: str = "./BestModels",
        sequence_length: int = 300,
        telemetry_dim: int = 8,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.models_dir = Path(models_dir)
        self.device = device
        self.sequence_length = sequence_length
        self.telemetry_dim = telemetry_dim
        
        # Initialize all specialist models
        print("🏛️ Initializing OLYMPUS Racing Intelligence Ensemble...")
        
        # Create specialist models
        self.minerva = create_minerva_model(device)
        self.atlas = create_atlas_model(device)
        self.iris = create_iris_model(device)
        self.chronos = create_chronos_model(device)
        self.prometheus = create_prometheus_model(device)
        
        # Load pre-trained weights
        self._load_specialist_weights()
        
        # Set specialists to evaluation mode
        self.minerva.eval()
        self.atlas.eval()
        self.iris.eval()
        self.chronos.eval()
        self.prometheus.eval()
        
        # Ensemble coordination system
        self.coordinator = EnsembleCoordinator(hidden_dim=256)
        
        # Master performance assessment
        self.performance_assessor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # [overall_performance, strategic_score, spatial_score, dynamics_score, timing_score, predictive_score, coordination_score, optimization_potential]
        )
        
        print(f"✅ OLYMPUS Ensemble initialized with {self._count_parameters():,} total parameters")
    
    def _load_specialist_weights(self):
        """Load pre-trained weights for all specialist models"""
        specialist_files = {
            'minerva': 'minerva_best.pt',
            'atlas': 'atlas_best.pt',
            'iris': 'iris_best.pt',
            'chronos': 'chronos_best.pt',
            'prometheus': 'prometheus_best.pt'
        }
        
        specialists = {
            'minerva': self.minerva,
            'atlas': self.atlas,
            'iris': self.iris,
            'chronos': self.chronos,
            'prometheus': self.prometheus
        }
        
        loaded_count = 0
        
        for name, filename in specialist_files.items():
            model_path = self.models_dir / filename
            
            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    
                    # Handle different checkpoint formats
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Load weights into specialist model
                    specialists[name].load_state_dict(state_dict, strict=False)
                    print(f"✅ Loaded {name.upper()} weights from {filename}")
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Failed to load {name.upper()} weights: {e}")
            else:
                print(f"⚠️ {name.upper()} weights not found: {model_path}")
        
        if loaded_count == 0:
            print("🆕 No pre-trained weights found - using fresh initialization")
        else:
            print(f"📊 Loaded weights for {loaded_count}/5 specialists")
    
    def _count_parameters(self) -> int:
        """Count total parameters across all models"""
        total = 0
        total += sum(p.numel() for p in self.minerva.parameters())
        total += sum(p.numel() for p in self.atlas.parameters())
        total += sum(p.numel() for p in self.iris.parameters())
        total += sum(p.numel() for p in self.chronos.parameters())
        total += sum(p.numel() for p in self.prometheus.parameters())
        total += sum(p.numel() for p in self.coordinator.parameters())
        total += sum(p.numel() for p in self.performance_assessor.parameters())
        return total
    
    def forward(
        self,
        telemetry_sequence: torch.Tensor,
        race_context: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Master forward pass coordinating all specialists
        
        Args:
            telemetry_sequence: (batch, seq_len, 8) - Toyota GR Cup telemetry
            race_context: Optional race context information
        
        Returns:
            olympus_output: Comprehensive racing intelligence output
        """
        batch_size, seq_len, _ = telemetry_sequence.shape
        
        # Run all specialist models
        with torch.no_grad():  # Specialists are frozen during ensemble operation
            minerva_output = self.minerva(telemetry_sequence, race_context)
            atlas_output = self.atlas(telemetry_sequence, race_context)
            iris_output = self.iris(telemetry_sequence, race_context)
            chronos_output = self.chronos(telemetry_sequence, race_context)
            prometheus_output = self.prometheus(telemetry_sequence, race_context)
        
        # Extract global states for coordination
        specialist_states = {
            'minerva': minerva_output['global_strategic_state'],
            'atlas': atlas_output['global_spatial_state'],
            'iris': iris_output['global_dynamics_state'],
            'chronos': chronos_output['global_timing_state'],
            'prometheus': prometheus_output['global_predictive_state']
        }
        
        # Coordinate specialist outputs
        coordination_output = self.coordinator(specialist_states)
        
        # Master performance assessment
        performance_scores = torch.sigmoid(self.performance_assessor(coordination_output['ensemble_state']))
        
        # Comprehensive OLYMPUS output
        olympus_output = {
            # Specialist outputs (full context)
            'specialist_outputs': {
                'minerva': minerva_output,
                'atlas': atlas_output,
                'iris': iris_output,
                'chronos': chronos_output,
                'prometheus': prometheus_output
            },
            
            # Coordination results
            'ensemble_coordination': coordination_output,
            'performance_assessment': performance_scores,
            
            # Master recommendations
            'master_recommendations': self._generate_master_recommendations(
                coordination_output, performance_scores, minerva_output, atlas_output, 
                iris_output, chronos_output, prometheus_output
            ),
            
            # Key performance indicators
            'kpi_summary': self._calculate_kpis(
                minerva_output, atlas_output, iris_output, chronos_output, prometheus_output, performance_scores
            ),
            
            # Real-time alerts
            'alerts': self._generate_alerts(
                minerva_output, atlas_output, iris_output, chronos_output, prometheus_output, coordination_output
            )
        }
        
        return olympus_output
    
    def _generate_master_recommendations(
        self, 
        coordination_output: Dict[str, torch.Tensor],
        performance_scores: torch.Tensor,
        minerva_out: Dict, atlas_out: Dict, iris_out: Dict, chronos_out: Dict, prometheus_out: Dict
    ) -> Dict[str, str]:
        """Generate master race engineering recommendations"""
        
        recommendations = {}
        
        # Extract key metrics (first batch item)
        master_rec = coordination_output['master_recommendations'][0]
        perf_scores = performance_scores[0]
        specialist_weights = coordination_output['specialist_weights'][0]
        
        # Priority action based on specialist consensus
        priority_action = master_rec[0].item()
        confidence = master_rec[1].item()
        urgency = master_rec[2].item()
        risk_level = master_rec[3].item()
        
        # Determine primary focus area
        dominant_specialist = torch.argmax(specialist_weights).item()
        specialist_names = ['MINERVA (Strategy)', 'ATLAS (Spatial)', 'IRIS (Dynamics)', 'CHRONOS (Timing)', 'PROMETHEUS (Predictive)']
        
        recommendations['primary_focus'] = f"{specialist_names[dominant_specialist]} - confidence: {specialist_weights[dominant_specialist]:.2f}"
        
        # Strategic recommendations (MINERVA)
        if hasattr(minerva_out, 'get_strategic_recommendations'):
            strategic_recs = minerva_out.get('strategic_recommendations', {})
        else:
            strategic_recs = {'pit_strategy': 'Available', 'pace_strategy': 'Available'}
        
        # Spatial recommendations (ATLAS) 
        if hasattr(atlas_out, 'get_spatial_recommendations'):
            spatial_recs = atlas_out.get('spatial_recommendations', {})
        else:
            spatial_recs = {'racing_line': 'Available', 'overtaking': 'Available'}
        
        # Priority recommendations based on urgency and risk
        if urgency > 0.8:
            recommendations['urgent_action'] = f"HIGH PRIORITY: Action required (urgency: {urgency:.2f})"
        elif risk_level > 0.7:
            recommendations['risk_warning'] = f"HIGH RISK detected (risk: {risk_level:.2f})"
        else:
            recommendations['status'] = f"Normal operation (confidence: {confidence:.2f})"
        
        # Performance optimization
        optimization_potential = master_rec[4].item()
        if optimization_potential > 0.7:
            recommendations['optimization'] = f"HIGH optimization potential: {optimization_potential:.2f}"
        
        # Coordination quality
        coordination_quality = coordination_output['coordination_quality'][0].item()
        recommendations['coordination'] = f"Specialist coordination: {coordination_quality:.2f}"
        
        return recommendations
    
    def _calculate_kpis(
        self, 
        minerva_out: Dict, atlas_out: Dict, iris_out: Dict, chronos_out: Dict, prometheus_out: Dict,
        performance_scores: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate key performance indicators"""
        
        kpis = {}
        perf = performance_scores[0]  # First batch item
        
        # Overall performance metrics
        kpis['overall_performance'] = perf[0].item()
        kpis['strategic_score'] = perf[1].item()
        kpis['spatial_score'] = perf[2].item()
        kpis['dynamics_score'] = perf[3].item()
        kpis['timing_score'] = perf[4].item()
        kpis['predictive_score'] = perf[5].item()
        kpis['coordination_score'] = perf[6].item()
        kpis['optimization_potential'] = perf[7].item()
        
        # Extract specialist-specific KPIs
        if 'confidence' in minerva_out:
            kpis['strategic_confidence'] = minerva_out['confidence'][0].item()
        
        if 'spatial_quality' in atlas_out:
            kpis['spatial_quality'] = atlas_out['spatial_quality'][0].item()
            
        if 'vehicle_stability' in iris_out:
            kpis['vehicle_stability'] = iris_out['vehicle_stability'][0].item()
            
        if 'consistency_score' in chronos_out:
            kpis['timing_consistency'] = chronos_out['consistency_score'][0].item()
            
        if 'prediction_confidence' in prometheus_out:
            kpis['prediction_confidence'] = prometheus_out['prediction_confidence'][0, 0].item()
        
        return kpis
    
    def _generate_alerts(
        self,
        minerva_out: Dict, atlas_out: Dict, iris_out: Dict, chronos_out: Dict, prometheus_out: Dict,
        coordination_output: Dict[str, torch.Tensor]
    ) -> List[Dict[str, Any]]:
        """Generate real-time alerts for race engineering"""
        
        alerts = []
        
        # Strategic alerts (MINERVA)
        if 'pit_window_open' in minerva_out and minerva_out['pit_window_open'][0].item() > 0.8:
            alerts.append({
                'type': 'strategic',
                'priority': 'high',
                'message': 'Optimal pit window detected',
                'specialist': 'MINERVA'
            })
        
        # Spatial alerts (ATLAS)
        if 'track_limits_warning' in atlas_out and atlas_out['track_limits_warning'][0].item() > 0.7:
            alerts.append({
                'type': 'spatial',
                'priority': 'medium',
                'message': 'Track limits risk detected',
                'specialist': 'ATLAS'
            })
        
        # Vehicle dynamics alerts (IRIS)
        if 'vehicle_stability' in iris_out and iris_out['vehicle_stability'][0].item() < 0.4:
            alerts.append({
                'type': 'dynamics',
                'priority': 'high',
                'message': 'Vehicle stability concern',
                'specialist': 'IRIS'
            })
        
        # Timing alerts (CHRONOS)
        if 'timing_improvement' in chronos_out and chronos_out['timing_improvement'][0].item() > 0.8:
            alerts.append({
                'type': 'timing',
                'priority': 'medium',
                'message': 'High improvement potential detected',
                'specialist': 'CHRONOS'
            })
        
        # Predictive alerts (PROMETHEUS)
        if 'tire_change_recommendation' in prometheus_out and prometheus_out['tire_change_recommendation'][0].item() > 0.8:
            alerts.append({
                'type': 'predictive',
                'priority': 'high',
                'message': 'Tire change recommended',
                'specialist': 'PROMETHEUS'
            })
        
        # Coordination alerts
        coord_quality = coordination_output['coordination_quality'][0].item()
        if coord_quality < 0.5:
            alerts.append({
                'type': 'coordination',
                'priority': 'low',
                'message': f'Specialist coordination suboptimal: {coord_quality:.2f}',
                'specialist': 'OLYMPUS'
            })
        
        return alerts
    
    def get_master_summary(self, olympus_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a master summary of all racing intelligence"""
        
        kpis = olympus_output['kpi_summary']
        recommendations = olympus_output['master_recommendations']
        alerts = olympus_output['alerts']
        
        # Count alerts by priority
        alert_counts = {'high': 0, 'medium': 0, 'low': 0}
        for alert in alerts:
            alert_counts[alert['priority']] += 1
        
        summary = {
            'overall_performance': f"{kpis.get('overall_performance', 0):.2f}",
            'primary_focus': recommendations.get('primary_focus', 'Balanced operation'),
            'urgent_actions': [alert['message'] for alert in alerts if alert['priority'] == 'high'],
            'optimization_potential': f"{kpis.get('optimization_potential', 0):.2f}",
            'coordination_quality': f"{kpis.get('coordination_score', 0):.2f}",
            'alert_summary': f"High: {alert_counts['high']}, Medium: {alert_counts['medium']}, Low: {alert_counts['low']}",
            'specialist_performance': {
                'strategic': f"{kpis.get('strategic_score', 0):.2f}",
                'spatial': f"{kpis.get('spatial_score', 0):.2f}",
                'dynamics': f"{kpis.get('dynamics_score', 0):.2f}",
                'timing': f"{kpis.get('timing_score', 0):.2f}",
                'predictive': f"{kpis.get('predictive_score', 0):.2f}"
            }
        }
        
        return summary


def create_olympus_ensemble(models_dir: str = "./BestModels", device: str = 'cuda') -> OlympusEnsembleModel:
    """Create and initialize OLYMPUS ensemble with pre-trained specialist models"""
    ensemble = OlympusEnsembleModel(
        models_dir=models_dir,
        sequence_length=300,
        telemetry_dim=8,
        device=device
    )
    
    return ensemble


if __name__ == "__main__":
    # Test OLYMPUS ensemble creation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create ensemble (will load pre-trained weights if available)
    ensemble = create_olympus_ensemble("./BestModels", device)
    
    # Test with sample telemetry data
    batch_size = 2
    seq_length = 300
    telemetry_dim = 8
    
    sample_telemetry = torch.randn(batch_size, seq_length, telemetry_dim).to(device)
    
    with torch.no_grad():
        output = ensemble(sample_telemetry)
        print(f"\n🏛️ OLYMPUS Ensemble operational!")
        print(f"📊 Total parameters: {ensemble._count_parameters():,}")
        
        # Get master summary
        summary = ensemble.get_master_summary(output)
        print("\n🏁 Master Race Intelligence Summary:")
        print(f"  Overall Performance: {summary['overall_performance']}")
        print(f"  Primary Focus: {summary['primary_focus']}")
        print(f"  Optimization Potential: {summary['optimization_potential']}")
        print(f"  Coordination Quality: {summary['coordination_quality']}")
        print(f"  Alerts: {summary['alert_summary']}")
        
        if summary['urgent_actions']:
            print("🚨 Urgent Actions:")
            for action in summary['urgent_actions']:
                print(f"    - {action}")
        
        print("\n🎯 Specialist Performance:")
        for specialist, score in summary['specialist_performance'].items():
            print(f"    {specialist.capitalize()}: {score}")
            
        print(f"\n✅ OLYMPUS Racing Intelligence System ready for deployment!")