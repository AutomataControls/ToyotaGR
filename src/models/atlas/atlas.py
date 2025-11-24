"""
ATLAS - Spatial Track Intelligence Model for Toyota GR Cup Series
Real-Time Analytics for Race Engineering - Track Position & Racing Line Expert

Specializes in:
- Optimal racing line analysis
- Track position optimization
- Overtaking opportunity detection
- Corner entry/exit strategy
- Track limits monitoring
- Spatial positioning relative to other cars

Processes Toyota GR Cup telemetry to provide spatial intelligence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math


class SpatialTelemetryEncoder(nn.Module):
    """Encodes Toyota GR Cup telemetry for spatial analysis"""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 256):
        super().__init__()
        
        # Spatial signal processing - focus on position-related telemetry
        # Input: [speed, throttle, brake_f, brake_r, gear, steering, accx, accy]
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Track-specific spatial embeddings
        self.track_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, telemetry_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            telemetry_sequence: (batch, sequence_len, 8) - time series telemetry
        Returns:
            spatial_features: (batch, sequence_len, hidden_dim)
        """
        # Encode spatial telemetry signals
        encoded = self.spatial_encoder(telemetry_sequence)
        
        # Add track-specific spatial context
        spatial_features = self.track_embedding(encoded)
        
        return spatial_features


class SpatialAttention(nn.Module):
    """Multi-head attention for spatial pattern recognition"""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Position-aware attention weights
        self.position_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            output: (batch, seq_len, hidden_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        attended, attn_weights = self.spatial_attention(x, x, x)
        
        # Apply position bias for spatial awareness
        if attn_weights.shape[-2:] == self.position_bias.shape[-2:]:
            attn_weights = attn_weights + self.position_bias
        
        output = self.norm(attended + x)  # Residual connection
        
        return output, attn_weights


class RacingLineAnalyzer(nn.Module):
    """Analyzes optimal racing line and track positioning"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Racing line optimization
        self.racing_line_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # [current_line_quality, optimal_line_deviation, corner_entry, corner_apex, corner_exit, straight_line]
        )
        
        # Track position analysis
        self.position_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # [track_position, racing_line_adherence, track_limits_risk, position_quality]
        )
        
        # Corner analysis
        self.corner_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # [corner_speed, braking_point, turn_in_point, apex_speed, exit_acceleration]
        )
        
    def forward(self, spatial_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze racing line and track positioning"""
        racing_line = self.racing_line_analyzer(spatial_features)
        position = self.position_analyzer(spatial_features)
        corner_analysis = self.corner_analyzer(spatial_features)
        
        return {
            'racing_line_analysis': racing_line,
            'track_position': position,
            'corner_analysis': corner_analysis
        }


class OvertakingAnalyzer(nn.Module):
    """Analyzes overtaking opportunities and defensive positioning"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Overtaking opportunity detection
        self.overtaking_detector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # [overtake_probability, drs_zone_advantage, slipstream_available, track_position_advantage]
        )
        
        # Defensive positioning
        self.defensive_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # [defensive_line, block_effectiveness, position_compromise]
        )
        
        # Gap analysis
        self.gap_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [gap_closing_rate, time_to_contact]
        )
        
    def forward(self, spatial_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze overtaking opportunities and defensive positions"""
        overtaking = torch.sigmoid(self.overtaking_detector(spatial_features))
        defensive = torch.sigmoid(self.defensive_analyzer(spatial_features))
        gap_analysis = self.gap_analyzer(spatial_features)
        
        return {
            'overtaking_analysis': overtaking,
            'defensive_analysis': defensive,
            'gap_analysis': gap_analysis
        }


class TrackLimitsMonitor(nn.Module):
    """Monitors track limits and spatial boundaries"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Track limits detection
        self.limits_detector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # [track_limits_risk, kerb_usage, off_track_probability]
        )
        
        # Spatial boundaries
        self.boundary_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [boundary_distance, safety_margin]
        )
        
    def forward(self, spatial_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Monitor track limits and spatial boundaries"""
        limits = torch.sigmoid(self.limits_detector(spatial_features))
        boundaries = self.boundary_analyzer(spatial_features)
        
        return {
            'track_limits': limits,
            'spatial_boundaries': boundaries
        }


class AtlasRacingModel(nn.Module):
    """
    ATLAS - Spatial Track Intelligence for Toyota GR Cup Series
    
    Processes telemetry sequences to provide spatial intelligence:
    - Optimal racing line analysis
    - Track positioning optimization
    - Overtaking opportunity detection
    - Corner entry/exit strategy
    - Track limits monitoring
    """
    
    def __init__(
        self,
        sequence_length: int = 300,  # 5 minutes at 60Hz telemetry
        telemetry_dim: int = 8,      # 8 main telemetry signals
        hidden_dim: int = 256,
        num_attention_heads: int = 8,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Core spatial components
        self.telemetry_encoder = SpatialTelemetryEncoder(telemetry_dim, hidden_dim)
        
        # Multi-layer spatial attention
        self.attention_layers = nn.ModuleList([
            SpatialAttention(hidden_dim, num_attention_heads)
            for _ in range(num_layers)
        ])
        
        # Global spatial feature aggregation
        self.spatial_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Spatial analysis modules
        self.racing_line_analyzer = RacingLineAnalyzer(hidden_dim)
        self.overtaking_analyzer = OvertakingAnalyzer(hidden_dim)
        self.track_limits_monitor = TrackLimitsMonitor(hidden_dim)
        
        # Spatial memory for track learning
        self.track_memory = nn.Parameter(torch.randn(100, hidden_dim) * 0.02)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Spatial quality score
        self.quality_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        telemetry_sequence: torch.Tensor,
        track_context: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            telemetry_sequence: (batch, seq_len, 8) - Toyota GR Cup telemetry
                [speed, throttle, brake_f, brake_r, gear, steering, accx, accy]
            track_context: Optional track context (sector, weather, etc.)
        
        Returns:
            spatial_output: Dictionary containing all spatial analysis
        """
        batch_size, seq_len, _ = telemetry_sequence.shape
        
        # Encode telemetry into spatial features
        encoded_features = self.telemetry_encoder(telemetry_sequence)
        
        # Apply spatial attention layers
        attention_weights = []
        x = encoded_features
        
        for attention_layer in self.attention_layers:
            x, attn = attention_layer(x)
            attention_weights.append(attn)
        
        # Global spatial state (aggregate sequence information)
        global_features = torch.mean(x, dim=1)  # (batch, hidden_dim)
        global_spatial_state = self.spatial_aggregator(global_features)
        
        # Track memory integration
        memory_features, memory_attn = self.memory_attention(
            global_spatial_state.unsqueeze(1),  # (batch, 1, hidden_dim)
            self.track_memory.unsqueeze(0).expand(batch_size, -1, -1),  # (batch, 100, hidden_dim)
            self.track_memory.unsqueeze(0).expand(batch_size, -1, -1)
        )
        enhanced_spatial_state = global_spatial_state + memory_features.squeeze(1)
        
        # Spatial analysis
        racing_line_analysis = self.racing_line_analyzer(enhanced_spatial_state)
        overtaking_analysis = self.overtaking_analyzer(enhanced_spatial_state)
        track_limits_analysis = self.track_limits_monitor(enhanced_spatial_state)
        
        # Confidence and quality estimation
        confidence = self.confidence_estimator(enhanced_spatial_state)
        spatial_quality = self.quality_estimator(enhanced_spatial_state)
        
        # Comprehensive spatial output
        output = {
            # Core spatial analysis
            **racing_line_analysis,
            **overtaking_analysis,
            **track_limits_analysis,
            
            # Model insights
            'spatial_features': x,  # (batch, seq_len, hidden_dim)
            'global_spatial_state': enhanced_spatial_state,  # (batch, hidden_dim)
            'attention_weights': attention_weights,  # List of attention matrices
            'memory_attention': memory_attn,  # Track memory attention
            'confidence': confidence,  # (batch, 1)
            'spatial_quality': spatial_quality,  # (batch, 1)
            
            # Spatial expertise indicators
            'racing_line_quality': self._assess_racing_line_quality(racing_line_analysis),
            'overtaking_opportunity': self._detect_overtaking_opportunity(overtaking_analysis),
            'track_limits_warning': self._assess_track_limits_risk(track_limits_analysis),
            'spatial_advantage': self._calculate_spatial_advantage(racing_line_analysis, overtaking_analysis)
        }
        
        return output
    
    def _assess_racing_line_quality(self, racing_line_analysis: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Assess overall racing line quality"""
        racing_line = racing_line_analysis['racing_line_analysis']
        # racing_line: [current_line_quality, optimal_line_deviation, corner_entry, corner_apex, corner_exit, straight_line]
        line_quality = racing_line[:, 0]  # Current line quality
        deviation_penalty = 1.0 - torch.abs(racing_line[:, 1])  # Less deviation = better
        overall_quality = (line_quality + deviation_penalty) / 2.0
        return overall_quality
    
    def _detect_overtaking_opportunity(self, overtaking_analysis: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Detect strong overtaking opportunities"""
        overtaking = overtaking_analysis['overtaking_analysis']
        # overtaking: [overtake_probability, drs_zone_advantage, slipstream_available, track_position_advantage]
        opportunity_score = torch.mean(overtaking, dim=1)  # Average all factors
        return (opportunity_score > 0.7).float()  # Strong opportunity threshold
    
    def _assess_track_limits_risk(self, track_limits_analysis: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Assess track limits violation risk"""
        limits = track_limits_analysis['track_limits']
        # limits: [track_limits_risk, kerb_usage, off_track_probability]
        risk_score = torch.max(limits, dim=1)[0]  # Take highest risk factor
        return (risk_score > 0.6).float()  # High risk threshold
    
    def _calculate_spatial_advantage(
        self,
        racing_line_analysis: Dict[str, torch.Tensor],
        overtaking_analysis: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate overall spatial advantage"""
        line_quality = self._assess_racing_line_quality(racing_line_analysis)
        overtake_opportunity = self._detect_overtaking_opportunity(overtaking_analysis)
        
        # Combined spatial advantage
        advantage = (line_quality + overtake_opportunity) / 2.0
        return advantage
    
    def get_spatial_recommendations(self, output: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """Convert model output to human-readable spatial recommendations"""
        recommendations = {}
        
        # Racing line recommendation
        racing_line = output['racing_line_analysis'][0]  # First batch item
        line_quality = racing_line[0].item()
        deviation = racing_line[1].item()
        
        if line_quality > 0.8:
            recommendations['racing_line'] = f"EXCELLENT line quality: {line_quality:.2f}"
        elif line_quality > 0.6:
            recommendations['racing_line'] = f"GOOD line quality: {line_quality:.2f}, deviation: {abs(deviation):.2f}"
        else:
            recommendations['racing_line'] = f"POOR line quality: {line_quality:.2f}, adjust racing line"
        
        # Overtaking recommendation
        overtaking = output['overtaking_analysis'][0]
        overtake_prob = overtaking[0].item()
        
        if overtake_prob > 0.7:
            recommendations['overtaking'] = f"STRONG overtaking opportunity: {overtake_prob:.2f}"
        elif overtake_prob > 0.4:
            recommendations['overtaking'] = f"MODERATE overtaking chance: {overtake_prob:.2f}"
        else:
            recommendations['overtaking'] = f"LOW overtaking opportunity: {overtake_prob:.2f}"
        
        # Track limits warning
        limits = output['track_limits'][0]
        limits_risk = limits[0].item()
        kerb_usage = limits[1].item()
        
        if limits_risk > 0.7:
            recommendations['track_limits'] = f"HIGH track limits risk: {limits_risk:.2f}"
        elif kerb_usage > 0.6:
            recommendations['track_limits'] = f"MODERATE kerb usage: {kerb_usage:.2f}"
        else:
            recommendations['track_limits'] = f"SAFE track position: {limits_risk:.2f}"
        
        # Corner analysis
        corner = output['corner_analysis'][0]
        corner_speed = corner[0].item()
        braking_point = corner[1].item()
        
        if corner_speed > 0.8:
            recommendations['corner_performance'] = f"OPTIMAL corner speed: {corner_speed:.2f}"
        else:
            recommendations['corner_performance'] = f"Corner speed improvement possible: {corner_speed:.2f}, braking: {braking_point:.2f}"
        
        return recommendations


def create_atlas_model(device: str = 'cuda') -> AtlasRacingModel:
    """Create and initialize ATLAS spatial intelligence model"""
    model = AtlasRacingModel(
        sequence_length=300,    # 5 minutes of telemetry
        telemetry_dim=8,       # 8 main telemetry signals
        hidden_dim=256,        # Spatial feature dimension
        num_attention_heads=8, # Multi-head attention
        num_layers=4          # Spatial attention layers
    )
    
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test ATLAS model creation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_atlas_model(device)
    
    # Test with sample telemetry data
    batch_size = 2
    seq_length = 300
    telemetry_dim = 8
    
    sample_telemetry = torch.randn(batch_size, seq_length, telemetry_dim).to(device)
    
    with torch.no_grad():
        output = model(sample_telemetry)
        print(f"ATLAS Model loaded successfully!")
        print(f"Spatial output keys: {list(output.keys())}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Get spatial recommendations
        recommendations = model.get_spatial_recommendations(output)
        print("\nSample Spatial Recommendations:")
        for key, value in recommendations.items():
            print(f"  {key}: {value}")