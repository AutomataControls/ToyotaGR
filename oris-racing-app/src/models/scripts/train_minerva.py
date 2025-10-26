#!/usr/bin/env python3
"""
MINERVA V6 Enhanced Racing Training Script
Advanced multi-stage training with progressive curriculum learning
Optimized for A100 GPU with mixed precision and advanced techniques
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import glob
from tqdm import tqdm
import time
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Fix the path to ensure imports work
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import MINERVA components
from minerva.minerva import MinervaV6Enhanced
from minerva.train_racing import MinervaRacingAdapter


# =====================================
# Configuration and Constants
# =====================================

class TrainingConfig:
    """Advanced training configuration for MINERVA V6"""
    # Model configuration
    model_name = "MINERVA V6 Enhanced"
    model_description = "Ultimate Strategic Intelligence Master for Racing"
    
    # Data paths
    data_root = "/content/ToyotaGR/src/data/tracks/tracks"
    tracks = ["Sonoma", "COTA", "Sebring", "Road America", "VIR", "barber"]
    
    # Training stages (progressive curriculum) - reduced learning rates to prevent NaN
    stages = [
        {"name": "foundation", "epochs": 5, "lr": 1e-4, "focus": "basic_patterns"},
        {"name": "strategic", "epochs": 5, "lr": 5e-5, "focus": "strategic_planning"},
        {"name": "advanced", "epochs": 10, "lr": 1e-5, "focus": "complex_scenarios"},
        {"name": "mastery", "epochs": 5, "lr": 5e-6, "focus": "fine_tuning"}
    ]
    
    # A100 optimized settings
    batch_size = 2  # Very small for tiny dataset
    gradient_accumulation = 1  # Keep simple for now
    num_workers = 2  # Reduced for small dataset
    pin_memory = True
    persistent_workers = False  # Disable for small dataset
    
    # Mixed precision settings
    use_amp = True
    amp_dtype = torch.float16
    
    # Optimization settings
    warmup_epochs = 5
    weight_decay = 1e-5
    gradient_clip = 0.5  # Reduced to prevent gradient explosion
    ema_decay = 0.995
    
    # Advanced features
    use_data_augmentation = True
    augmentation_factor = 3
    use_mixup = True
    mixup_alpha = 0.2
    
    # Telemetry channels to use (mapped to actual column names)
    telemetry_channels = [
        'Speed', 'nmot', 'Gear', 'ath', 'pbrake_f', 'pbrake_r',
        'Steering_Angle', 'accy_can', 'accx_can', 
        'VBOX_Long_Minutes', 'VBOX_Lat_Min', 'Laptrigger_lapdist_dls'
    ]
    
    # Column mappings for compatibility
    column_mappings = {
        'speed': 'Speed',
        'rpm': 'nmot',
        'gear': 'Gear', 
        'throttle': 'ath',
        'brake': 'pbrake_f',
        'steer_angle': 'Steering_Angle',
        'lat_g': 'accy_can',
        'long_g': 'accx_can'
    }
    
    # Grid encoding settings
    grid_size = 30  # MINERVA's max grid size
    sequence_length = 128  # Temporal sequence for patterns
    
    # Checkpointing
    checkpoint_dir = "racing_models/checkpoints"
    save_every = 10
    
    # Logging
    log_interval = 1
    val_interval = 5  # Less frequent validation for small dataset


# =====================================
# Racing Dataset with Advanced Features
# =====================================

class MinervaRacingDataset(Dataset):
    """Advanced racing dataset with telemetry processing and augmentation"""
    
    def __init__(self, config: TrainingConfig, stage: str = "train", augment: bool = True):
        self.config = config
        self.stage = stage
        self.augment = augment and stage == "train"
        self.samples = []
        self.telemetry_cache = {}
        
        # Load all telemetry data
        self._load_telemetry_data()
        
        # Create training samples with strategic patterns
        self._create_strategic_samples()
        
        # Split data for train/val
        if len(self.samples) > 0:
            if stage == "train":
                # Take 80% for training
                split_idx = int(len(self.samples) * 0.8)
                self.samples = self.samples[:split_idx]
            elif stage == "val":
                # Take 20% for validation
                split_idx = int(len(self.samples) * 0.8)
                self.samples = self.samples[split_idx:]
        
        print(f"Created {len(self.samples)} samples for {stage}")
    
    def _load_telemetry_data(self):
        """Load telemetry CSV files from all tracks"""
        
        for track in self.config.tracks:
            track_path = os.path.join(self.config.data_root, track)
            if not os.path.exists(track_path):
                print(f"  Track folder not found: {track_path}")
                continue
            
            
            # Find all telemetry files
            telemetry_files = glob.glob(os.path.join(track_path, "**/telemetry*.csv"), recursive=True)
            telemetry_files.extend(glob.glob(os.path.join(track_path, "**/*telemetry*.csv"), recursive=True))
            
            # Also look for any CSV files if no telemetry files found
            if not telemetry_files:
                csv_files = glob.glob(os.path.join(track_path, "**/*.csv"), recursive=True)
                telemetry_files.extend(csv_files[:5])  # Take first 5 CSV files
            
            for tel_file in telemetry_files:
                try:
                    # Read telemetry data
                    df = pd.read_csv(tel_file, nrows=50000)  # Increased for more data
                    if len(df) > 0:
                        # Convert long format to wide format if needed
                        df = self._convert_to_wide_format(df)
                        self.telemetry_cache[tel_file] = df
                except Exception as e:
                    print(f"    Error loading {tel_file}: {e}")
        
        if not self.telemetry_cache:
            print(f"  No telemetry data found! Creating synthetic data...")
            self._create_synthetic_data()
    
    def _convert_to_wide_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert long format telemetry data to wide format"""
        # Check if data is in long format (has telemetry_name and telemetry_value columns)
        if 'telemetry_name' in df.columns and 'telemetry_value' in df.columns:
            # Pivot the data to wide format
            try:
                # Create a unique index for each row
                if 'lap' in df.columns and 'timestamp' in df.columns:
                    df['row_id'] = df.groupby(['lap', 'timestamp']).cumcount()
                    wide_df = df.pivot_table(
                        index=['lap', 'timestamp', 'row_id'],
                        columns='telemetry_name',
                        values='telemetry_value',
                        aggfunc='first'
                    ).reset_index()
                    wide_df = wide_df.drop('row_id', axis=1)
                elif 'timestamp' in df.columns:
                    df['row_id'] = df.groupby('timestamp').cumcount()
                    wide_df = df.pivot_table(
                        index=['timestamp', 'row_id'],
                        columns='telemetry_name',
                        values='telemetry_value',
                        aggfunc='first'
                    ).reset_index()
                    wide_df = wide_df.drop('row_id', axis=1)
                else:
                    # Fallback - group by index
                    return df
                
                # Rename columns to match expected names
                column_map = {
                    'vehicle_speed': 'Speed',
                    'speed': 'Speed',
                    'Speed': 'Speed',  # Already correct
                    'engine_rpm': 'nmot',
                    'rpm': 'nmot',
                    'nmot': 'nmot',  # Already correct
                    'throttle_position': 'ath',
                    'ath': 'ath',  # Already correct
                    'brake_pressure': 'pbrake_f',
                    'pbrake_f': 'pbrake_f',  # Already correct
                    'steering_angle': 'Steering_Angle',
                    'Steering_Angle': 'Steering_Angle',  # Already correct
                    'lateral_g': 'accy_can',
                    'longitudinal_g': 'accx_can',
                    'accx_can': 'accx_can',  # Already correct
                    'accy_can': 'accy_can'  # Already correct
                }
                
                # Only rename columns that exist
                rename_dict = {}
                for col in wide_df.columns:
                    if col in column_map and column_map[col] != col:
                        rename_dict[col] = column_map[col]
                
                if rename_dict:
                    wide_df = wide_df.rename(columns=rename_dict)
                
                # Fill missing values
                wide_df = wide_df.ffill().fillna(0)
                
                # Generate Speed column if missing (estimate from acceleration data)
                if 'Speed' not in wide_df.columns and 'accx_can' in wide_df.columns:
                    # Estimate speed from longitudinal acceleration
                    # Integrate acceleration over time to get velocity
                    if 'timestamp' in wide_df.columns:
                        time_diff = wide_df['timestamp'].diff().fillna(0.01)  # Assume 10ms intervals
                        wide_df['Speed'] = (wide_df['accx_can'] * 9.81 * time_diff).cumsum() * 3.6  # m/s to km/h
                        wide_df['Speed'] = wide_df['Speed'].clip(0, 300)  # Realistic speed range
                    else:
                        # Fallback: use acceleration magnitude as proxy
                        wide_df['Speed'] = (wide_df['accx_can'].abs() * 100).clip(0, 300)
                
                # Add other missing critical columns with synthetic data
                if 'nmot' not in wide_df.columns:
                    # Estimate RPM from speed if available
                    if 'Speed' in wide_df.columns:
                        wide_df['nmot'] = wide_df['Speed'] * 50 + np.random.normal(2000, 500, len(wide_df))
                        wide_df['nmot'] = wide_df['nmot'].clip(1000, 12000)
                
                if 'ath' not in wide_df.columns:
                    # Estimate throttle from acceleration
                    if 'accx_can' in wide_df.columns:
                        wide_df['ath'] = (wide_df['accx_can'].clip(0, 2) * 50).clip(0, 100)
                
                if 'pbrake_f' not in wide_df.columns:
                    # Estimate brake from negative acceleration
                    if 'accx_can' in wide_df.columns:
                        wide_df['pbrake_f'] = (-wide_df['accx_can'].clip(-3, 0) * 33).clip(0, 100)
                
                return wide_df
            except Exception as e:
                print(f"    Could not convert to wide format: {e}")
                return df
        else:
            # Already in wide format or different format
            return df
    
    def _create_synthetic_data(self):
        """Create synthetic telemetry data for testing"""
        for i, track in enumerate(self.config.tracks[:3]):  # Create data for first 3 tracks
            # Create synthetic DataFrame
            num_laps = 30
            for lap in range(1, num_laps + 1):
                lap_data = []
                for sample in range(500):  # 500 samples per lap
                    row = {
                        'lap': lap,
                        'timestamp': sample / 100.0,
                        'speed': np.random.uniform(100, 250),
                        'rpm': np.random.uniform(5000, 12000),
                        'throttle': np.random.uniform(0, 100),
                        'brake': np.random.uniform(0, 100) if sample % 10 < 2 else 0,
                        'steer_angle': np.random.uniform(-45, 45),
                        'gear': np.random.randint(1, 8),
                        'tire_temp_fl': np.random.uniform(70, 110),
                        'tire_temp_fr': np.random.uniform(70, 110),
                        'tire_temp_rl': np.random.uniform(70, 110),
                        'tire_temp_rr': np.random.uniform(70, 110)
                    }
                    lap_data.append(row)
                
                df = pd.DataFrame(lap_data)
                filename = f"synthetic_{track}_lap_{lap}.csv"
                self.telemetry_cache[filename] = df
        
        print(f"  Created {len(self.telemetry_cache)} synthetic telemetry files")
    
    def _create_strategic_samples(self):
        """Create training samples with strategic racing scenarios"""
        for tel_file, df in self.telemetry_cache.items():
            # Extract track info
            track_name = self._extract_track_name(tel_file)
            track_id = self.config.tracks.index(track_name) if track_name in self.config.tracks else 0
            
            # Group by lap for strategic analysis
            if 'lap' in df.columns and df['lap'].nunique() > 1:
                laps = df['lap'].unique()
                for lap in laps[:30]:  # Process up to 30 laps
                    lap_data = df[df['lap'] == lap]
                    if len(lap_data) > self.config.sequence_length:
                        # Create multiple samples per lap for different strategic scenarios
                        for scenario in self._generate_scenarios(lap_data, track_id, lap):
                            if scenario:
                                self.samples.append(scenario)
            else:
                # If no lap column or single lap, create samples from chunks of data
                chunk_size = self.config.sequence_length * 2
                num_chunks = min(50, (len(df) - chunk_size) // (chunk_size // 2))  # Limit chunks
                for i in range(num_chunks):
                    start_idx = i * (chunk_size // 2)
                    chunk = df.iloc[start_idx:start_idx+chunk_size]
                    lap_num = i + 1
                    for scenario in self._generate_scenarios(chunk, track_id, lap_num):
                        if scenario:
                            self.samples.append(scenario)
    
    def _extract_track_name(self, filepath: str) -> str:
        """Extract track name from file path"""
        for track in self.config.tracks:
            if track.lower() in filepath.lower():
                return track
        return "Unknown"
    
    def _generate_scenarios(self, lap_data: pd.DataFrame, track_id: int, lap_num: int) -> List[Dict]:
        """Generate strategic racing scenarios from lap data"""
        scenarios = []
        
        # Scenario 1: Pit window analysis (more frequent)
        if lap_num % 3 == 0 or lap_num == 1:  # Every 3 laps or first lap
            scenario = self._create_pit_scenario(lap_data, track_id, lap_num)
            if scenario:
                scenarios.append(scenario)
        
        # Scenario 2: Overtaking opportunities
        if 'Speed' in lap_data.columns or 'speed' in lap_data.columns:
            overtake_points = self._find_overtake_opportunities(lap_data)
            for point in overtake_points[:3]:  # Top 3 opportunities
                scenario = self._create_overtake_scenario(lap_data, track_id, point)
                if scenario:
                    scenarios.append(scenario)
        
        # Scenario 3: Tire degradation patterns
        tire_scenario = self._create_tire_scenario(lap_data, track_id, lap_num)
        if tire_scenario:
            scenarios.append(tire_scenario)
        
        # Scenario 4: Fuel saving strategies
        if lap_num > 3:  # After initial laps
            fuel_scenario = self._create_fuel_scenario(lap_data, track_id, lap_num)
            if fuel_scenario:
                scenarios.append(fuel_scenario)
        
        return scenarios
    
    def _create_pit_scenario(self, lap_data: pd.DataFrame, track_id: int, lap_num: int) -> Optional[Dict]:
        """Create pit stop decision scenario"""
        try:
            # Extract relevant telemetry
            telemetry = self._extract_telemetry_window(lap_data, start_idx=0, window_size=self.config.sequence_length)
            
            # Calculate tire degradation
            tire_deg = self._calculate_tire_degradation(lap_data, lap_num)
            
            # Calculate fuel level
            fuel_level = max(0.1, 1.0 - (lap_num / 50))  # Simplified fuel model
            
            # Create strategic grid encoding
            grid = self._encode_strategic_grid(telemetry, tire_deg, fuel_level, track_id)
            
            # Pit strategy labels
            pit_decision = 1 if (tire_deg['avg'] < 0.5 or fuel_level < 0.2) else 0
            tire_choice = 0 if lap_num < 20 else 1  # 0: Soft, 1: Medium, 2: Hard
            
            return {
                'grid': grid,
                'track_id': track_id,
                'scenario_type': 'pit_strategy',
                'labels': {
                    'pit_decision': pit_decision,
                    'tire_choice': tire_choice,
                    'fuel_save': 1 if fuel_level < 0.3 else 0
                },
                'metadata': {
                    'lap': lap_num,
                    'tire_deg': tire_deg,
                    'fuel_level': fuel_level
                }
            }
        except Exception:
            return None
    
    def _create_overtake_scenario(self, lap_data: pd.DataFrame, track_id: int, position: int) -> Optional[Dict]:
        """Create overtaking opportunity scenario"""
        try:
            window_size = min(self.config.sequence_length, len(lap_data) - position)
            telemetry = self._extract_telemetry_window(lap_data, position, window_size)
            
            # Simulate competitor positions
            competitors = self._generate_competitor_positions(position, len(lap_data))
            
            # Create grid with competitor awareness
            grid = self._encode_competitive_grid(telemetry, competitors, track_id)
            
            # Overtake decision based on track position
            overtake_decision = 1 if self._is_good_overtake_point(lap_data, position) else 0
            
            return {
                'grid': grid,
                'track_id': track_id,
                'scenario_type': 'overtake',
                'labels': {
                    'overtake_decision': overtake_decision,
                    'risk_level': random.choice([0, 1, 2]),  # Low, Medium, High
                    'drs_usage': 1 if overtake_decision else 0
                },
                'metadata': {
                    'position': position,
                    'competitors': len(competitors)
                }
            }
        except Exception:
            return None
    
    def _create_tire_scenario(self, lap_data: pd.DataFrame, track_id: int, lap_num: int) -> Optional[Dict]:
        """Create tire management scenario"""
        try:
            telemetry = self._extract_telemetry_window(lap_data, 0, self.config.sequence_length)
            
            # Calculate detailed tire metrics
            tire_temps = self._extract_tire_temperatures(lap_data)
            tire_deg = self._calculate_tire_degradation(lap_data, lap_num)
            
            # Encode with tire focus
            grid = self._encode_tire_grid(telemetry, tire_temps, tire_deg, track_id)
            
            # Tire management decisions
            push_level = 2 if tire_deg['avg'] > 0.7 else (1 if tire_deg['avg'] > 0.4 else 0)
            
            return {
                'grid': grid,
                'track_id': track_id,
                'scenario_type': 'tire_management',
                'labels': {
                    'push_level': push_level,
                    'tire_saving': 1 if tire_deg['avg'] < 0.5 else 0,
                    'rotation_needed': 1 if tire_deg['variance'] > 0.2 else 0
                },
                'metadata': {
                    'lap': lap_num,
                    'tire_temps': tire_temps,
                    'tire_deg': tire_deg
                }
            }
        except Exception:
            return None
    
    def _create_fuel_scenario(self, lap_data: pd.DataFrame, track_id: int, lap_num: int) -> Optional[Dict]:
        """Create fuel management scenario"""
        try:
            telemetry = self._extract_telemetry_window(lap_data, 0, self.config.sequence_length)
            
            # Estimate fuel consumption
            fuel_level = max(0.1, 1.0 - (lap_num / 50))
            fuel_rate = self._calculate_fuel_rate(lap_data)
            laps_remaining = 50 - lap_num
            
            # Strategic grid for fuel management
            grid = self._encode_fuel_grid(telemetry, fuel_level, fuel_rate, track_id)
            
            # Fuel saving decisions
            fuel_mode = 0  # 0: Normal, 1: Save, 2: Critical
            if fuel_level * 50 < laps_remaining * fuel_rate:
                fuel_mode = 2 if fuel_level < 0.15 else 1
            
            return {
                'grid': grid,
                'track_id': track_id,
                'scenario_type': 'fuel_management',
                'labels': {
                    'fuel_mode': fuel_mode,
                    'lift_and_coast': 1 if fuel_mode > 0 else 0,
                    'engine_mode': max(0, 3 - fuel_mode)  # 3: Full power, 0: Maximum save
                },
                'metadata': {
                    'lap': lap_num,
                    'fuel_level': fuel_level,
                    'fuel_rate': fuel_rate
                }
            }
        except Exception:
            return None
    
    def _encode_strategic_grid(self, telemetry: np.ndarray, tire_deg: Dict, 
                              fuel_level: float, track_id: int) -> torch.Tensor:
        """Encode race state into MINERVA's 30x30 grid format"""
        grid = torch.zeros(30, 30)
        
        # Ensure valid values
        fuel_level = max(0.0, min(1.0, fuel_level))
        for key in tire_deg:
            if isinstance(tire_deg[key], (int, float)):
                tire_deg[key] = max(0.0, min(1.0, tire_deg[key]))
        
        # Rows 0-9: Speed and position data
        if telemetry.shape[0] > 0 and telemetry.shape[1] > 0:
            speed_profile = telemetry[:, 0] if telemetry.shape[1] > 0 else telemetry
            speed_norm = (speed_profile - speed_profile.min()) / (speed_profile.max() - speed_profile.min() + 1e-6)
            for i in range(min(10, len(speed_norm))):
                grid[i, :len(speed_norm)] = torch.tensor(speed_norm)
        
        # Rows 10-14: Tire degradation visualization
        grid[10, :int(tire_deg['fl'] * 30)] = 1.0
        grid[11, :int(tire_deg['fr'] * 30)] = 1.0
        grid[12, :int(tire_deg['rl'] * 30)] = 1.0
        grid[13, :int(tire_deg['rr'] * 30)] = 1.0
        grid[14, :] = tire_deg['avg']  # Average degradation
        
        # Rows 15-19: Fuel visualization
        grid[15:20, :int(fuel_level * 30)] = fuel_level
        
        # Rows 20-24: Track characteristics
        grid[20, track_id * 5:(track_id + 1) * 5] = 1.0  # Track identifier
        
        # Rows 25-29: Strategic patterns
        # Add sinusoidal patterns for MINERVA's pattern recognition
        x = torch.linspace(0, 2 * np.pi, 30)
        grid[25, :] = torch.sin(x * (track_id + 1)) * 0.5 + 0.5
        grid[26, :] = torch.cos(x * (track_id + 2)) * 0.5 + 0.5
        
        return grid
    
    def _encode_competitive_grid(self, telemetry: np.ndarray, competitors: List[float], 
                                track_id: int) -> torch.Tensor:
        """Encode competitive scenario into grid"""
        grid = torch.zeros(30, 30)
        
        # Base telemetry encoding
        if telemetry.shape[0] > 0:
            grid[:10, :] = self._encode_telemetry_block(telemetry[:10])
        
        # Competitor positions (rows 20-24)
        for i, comp_pos in enumerate(competitors[:5]):
            grid[20 + i, int(comp_pos * 29)] = 1.0
            # Add proximity indicators
            for j in range(-2, 3):
                pos = int(comp_pos * 29) + j
                if 0 <= pos < 30:
                    grid[20 + i, pos] = max(0, 1 - abs(j) * 0.3)
        
        return grid
    
    def _encode_tire_grid(self, telemetry: np.ndarray, tire_temps: Dict, 
                         tire_deg: Dict, track_id: int) -> torch.Tensor:
        """Encode tire-focused scenario into grid"""
        grid = torch.zeros(30, 30)
        
        # Tire temperature patterns (rows 0-7)
        temp_positions = ['fl', 'fr', 'rl', 'rr']
        for i, pos in enumerate(temp_positions):
            if pos in tire_temps:
                # Inner, Middle, Outer temperatures
                temps = tire_temps[pos]
                grid[i*2, :10] = temps['inner'] / 120.0  # Normalize to ~120°C max
                grid[i*2, 10:20] = temps['middle'] / 120.0
                grid[i*2, 20:30] = temps['outer'] / 120.0
                
                # Temperature gradients
                grid[i*2+1, :] = torch.tensor(np.gradient(grid[i*2, :].numpy()))
        
        # Degradation patterns (rows 8-15)
        for i, pos in enumerate(temp_positions):
            if pos in tire_deg:
                deg_value = tire_deg[pos]
                grid[8 + i, :int(deg_value * 30)] = deg_value
                # Add wear pattern
                wear_pattern = torch.linspace(deg_value, deg_value * 0.8, 30)
                grid[12 + i, :] = wear_pattern
        
        # Telemetry context (rows 16-29)
        if telemetry.shape[0] > 0:
            grid[16:26, :] = self._encode_telemetry_block(telemetry[:10])
        
        return grid
    
    def _encode_fuel_grid(self, telemetry: np.ndarray, fuel_level: float, 
                         fuel_rate: float, track_id: int) -> torch.Tensor:
        """Encode fuel management scenario into grid"""
        grid = torch.zeros(30, 30)
        
        # Fuel visualization (rows 0-9)
        # Current fuel level
        grid[0:3, :int(fuel_level * 30)] = fuel_level
        
        # Fuel consumption rate pattern
        consumption_profile = torch.linspace(fuel_rate, fuel_rate * 1.2, 30)
        grid[3, :] = consumption_profile
        grid[4, :] = torch.cumsum(consumption_profile, dim=0) / 30  # Cumulative
        
        # Predicted fuel levels
        for i in range(5):
            future_fuel = max(0, fuel_level - (i + 1) * fuel_rate)
            grid[5 + i, :int(future_fuel * 30)] = future_fuel
        
        # Throttle and brake patterns (rows 10-19)
        if telemetry.shape[0] > 0 and telemetry.shape[1] > 3:
            throttle_data = telemetry[:, 3] if telemetry.shape[1] > 3 else np.ones(len(telemetry))
            brake_data = telemetry[:, 4] if telemetry.shape[1] > 4 else np.zeros(len(telemetry))
            
            grid[10:15, :len(throttle_data)] = torch.tensor(throttle_data / 100.0).unsqueeze(0)
            grid[15:20, :len(brake_data)] = torch.tensor(brake_data / 100.0).unsqueeze(0)
        
        # Strategic patterns (rows 20-29)
        # Optimal fuel saving opportunities
        track_profile = self._get_track_profile(track_id)
        grid[20:25, :] = track_profile['elevation']
        grid[25:30, :] = track_profile['corners']
        
        return grid
    
    def _extract_telemetry_window(self, lap_data: pd.DataFrame, start_idx: int, 
                                 window_size: int) -> np.ndarray:
        """Extract telemetry window from lap data"""
        # Get available telemetry channels
        available_channels = []
        for channel in self.config.telemetry_channels:
            if channel in lap_data.columns:
                available_channels.append(channel)
        
        if not available_channels:
            # Fallback: use any numeric columns
            numeric_cols = lap_data.select_dtypes(include=[np.number]).columns
            available_channels = list(numeric_cols)[:len(self.config.telemetry_channels)]
        
        if not available_channels:
            return np.zeros((window_size, 1))
        
        # Extract window
        end_idx = min(start_idx + window_size, len(lap_data))
        window_data = lap_data.iloc[start_idx:end_idx][available_channels].values
        
        # Pad if necessary
        if len(window_data) < window_size:
            padding = np.zeros((window_size - len(window_data), len(available_channels)))
            window_data = np.vstack([window_data, padding])
        
        return window_data.astype(np.float32)
    
    def _calculate_tire_degradation(self, lap_data: pd.DataFrame, lap_num: int) -> Dict[str, float]:
        """Calculate tire degradation based on lap number and telemetry"""
        base_deg = min(1.0, lap_num * 0.02)  # 2% per lap base degradation
        
        # Add variance based on driving style
        variance = np.random.uniform(0.0, 0.1)
        
        deg = {
            'fl': min(1.0, base_deg + np.random.uniform(-variance, variance)),
            'fr': min(1.0, base_deg + np.random.uniform(-variance, variance)),
            'rl': min(1.0, base_deg + np.random.uniform(-variance, variance)),
            'rr': min(1.0, base_deg + np.random.uniform(-variance, variance))
        }
        
        deg['avg'] = np.mean(list(deg.values()))
        deg['variance'] = np.std(list(deg.values()))
        
        return deg
    
    def _extract_tire_temperatures(self, lap_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Extract tire temperature data"""
        temps = {}
        positions = ['fl', 'fr', 'rl', 'rr']
        
        for pos in positions:
            # Simulate temperature distribution
            base_temp = np.random.uniform(80, 100)
            temps[pos] = {
                'inner': base_temp + np.random.uniform(-5, 10),
                'middle': base_temp + np.random.uniform(-2, 5),
                'outer': base_temp + np.random.uniform(-5, 10)
            }
        
        return temps
    
    def _find_overtake_opportunities(self, lap_data: pd.DataFrame) -> List[int]:
        """Find potential overtaking points based on speed profiles"""
        speed_col = 'Speed' if 'Speed' in lap_data.columns else 'speed'
        if speed_col not in lap_data.columns or len(lap_data) < 10:
            return []
        
        speed_data = lap_data[speed_col].values
        
        # Find braking zones (speed decreasing)
        braking_zones = []
        for i in range(1, len(speed_data) - 1):
            if speed_data[i] < speed_data[i-1] * 0.9:  # 10% speed reduction
                braking_zones.append(i)
        
        # Return top overtaking opportunities
        return braking_zones[:5]
    
    def _is_good_overtake_point(self, lap_data: pd.DataFrame, position: int) -> bool:
        """Determine if position is good for overtaking"""
        speed_col = 'Speed' if 'Speed' in lap_data.columns else 'speed'
        if speed_col not in lap_data.columns:
            return False
        
        # Check if it's a braking zone or corner entry
        if position > 5 and position < len(lap_data) - 5:
            speed_before = lap_data.iloc[position-5:position][speed_col].mean()
            speed_at = lap_data.iloc[position][speed_col]
            return speed_at < speed_before * 0.85
        
        return False
    
    def _generate_competitor_positions(self, current_pos: int, track_length: int) -> List[float]:
        """Generate simulated competitor positions"""
        competitors = []
        num_competitors = np.random.randint(2, 6)
        
        for _ in range(num_competitors):
            # Position relative to current position
            relative_pos = np.random.uniform(-0.1, 0.1)
            comp_pos = (current_pos / track_length) + relative_pos
            comp_pos = max(0, min(1, comp_pos))  # Clamp to [0, 1]
            competitors.append(comp_pos)
        
        return sorted(competitors)
    
    def _calculate_fuel_rate(self, lap_data: pd.DataFrame) -> float:
        """Calculate fuel consumption rate"""
        # Simplified model based on throttle usage
        if 'throttle' in lap_data.columns:
            avg_throttle = lap_data['throttle'].mean()
            return 0.02 * (avg_throttle / 100)  # 2% per lap at full throttle
        return 0.015  # Default 1.5% per lap
    
    def _encode_telemetry_block(self, telemetry: np.ndarray) -> torch.Tensor:
        """Encode telemetry data into grid block"""
        if len(telemetry.shape) == 1:
            telemetry = telemetry.reshape(-1, 1)
        
        rows, cols = telemetry.shape
        block = torch.zeros(min(rows, 10), 30)
        
        # Normalize and encode
        for i in range(min(rows, 10)):
            if cols > 0:
                data = telemetry[i, :]
                # Normalize
                if data.max() > data.min():
                    data_norm = (data - data.min()) / (data.max() - data.min())
                else:
                    data_norm = data
                
                # Spread across grid columns
                indices = np.linspace(0, 29, len(data_norm), dtype=int)
                block[i, indices] = torch.tensor(data_norm)
        
        return block
    
    def _get_track_profile(self, track_id: int) -> Dict[str, torch.Tensor]:
        """Get track-specific profile for strategic planning"""
        # Simplified track profiles
        profiles = {
            0: {'elevation': torch.sin(torch.linspace(0, 4*np.pi, 30)) * 0.5 + 0.5,
                'corners': torch.abs(torch.cos(torch.linspace(0, 6*np.pi, 30)))},
            1: {'elevation': torch.cos(torch.linspace(0, 3*np.pi, 30)) * 0.5 + 0.5,
                'corners': torch.abs(torch.sin(torch.linspace(0, 8*np.pi, 30)))},
            2: {'elevation': torch.ones(30) * 0.5,  # Flat track
                'corners': torch.abs(torch.sin(torch.linspace(0, 10*np.pi, 30)))},
            3: {'elevation': torch.sigmoid(torch.linspace(-3, 3, 30)),
                'corners': torch.abs(torch.cos(torch.linspace(0, 5*np.pi, 30)))},
            4: {'elevation': torch.sin(torch.linspace(0, 2*np.pi, 30)) * 0.7 + 0.3,
                'corners': torch.abs(torch.sin(torch.linspace(0, 7*np.pi, 30)))},
            5: {'elevation': torch.cos(torch.linspace(0, 5*np.pi, 30)) * 0.4 + 0.6,
                'corners': torch.abs(torch.cos(torch.linspace(0, 9*np.pi, 30)))}
        }
        
        return profiles.get(track_id, profiles[0])
    
    def __len__(self) -> int:
        return len(self.samples) * (self.config.augmentation_factor if self.augment else 1)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Handle augmentation indexing
        if self.augment:
            sample_idx = idx // self.config.augmentation_factor
            aug_idx = idx % self.config.augmentation_factor
        else:
            sample_idx = idx
            aug_idx = 0
        
        sample = self.samples[sample_idx].copy()
        
        # Apply augmentation
        if self.augment and aug_idx > 0:
            sample = self._augment_sample(sample, aug_idx)
        
        # Convert to tensors
        grid = sample['grid']
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(grid, dtype=torch.float32)
        
        # Prepare labels
        labels = {}
        for key, value in sample['labels'].items():
            if isinstance(value, (int, float)):
                labels[key] = torch.tensor(value, dtype=torch.long if isinstance(value, int) else torch.float32)
            else:
                labels[key] = torch.tensor(value)
        
        return {
            'grid': grid,
            'track_id': torch.tensor(sample['track_id'], dtype=torch.long),
            'scenario_type': sample['scenario_type'],
            'labels': labels,
            'metadata': sample.get('metadata', {})
        }
    
    def _augment_sample(self, sample: Dict, aug_idx: int) -> Dict:
        """Apply data augmentation to sample"""
        augmented = sample.copy()
        grid = augmented['grid'].clone() if isinstance(augmented['grid'], torch.Tensor) else torch.tensor(augmented['grid'])
        
        if aug_idx == 1:
            # Add noise
            noise = torch.randn_like(grid) * 0.05
            grid = torch.clamp(grid + noise, 0, 1)
        elif aug_idx == 2:
            # Shift pattern
            shift = np.random.randint(1, 5)
            grid = torch.roll(grid, shift, dims=1)
        
        augmented['grid'] = grid
        return augmented


# =====================================
# Custom Data Collation
# =====================================

def custom_collate_fn(batch):
    """Custom collate function to handle different scenario types with different labels"""
    # Separate components
    grids = torch.stack([torch.tensor(item['grid']) if not isinstance(item['grid'], torch.Tensor) else item['grid'] for item in batch])
    track_ids = torch.stack([item['track_id'] for item in batch])
    scenario_types = [item['scenario_type'] for item in batch]
    
    # Collate labels - handle different label keys
    all_label_keys = set()
    for item in batch:
        all_label_keys.update(item['labels'].keys())
    
    labels = {}
    for key in all_label_keys:
        # Only include samples that have this label
        label_values = []
        for item in batch:
            if key in item['labels']:
                label_values.append(item['labels'][key])
            else:
                # Add default value based on label type
                if key in ['pit_decision', 'tire_choice', 'fuel_mode', 'overtake_decision', 'risk_level', 'push_level']:
                    label_values.append(torch.tensor(0, dtype=torch.long))
                else:
                    label_values.append(torch.tensor(0.0, dtype=torch.float32))
        
        labels[key] = torch.stack(label_values)
    
    # Metadata is optional
    metadata = [item.get('metadata', {}) for item in batch]
    
    return {
        'grid': grids,
        'track_id': track_ids,
        'scenario_type': scenario_types,
        'labels': labels,
        'metadata': metadata
    }


# =====================================
# Advanced Training Logic
# =====================================

class MinervaTrainer:
    """Advanced trainer for MINERVA V6 Racing with progressive curriculum"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enable A100 optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        print(f"\033[96mInitializing MINERVA V6 Enhanced Racing Trainer\033[0m")
        print(f"\033[93mDevice: {self.device}\033[0m")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Initialize model
        self._initialize_model()
        
        # Initialize datasets  
        print(f"Loading racing datasets...")
        self._initialize_datasets()
        
        # Initialize training components
        self._initialize_training_components()
        
        # Training state
        self.global_step = 0
        self.best_performance = 0.0
        self.stage_performances = {}
        self.start_epoch = 0
        
    def _initialize_model(self):
        """Initialize MINERVA model and racing adapter"""
        print(f"\033[96mLoading {self.config.model_name}...\033[0m")
        
        # Create base MINERVA model
        self.base_model = MinervaV6Enhanced(
            max_grid_size=self.config.grid_size,
            hidden_dim=256,
            preserve_weights=True
        )
        
        # Load pre-trained weights if available
        checkpoint_path = os.path.join(self.config.checkpoint_dir, "minerva_v6_base.pt")
        if os.path.exists(checkpoint_path):
            self.base_model.load_compatible_weights(checkpoint_path)
            print(f"Loaded pre-trained MINERVA weights")
        
        # Create racing adapter
        self.model = MinervaRacingAdapter(
            base_model=self.base_model,
            num_tracks=len(self.config.tracks)
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Calculate parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Load best checkpoint if available
        best_checkpoint_path = os.path.join(self.config.checkpoint_dir, "minerva_best.pt")
        if os.path.exists(best_checkpoint_path):
            try:
                checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.best_performance = checkpoint.get('performance', checkpoint.get('best_accuracy', 0.0))
                print(f"\033[92mLoaded best checkpoint with {self.best_performance:.2f}% accuracy\033[0m")
            except Exception as e:
                print(f"Warning: Could not load best checkpoint: {e}")
        
        # Initialize EMA model for stability
        self.ema_model = self._create_ema_model()
    
    def _create_ema_model(self):
        """Create exponential moving average model"""
        ema_model = MinervaRacingAdapter(
            base_model=MinervaV6Enhanced(
                max_grid_size=self.config.grid_size,
                hidden_dim=256,
                preserve_weights=True
            ),
            num_tracks=len(self.config.tracks)
        ).to(self.device)
        
        # Copy weights
        for param, ema_param in zip(self.model.parameters(), ema_model.parameters()):
            ema_param.data.copy_(param.data)
        
        return ema_model
    
    def _update_ema_model(self):
        """Update EMA model weights"""
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.mul_(self.config.ema_decay).add_(param.data, alpha=1 - self.config.ema_decay)
    
    def _initialize_datasets(self):
        """Initialize training and validation datasets"""
        
        # Create datasets
        self.train_dataset = MinervaRacingDataset(
            self.config, 
            stage="train", 
            augment=self.config.use_data_augmentation
        )
        
        self.val_dataset = MinervaRacingDataset(
            self.config, 
            stage="val", 
            augment=False
        )
        
        # Create data loaders
        # Adjust batch size if we have fewer samples
        train_batch_size = min(self.config.batch_size, len(self.train_dataset))
        if train_batch_size < self.config.batch_size:
            print(f"Adjusting batch size from {self.config.batch_size} to {train_batch_size} due to limited samples")
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers if len(self.train_dataset) > 100 else 0,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers if len(self.train_dataset) > 100 else False,
            drop_last=False,  # Don't drop last with small datasets
            collate_fn=custom_collate_fn
        )
        
        # Handle small validation set
        val_batch_size = min(self.config.batch_size * 2, max(1, len(self.val_dataset)))
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,  # Adaptive batch size
            shuffle=False,
            num_workers=0 if len(self.val_dataset) < 10 else self.config.num_workers // 2,
            pin_memory=self.config.pin_memory if len(self.val_dataset) > 10 else False,
            persistent_workers=False,  # Disable for small datasets
            collate_fn=custom_collate_fn
        ) if len(self.val_dataset) > 0 else None
        
        print(f"Train samples: {len(self.train_dataset):,}")
        print(f"Val samples: {len(self.val_dataset):,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation}")
    
    def _initialize_training_components(self):
        """Initialize optimizer, scheduler, scaler, and loss functions"""
        # Parameter groups for different learning rates
        # Get all available heads from the model
        param_groups = []
        
        # Standard heads that should always exist
        standard_heads = ['pit_strategy_head', 'tire_strategy_head', 'fuel_optimization_head']
        for head_name in standard_heads:
            if hasattr(self.model, head_name):
                param_groups.append({
                    'params': getattr(self.model, head_name).parameters(),
                    'lr': self.config.stages[0]['lr'],
                    'name': head_name.replace('_head', '')
                })
        
        # Additional heads for specific scenarios
        additional_heads = ['overtake_decision_head', 'risk_level_head', 'push_level_head', 'fuel_mode_head']
        for head_name in additional_heads:
            if hasattr(self.model, head_name):
                param_groups.append({
                    'params': getattr(self.model, head_name).parameters(),
                    'lr': self.config.stages[0]['lr'],
                    'name': head_name.replace('_head', '')
                })
        
        # Track embedding
        if hasattr(self.model, 'track_embedding'):
            param_groups.append({
                'params': self.model.track_embedding.parameters(),
                'lr': self.config.stages[0]['lr'] * 2,  # Higher LR for embeddings
                'name': 'track_embedding'
            })
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warm restarts
        steps_per_epoch = max(1, len(self.train_loader))  # At least 1
        total_steps = sum(stage['epochs'] for stage in self.config.stages) * steps_per_epoch
        T_0 = max(1, steps_per_epoch * self.config.warmup_epochs)  # At least 1
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_0,
            T_mult=2,
            eta_min=1e-7
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Loss functions
        self.criterion_pit = nn.CrossEntropyLoss(label_smoothing=0.1)  # 4 classes
        self.criterion_tire = nn.CrossEntropyLoss(label_smoothing=0.1)  # 3 classes
        self.criterion_fuel = nn.BCEWithLogitsLoss()  # Binary
        
        # Additional losses for advanced scenarios
        self.criterion_overtake = nn.CrossEntropyLoss()  # 2 classes (binary)
        self.criterion_risk = nn.CrossEntropyLoss()  # 3 classes
        self.criterion_push = nn.CrossEntropyLoss()  # 3 classes
        self.criterion_fuel_mode = nn.CrossEntropyLoss()  # 3 classes
        
        print(f"Optimizer: AdamW with {len(param_groups)} parameter groups")
        print(f"Scheduler: CosineAnnealingWarmRestarts")
        print(f"Mixed Precision: {'Enabled' if self.config.use_amp else 'Disabled'}")
        
        # Load optimizer state if resuming
        best_checkpoint_path = os.path.join(self.config.checkpoint_dir, "minerva_best.pt")
        if os.path.exists(best_checkpoint_path):
            try:
                checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"Loaded optimizer state from checkpoint")
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print(f"Loaded scheduler state from checkpoint")
                if 'epoch' in checkpoint:
                    self.start_epoch = checkpoint['epoch']
                    print(f"Resuming from epoch {self.start_epoch}")
            except Exception as e:
                print(f"Warning: Could not load optimizer/scheduler state: {e}")
    
    def train(self):
        """Execute multi-stage progressive training"""
        print(f"\n\033[96m{'='*125}\033[0m")
        print(f"\033[96mStarting MINERVA V6 Enhanced Racing Training\033[0m")
        print(f"\033[96m{'='*125}\033[0m\n")
        
        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # Progressive training stages
        for stage_idx, stage in enumerate(self.config.stages):
            print(f"\n\033[94m{'='*125}\033[0m")
            print(f"\033[94mStage {stage_idx + 1}: {stage['name'].upper()} - {stage['focus']}\033[0m")
            print(f"\033[94mEpochs: {stage['epochs']} | Learning Rate: {stage['lr']}\033[0m")
            print(f"\033[94m{'='*125}\033[0m\n")
            
            # Update learning rates for new stage
            self._update_learning_rates(stage['lr'])
            
            # Unfreeze layers based on stage
            self._configure_stage_training(stage_idx)
            
            # Train stage
            stage_performance = self._train_stage(stage, stage_idx)
            self.stage_performances[stage['name']] = stage_performance
            
            # Save stage checkpoint
            self._save_checkpoint(f"minerva_stage_{stage['name']}.pt", stage_performance)
            
            print(f"\n\033[92mStage {stage['name']} complete! Best performance: {stage_performance:.2f}%\033[0m")
        
        # Final evaluation
        self._final_evaluation()
        
        print(f"\n\033[92m{'='*125}\033[0m")
        print(f"\033[92mTraining Complete!\033[0m")
        print(f"\033[92mBest Overall Performance: {self.best_performance:.2f}%\033[0m")
        print(f"\033[92m{'='*125}\033[0m\n")
    
    def _update_learning_rates(self, base_lr: float):
        """Update learning rates for parameter groups"""
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'track_embedding':
                param_group['lr'] = base_lr * 2
            else:
                param_group['lr'] = base_lr
    
    def _configure_stage_training(self, stage_idx: int):
        """Configure model for specific training stage"""
        # Progressive unfreezing
        if stage_idx == 0:
            # Stage 1: Only train heads
            for param in self.model.minerva.parameters():
                param.requires_grad = False
            print(f"\033[93mBase model frozen - training heads only\033[0m")
            
        elif stage_idx == 1:
            # Stage 2: Unfreeze strategic components
            for name, param in self.model.minerva.named_parameters():
                if 'strategic' in name or 'decision' in name:
                    param.requires_grad = True
            print(f"\033[93mStrategic components unfrozen\033[0m")
            
        elif stage_idx == 2:
            # Stage 3: Unfreeze pattern memory
            for name, param in self.model.minerva.named_parameters():
                if 'pattern' in name or 'memory' in name:
                    param.requires_grad = True
            print(f"\033[93mPattern memory unfrozen\033[0m")
            
        else:
            # Stage 4: Full fine-tuning
            for param in self.model.minerva.parameters():
                param.requires_grad = True
            print(f"\033[93mFull model unfrozen for fine-tuning\033[0m")
    
    def _train_stage(self, stage: Dict, stage_idx: int) -> float:
        """Train a single stage"""
        best_stage_performance = 0.0
        
        for epoch in range(stage['epochs']):
            # Training epoch
            train_metrics = self._train_epoch(epoch, stage['epochs'], stage_idx)
            
            # Validation (always validate with small datasets)
            if (epoch + 1) % self.config.val_interval == 0 or len(self.train_dataset) < 100:
                val_metrics = self._validate()
                
                # Track best performance
                if val_metrics['accuracy'] > best_stage_performance:
                    best_stage_performance = val_metrics['accuracy']
                    
                    if val_metrics['accuracy'] > self.best_performance:
                        self.best_performance = val_metrics['accuracy']
                        self._save_checkpoint("minerva_best.pt", val_metrics['accuracy'])
                        print(f"\033[92mNew global best performance: {self.best_performance:.2f}% - Saved!\033[0m")
                
                # Log metrics
                print(f"\n\033[93mStage {stage_idx + 1}, Epoch {epoch + 1} (Global: {self.global_step}):\033[0m")
                print(f"   \033[95mTrain Loss: {train_metrics['loss']:.4f}\033[0m")
                print(f"   \033[95mVal Accuracy: {val_metrics['accuracy']:.2f}%\033[0m")
                print(f"   \033[94mBest Stage: {best_stage_performance:.2f}%\033[0m")
                print(f"   \033[92mBest Overall: {self.best_performance:.2f}%\033[0m")
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f"minerva_epoch_{self.global_step}.pt", best_stage_performance)
        
        return best_stage_performance
    
    def _train_epoch(self, epoch: int, total_epochs: int, stage_idx: int) -> Dict[str, float]:
        """Train a single epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_pit_loss = 0.0
        epoch_tire_loss = 0.0
        epoch_fuel_loss = 0.0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}/{total_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            grids = batch['grid'].to(self.device)
            track_ids = batch['track_id'].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
            scenario_types = batch['scenario_type']
            
            # Process batch based on scenario types
            batch_loss = 0.0
            
            # Group by scenario type for efficient processing
            scenario_groups = defaultdict(list)
            for i, scenario in enumerate(scenario_types):
                scenario_groups[scenario].append(i)
            
            # Process each scenario group
            for scenario_type, indices in scenario_groups.items():
                if not indices:
                    continue
                
                # Create mini-batch for this scenario
                scenario_grids = grids[indices]
                scenario_track_ids = track_ids[indices]
                
                # Create race data for adapter
                for i, (grid, track_id) in enumerate(zip(scenario_grids, scenario_track_ids)):
                    race_data = self._grid_to_race_data(grid, track_id.item(), scenario_type)
                    
                    # Forward pass with mixed precision
                    if self.config.use_amp:
                        with autocast():
                            predictions = self.model(race_data)
                            loss = self._compute_scenario_loss(
                                predictions, labels, indices[i], scenario_type
                            )
                    else:
                        predictions = self.model(race_data)
                        loss = self._compute_scenario_loss(
                            predictions, labels, indices[i], scenario_type
                        )
                    
                    batch_loss += loss / len(indices)
            
            # Scale loss for gradient accumulation
            batch_loss = batch_loss / self.config.gradient_accumulation
            
            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(batch_loss).backward()
            else:
                batch_loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                # Check for NaN loss first
                if torch.isnan(batch_loss):
                    print(f"Warning: NaN loss detected at step {self.global_step}, skipping update")
                    self.optimizer.zero_grad()
                    if self.config.use_amp:
                        self.scaler.update()  # Still need to update scaler
                    continue
                
                # Gradient clipping
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # Clip gradients to prevent NaN
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
                
                # Check if gradients are valid
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: Invalid gradients detected at step {self.global_step}, skipping update")
                    self.optimizer.zero_grad()
                    if self.config.use_amp:
                        self.scaler.update()  # Still need to update scaler
                    continue
                
                # Optimizer step
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA model
                self._update_ema_model()
                
                # Update scheduler
                self.scheduler.step()
            
            # Update metrics
            epoch_loss += batch_loss.item() * self.config.gradient_accumulation
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{epoch_loss / (batch_idx + 1):.4f}",
                    'lr': f"{current_lr:.2e}"
                })
            
            self.global_step += 1
            
            # Clear cache periodically
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # Return epoch metrics
        return {
            'loss': epoch_loss / len(self.train_loader),
            'pit_loss': epoch_pit_loss / max(1, len(self.train_loader)),
            'tire_loss': epoch_tire_loss / max(1, len(self.train_loader)),
            'fuel_loss': epoch_fuel_loss / max(1, len(self.train_loader))
        }
    
    def _grid_to_race_data(self, grid: torch.Tensor, track_id: int, scenario_type: str) -> Dict:
        """Convert grid back to race data format for model input"""
        # Extract information from grid encoding
        # Handle NaN values
        def safe_float(value):
            result = float(value)
            return result if not (np.isnan(result) or np.isinf(result)) else 0.5
            
        race_data = {
            'track_position': safe_float(grid[:10].mean().item()),  # Simplified
            'speed': safe_float(grid[:10].max().item() * 300),  # Denormalize
            'tire_degradation': {
                'FL': safe_float(grid[10].sum().item() / 30),
                'FR': safe_float(grid[11].sum().item() / 30),
                'RL': safe_float(grid[12].sum().item() / 30),
                'RR': safe_float(grid[13].sum().item() / 30)
            },
            'fuel_percentage': safe_float(grid[15:20].mean().item()),
            'track_id': track_id,
            'competitors': [
                {'position': safe_float(grid[20 + i].argmax().item() / 30)}
                for i in range(5) if grid[20 + i].sum() > 0
            ]
        }
        
        # Add scenario-specific data
        if scenario_type == 'overtake':
            race_data['weather_grid'] = grid[25:30].cpu().numpy()
        
        return race_data
    
    def _compute_scenario_loss(self, predictions: Dict[str, torch.Tensor], 
                              labels: Dict[str, torch.Tensor], 
                              idx: int, scenario_type: str) -> torch.Tensor:
        """Compute loss based on scenario type"""
        loss = 0.0
        
        # Always compute primary losses
        if 'pit_strategy' in predictions and 'pit_decision' in labels:
            if 'pit_decision' in labels:
                target = labels['pit_decision'][idx].long()
                loss += self.criterion_pit(predictions['pit_strategy'].unsqueeze(0), target.unsqueeze(0))
        
        if 'tire_choice' in predictions and 'tire_choice' in labels:
            target = labels['tire_choice'][idx].long()
            loss += self.criterion_tire(predictions['tire_choice'].unsqueeze(0), target.unsqueeze(0))
        
        if 'fuel_save' in predictions and 'fuel_save' in labels:
            target = labels['fuel_save'][idx:idx+1].float()
            loss += self.criterion_fuel(predictions['fuel_save'].squeeze(), target.squeeze())
        
        # Scenario-specific losses - use the CORRECT prediction heads
        if scenario_type == 'overtake':
            if 'overtake_decision' in labels and 'overtake_decision' in predictions:
                target = labels['overtake_decision'][idx].long()
                loss += self.criterion_overtake(predictions['overtake_decision'].unsqueeze(0), target.unsqueeze(0))
            if 'risk_level' in labels and 'risk_level' in predictions:
                target = labels['risk_level'][idx].long()
                loss += 0.5 * self.criterion_risk(predictions['risk_level'].unsqueeze(0), target.unsqueeze(0))
        
        elif scenario_type == 'tire_management':
            if 'push_level' in labels and 'push_level' in predictions:
                target = labels['push_level'][idx].long()
                loss += self.criterion_push(predictions['push_level'].unsqueeze(0), target.unsqueeze(0))
            # Use any binary labels with overtake head
            if 'tire_saving' in labels and 'overtake_decision' in predictions:
                target = labels['tire_saving'][idx].long()
                loss += 0.5 * self.criterion_overtake(predictions['overtake_decision'].unsqueeze(0), target.unsqueeze(0))
        
        elif scenario_type == 'fuel_management':
            if 'fuel_mode' in labels and 'fuel_mode' in predictions:
                target = labels['fuel_mode'][idx].long()
                loss += self.criterion_fuel_mode(predictions['fuel_mode'].unsqueeze(0), target.unsqueeze(0))
            # Use any binary labels with overtake head
            if 'lift_and_coast' in labels and 'overtake_decision' in predictions:
                target = labels['lift_and_coast'][idx].long()
                loss += 0.5 * self.criterion_overtake(predictions['overtake_decision'].unsqueeze(0), target.unsqueeze(0))
        
        return loss
    
    def _validate(self) -> Dict[str, float]:
        """Validate model performance"""
        self.model.eval()
        
        # Skip validation if no validation data
        if self.val_loader is None or len(self.val_dataset) == 0:
            print("No validation data available - skipping validation")
            return {'accuracy': 0.0, 'loss': 0.0}
        
        total_correct = 0
        total_samples = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                grids = batch['grid'].to(self.device)
                track_ids = batch['track_id'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                scenario_types = batch['scenario_type']
                
                # Process each sample
                for i in range(len(grids)):
                    race_data = self._grid_to_race_data(
                        grids[i], track_ids[i].item(), scenario_types[i]
                    )
                    
                    # Forward pass
                    predictions = self.model(race_data)
                    
                    # Compute accuracy based on scenario type using CORRECT heads
                    correct = False
                    if scenario_types[i] == 'pit_strategy' and 'pit_decision' in labels:
                        pred = predictions['pit_strategy'].argmax(dim=-1)
                        target = labels['pit_decision'][i:i+1]
                        correct = (pred == target).item()
                    elif scenario_types[i] == 'tire_management':
                        if 'push_level' in labels and 'push_level' in predictions:
                            pred = predictions['push_level'].argmax(dim=-1)
                            target = labels['push_level'][i:i+1]
                            correct = (pred == target).item()
                        elif 'tire_saving' in labels and 'overtake_decision' in predictions:
                            pred = predictions['overtake_decision'].argmax(dim=-1)
                            target = labels['tire_saving'][i:i+1]
                            correct = (pred == target).item()
                    elif scenario_types[i] == 'overtake':
                        if 'overtake_decision' in labels and 'overtake_decision' in predictions:
                            pred = predictions['overtake_decision'].argmax(dim=-1)
                            target = labels['overtake_decision'][i:i+1]
                            correct = (pred == target).item()
                    elif scenario_types[i] == 'fuel_management':
                        if 'fuel_mode' in labels and 'fuel_mode' in predictions:
                            pred = predictions['fuel_mode'].argmax(dim=-1)
                            target = labels['fuel_mode'][i:i+1]
                            correct = (pred == target).item()
                        elif 'lift_and_coast' in labels and 'overtake_decision' in predictions:
                            pred = predictions['overtake_decision'].argmax(dim=-1)
                            target = labels['lift_and_coast'][i:i+1]
                            correct = (pred == target).item()
                    
                    if correct:
                        total_correct += 1
                    total_samples += 1
                    
                    # Compute loss
                    loss = self._compute_scenario_loss(
                        predictions, labels, i, scenario_types[i]
                    )
                    val_loss += loss.item()
        
        accuracy = (total_correct / max(1, total_samples)) * 100
        avg_loss = val_loss / len(self.val_loader.dataset)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss
        }
    
    def _save_checkpoint(self, filename: str, performance: float):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'performance': performance,
            'best_accuracy': self.best_performance,
            'epoch': self.global_step // len(self.train_loader),
            'config': self.config.__dict__
        }
        
        if self.config.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filename}")
    
    def _final_evaluation(self):
        """Perform final model evaluation"""
        print(f"\n\033[96mFinal Evaluation...\033[0m")
        
        # Use EMA model for final evaluation
        self.ema_model.eval()
        
        # Comprehensive evaluation metrics
        scenario_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Final Evaluation"):
                grids = batch['grid'].to(self.device)
                track_ids = batch['track_id'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                scenario_types = batch['scenario_type']
                
                for i in range(len(grids)):
                    race_data = self._grid_to_race_data(
                        grids[i], track_ids[i].item(), scenario_types[i]
                    )
                    
                    # Use EMA model
                    predictions = self.ema_model(race_data)
                    
                    # Evaluate by scenario
                    scenario = scenario_types[i]
                    scenario_metrics[scenario]['total'] += 1
                    
                    # Check primary decision
                    correct = False
                    if scenario == 'pit_strategy' and 'pit_decision' in labels:
                        pred = predictions['pit_strategy'].argmax(dim=-1)
                        target = labels['pit_decision'][i:i+1]
                        correct = (pred == target).item()
                    elif scenario == 'overtake' and 'overtake_decision' in labels:
                        # Use pit strategy as proxy
                        pred = predictions['pit_strategy'].argmax(dim=-1) > 1
                        target = labels['overtake_decision'][i:i+1] > 0
                        correct = (pred == target).item()
                    elif scenario == 'tire_management' and 'push_level' in labels:
                        pred = predictions['pit_strategy'].argmax(dim=-1)
                        target = labels['push_level'][i:i+1]
                        correct = (pred == target).item()
                    elif scenario == 'fuel_management' and 'fuel_mode' in labels:
                        pred = predictions['tire_choice'].argmax(dim=-1)
                        target = labels['fuel_mode'][i:i+1]
                        correct = (pred == target).item()
                    
                    if correct:
                        scenario_metrics[scenario]['correct'] += 1
        
        # Print final results
        print(f"\n\033[93m{'='*80}\033[0m")
        print(f"\033[93mFinal Evaluation Results\033[0m")
        print(f"\033[93m{'='*80}\033[0m")
        
        total_correct = 0
        total_samples = 0
        
        for scenario, metrics in scenario_metrics.items():
            accuracy = (metrics['correct'] / max(1, metrics['total'])) * 100
            print(f"{scenario:.<30} {accuracy:>6.2f}% ({metrics['correct']}/{metrics['total']})")
            total_correct += metrics['correct']
            total_samples += metrics['total']
        
        overall_accuracy = (total_correct / max(1, total_samples)) * 100
        print(f"\033[93m{'='*80}\033[0m")
        print(f"\033[92m{'Overall Accuracy':.<30} {overall_accuracy:>6.2f}%\033[0m")
        print(f"\033[93m{'='*80}\033[0m")
        
        # Save final results
        results = {
            'scenario_metrics': dict(scenario_metrics),
            'overall_accuracy': overall_accuracy,
            'stage_performances': self.stage_performances
        }
        
        with open(os.path.join(self.config.checkpoint_dir, "final_results.json"), 'w') as f:
            json.dump(results, f, indent=2)


# =====================================
# Main Training Entry Point
# =====================================

def main():
    """Main training entry point"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create configuration
    config = TrainingConfig()
    
    # Print banner
    print(f"\n\033[96m{'='*125}\033[0m")
    print(f"\033[96mMINERVA V6 Enhanced Racing Training System\033[0m")
    print(f"\033[96mAdvanced Multi-Stage Progressive Curriculum Learning\033[0m")
    print(f"\033[96mOptimized for A100 GPU with Mixed Precision\033[0m")
    print(f"\033[96m{'='*125}\033[0m\n")
    
    # Create trainer
    trainer = MinervaTrainer(config)
    
    # Start training
    trainer.train()
    
    print(f"\nTraining complete!")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()