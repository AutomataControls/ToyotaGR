import React, { useState, useCallback, useEffect } from 'react';
import { Card, StatusIndicator } from '../common';
import styles from './StrategyAdvisor.module.css';
import { aiModelClient } from '../../services/aiModelClient';
import type { MinervaResponse } from '../../services/aiModelClient';
import { raceDataManager } from '../../services/raceDataManager';

interface StrategyItem {
  icon: string;
  label: string;
  value: string;
  status: 'success' | 'warning' | 'danger' | 'info';
  recommendation?: string;
}

type StrategyMode = 'Standard' | 'Push' | 'Conserve';

interface StrategyModeData {
  mode: StrategyMode;
  description: string;
  impacts: {
    fuelConsumption: string;
    tireWear: string;
    lapTime: string;
    risk: string;
  };
}

// Strategy mode configurations
const strategyModes: Record<StrategyMode, StrategyModeData> = {
  Standard: {
    mode: 'Standard',
    description: 'Balanced pace for optimal race management',
    impacts: {
      fuelConsumption: 'Normal',
      tireWear: 'Controlled',
      lapTime: 'Target pace',
      risk: 'Low'
    }
  },
  Push: {
    mode: 'Push',
    description: 'Maximum attack mode for overtaking/gap building',
    impacts: {
      fuelConsumption: 'High (+15%)',
      tireWear: 'Increased',
      lapTime: '-0.5 to -1.2s',
      risk: 'Medium'
    }
  },
  Conserve: {
    mode: 'Conserve',
    description: 'Fuel/tire saving mode for stint extension',
    impacts: {
      fuelConsumption: 'Reduced (-12%)',
      tireWear: 'Minimal',
      lapTime: '+0.3 to +0.8s',
      risk: 'Low'
    }
  }
};

// Dynamic strategy data based on selected mode
const getStrategyData = (mode: StrategyMode): StrategyItem[] => {
  const baseData: StrategyItem[] = [
    {
      icon: '⚡',
      label: 'PIT WINDOW',
      value: mode === 'Push' ? 'Lap 40-42' : mode === 'Conserve' ? 'Lap 45-48' : 'Lap 42-45',
      status: mode === 'Push' ? 'danger' : 'warning',
      recommendation: mode === 'Push' ? 'Early window for undercut' : mode === 'Conserve' ? 'Extended stint possible' : 'Optimal window approaching'
    },
    {
      icon: '🛞',
      label: 'Tire Life',
      value: mode === 'Push' ? '68% (↓)' : mode === 'Conserve' ? '78% (→)' : '73%',
      status: mode === 'Push' ? 'warning' : 'success',
      recommendation: mode === 'Push' ? 'Higher degradation rate' : mode === 'Conserve' ? 'Preserving tire life' : 'Degradation normal'
    },
    {
      icon: '⛽',
      label: 'Fuel to End',
      value: mode === 'Push' ? 'TIGHT (-0.8L)' : mode === 'Conserve' ? 'YES (+3.2L)' : 'YES (+2.1L)',
      status: mode === 'Push' ? 'danger' : 'success',
      recommendation: mode === 'Push' ? 'Monitor consumption' : mode === 'Conserve' ? 'Fuel saving active' : 'Lift and coast T1-T3'
    },
    {
      icon: '🎯',
      label: 'Target Delta',
      value: mode === 'Push' ? '-0.8s' : mode === 'Conserve' ? '+0.5s' : '+0.234s',
      status: mode === 'Push' ? 'success' : mode === 'Conserve' ? 'warning' : 'info',
      recommendation: mode === 'Push' ? 'Maximum attack!' : mode === 'Conserve' ? 'Saving resources' : 'Push mode available'
    }
  ];
  
  return baseData;
};

export const StrategyAdvisor: React.FC = () => {
  const [selectedMode, setSelectedMode] = useState<StrategyMode>('Standard');
  const [aiPredictions, setAiPredictions] = useState<MinervaResponse | null>(null);
  const [isLoadingAI, setIsLoadingAI] = useState(false);
  const [telemetryHistory, setTelemetryHistory] = useState<any[]>([]);
  
  const handleModeChange = useCallback((mode: StrategyMode) => {
    setSelectedMode(mode);
    console.log(`🏁 Strategy mode changed to: ${mode}`);
    // Here you could send the mode change to your race data system
  }, []);

  // Get AI predictions from MINERVA
  const updateAIPredictions = useCallback(async () => {
    if (telemetryHistory.length < 10) return; // Need enough data
    
    setIsLoadingAI(true);
    try {
      const predictions = await aiModelClient.getStrategyPredictions(telemetryHistory);
      setAiPredictions(predictions);
      console.log('🤖 MINERVA predictions updated:', predictions.recommendations);
    } catch (error) {
      console.error('❌ Failed to get AI predictions:', error);
    } finally {
      setIsLoadingAI(false);
    }
  }, [telemetryHistory]);

  // Subscribe to telemetry data
  useEffect(() => {
    const unsubscribe = raceDataManager.subscribeToTelemetry((telemetryData) => {
      setTelemetryHistory(prev => {
        const newHistory = [...prev, {
          timestamp: new Date().toISOString(),
          sessionId: 'COTA_R2_2024',
          trackId: 'cota',
          carNumber: 7,
          currentLap: 23,
          telemetry: telemetryData
        }];
        // Keep last 300 data points (5 minutes at 60Hz)
        return newHistory.slice(-300);
      });
    });

    return unsubscribe;
  }, []);

  // Update AI predictions every 5 seconds
  useEffect(() => {
    const interval = setInterval(updateAIPredictions, 5000);
    return () => clearInterval(interval);
  }, [updateAIPredictions]);
  
  const currentStrategyData = getStrategyData(selectedMode);
  const currentModeData = strategyModes[selectedMode];
  
  return (
    <Card title="STRATEGY ADVISOR">
      <div className={styles.strategyGrid}>
        {currentStrategyData.map((item, index) => (
          <div key={index} className={styles.strategyItem}>
            <div className={styles.header}>
              <span className={styles.icon}>{item.icon}</span>
              <StatusIndicator status={item.status} size="sm" />
            </div>
            
            <div className={styles.content}>
              <span className={styles.label}>{item.label}</span>
              <span className={styles.value}>{item.value}</span>
              {item.recommendation && (
                <span className={styles.recommendation}>{item.recommendation}</span>
              )}
            </div>
          </div>
        ))}
      </div>
      
      <div className={styles.alerts}>
        <div className={styles.alert}>
          <StatusIndicator 
            status={isLoadingAI ? "warning" : "success"} 
            size="sm" 
            pulse={isLoadingAI} 
          />
          <span className={styles.alertText}>
            {isLoadingAI ? 'MINERVA analyzing strategy...' : 
             aiPredictions ? aiPredictions.recommendations.pit_strategy :
             selectedMode === 'Push' 
              ? 'ATTACK MODE: Monitor fuel consumption closely'
              : selectedMode === 'Conserve'
              ? 'CONSERVE MODE: Extend stint, manage gap to cars behind'
              : 'Undercut window: Lap 40-41 | Risk: Medium'
            }
          </span>
        </div>
        
        {aiPredictions && (
          <div className={styles.alert}>
            <StatusIndicator status="info" size="sm" />
            <span className={styles.alertText}>
              🤖 MINERVA: {aiPredictions.recommendations.pace_strategy} | {aiPredictions.recommendations.tire_warning}
            </span>
          </div>
        )}
      </div>
      
      {/* Mode Impact Display */}
      <div className={styles.modeImpact}>
        <div className={styles.modeDescription}>
          <strong>{currentModeData.description}</strong>
        </div>
        <div className={styles.impacts}>
          <span>Fuel: {currentModeData.impacts.fuelConsumption}</span>
          <span>Pace: {currentModeData.impacts.lapTime}</span>
          <span>Risk: {currentModeData.impacts.risk}</span>
        </div>
      </div>
      
      <div className={styles.modeSelector}>
        {Object.keys(strategyModes).map((mode) => (
          <button 
            key={mode}
            className={`${styles.modeButton} ${selectedMode === mode ? styles.active : ''}`}
            onClick={() => handleModeChange(mode as StrategyMode)}
          >
            {mode}
          </button>
        ))}
      </div>
    </Card>
  );
};