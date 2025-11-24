import React, { useState, useEffect } from 'react';
import { Card } from '../common';
import styles from './PredictionsPanel.module.css';
import { aiModelClient } from '../../services/aiModelClient';
import type { ChronosResponse, PrometheusResponse } from '../../services/aiModelClient';

interface Prediction {
  label: string;
  value: string;
  confidence: number;
  trend?: 'up' | 'down' | 'stable';
  details?: string;
}

const predictions: Prediction[] = [
  {
    label: 'P2 by Lap 45',
    value: '78%',
    confidence: 78,
    trend: 'up',
    details: 'Based on current pace delta'
  },
  {
    label: 'Tire Cliff',
    value: '~8 Laps',
    confidence: 85,
    trend: 'stable',
    details: 'Front left showing higher deg'
  },
  {
    label: 'Next SC',
    value: 'Low',
    confidence: 92,
    details: 'No incidents expected'
  },
  {
    label: 'Rain Risk',
    value: '12%',
    confidence: 95,
    trend: 'down',
    details: 'Weather stable'
  },
  {
    label: 'Optimal Stop',
    value: 'Lap 42',
    confidence: 88,
    trend: 'stable',
    details: 'Clear air guaranteed'
  }
];

export const PredictionsPanel: React.FC = () => {
  const [aiPredictions, setAiPredictions] = useState<{
    chronos?: ChronosResponse;
    prometheus?: PrometheusResponse;
  }>({});
  const [telemetryHistory, setTelemetryHistory] = useState<any[]>([]);
  const [dynamicPredictions, setDynamicPredictions] = useState<Prediction[]>(predictions);

  // Get AI predictions
  useEffect(() => {
    const updatePredictions = async () => {
      if (telemetryHistory.length < 20) return;

      try {
        const [chronosResult, prometheusResult] = await Promise.all([
          aiModelClient.getTimingPredictions(telemetryHistory),
          aiModelClient.getFuturePredictions(telemetryHistory)
        ]);

        setAiPredictions({ chronos: chronosResult, prometheus: prometheusResult });
        
        // Update dynamic predictions with AI data
        const updatedPredictions: Prediction[] = [
          {
            label: 'P2 by Lap 45',
            value: `${Math.round(prometheusResult.predictions.position_forecast.lap_45.probability * 100)}%`,
            confidence: Math.round(prometheusResult.predictions.position_forecast.lap_45.probability * 100),
            trend: prometheusResult.predictions.position_forecast.lap_45.probability > 0.7 ? 'up' : 'stable',
            details: 'AI model prediction'
          },
          {
            label: 'Lap Time',
            value: chronosResult.predictions.predicted_lap_time,
            confidence: Math.round(chronosResult.predictions.confidence * 100),
            trend: chronosResult.predictions.time_delta_to_optimal.includes('-') ? 'up' : 'down',
            details: `Delta: ${chronosResult.predictions.time_delta_to_optimal}`
          },
          {
            label: 'Incident Risk',
            value: `${Math.round(prometheusResult.predictions.incident_probability * 100)}%`,
            confidence: Math.round(prometheusResult.predictions.confidence * 100),
            trend: prometheusResult.predictions.incident_probability < 0.15 ? 'down' : 'up',
            details: 'Safety car probability'
          },
          {
            label: 'Rain Risk',
            value: `${Math.round(prometheusResult.predictions.weather_forecast.rain_probability * 100)}%`,
            confidence: 95,
            trend: prometheusResult.predictions.weather_forecast.rain_probability < 0.2 ? 'down' : 'up',
            details: `Track: ${prometheusResult.predictions.weather_forecast.track_temperature_trend}`
          },
          {
            label: 'Undercut Window',
            value: prometheusResult.predictions.strategic_opportunities.find(o => o.event === 'undercut_opportunity')?.lap?.toString() || 'Lap 42',
            confidence: Math.round((prometheusResult.predictions.strategic_opportunities.find(o => o.event === 'undercut_opportunity')?.probability || 0.67) * 100),
            trend: 'stable',
            details: 'Optimal undercut timing'
          }
        ];

        setDynamicPredictions(updatedPredictions);

      } catch (error) {
        console.error('❌ AI predictions failed:', error);
      }
    };

    const interval = setInterval(updatePredictions, 8000); // Every 8 seconds
    return () => clearInterval(interval);
  }, [telemetryHistory]);

  // Mock telemetry data for predictions
  useEffect(() => {
    const interval = setInterval(() => {
      const mockTelemetry = {
        timestamp: new Date().toISOString(),
        sessionId: 'COTA_R2_2024',
        trackId: 'cota',
        carNumber: 7,
        currentLap: 23 + Math.floor(Date.now() / 100000) % 5,
        telemetry: {
          speed: 120 + Math.random() * 30,
          rpm: 7000 + Math.random() * 1000,
          gear: 4,
          throttle: 70 + Math.random() * 30,
          brake: Math.random() * 20,
          steeringAngle: (Math.random() - 0.5) * 30,
          gForce: { lateral: (Math.random() - 0.5) * 2, longitudinal: (Math.random() - 0.5), vertical: 0.98 },
          temperatures: {
            tires: { frontLeft: 90 + Math.random() * 10, frontRight: 90 + Math.random() * 10, rearLeft: 85 + Math.random() * 10, rearRight: 85 + Math.random() * 10 },
            brakes: { frontLeft: 350 + Math.random() * 50, frontRight: 350 + Math.random() * 50, rearLeft: 300 + Math.random() * 50, rearRight: 300 + Math.random() * 50 },
            engine: 95 + Math.random() * 10, oil: 100 + Math.random() * 10, coolant: 80 + Math.random() * 10
          },
          fuel: { level: 60 + Math.random() * 10, consumption: 2 + Math.random(), lapsRemaining: 25 + Math.random() * 5 }
        }
      };

      setTelemetryHistory(prev => [...prev, mockTelemetry].slice(-100));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <Card title="PREDICTIONS">
      <div className={styles.predictionsList}>
        {dynamicPredictions.map((prediction, index) => (
          <div key={index} className={styles.prediction}>
            <div className={styles.header}>
              <span className={styles.label}>{prediction.label}</span>
              {prediction.trend && (
                <span className={`${styles.trend} ${styles[prediction.trend]}`}>
                  {prediction.trend === 'up' && '↑'}
                  {prediction.trend === 'down' && '↓'}
                  {prediction.trend === 'stable' && '→'}
                </span>
              )}
            </div>
            
            <div className={styles.content}>
              <span className={styles.value}>{prediction.value}</span>
              <div className={styles.confidence}>
                <div className={styles.confidenceBar}>
                  <div 
                    className={styles.confidenceFill}
                    style={{ width: `${prediction.confidence}%` }}
                  />
                </div>
                <span className={styles.confidenceText}>{prediction.confidence}%</span>
              </div>
            </div>
            
            {prediction.details && (
              <span className={styles.details}>{prediction.details}</span>
            )}
          </div>
        ))}
      </div>
      
      <div className={styles.footer}>
        <span className={styles.updateTime}>Updated: {new Date().toLocaleTimeString()}</span>
        <span className={styles.accuracy}>
          🤖 {Object.keys(aiPredictions).length > 0 ? 
            `CHRONOS + PROMETHEUS Active (${Math.round((aiPredictions.chronos?.predictions?.confidence || 0.85) * 100)}% conf)` : 
            'AI Models Loading...'}
        </span>
      </div>
    </Card>
  );
};