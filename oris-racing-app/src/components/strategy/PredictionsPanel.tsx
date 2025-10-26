import React from 'react';
import { Card } from '../common';
import styles from './PredictionsPanel.module.css';

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
  return (
    <Card title="PREDICTIONS">
      <div className={styles.predictionsList}>
        {predictions.map((prediction, index) => (
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
        <span className={styles.updateTime}>Updated: 2s ago</span>
        <span className={styles.accuracy}>Model Accuracy: 94.2%</span>
      </div>
    </Card>
  );
};