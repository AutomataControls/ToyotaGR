import React from 'react';
import { Card, StatusIndicator } from '../common';
import styles from './StrategyAdvisor.module.css';

interface StrategyItem {
  icon: string;
  label: string;
  value: string;
  status: 'success' | 'warning' | 'danger' | 'info';
  recommendation?: string;
}

const strategyData: StrategyItem[] = [
  {
    icon: '⚡',
    label: 'PIT WINDOW',
    value: 'Lap 42-45',
    status: 'warning',
    recommendation: 'Optimal window approaching'
  },
  {
    icon: '🛞',
    label: 'Tire Life',
    value: '73%',
    status: 'success',
    recommendation: 'Degradation normal'
  },
  {
    icon: '⛽',
    label: 'Fuel to End',
    value: 'YES (+2.1L)',
    status: 'success',
    recommendation: 'Lift and coast T1-T3'
  },
  {
    icon: '🎯',
    label: 'Target Delta',
    value: '+0.234s',
    status: 'info',
    recommendation: 'Push mode available'
  }
];

export const StrategyAdvisor: React.FC = () => {
  return (
    <Card title="STRATEGY ADVISOR">
      <div className={styles.strategyGrid}>
        {strategyData.map((item, index) => (
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
          <StatusIndicator status="warning" size="sm" pulse />
          <span className={styles.alertText}>
            Undercut window: Lap 40-41 | Risk: Medium
          </span>
        </div>
      </div>
      
      <div className={styles.modeSelector}>
        <button className={`${styles.modeButton} ${styles.active}`}>
          Standard
        </button>
        <button className={styles.modeButton}>
          Push
        </button>
        <button className={styles.modeButton}>
          Conserve
        </button>
      </div>
    </Card>
  );
};