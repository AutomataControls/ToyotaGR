import React, { useState, useEffect } from 'react';
import { Card, StatusIndicator } from '../common';
import styles from './TrackPosition.module.css';

interface Driver {
  position: number;
  name: string;
  team: string;
  gap: string;
  status: 'running' | 'pit' | 'out';
  lastLap: string;
  trend: 'up' | 'down' | 'stable';
}

interface TrackPositionProps {
  trackId: string;
  raceNumber: 1 | 2;
}

const mockDrivers: Driver[] = [
  { position: 1, name: 'VER', team: 'RBR', gap: 'Leader', status: 'running', lastLap: '1:23.145', trend: 'stable' },
  { position: 2, name: 'HAM', team: 'MER', gap: '+2.341', status: 'running', lastLap: '1:23.487', trend: 'stable' },
  { position: 3, name: 'LEC', team: 'FER', gap: '+5.123', status: 'running', lastLap: '1:23.654', trend: 'up' },
  { position: 4, name: 'PER', team: 'RBR', gap: '+7.456', status: 'pit', lastLap: '1:24.123', trend: 'down' },
  { position: 5, name: 'SAI', team: 'FER', gap: '+12.789', status: 'running', lastLap: '1:24.234', trend: 'stable' },
];

export const TrackPosition: React.FC<TrackPositionProps> = ({ trackId, raceNumber }) => {
  const [drivers, setDrivers] = useState(mockDrivers);
  const [ourDriver] = useState(3); // We are LEC in P3
  const [driverProgress, setDriverProgress] = useState(65);
  const [currentLap, setCurrentLap] = useState(15);

  // Simulate position updates
  useEffect(() => {
    const interval = setInterval(() => {
      setDrivers(prev => prev.map(driver => ({
        ...driver,
        gap: driver.position === 1 ? 'Leader' : 
              `+${(parseFloat(driver.gap.replace('+', '')) + Math.random() * 0.1 - 0.05).toFixed(3)}`,
        lastLap: `1:${(23 + Math.random()).toFixed(3)}`
      })));
      
      // Update driver progress
      setDriverProgress(prev => {
        const newProgress = prev + 2;
        if (newProgress >= 100) {
          setCurrentLap(lap => lap + 1);
          return 0;
        }
        return newProgress;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <Card title="TRACK POSITION">
      <div className={styles.positionList}>
        {drivers.map((driver) => (
          <div 
            key={driver.position}
            className={`${styles.driverRow} ${driver.position === ourDriver ? styles.ourDriver : ''}`}
          >
            <div className={styles.position}>P{driver.position}</div>
            <div className={styles.driverInfo}>
              <span className={styles.name}>{driver.name}</span>
              <span className={styles.team}>{driver.team}</span>
            </div>
            <div className={styles.gap}>{driver.gap}</div>
            <div className={styles.lastLap}>{driver.lastLap}</div>
            <div className={styles.status}>
              {driver.status === 'pit' && <span className={styles.pitTag}>PIT</span>}
              {driver.trend === 'up' && <span className={styles.trend}>↑</span>}
              {driver.trend === 'down' && <span className={styles.trend}>↓</span>}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};