import React, { useState, useEffect } from 'react';
import { Card } from '../common';
import styles from './LapComparison.module.css';

interface LapTime {
  sector1: number;
  sector2: number;
  sector3: number;
  total: number;
}

interface LapData {
  current: LapTime;
  best: LapTime;
  theoretical: LapTime;
}

const generateLapTime = (base: number, variance: number): LapTime => {
  const s1 = base * 0.3 + (Math.random() - 0.5) * variance;
  const s2 = base * 0.35 + (Math.random() - 0.5) * variance;
  const s3 = base * 0.35 + (Math.random() - 0.5) * variance;
  return {
    sector1: s1,
    sector2: s2,
    sector3: s3,
    total: s1 + s2 + s3
  };
};

export const LapComparison: React.FC = () => {
  const [lapData, setLapData] = useState<LapData>({
    current: generateLapTime(83.456, 0.5),
    best: generateLapTime(83.123, 0.1),
    theoretical: { sector1: 24.789, sector2: 29.012, sector3: 29.123, total: 82.924 }
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setLapData(prev => ({
        ...prev,
        current: generateLapTime(83.456, 0.5)
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(3);
    return `${minutes}:${secs.padStart(6, '0')}`;
  };

  const getDelta = (current: number, reference: number): string => {
    const diff = current - reference;
    return diff >= 0 ? `+${diff.toFixed(3)}` : diff.toFixed(3);
  };

  const getDeltaColor = (current: number, reference: number): string => {
    const diff = current - reference;
    if (Math.abs(diff) < 0.01) return styles.neutral;
    return diff > 0 ? styles.slower : styles.faster;
  };

  return (
    <Card title="LAP COMPARISON">
      <div className={styles.timeDisplay}>
        <div className={styles.mainTime}>
          <span className={styles.label}>Current Lap</span>
          <span className={styles.time}>{formatTime(lapData.current.total)}</span>
        </div>
        
        <div className={styles.comparison}>
          <div className={styles.comparisonRow}>
            <span className={styles.compLabel}>Best Lap</span>
            <span className={styles.compTime}>{formatTime(lapData.best.total)}</span>
            <span className={`${styles.delta} ${getDeltaColor(lapData.current.total, lapData.best.total)}`}>
              {getDelta(lapData.current.total, lapData.best.total)}
            </span>
          </div>
          
          <div className={styles.comparisonRow}>
            <span className={styles.compLabel}>Theoretical</span>
            <span className={styles.compTime}>{formatTime(lapData.theoretical.total)}</span>
            <span className={`${styles.delta} ${getDeltaColor(lapData.current.total, lapData.theoretical.total)}`}>
              {getDelta(lapData.current.total, lapData.theoretical.total)}
            </span>
          </div>
        </div>
      </div>
      
      <div className={styles.sectors}>
        <h4 className={styles.sectorTitle}>Sector Times</h4>
        <div className={styles.sectorGrid}>
          {[1, 2, 3].map((sector) => {
            const key = `sector${sector}` as keyof LapTime;
            return (
              <div key={sector} className={styles.sector}>
                <span className={styles.sectorLabel}>S{sector}</span>
                <span className={styles.sectorTime}>
                  {lapData.current[key].toFixed(3)}
                </span>
                <span className={`${styles.sectorDelta} ${getDeltaColor(lapData.current[key], lapData.best[key])}`}>
                  {getDelta(lapData.current[key], lapData.best[key])}
                </span>
              </div>
            );
          })}
        </div>
      </div>
      
      <div className={styles.miniMap}>
        <div className={styles.progressBar}>
          <div className={styles.progress} style={{ width: '65%' }} />
        </div>
        <div className={styles.sectorMarkers}>
          <span>S1</span>
          <span>S2</span>
          <span>S3</span>
        </div>
      </div>
    </Card>
  );
};