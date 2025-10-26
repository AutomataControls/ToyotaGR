import React, { useState } from 'react';
import { Card } from '../components/common';
import { LapComparison } from '../components/timing/LapComparison';
import { TrackSelector } from '../components/TrackSelector';
import { Timer, TrendingDown, Flag, Clock } from 'lucide-react';
import styles from './Timing.module.css';

interface LapTime {
  lap: number;
  time: string;
  sector1: string;
  sector2: string;
  sector3: string;
  delta: string;
  tireCompound: 'S' | 'M' | 'H';
}

export const Timing: React.FC = () => {
  const [selectedTrack, setSelectedTrack] = useState('cota');
  const [selectedRace, setSelectedRace] = useState<1 | 2>(1);
  
  // Mock lap times data
  const lapTimes: LapTime[] = [
    { lap: 25, time: '1:23.456', sector1: '27.123', sector2: '28.789', sector3: '27.544', delta: '-0.234', tireCompound: 'M' },
    { lap: 24, time: '1:23.690', sector1: '27.234', sector2: '28.901', sector3: '27.555', delta: '+0.123', tireCompound: 'M' },
    { lap: 23, time: '1:23.567', sector1: '27.189', sector2: '28.834', sector3: '27.544', delta: '-0.456', tireCompound: 'M' },
    { lap: 22, time: '1:24.023', sector1: '27.456', sector2: '29.012', sector3: '27.555', delta: '+0.234', tireCompound: 'M' },
    { lap: 21, time: '1:23.789', sector1: '27.234', sector2: '28.999', sector3: '27.556', delta: '-0.123', tireCompound: 'M' },
  ];

  const bestLap = {
    time: '1:23.234',
    lap: 15,
    sector1: '27.012',
    sector2: '28.678',
    sector3: '27.544'
  };

  const sessionBest = {
    time: '1:22.987',
    driver: 'Driver 7',
    lap: 18
  };

  return (
    <div className={styles.timingPage}>
      <div className={styles.header}>
        <h1 className={styles.title}>
          <Timer size={24} />
          Lap Times & Analysis
        </h1>
        <TrackSelector
          selectedTrack={selectedTrack}
          onTrackChange={setSelectedTrack}
          selectedRace={selectedRace}
          onRaceChange={setSelectedRace}
        />
      </div>

      <div className={styles.grid}>
        <Card className={styles.sessionInfo}>
          <h2><Clock size={18} /> Session Information</h2>
          <div className={styles.sessionStats}>
            <div className={styles.stat}>
              <span className={styles.statLabel}>Current Lap</span>
              <span className={styles.statValue}>26</span>
            </div>
            <div className={styles.stat}>
              <span className={styles.statLabel}>Laps Remaining</span>
              <span className={styles.statValue}>14</span>
            </div>
            <div className={styles.stat}>
              <span className={styles.statLabel}>Session Time</span>
              <span className={styles.statValue}>35:42</span>
            </div>
            <div className={styles.stat}>
              <span className={styles.statLabel}>Position</span>
              <span className={styles.statValue}>P4</span>
            </div>
          </div>
        </Card>

        <Card className={styles.bestTimes}>
          <h2><TrendingDown size={18} /> Best Times</h2>
          <div className={styles.bestTimesList}>
            <div className={styles.bestTime}>
              <div className={styles.bestTimeHeader}>
                <span>Personal Best</span>
                <Flag size={16} color="#14b8a6" />
              </div>
              <div className={styles.bestTimeValue}>{bestLap.time}</div>
              <div className={styles.bestTimeDetails}>
                Lap {bestLap.lap} • S1: {bestLap.sector1} • S2: {bestLap.sector2} • S3: {bestLap.sector3}
              </div>
            </div>
            <div className={styles.bestTime}>
              <div className={styles.bestTimeHeader}>
                <span>Session Best</span>
                <Flag size={16} color="#f59e0b" />
              </div>
              <div className={styles.bestTimeValue}>{sessionBest.time}</div>
              <div className={styles.bestTimeDetails}>
                {sessionBest.driver} • Lap {sessionBest.lap}
              </div>
            </div>
          </div>
        </Card>

        <div className={styles.lapTimesSection}>
          <Card>
            <h2>Recent Lap Times</h2>
            <div className={styles.lapTimesTable}>
              <div className={styles.tableHeader}>
                <span>Lap</span>
                <span>Time</span>
                <span>S1</span>
                <span>S2</span>
                <span>S3</span>
                <span>Delta</span>
                <span>Tire</span>
              </div>
              {lapTimes.map((lap) => (
                <div key={lap.lap} className={styles.tableRow}>
                  <span>{lap.lap}</span>
                  <span className={styles.lapTime}>{lap.time}</span>
                  <span className={styles.sector}>{lap.sector1}</span>
                  <span className={styles.sector}>{lap.sector2}</span>
                  <span className={styles.sector}>{lap.sector3}</span>
                  <span className={`${styles.delta} ${lap.delta.startsWith('-') ? styles.negative : styles.positive}`}>
                    {lap.delta}
                  </span>
                  <span className={styles.tire}>{lap.tireCompound}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>

        <div className={styles.comparisonSection}>
          <LapComparison />
        </div>
      </div>
    </div>
  );
};