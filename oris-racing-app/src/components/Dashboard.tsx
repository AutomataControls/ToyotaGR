import React, { useState } from 'react';
import { Card } from './common';
import { TrackSelector } from './TrackSelector';
import { TrackPosition } from './telemetry/TrackPosition';
import { TrackVisualization3D } from './telemetry/TrackVisualization3D';
import { LapComparison } from './timing/LapComparison';
import { StrategyAdvisor } from './strategy/StrategyAdvisor';
import { TelemetryGraphs } from './telemetry/TelemetryGraphs';
import { PredictionsPanel } from './strategy/PredictionsPanel';
import { SpecialistConsensus } from './strategy/SpecialistConsensus';
import styles from './Dashboard.module.css';

export const Dashboard: React.FC = () => {
  const [selectedTrack, setSelectedTrack] = useState('cota');
  const [selectedRace, setSelectedRace] = useState<1 | 2>(1);
  const [driverProgress, setDriverProgress] = useState(65);
  const [currentLap, setCurrentLap] = useState(15);

  // Simulate driver progress
  React.useEffect(() => {
    const interval = setInterval(() => {
      setDriverProgress(prev => {
        const newProgress = prev + 0.5;
        if (newProgress >= 100) {
          setCurrentLap(lap => lap + 1);
          return 0;
        }
        return newProgress;
      });
    }, 500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className={styles.dashboard}>
      <TrackSelector 
        selectedTrack={selectedTrack}
        onTrackChange={setSelectedTrack}
        selectedRace={selectedRace}
        onRaceChange={setSelectedRace}
      />
      
      <div className={styles.trackMapFullWidth}>
        <TrackVisualization3D 
          trackId={selectedTrack}
          driverProgress={driverProgress}
          currentLap={currentLap}
          totalLaps={50}
        />
      </div>
      
      <div className={styles.grid}>
        <div className={styles.trackPositionSection}>
          <TrackPosition trackId={selectedTrack} raceNumber={selectedRace} />
        </div>
        <div className={styles.lapComparisonSection}>
          <LapComparison />
        </div>
        
        <div className={styles.strategySection}>
          <StrategyAdvisor />
        </div>
        
        <div className={styles.telemetrySection}>
          <TelemetryGraphs />
        </div>
        
        <div className={styles.predictionsSection}>
          <PredictionsPanel />
        </div>
        
        <div className={styles.specialistSection}>
          <SpecialistConsensus />
        </div>
      </div>
    </div>
  );
};