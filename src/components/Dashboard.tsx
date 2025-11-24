import React, { useState, useEffect } from 'react';
import { Card } from './common';
import { TrackSelector } from './TrackSelector';
import { TrackPosition } from './telemetry/TrackPosition';
import { LapComparison } from './timing/LapComparison';
import { StrategyAdvisor } from './strategy/StrategyAdvisor';
import { TelemetryGraphs } from './telemetry/TelemetryGraphs';
import { PredictionsPanel } from './strategy/PredictionsPanel';
import { SpecialistConsensus } from './strategy/SpecialistConsensus';
import styles from './Dashboard.module.css';
import { toyotaDataLoader } from '../services/toyotaDataLoader';

export const Dashboard: React.FC = () => {
  const [selectedTrack, setSelectedTrack] = useState('cota');
  const [selectedRace, setSelectedRace] = useState<1 | 2>(1);
  useEffect(() => {
    console.log('🏁 Dashboard initialized with Toyota GR Cup data');
  }, []);


  return (
    <div className={styles.dashboard}>

      <TrackSelector 
        selectedTrack={selectedTrack}
        onTrackChange={setSelectedTrack}
        selectedRace={selectedRace}
        onRaceChange={setSelectedRace}
      />
      
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