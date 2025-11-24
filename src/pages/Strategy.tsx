import React, { useState } from 'react';
import { Card } from '../components/common';
import { StrategyAdvisor } from '../components/strategy/StrategyAdvisor';
import { PredictionsPanel } from '../components/strategy/PredictionsPanel';
import { SpecialistConsensus } from '../components/strategy/SpecialistConsensus';
import { TrackSelector } from '../components/TrackSelector';
import { Target, Brain, Fuel, TrendingUp } from 'lucide-react';
import styles from './Strategy.module.css';

export const Strategy: React.FC = () => {
  const [selectedTrack, setSelectedTrack] = useState('cota');
  const [selectedRace, setSelectedRace] = useState<1 | 2>(1);
  const [pitWindow, setPitWindow] = useState({ start: 12, end: 18 });

  return (
    <div className={styles.strategyPage}>
      <div className={styles.header}>
        <h1 className={styles.title}>
          <Target size={24} />
          Race Strategy
        </h1>
        <TrackSelector
          selectedTrack={selectedTrack}
          onTrackChange={setSelectedTrack}
          selectedRace={selectedRace}
          onRaceChange={setSelectedRace}
        />
      </div>

      <div className={styles.mainContent}>
        <div className={styles.leftColumn}>
          <Card className={styles.strategyOverview}>
            <h2><Brain size={18} /> AI Strategy Recommendation</h2>
            <StrategyAdvisor />
          </Card>

          <Card className={styles.pitStrategy}>
            <h2><Fuel size={18} /> Pit Window Optimization</h2>
            <div className={styles.pitWindowDisplay}>
              <div className={styles.pitWindow}>
                <span className={styles.label}>Optimal Window</span>
                <span className={styles.value}>Lap {pitWindow.start} - {pitWindow.end}</span>
              </div>
              <div className={styles.pitFactors}>
                <div className={styles.factor}>
                  <span>Tire Degradation</span>
                  <div className={styles.progressBar}>
                    <div className={styles.progress} style={{ width: '65%' }} />
                  </div>
                </div>
                <div className={styles.factor}>
                  <span>Fuel Remaining</span>
                  <div className={styles.progressBar}>
                    <div className={styles.progress} style={{ width: '42%' }} />
                  </div>
                </div>
                <div className={styles.factor}>
                  <span>Track Position Risk</span>
                  <div className={styles.progressBar}>
                    <div className={styles.progress} style={{ width: '78%' }} />
                  </div>
                </div>
              </div>
            </div>
          </Card>

          <Card className={styles.scenarioAnalysis}>
            <h2><TrendingUp size={18} /> Scenario Analysis</h2>
            <div className={styles.scenarios}>
              <div className={styles.scenario}>
                <h3>Early Stop (Lap 10-12)</h3>
                <ul>
                  <li>+ Clear track ahead</li>
                  <li>+ Fresh tires for push</li>
                  <li>- Risk of late degradation</li>
                </ul>
                <span className={styles.probability}>Success: 72%</span>
              </div>
              <div className={styles.scenario}>
                <h3>Standard Stop (Lap 14-16)</h3>
                <ul>
                  <li>+ Balanced strategy</li>
                  <li>+ Flexible for safety car</li>
                  <li>- Traffic on exit</li>
                </ul>
                <span className={styles.probability}>Success: 85%</span>
              </div>
              <div className={styles.scenario}>
                <h3>Late Stop (Lap 18-20)</h3>
                <ul>
                  <li>+ Track position</li>
                  <li>+ Tire advantage at end</li>
                  <li>- High degradation risk</li>
                </ul>
                <span className={styles.probability}>Success: 63%</span>
              </div>
            </div>
          </Card>
        </div>

        <div className={styles.rightColumn}>
          <PredictionsPanel />
          <SpecialistConsensus />
        </div>
      </div>
    </div>
  );
};