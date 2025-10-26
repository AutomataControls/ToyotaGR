import React, { useState } from 'react';
import { Card } from '../components/common';
import { TrackSelector } from '../components/TrackSelector';
import { GraduationCap, Award, TrendingUp, Target } from 'lucide-react';
import styles from './Training.module.css';

interface PerformanceMetric {
  category: string;
  score: number;
  trend: 'up' | 'down' | 'stable';
  benchmark: number;
}

export const Training: React.FC = () => {
  const [selectedTrack, setSelectedTrack] = useState('cota');
  const [selectedRace, setSelectedRace] = useState<1 | 2>(1);

  const performanceMetrics: PerformanceMetric[] = [
    { category: 'Braking Consistency', score: 87, trend: 'up', benchmark: 85 },
    { category: 'Cornering Speed', score: 82, trend: 'stable', benchmark: 88 },
    { category: 'Throttle Control', score: 91, trend: 'up', benchmark: 90 },
    { category: 'Racing Line', score: 79, trend: 'down', benchmark: 85 },
    { category: 'Tire Management', score: 85, trend: 'up', benchmark: 82 },
    { category: 'Fuel Efficiency', score: 88, trend: 'stable', benchmark: 86 },
  ];

  const trainingRecommendations = [
    {
      priority: 'high',
      area: 'Racing Line Optimization',
      description: 'Focus on late apex corners (T4, T7, T11). You\'re turning in too early, losing exit speed.',
      improvement: '+0.3s per lap'
    },
    {
      priority: 'medium',
      area: 'Trail Braking Technique',
      description: 'Practice carrying more brake pressure into corner entry, especially at T1 and T9.',
      improvement: '+0.2s per lap'
    },
    {
      priority: 'low',
      area: 'Sector 3 Consistency',
      description: 'Maintain focus in final sector. Lap times vary by ±0.4s, aim for ±0.1s.',
      improvement: 'Better race pace'
    },
  ];

  return (
    <div className={styles.trainingPage}>
      <div className={styles.header}>
        <h1 className={styles.title}>
          <GraduationCap size={24} />
          Driver Training & Analysis
        </h1>
        <TrackSelector
          selectedTrack={selectedTrack}
          onTrackChange={setSelectedTrack}
          selectedRace={selectedRace}
          onRaceChange={setSelectedRace}
        />
      </div>

      <div className={styles.grid}>
        <Card className={styles.performanceCard}>
          <h2><Award size={18} /> Performance Metrics</h2>
          <div className={styles.metricsGrid}>
            {performanceMetrics.map((metric) => (
              <div key={metric.category} className={styles.metric}>
                <div className={styles.metricHeader}>
                  <span className={styles.metricName}>{metric.category}</span>
                  <TrendingUp 
                    size={16} 
                    className={`${styles.trend} ${styles[metric.trend]}`}
                  />
                </div>
                <div className={styles.metricScore}>
                  <div className={styles.scoreBar}>
                    <div 
                      className={styles.scoreProgress} 
                      style={{ width: `${metric.score}%` }}
                    />
                    <div 
                      className={styles.benchmarkLine} 
                      style={{ left: `${metric.benchmark}%` }}
                    />
                  </div>
                  <span className={styles.scoreValue}>{metric.score}%</span>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card className={styles.recommendationsCard}>
          <h2><Target size={18} /> AI Training Recommendations</h2>
          <div className={styles.recommendations}>
            {trainingRecommendations.map((rec, index) => (
              <div key={index} className={`${styles.recommendation} ${styles[rec.priority]}`}>
                <div className={styles.recommendationHeader}>
                  <h3>{rec.area}</h3>
                  <span className={styles.improvement}>{rec.improvement}</span>
                </div>
                <p>{rec.description}</p>
              </div>
            ))}
          </div>
        </Card>

        <Card className={styles.comparisonCard}>
          <h2>Performance vs Top Drivers</h2>
          <div className={styles.comparisonChart}>
            <div className={styles.chartHeader}>
              <span>Sector</span>
              <span>Your Time</span>
              <span>Best Time</span>
              <span>Delta</span>
            </div>
            <div className={styles.chartRow}>
              <span>Sector 1</span>
              <span>27.234</span>
              <span>26.987</span>
              <span className={styles.deltaPositive}>+0.247</span>
            </div>
            <div className={styles.chartRow}>
              <span>Sector 2</span>
              <span>28.901</span>
              <span>28.456</span>
              <span className={styles.deltaPositive}>+0.445</span>
            </div>
            <div className={styles.chartRow}>
              <span>Sector 3</span>
              <span>27.555</span>
              <span>27.544</span>
              <span className={styles.deltaPositive}>+0.011</span>
            </div>
            <div className={styles.chartRow}>
              <span>Total</span>
              <span className={styles.totalTime}>1:23.690</span>
              <span className={styles.totalTime}>1:22.987</span>
              <span className={styles.deltaPositive}>+0.703</span>
            </div>
          </div>
        </Card>

        <Card className={styles.progressCard}>
          <h2>Weekly Progress</h2>
          <div className={styles.progressStats}>
            <div className={styles.progressStat}>
              <span className={styles.progressLabel}>Avg Lap Time</span>
              <span className={styles.progressValue}>1:23.456</span>
              <span className={styles.progressChange}>-0.234s</span>
            </div>
            <div className={styles.progressStat}>
              <span className={styles.progressLabel}>Consistency</span>
              <span className={styles.progressValue}>92%</span>
              <span className={styles.progressChange}>+5%</span>
            </div>
            <div className={styles.progressStat}>
              <span className={styles.progressLabel}>Incidents</span>
              <span className={styles.progressValue}>2</span>
              <span className={styles.progressChange}>-3</span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};