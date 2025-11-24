import React, { useState, useEffect } from 'react';
import { Card, StatusIndicator } from '../common';
import styles from './SpecialistConsensus.module.css';

interface Specialist {
  name: string;
  role: string;
  recommendation: string;
  confidence: 'high' | 'medium' | 'low';
  status: 'active' | 'processing' | 'standby';
}

const specialists: Specialist[] = [
  {
    name: 'MINERVA',
    role: 'Strategic Analysis',
    recommendation: 'PIT LAP 42',
    confidence: 'high',
    status: 'active'
  },
  {
    name: 'ATLAS',
    role: 'Track Positioning',
    recommendation: 'T4 WIDER LINE',
    confidence: 'medium',
    status: 'active'
  },
  {
    name: 'IRIS',
    role: 'Vehicle Dynamics',
    recommendation: 'TEMPS OPTIMAL',
    confidence: 'high',
    status: 'active'
  },
  {
    name: 'CHRONOS',
    role: 'Timing Analysis',
    recommendation: 'PUSH NOW',
    confidence: 'high',
    status: 'processing'
  },
  {
    name: 'PROMETHEUS',
    role: 'Predictive Strategy',
    recommendation: 'UNDERCUT P2',
    confidence: 'medium',
    status: 'active'
  }
];

export const SpecialistConsensus: React.FC = () => {
  const [activeSpecialists, setActiveSpecialists] = useState(specialists);
  const [consensusScore, setConsensusScore] = useState(87);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveSpecialists(prev => 
        prev.map(specialist => ({
          ...specialist,
          status: Math.random() > 0.7 ? 'processing' : 'active'
        }))
      );
      setConsensusScore(prev => Math.min(100, Math.max(0, prev + (Math.random() - 0.5) * 5)));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string): 'success' | 'warning' | 'info' => {
    switch (status) {
      case 'active': return 'success';
      case 'processing': return 'warning';
      default: return 'info';
    }
  };

  const getConfidenceColor = (confidence: string): string => {
    switch (confidence) {
      case 'high': return styles.highConfidence;
      case 'medium': return styles.mediumConfidence;
      case 'low': return styles.lowConfidence;
      default: return '';
    }
  };

  return (
    <Card title="SPECIALIST CONSENSUS">
      <div className={styles.consensusHeader}>
        <div className={styles.scoreContainer}>
          <span className={styles.scoreLabel}>Consensus Score</span>
          <span className={styles.scoreValue}>{consensusScore}%</span>
        </div>
        <div className={styles.consensusBar}>
          <div 
            className={styles.consensusFill}
            style={{ width: `${consensusScore}%` }}
          />
        </div>
      </div>
      
      <div className={styles.specialistList}>
        {activeSpecialists.map((specialist, index) => (
          <div key={index} className={styles.specialist}>
            <div className={styles.specialistHeader}>
              <div className={styles.specialistInfo}>
                <span className={styles.name}>{specialist.name}</span>
                <StatusIndicator 
                  status={getStatusColor(specialist.status)} 
                  size="sm"
                  pulse={specialist.status === 'processing'}
                />
              </div>
              <span className={`${styles.confidence} ${getConfidenceColor(specialist.confidence)}`}>
                {specialist.confidence.toUpperCase()}
              </span>
            </div>
            
            <div className={styles.specialistContent}>
              <span className={styles.role}>{specialist.role}</span>
              <span className={styles.recommendation}>{specialist.recommendation}</span>
            </div>
          </div>
        ))}
      </div>
      
      <div className={styles.footer}>
        <div className={styles.primaryAction}>
          <StatusIndicator status="success" size="sm" pulse />
          <span className={styles.actionText}>
            Primary: Execute undercut strategy at lap 42
          </span>
        </div>
      </div>
    </Card>
  );
};