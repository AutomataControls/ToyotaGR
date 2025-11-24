import React from 'react';
import { getTrackConfig, getCurrentSector } from '../../data/trackMaps';
import styles from './TrackVisualization.module.css';

interface TrackVisualizationProps {
  trackId: string;
  driverProgress: number; // 0-100 percentage
  currentLap: number;
  totalLaps: number;
}

export const TrackVisualization: React.FC<TrackVisualizationProps> = ({
  trackId,
  driverProgress,
  currentLap,
  totalLaps
}) => {
  const track = getTrackConfig(trackId);
  if (!track) return null;

  const currentSector = getCurrentSector(driverProgress, track.sectors);
  
  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h4 className={styles.trackName}>{track.name}</h4>
        <span className={styles.lapInfo}>Lap {currentLap}/{totalLaps}</span>
      </div>
      
      <div className={styles.trackMap}>
        {track.svgPath && (
          <svg viewBox="0 0 300 250" className={styles.svg}>
            {/* Track outline */}
            <path 
              id="trackPath"
              d={track.svgPath} 
              fill="none" 
              stroke="#e5e7eb" 
              strokeWidth="15"
              className={styles.trackPath}
            />
            
            {/* Progress indicator */}
            <path 
              d={track.svgPath} 
              fill="none" 
              stroke="#14b8a6" 
              strokeWidth="15"
              strokeDasharray={`${driverProgress * 10} 1000`}
              className={styles.progressPath}
            />
            
            {/* Car position - animated based on progress */}
            <circle
              r="10"
              fill="#ef4444"
              className={styles.carDot}
            >
              <animateMotion
                dur="0.001s"
                fill="freeze"
                keyPoints={`0;${driverProgress / 100}`}
                keyTimes="0;1"
              >
                <mpath href="#trackPath" />
              </animateMotion>
              <animate
                attributeName="r"
                values="10;12;10"
                dur="1.5s"
                repeatCount="indefinite"
              />
            </circle>
            
            {/* Start/Finish line */}
            <line 
              x1="80" 
              y1="175" 
              x2="80" 
              y2="185" 
              stroke="#000" 
              strokeWidth="3"
              strokeDasharray="3,3"
            />
            <text 
              x="80" 
              y="170" 
              textAnchor="middle" 
              fontSize="10" 
              fontWeight="bold"
            >
              S/F
            </text>
          </svg>
        )}
      </div>
      
      <div className={styles.info}>
        <div className={styles.infoRow}>
          <span className={styles.infoLabel}>Current Sector</span>
          <span className={`${styles.infoValue} ${styles[currentSector.difficulty]}`}>
            {currentSector.name}
          </span>
        </div>
        <div className={styles.infoRow}>
          <span className={styles.infoLabel}>Track Length</span>
          <span className={styles.infoValue}>{track.length} miles</span>
        </div>
        <div className={styles.infoRow}>
          <span className={styles.infoLabel}>Progress</span>
          <div className={styles.progressBar}>
            <div 
              className={styles.progressFill}
              style={{ width: `${driverProgress}%` }}
            />
          </div>
          <span className={styles.progressText}>{driverProgress.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
};