import React from 'react';
import { trackConfigs } from '../data/trackMaps';
import styles from './TrackSelector.module.css';

interface TrackSelectorProps {
  selectedTrack: string;
  onTrackChange: (trackId: string) => void;
  selectedRace: 1 | 2;
  onRaceChange: (race: 1 | 2) => void;
}

export const TrackSelector: React.FC<TrackSelectorProps> = ({
  selectedTrack,
  onTrackChange,
  selectedRace,
  onRaceChange
}) => {
  return (
    <div className={styles.container}>
      <div className={styles.selectorGroup}>
        <label className={styles.label}>Track</label>
        <select 
          className={styles.select}
          value={selectedTrack}
          onChange={(e) => onTrackChange(e.target.value)}
        >
          {trackConfigs.map(track => (
            <option key={track.id} value={track.id}>
              {track.name} ({track.length} mi)
            </option>
          ))}
        </select>
      </div>
      
      <div className={styles.selectorGroup}>
        <label className={styles.label}>Race</label>
        <div className={styles.raceButtons}>
          <button
            className={`${styles.raceButton} ${selectedRace === 1 ? styles.active : ''}`}
            onClick={() => onRaceChange(1)}
          >
            Race 1
          </button>
          <button
            className={`${styles.raceButton} ${selectedRace === 2 ? styles.active : ''}`}
            onClick={() => onRaceChange(2)}
          >
            Race 2
          </button>
        </div>
      </div>
    </div>
  );
};