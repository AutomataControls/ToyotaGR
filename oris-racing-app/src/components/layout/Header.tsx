import React from 'react';
import styles from './Header.module.css';

interface HeaderProps {
  isLive?: boolean;
  sessionType?: string;
}

export const Header: React.FC<HeaderProps> = ({ isLive = false, sessionType = 'Practice' }) => {
  return (
    <header className={styles.header}>
      <div className={styles.logoSection}>
        <h1 className={styles.title}>ORIS</h1>
        <span className={styles.subtitle}>OLYMPUS Racing Intelligence System</span>
      </div>
      
      <div className={styles.sessionInfo}>
        <span className={styles.sessionType}>{sessionType}</span>
      </div>
      
      <div className={styles.statusSection}>
        <div className={`${styles.liveIndicator} ${isLive ? styles.live : ''}`}>
          {isLive ? 'LIVE' : 'OFFLINE'}
          <span className={styles.statusDot}></span>
        </div>
      </div>
    </header>
  );
};