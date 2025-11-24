import React from 'react';
import styles from './StatusIndicator.module.css';

type Status = 'success' | 'warning' | 'danger' | 'info' | 'neutral';

interface StatusIndicatorProps {
  status: Status;
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  pulse?: boolean;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  label,
  size = 'md',
  pulse = false
}) => {
  return (
    <div className={styles.container}>
      <span 
        className={`
          ${styles.indicator} 
          ${styles[status]} 
          ${styles[size]}
          ${pulse ? styles.pulse : ''}
        `}
      />
      {label && <span className={styles.label}>{label}</span>}
    </div>
  );
};