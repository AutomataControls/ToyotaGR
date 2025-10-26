import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card } from '../common';
import styles from './TelemetryGraphs.module.css';

interface TelemetryData {
  time: number;
  speed: number;
  brake: number;
  throttle: number;
  steering: number;
}

const generateTelemetryData = (): TelemetryData[] => {
  const data: TelemetryData[] = [];
  for (let i = 0; i < 50; i++) {
    data.push({
      time: i,
      speed: 180 + Math.sin(i * 0.2) * 60 + Math.random() * 10,
      brake: Math.max(0, Math.sin(i * 0.2 + Math.PI) * 100),
      throttle: Math.max(0, Math.cos(i * 0.2) * 100),
      steering: Math.sin(i * 0.15) * 45
    });
  }
  return data;
};

export const TelemetryGraphs: React.FC = () => {
  const [data, setData] = useState<TelemetryData[]>(generateTelemetryData());
  const [activeMetric, setActiveMetric] = useState<'speed' | 'inputs'>('speed');

  useEffect(() => {
    const interval = setInterval(() => {
      setData(prevData => {
        const newData = [...prevData.slice(1)];
        const lastTime = newData[newData.length - 1]?.time || 0;
        newData.push({
          time: lastTime + 1,
          speed: 180 + Math.sin((lastTime + 1) * 0.2) * 60 + Math.random() * 10,
          brake: Math.max(0, Math.sin((lastTime + 1) * 0.2 + Math.PI) * 100),
          throttle: Math.max(0, Math.cos((lastTime + 1) * 0.2) * 100),
          steering: Math.sin((lastTime + 1) * 0.15) * 45
        });
        return newData;
      });
    }, 100);

    return () => clearInterval(interval);
  }, []);

  const currentValues = data[data.length - 1] || {
    speed: 0,
    brake: 0,
    throttle: 0,
    steering: 0
  };

  return (
    <Card title="TELEMETRY">
      <div className={styles.metricTabs}>
        <button
          className={`${styles.tab} ${activeMetric === 'speed' ? styles.active : ''}`}
          onClick={() => setActiveMetric('speed')}
        >
          Speed
        </button>
        <button
          className={`${styles.tab} ${activeMetric === 'inputs' ? styles.active : ''}`}
          onClick={() => setActiveMetric('inputs')}
        >
          Inputs
        </button>
      </div>

      <div className={styles.chartContainer}>
        {activeMetric === 'speed' ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" stroke="#9ca3af" hide />
              <YAxis stroke="#9ca3af" domain={[100, 300]} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#ffffff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '0.375rem'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="speed" 
                stroke="#14b8a6" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" stroke="#9ca3af" hide />
              <YAxis stroke="#9ca3af" domain={[0, 100]} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#ffffff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '0.375rem'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="throttle" 
                stroke="#10b981" 
                strokeWidth={2}
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="brake" 
                stroke="#ef4444" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className={styles.liveMetrics}>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Speed</span>
          <span className={styles.metricValue}>{Math.round(currentValues.speed)} km/h</span>
          <div className={styles.metricBar}>
            <div 
              className={styles.metricFill} 
              style={{ 
                width: `${(currentValues.speed / 300) * 100}%`,
                backgroundColor: '#14b8a6' 
              }}
            />
          </div>
        </div>
        
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Brake</span>
          <span className={styles.metricValue}>{Math.round(currentValues.brake)}%</span>
          <div className={styles.metricBar}>
            <div 
              className={styles.metricFill} 
              style={{ 
                width: `${currentValues.brake}%`,
                backgroundColor: '#ef4444' 
              }}
            />
          </div>
        </div>
        
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Throttle</span>
          <span className={styles.metricValue}>{Math.round(currentValues.throttle)}%</span>
          <div className={styles.metricBar}>
            <div 
              className={styles.metricFill} 
              style={{ 
                width: `${currentValues.throttle}%`,
                backgroundColor: '#10b981' 
              }}
            />
          </div>
        </div>
      </div>
    </Card>
  );
};