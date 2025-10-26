import React, { useState, useEffect } from 'react';
import { Card } from '../components/common';
import { TelemetryGraphs } from '../components/telemetry/TelemetryGraphs';
import { TrackSelector } from '../components/TrackSelector';
import { Activity, Gauge, Thermometer, Zap } from 'lucide-react';
import styles from './Telemetry.module.css';

interface TelemetryData {
  speed: number;
  rpm: number;
  gear: number;
  throttle: number;
  brake: number;
  steeringAngle: number;
  gForceLateral: number;
  gForceLongitudinal: number;
  tireTemps: {
    fl: number;
    fr: number;
    rl: number;
    rr: number;
  };
  brakeTemps: {
    fl: number;
    fr: number;
    rl: number;
    rr: number;
  };
  engineTemp: number;
  oilPressure: number;
  fuelLevel: number;
}

export const Telemetry: React.FC = () => {
  const [selectedTrack, setSelectedTrack] = useState('cota');
  const [selectedRace, setSelectedRace] = useState<1 | 2>(1);
  const [liveData, setLiveData] = useState<TelemetryData | null>(null);

  useEffect(() => {
    // TODO: Connect to InfluxDB WebSocket for live data
    const mockData: TelemetryData = {
      speed: 156,
      rpm: 7200,
      gear: 4,
      throttle: 85,
      brake: 0,
      steeringAngle: -15,
      gForceLateral: 1.2,
      gForceLongitudinal: -0.3,
      tireTemps: { fl: 92, fr: 94, rl: 88, rr: 89 },
      brakeTemps: { fl: 380, fr: 390, rl: 350, rr: 360 },
      engineTemp: 98,
      oilPressure: 55,
      fuelLevel: 62
    };
    setLiveData(mockData);
  }, []);

  return (
    <div className={styles.telemetryPage}>
      <div className={styles.header}>
        <h1 className={styles.title}>
          <Activity size={24} />
          Live Telemetry
        </h1>
        <TrackSelector
          selectedTrack={selectedTrack}
          onTrackChange={setSelectedTrack}
          selectedRace={selectedRace}
          onRaceChange={setSelectedRace}
        />
      </div>

      <div className={styles.grid}>
        <Card className={styles.gaugeSection}>
          <h2><Gauge size={18} /> Performance Metrics</h2>
          {liveData && (
            <div className={styles.gaugeGrid}>
              <div className={styles.gauge}>
                <span className={styles.gaugeLabel}>Speed</span>
                <span className={styles.gaugeValue}>{liveData.speed} km/h</span>
              </div>
              <div className={styles.gauge}>
                <span className={styles.gaugeLabel}>RPM</span>
                <span className={styles.gaugeValue}>{liveData.rpm}</span>
              </div>
              <div className={styles.gauge}>
                <span className={styles.gaugeLabel}>Gear</span>
                <span className={styles.gaugeValue}>{liveData.gear}</span>
              </div>
              <div className={styles.gauge}>
                <span className={styles.gaugeLabel}>Throttle</span>
                <span className={styles.gaugeValue}>{liveData.throttle}%</span>
              </div>
            </div>
          )}
        </Card>

        <Card className={styles.temperatureSection}>
          <h2><Thermometer size={18} /> Temperatures</h2>
          {liveData && (
            <div className={styles.tempGrid}>
              <div className={styles.tempGroup}>
                <h3>Tires (°C)</h3>
                <div className={styles.tireTemps}>
                  <div>FL: {liveData.tireTemps.fl}</div>
                  <div>FR: {liveData.tireTemps.fr}</div>
                  <div>RL: {liveData.tireTemps.rl}</div>
                  <div>RR: {liveData.tireTemps.rr}</div>
                </div>
              </div>
              <div className={styles.tempGroup}>
                <h3>Brakes (°C)</h3>
                <div className={styles.brakeTemps}>
                  <div>FL: {liveData.brakeTemps.fl}</div>
                  <div>FR: {liveData.brakeTemps.fr}</div>
                  <div>RL: {liveData.brakeTemps.rl}</div>
                  <div>RR: {liveData.brakeTemps.rr}</div>
                </div>
              </div>
            </div>
          )}
        </Card>

        <Card className={styles.gForceSection}>
          <h2><Zap size={18} /> G-Forces</h2>
          {liveData && (
            <div className={styles.gForceDisplay}>
              <div className={styles.gForce}>
                <span>Lateral</span>
                <span>{liveData.gForceLateral.toFixed(2)}g</span>
              </div>
              <div className={styles.gForce}>
                <span>Longitudinal</span>
                <span>{liveData.gForceLongitudinal.toFixed(2)}g</span>
              </div>
            </div>
          )}
        </Card>

        <div className={styles.graphsSection}>
          <TelemetryGraphs />
        </div>
      </div>
    </div>
  );
};