import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card } from '../common';
import styles from './TelemetryGraphs.module.css';
import { aiModelClient } from '../../services/aiModelClient';
import type { IrisResponse } from '../../services/aiModelClient';
import { toyotaDataLoader } from '../../services/toyotaDataLoader';

interface TelemetryData {
  time: number;
  speed: number;
  brake: number;
  throttle: number;
  steering: number;
}

// Simulate realistic racing telemetry for a lap at COTA
const generateTelemetryData = (): TelemetryData[] => {
  const data: TelemetryData[] = [];
  const lapProgress = 50; // 50 data points representing a lap
  
  for (let i = 0; i < lapProgress; i++) {
    const progress = i / lapProgress; // 0 to 1
    
    // Simulate different track sections (straights, corners, braking zones)
    let speed: number;
    let throttle: number;
    let brake: number;
    let steering: number;
    
    // Define track sections based on progress through lap (speeds in MPH)
    if (progress < 0.1) {
      // Start/Finish straight - acceleration
      speed = 75 + (progress * 10) * 50; // 75-125 mph
      throttle = 85 + Math.random() * 15; // High throttle
      brake = 0; // No braking
      steering = -2 + Math.random() * 4; // Minimal steering
    } else if (progress < 0.15) {
      // Turn 1 approach - heavy braking
      const brakeProgress = (progress - 0.1) / 0.05;
      speed = 125 - brakeProgress * 75; // 125-50 mph
      throttle = Math.max(0, 85 - brakeProgress * 85); // Lift off throttle
      brake = brakeProgress * 100; // Progressive braking
      steering = -15 + brakeProgress * -10; // Turn in
    } else if (progress < 0.25) {
      // Turn 1-3 complex - slow corners
      speed = 50 + Math.sin((progress - 0.15) * 20) * 10; // 40-60 mph
      throttle = 35 + Math.random() * 20; // Moderate throttle
      brake = 0; // No braking in corner
      steering = -25 + Math.sin((progress - 0.15) * 15) * 20; // Steering input
    } else if (progress < 0.4) {
      // Back straight acceleration
      const accelProgress = (progress - 0.25) / 0.15;
      speed = 60 + accelProgress * 55; // 60-115 mph
      throttle = 70 + accelProgress * 30; // Building speed
      brake = 0;
      steering = Math.sin((progress - 0.25) * 30) * 8; // Small corrections
    } else if (progress < 0.6) {
      // Technical middle section
      speed = 85 + Math.sin((progress - 0.4) * 25) * 25; // 60-110 mph
      const cornering = Math.abs(Math.sin((progress - 0.4) * 20));
      throttle = cornering > 0.5 ? 20 + Math.random() * 30 : 60 + Math.random() * 25;
      brake = cornering > 0.7 ? 40 + Math.random() * 30 : 0;
      steering = Math.sin((progress - 0.4) * 20) * 30;
    } else {
      // Final sector back to start/finish
      speed = 75 + (1 - progress) * 35 + Math.random() * 15; // 75-110 mph
      throttle = 75 + Math.random() * 20;
      brake = 0;
      steering = Math.sin((progress - 0.6) * 15) * 12;
    }
    
    // Ensure brake and throttle are mutually exclusive (realistic)
    if (brake > 5) {
      throttle = 0; // Can't brake and accelerate simultaneously
    }
    
    // Add some realistic noise
    speed = Math.max(20, speed + (Math.random() - 0.5) * 5);
    throttle = Math.max(0, Math.min(100, throttle + (Math.random() - 0.5) * 3));
    brake = Math.max(0, Math.min(100, brake + (Math.random() - 0.5) * 5));
    steering = Math.max(-45, Math.min(45, steering + (Math.random() - 0.5) * 2));
    
    data.push({
      time: i,
      speed: Math.round(speed),
      brake: Math.round(brake),
      throttle: Math.round(throttle),
      steering: Math.round(steering)
    });
  }
  return data;
};

export const TelemetryGraphs: React.FC = () => {
  const [data, setData] = useState<TelemetryData[]>([]);
  const [activeMetric, setActiveMetric] = useState<'speed' | 'inputs'>('speed');
  const [irisAnalysis, setIrisAnalysis] = useState<IrisResponse | null>(null);
  const [telemetryHistory, setTelemetryHistory] = useState<any[]>([]);
  const [isUsingToyotaData, setIsUsingToyotaData] = useState(false);

  // Load real Toyota data on mount
  useEffect(() => {
    const loadToyotaData = async () => {
      try {
        console.log('🏁 Loading real Toyota GR Cup telemetry data...');
        const toyotaData = await toyotaDataLoader.loadTelemetryData('COTA', 1);
        
        if (toyotaData.length > 0) {
          console.log(`✅ Loaded ${toyotaData.length} Toyota telemetry points`);
          
          // Convert Toyota data to component format
          const convertedData: TelemetryData[] = toyotaData.slice(0, 300).map((point, index) => ({
            time: index,
            speed: Math.round(point.speed * 0.621371), // km/h to mph
            brake: Math.max(point.brake_front, point.brake_rear),
            throttle: point.throttle,
            steering: point.steering_angle,
            gear: point.gear,
            rpm: point.engine_rpm
          }));
          
          setData(convertedData);
          setIsUsingToyotaData(true);
        } else {
          console.log('⚠️ No Toyota data available, using generated data');
          setData(generateTelemetryData());
        }
        
      } catch (error) {
        console.error('❌ Failed to load Toyota data:', error);
        setData(generateTelemetryData());
      }
    };

    loadToyotaData();
  }, []);

  useEffect(() => {
    if (isUsingToyotaData) return; // Don't update if using real Toyota data
    
    const interval = setInterval(() => {
      setData(prevData => {
        const newData = [...prevData.slice(1)];
        const lastTime = newData[newData.length - 1]?.time || 0;
        const lastPoint = newData[newData.length - 1];
        
        // Generate realistic next data point based on previous values
        let newSpeed = lastPoint?.speed || 120;
        let newThrottle = lastPoint?.throttle || 50;
        let newBrake = lastPoint?.brake || 0;
        let newSteering = lastPoint?.steering || 0;
        
        // Simulate realistic transitions
        const random = Math.random();
        
        if (random < 0.3) {
          // Braking zone
          newSpeed = Math.max(40, newSpeed - 10 - Math.random() * 15); // mph
          newThrottle = 0; // Lift off throttle
          newBrake = 70 + Math.random() * 30; // Heavy braking
          newSteering = newSteering + (Math.random() - 0.5) * 10;
        } else if (random < 0.6) {
          // Cornering
          newSpeed = Math.max(50, Math.min(85, newSpeed + (Math.random() - 0.5) * 8)); // mph
          newThrottle = 30 + Math.random() * 40; // Partial throttle
          newBrake = 0; // No braking in corners
          newSteering = Math.max(-35, Math.min(35, newSteering + (Math.random() - 0.5) * 20));
        } else {
          // Acceleration/straight
          newSpeed = Math.min(125, newSpeed + 5 + Math.random() * 8); // mph
          newThrottle = 70 + Math.random() * 30; // High throttle
          newBrake = 0; // No braking
          newSteering = newSteering * 0.8 + (Math.random() - 0.5) * 3; // Straighten out
        }
        
        // Ensure realistic constraints
        if (newBrake > 5) {
          newThrottle = 0; // Can't brake and accelerate
        }
        
        // Smooth transitions to avoid jarring jumps
        if (lastPoint) {
          newSpeed = lastPoint.speed + (newSpeed - lastPoint.speed) * 0.3;
          newThrottle = lastPoint.throttle + (newThrottle - lastPoint.throttle) * 0.4;
          newBrake = lastPoint.brake + (newBrake - lastPoint.brake) * 0.5;
          newSteering = lastPoint.steering + (newSteering - lastPoint.steering) * 0.3;
        }
        
        newData.push({
          time: lastTime + 1,
          speed: Math.round(Math.max(15, Math.min(140, newSpeed))), // mph range
          brake: Math.round(Math.max(0, Math.min(100, newBrake))),
          throttle: Math.round(Math.max(0, Math.min(100, newThrottle))),
          steering: Math.round(Math.max(-45, Math.min(45, newSteering)))
        });
        
        return newData;
      });
    }, 150); // Slightly slower update for realism

    return () => clearInterval(interval);
  }, []);

  // Get IRIS vehicle dynamics analysis
  useEffect(() => {
    const updateIrisAnalysis = async () => {
      if (telemetryHistory.length < 20) return;
      
      try {
        const analysis = await aiModelClient.getDynamicsPredictions(telemetryHistory);
        setIrisAnalysis(analysis);
        console.log('🤖 IRIS dynamics analysis:', analysis.predictions.setup_recommendations);
      } catch (error) {
        console.error('❌ IRIS analysis failed:', error);
      }
    };

    const interval = setInterval(updateIrisAnalysis, 10000); // Every 10 seconds
    return () => clearInterval(interval);
  }, [telemetryHistory]);

  // Update telemetry history for AI analysis
  useEffect(() => {
    const newTelemetryPoint = {
      timestamp: new Date().toISOString(),
      sessionId: 'COTA_R2_2024',
      trackId: 'cota', 
      carNumber: 7,
      currentLap: 23,
      telemetry: {
        speed: data[data.length - 1]?.speed || 0,
        rpm: 7200,
        gear: 4,
        throttle: data[data.length - 1]?.throttle || 0,
        brake: data[data.length - 1]?.brake || 0,
        steeringAngle: data[data.length - 1]?.steering || 0,
        gForce: { lateral: 1.2, longitudinal: -0.3, vertical: 0.98 },
        temperatures: {
          tires: { frontLeft: 92.5, frontRight: 94.2, rearLeft: 88.1, rearRight: 89.7 },
          brakes: { frontLeft: 380.5, frontRight: 390.2, rearLeft: 350.1, rearRight: 360.7 },
          engine: 98.5, oil: 102.3, coolant: 85.7
        },
        fuel: { level: 62.5, consumption: 2.3, lapsRemaining: 27 }
      }
    };

    setTelemetryHistory(prev => [...prev, newTelemetryPoint].slice(-200));
  }, [data]);

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
              <YAxis stroke="#9ca3af" domain={[30, 140]} />
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
          <span className={styles.metricValue}>{Math.round(currentValues.speed)} mph</span>
          <div className={styles.metricBar}>
            <div 
              className={styles.metricFill} 
              style={{ 
                width: `${(currentValues.speed / 140) * 100}%`,
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

      {/* Data Source Indicator */}
      <div className={styles.dataSource}>
        <h4>📊 Data Source</h4>
        <div className={styles.sourceInfo}>
          <span className={`${styles.indicator} ${isUsingToyotaData ? styles.live : styles.simulated}`}></span>
          <span>{isUsingToyotaData ? 'Real Toyota GR Cup Race Data (COTA Race 1)' : 'Simulated Racing Data'}</span>
        </div>
      </div>

      {/* IRIS AI Analysis */}
      {irisAnalysis && (
        <div className={styles.aiAnalysis}>
          <h4>🤖 IRIS Dynamics Analysis</h4>
          <div className={styles.setupRecommendations}>
            <span><strong>Vehicle Balance:</strong> {Math.round(irisAnalysis.predictions.vehicle_balance * 100)}%</span>
            <span><strong>Understeer:</strong> {Math.round(irisAnalysis.predictions.handling_analysis.understeer_tendency * 100)}%</span>
            <span><strong>Setup:</strong> {irisAnalysis.predictions.setup_recommendations.front_wing}</span>
          </div>
        </div>
      )}
    </Card>
  );
};