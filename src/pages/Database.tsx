import React, { useState, useEffect } from 'react';
import { Card } from '../components/common';
import { liveDataService } from '../services/liveDataService';
import type { LiveCarConnection, LiveTelemetryData, LiveDataStatus } from '../types/liveData';
import { 
  Database as DatabaseIcon, 
  Wifi, 
  WifiOff,
  Play,
  Pause,
  Activity,
  Server,
  Car,
  Users,
  Clock,
  Signal,
  AlertCircle,
  CheckCircle
} from 'lucide-react';
import styles from './Database.module.css';

// Interfaces moved to liveDataService.ts

export const Database: React.FC = () => {
  const [isLiveDataEnabled, setIsLiveDataEnabled] = useState(false);
  const [connections, setConnections] = useState<LiveCarConnection[]>([]);
  const [recentPackets, setRecentPackets] = useState<LiveTelemetryData[]>([]);
  const [status, setStatus] = useState<LiveDataStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Initialize live data service
  useEffect(() => {
    const initializeService = async () => {
      try {
        await liveDataService.initialize();
        
        // Get initial status
        const initialStatus = await liveDataService.getStatus();
        setStatus(initialStatus);
        setIsLiveDataEnabled(initialStatus.enabled);
        
        // Get initial connections
        if (initialStatus.enabled) {
          const initialConnections = await liveDataService.getConnectedCars();
          setConnections(initialConnections);
          
          const initialPackets = await liveDataService.getRecentPackets(20);
          setRecentPackets(initialPackets);
        }
        
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to initialize live data service');
        console.error('❌ Live data service initialization failed:', err);
      }
    };

    initializeService();

    // Subscribe to live data updates
    const unsubscribeTelemetry = liveDataService.subscribeToTelemetry((data) => {
      setRecentPackets(prev => [data, ...prev.slice(0, 19)]);
    });

    const unsubscribeStatus = liveDataService.subscribeToStatus((newStatus) => {
      setStatus(newStatus);
      setIsLiveDataEnabled(newStatus.enabled);
    });

    return () => {
      unsubscribeTelemetry();
      unsubscribeStatus();
    };
  }, []);

  // Update connections periodically when enabled
  useEffect(() => {
    if (!isLiveDataEnabled) return;

    const updateConnections = async () => {
      try {
        const currentConnections = await liveDataService.getConnectedCars();
        setConnections(currentConnections);
      } catch (err) {
        console.warn('⚠️ Failed to update connections:', err);
      }
    };

    const interval = setInterval(updateConnections, 2000);
    return () => clearInterval(interval);
  }, [isLiveDataEnabled]);

  const toggleLiveData = async () => {
    try {
      setError(null);
      const newState = await liveDataService.toggleLiveData();
      setIsLiveDataEnabled(newState);
      
      // Clear data when disabled
      if (!newState) {
        setConnections([]);
        setRecentPackets([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle live data');
      console.error('❌ Failed to toggle live data:', err);
    }
  };

  const getStatusColor = (status: LiveCarConnection['status']) => {
    switch (status) {
      case 'connected': return '#10b981';
      case 'disconnected': return '#6b7280';
      case 'error': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getStatusIcon = (status: LiveCarConnection['status']) => {
    switch (status) {
      case 'connected': return <CheckCircle size={16} />;
      case 'disconnected': return <WifiOff size={16} />;
      case 'error': return <AlertCircle size={16} />;
      default: return <WifiOff size={16} />;
    }
  };

  return (
    <div className={styles.databasePage}>
      <div className={styles.header}>
        <h1 className={styles.title}>
          <DatabaseIcon size={24} />
          Live Data Feed
        </h1>
        
        <div className={styles.controls}>
          {error && (
            <div className={styles.error}>
              <AlertCircle size={16} />
              {error}
            </div>
          )}
          
          <div className={styles.toggleContainer}>
            <label className={styles.toggleLabel}>
              Live Data Feed
            </label>
            <button
              className={`${styles.toggle} ${isLiveDataEnabled ? styles.active : ''}`}
              onClick={toggleLiveData}
            >
              <div className={styles.toggleSlider}>
                {isLiveDataEnabled ? <Play size={12} /> : <Pause size={12} />}
              </div>
            </button>
          </div>
          
          {isLiveDataEnabled && (
            <button 
              className={styles.testButton}
              onClick={() => liveDataService.simulateTelemetry(99, 'Test Driver')}
            >
              Send Test Data
            </button>
          )}
        </div>
      </div>

      <div className={styles.statsGrid}>
        <Card className={styles.statCard}>
          <div className={styles.statHeader}>
            <Server size={18} />
            <span>Connection Status</span>
          </div>
          <div className={styles.statValue}>
            {isLiveDataEnabled ? 'ACTIVE' : 'INACTIVE'}
          </div>
          <div className={styles.statSubtext}>
            Live feed {isLiveDataEnabled ? 'enabled' : 'disabled'}
          </div>
        </Card>

        <Card className={styles.statCard}>
          <div className={styles.statHeader}>
            <Car size={18} />
            <span>Connected Cars</span>
          </div>
          <div className={styles.statValue}>
            {status?.connectedCars || 0}
          </div>
          <div className={styles.statSubtext}>
            connected cars
          </div>
        </Card>

        <Card className={styles.statCard}>
          <div className={styles.statHeader}>
            <Signal size={18} />
            <span>Data Rate</span>
          </div>
          <div className={styles.statValue}>
            {(status?.packetsPerSecond || 0).toFixed(0)}Hz
          </div>
          <div className={styles.statSubtext}>
            packets per second
          </div>
        </Card>

        <Card className={styles.statCard}>
          <div className={styles.statHeader}>
            <Activity size={18} />
            <span>Total Packets</span>
          </div>
          <div className={styles.statValue}>
            {(status?.totalPackets || 0).toLocaleString()}
          </div>
          <div className={styles.statSubtext}>
            this session
          </div>
        </Card>
      </div>

      <div className={styles.mainContent}>
        <div className={styles.leftColumn}>
          <Card className={styles.connectionsCard}>
            <h2 className={styles.cardTitle}>
              <Users size={18} />
              Field Car Connections
            </h2>
            
            <div className={styles.connectionsList}>
              {connections.map(connection => (
                <div key={connection.id} className={styles.connectionItem}>
                  <div className={styles.connectionInfo}>
                    <div className={styles.connectionHeader}>
                      <span className={styles.carNumber}>#{connection.carNumber}</span>
                      <span 
                        className={styles.connectionStatus}
                        style={{ color: getStatusColor(connection.status) }}
                      >
                        {getStatusIcon(connection.status)}
                        {connection.status.toUpperCase()}
                      </span>
                    </div>
                    <div className={styles.driverName}>{connection.driverName}</div>
                    <div className={styles.connectionStats}>
                      <div className={styles.stat}>
                        <Clock size={12} />
                        {Math.floor((Date.now() - connection.lastUpdate.getTime()) / 1000)}s ago
                      </div>
                      <div className={styles.stat}>
                        <Signal size={12} />
                        {connection.dataRate}Hz
                      </div>
                      <div className={styles.stat}>
                        Packets: {(connection.packetsReceived || 0).toLocaleString()}
                      </div>
                      <div className={styles.stat}>
                        Lost: {connection.packetsLost}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        <div className={styles.rightColumn}>
          <Card className={styles.packetsCard}>
            <h2 className={styles.cardTitle}>
              <Activity size={18} />
              Live Telemetry Stream
            </h2>
            
            <div className={styles.packetsContainer}>
              <div className={styles.packetsHeader}>
                <div>Timestamp</div>
                <div>Car</div>
                <div>Speed</div>
                <div>Throttle</div>
                <div>Brake</div>
                <div>Gear</div>
                <div>Lap/Sector</div>
              </div>
              
              <div className={styles.packetsList}>
                {recentPackets.map((packet, index) => (
                  <div key={index} className={styles.packetItem}>
                    <div>{packet.timestamp ? new Date(packet.timestamp).toLocaleTimeString() : 'N/A'}</div>
                    <div>#{packet.carNumber}</div>
                    <div>{(packet.speed || 0).toFixed(1)} km/h</div>
                    <div>{((packet.throttle || 0) * 100).toFixed(0)}%</div>
                    <div>{((packet.brake || 0) * 100).toFixed(0)}%</div>
                    <div>{packet.gear}</div>
                    <div>L{packet.lapNumber}/S{packet.sector}</div>
                  </div>
                ))}
                
                {recentPackets.length === 0 && (
                  <div className={styles.noData}>
                    {isLiveDataEnabled ? 
                      'Waiting for telemetry data...' : 
                      'Enable live data feed to see telemetry stream'
                    }
                  </div>
                )}
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};