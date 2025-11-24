/**
 * Race Data Manager - Central hub for live race data integration
 * Connects to Toyota GR Cup telemetry systems and feeds ORIS components
 */

import { TelemetryWebSocket } from './websocket/telemetryStream';

// Type definitions for race data
export interface LiveRaceData {
  timestamp: string;
  sessionId: string;
  trackId: string;
  carNumber: number;
  currentLap: number;
  totalLaps: number;
  position: {
    current: number;
    starting: number;
    classPosition: number;
  };
  timing: {
    currentLapTime: string | null;
    lastLapTime: string | null;
    bestLapTime: string | null;
    sectorTimes: {
      sector1: string | null;
      sector2: string | null;
      sector3: string | null;
    };
    deltaToLeader: string;
    deltaToAhead: string;
  };
  telemetry: {
    speed: number;
    rpm: number;
    gear: number;
    throttle: number;
    brake: number;
    steeringAngle: number;
    gForce: {
      lateral: number;
      longitudinal: number;
      vertical: number;
    };
    temperatures: {
      tires: {
        frontLeft: number;
        frontRight: number;
        rearLeft: number;
        rearRight: number;
      };
      brakes: {
        frontLeft: number;
        frontRight: number;
        rearLeft: number;
        rearRight: number;
      };
      engine: number;
      oil: number;
      coolant: number;
    };
    fuel: {
      level: number;
      consumption: number;
      lapsRemaining: number;
    };
  };
  strategy: {
    currentStrategy: string;
    nextStopLap: number | null;
    tireCompound: string;
    fuelRemaining: number;
    pitWindow: {
      optimal: { start: number; end: number; recommended: number };
      factors: {
        tireDegradation: number;
        fuelRemaining: number;
        trackPositionRisk: number;
        weatherRisk: number;
      };
    };
  };
}

// Data callback types for each component
export type DashboardCallback = (data: Partial<LiveRaceData>) => void;
export type TelemetryCallback = (data: LiveRaceData['telemetry']) => void;
export type StrategyCallback = (data: LiveRaceData['strategy']) => void;
export type TimingCallback = (data: LiveRaceData['timing']) => void;

export class RaceDataManager {
  private static instance: RaceDataManager;
  private wsConnections: Map<string, WebSocket> = new Map();
  private dataCallbacks: Map<string, Function[]> = new Map();
  private currentData: Partial<LiveRaceData> = {};
  private isConnected = false;

  private constructor() {
    this.initializeCallbacks();
  }

  public static getInstance(): RaceDataManager {
    if (!RaceDataManager.instance) {
      RaceDataManager.instance = new RaceDataManager();
    }
    return RaceDataManager.instance;
  }

  private initializeCallbacks() {
    this.dataCallbacks.set('dashboard', []);
    this.dataCallbacks.set('telemetry', []);
    this.dataCallbacks.set('strategy', []);
    this.dataCallbacks.set('timing', []);
    this.dataCallbacks.set('training', []);
  }

  /**
   * Connect to race data WebSocket servers
   */
  public async connectToRaceData(config: {
    baseUrl: string;
    sessionId: string;
    carNumber: number;
  }): Promise<void> {
    const { baseUrl, sessionId, carNumber } = config;

    console.log('🏁 Connecting to Toyota GR Cup race data...', { sessionId, carNumber });

    // Connect to different data streams
    const endpoints = [
      { name: 'dashboard', url: `${baseUrl}/live/dashboard` },
      { name: 'telemetry', url: `${baseUrl}/live/telemetry` },
      { name: 'strategy', url: `${baseUrl}/live/strategy` },
      { name: 'timing', url: `${baseUrl}/live/timing` },
      { name: 'training', url: `${baseUrl}/live/training` }
    ];

    for (const endpoint of endpoints) {
      await this.connectToEndpoint(endpoint.name, endpoint.url, sessionId, carNumber);
    }

    this.isConnected = true;
    console.log('✅ Connected to all race data streams');
  }

  private async connectToEndpoint(
    name: string, 
    url: string, 
    sessionId: string, 
    carNumber: number
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const ws = new WebSocket(url);
        
        ws.onopen = () => {
          console.log(`📡 Connected to ${name} stream`);
          
          // Send authentication/subscription message
          ws.send(JSON.stringify({
            type: 'subscribe',
            sessionId,
            carNumber,
            dataType: name
          }));
          
          this.wsConnections.set(name, ws);
          resolve();
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleIncomingData(name, data);
          } catch (error) {
            console.error(`Error parsing ${name} data:`, error);
          }
        };

        ws.onerror = (error) => {
          console.error(`WebSocket error for ${name}:`, error);
          reject(error);
        };

        ws.onclose = () => {
          console.log(`❌ Disconnected from ${name} stream`);
          this.wsConnections.delete(name);
          // Attempt reconnection
          setTimeout(() => {
            this.connectToEndpoint(name, url, sessionId, carNumber);
          }, 5000);
        };

      } catch (error) {
        console.error(`Failed to connect to ${name}:`, error);
        reject(error);
      }
    });
  }

  /**
   * Handle incoming race data and distribute to subscribers
   */
  private handleIncomingData(source: string, data: any): void {
    // Update current data state
    if (source === 'dashboard') {
      this.currentData = { ...this.currentData, ...data };
    } else {
      this.currentData[source as keyof LiveRaceData] = data;
    }

    // Notify all subscribers for this data type
    const callbacks = this.dataCallbacks.get(source) || [];
    callbacks.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Error in ${source} callback:`, error);
      }
    });

    // Feed data to AI models for predictions
    this.feedAIModels(source, data);
  }

  /**
   * Feed data to AI models for real-time predictions
   */
  private async feedAIModels(source: string, data: any): Promise<void> {
    try {
      // This would integrate with your ATLAS, MINERVA, etc. models
      const aiEndpoint = `${process.env.VITE_AI_MODEL_URL}/predict`;
      
      const modelInput = {
        source,
        data,
        timestamp: new Date().toISOString(),
        sessionId: this.currentData.sessionId,
        trackId: this.currentData.trackId
      };

      // Send to AI models for processing
      fetch(aiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(modelInput)
      }).catch(error => {
        console.warn('AI model prediction failed:', error);
      });

    } catch (error) {
      console.error('Error feeding AI models:', error);
    }
  }

  /**
   * Subscribe to specific data updates
   */
  public subscribeToDashboard(callback: DashboardCallback): () => void {
    const callbacks = this.dataCallbacks.get('dashboard') || [];
    callbacks.push(callback);
    this.dataCallbacks.set('dashboard', callbacks);

    // Send current data if available
    if (this.currentData) {
      callback(this.currentData);
    }

    // Return unsubscribe function
    return () => {
      const updatedCallbacks = callbacks.filter(cb => cb !== callback);
      this.dataCallbacks.set('dashboard', updatedCallbacks);
    };
  }

  public subscribeToTelemetry(callback: TelemetryCallback): () => void {
    const callbacks = this.dataCallbacks.get('telemetry') || [];
    callbacks.push(callback);
    this.dataCallbacks.set('telemetry', callbacks);

    if (this.currentData.telemetry) {
      callback(this.currentData.telemetry);
    }

    return () => {
      const updatedCallbacks = callbacks.filter(cb => cb !== callback);
      this.dataCallbacks.set('telemetry', updatedCallbacks);
    };
  }

  public subscribeToStrategy(callback: StrategyCallback): () => void {
    const callbacks = this.dataCallbacks.get('strategy') || [];
    callbacks.push(callback);
    this.dataCallbacks.set('strategy', callbacks);

    if (this.currentData.strategy) {
      callback(this.currentData.strategy);
    }

    return () => {
      const updatedCallbacks = callbacks.filter(cb => cb !== callback);
      this.dataCallbacks.set('strategy', updatedCallbacks);
    };
  }

  public subscribeToTiming(callback: TimingCallback): () => void {
    const callbacks = this.dataCallbacks.get('timing') || [];
    callbacks.push(callback);
    this.dataCallbacks.set('timing', callbacks);

    if (this.currentData.timing) {
      callback(this.currentData.timing);
    }

    return () => {
      const updatedCallbacks = callbacks.filter(cb => cb !== callback);
      this.dataCallbacks.set('timing', updatedCallbacks);
    };
  }

  /**
   * Get current race data
   */
  public getCurrentData(): Partial<LiveRaceData> {
    return { ...this.currentData };
  }

  /**
   * Check connection status
   */
  public isConnectedToRaceData(): boolean {
    return this.isConnected && this.wsConnections.size > 0;
  }

  /**
   * Disconnect from all streams
   */
  public disconnect(): void {
    this.wsConnections.forEach((ws, name) => {
      console.log(`🔌 Disconnecting from ${name}`);
      ws.close();
    });
    this.wsConnections.clear();
    this.isConnected = false;
  }

  /**
   * Send test data (for development/testing)
   */
  public sendTestData(): void {
    const testData: LiveRaceData = {
      timestamp: new Date().toISOString(),
      sessionId: 'COTA_R2_TEST',
      trackId: 'cota',
      carNumber: 7,
      currentLap: 23,
      totalLaps: 50,
      position: { current: 4, starting: 7, classPosition: 2 },
      timing: {
        currentLapTime: '1:23.456',
        lastLapTime: '1:23.234',
        bestLapTime: '1:22.987',
        sectorTimes: {
          sector1: '27.123',
          sector2: '28.789',
          sector3: '27.544'
        },
        deltaToLeader: '+12.456',
        deltaToAhead: '+2.345'
      },
      telemetry: {
        speed: 156.7,
        rpm: 7200,
        gear: 4,
        throttle: 85.2,
        brake: 0.0,
        steeringAngle: -15.3,
        gForce: { lateral: 1.24, longitudinal: -0.32, vertical: 0.98 },
        temperatures: {
          tires: { frontLeft: 92.5, frontRight: 94.2, rearLeft: 88.1, rearRight: 89.7 },
          brakes: { frontLeft: 380.5, frontRight: 390.2, rearLeft: 350.1, rearRight: 360.7 },
          engine: 98.5,
          oil: 102.3,
          coolant: 85.7
        },
        fuel: { level: 62.5, consumption: 2.3, lapsRemaining: 27 }
      },
      strategy: {
        currentStrategy: 'two_stop',
        nextStopLap: 42,
        tireCompound: 'MEDIUM',
        fuelRemaining: 62.5,
        pitWindow: {
          optimal: { start: 12, end: 18, recommended: 15 },
          factors: {
            tireDegradation: 65,
            fuelRemaining: 42,
            trackPositionRisk: 78,
            weatherRisk: 12
          }
        }
      }
    };

    // Distribute test data to all subscribers
    this.handleIncomingData('dashboard', testData);
    this.handleIncomingData('telemetry', testData.telemetry);
    this.handleIncomingData('strategy', testData.strategy);
    this.handleIncomingData('timing', testData.timing);
  }
}

// Export singleton instance
export const raceDataManager = RaceDataManager.getInstance();