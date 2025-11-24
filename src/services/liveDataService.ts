/**
 * Live Data Service - Manages real-time connection to field cars
 * Integrates with ORIS AI models and telemetry system
 */

import type { 
  LiveDataConfig, 
  LiveCarConnection, 
  LiveTelemetryData, 
  LiveDataStatus 
} from '../types/liveData';

class LiveDataService {
  private config: LiveDataConfig;
  private isEnabled: boolean = false;
  private connections: Map<number, LiveCarConnection> = new Map();
  private recentPackets: LiveTelemetryData[] = [];
  private subscribers: Array<(data: LiveTelemetryData) => void> = [];
  private statusSubscribers: Array<(status: LiveDataStatus) => void> = [];
  private pollInterval: NodeJS.Timeout | null = null;

  constructor(config: Partial<LiveDataConfig> = {}) {
    this.config = {
      apiUrl: config.apiUrl || 'http://localhost:8000',
      pollInterval: config.pollInterval || 1000,
      maxRetries: config.maxRetries || 3
    };
  }

  async initialize(): Promise<void> {
    try {
      await this.checkAPIConnection();
      console.log('🔗 Live Data Service initialized successfully');
    } catch (error) {
      console.warn('⚠️ Live Data Service initialization failed:', error);
      throw error;
    }
  }

  private async checkAPIConnection(): Promise<void> {
    const response = await fetch(`${this.config.apiUrl}/live-data/status`);
    if (!response.ok) {
      throw new Error(`API connection failed: ${response.statusText}`);
    }
  }

  async toggleLiveData(): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.apiUrl}/live-data/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`Toggle failed: ${response.statusText}`);
      }

      const result = await response.json();
      this.isEnabled = result.enabled;

      if (this.isEnabled) {
        this.startPolling();
        console.log('🟢 Live data feed enabled');
      } else {
        this.stopPolling();
        this.connections.clear();
        console.log('🔴 Live data feed disabled');
      }

      return this.isEnabled;
    } catch (error) {
      console.error('❌ Failed to toggle live data:', error);
      throw error;
    }
  }

  async getStatus(): Promise<LiveDataStatus> {
    try {
      const response = await fetch(`${this.config.apiUrl}/live-data/status`);
      if (!response.ok) {
        throw new Error(`Status check failed: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('❌ Failed to get live data status:', error);
      throw error;
    }
  }

  async getConnectedCars(): Promise<LiveCarConnection[]> {
    try {
      const response = await fetch(`${this.config.apiUrl}/live-data/cars`);
      if (!response.ok) {
        throw new Error(`Failed to get connected cars: ${response.statusText}`);
      }
      
      const data = await response.json();
      const connections: LiveCarConnection[] = [];

      for (const [carNumber, info] of Object.entries(data.cars as any)) {
        connections.push({
          id: `car-${carNumber}`,
          carNumber: parseInt(carNumber),
          driverName: (info as any).driver_name,
          status: (info as any).status,
          lastUpdate: new Date((info as any).last_update),
          dataRate: 60,
          packetsReceived: 0,
          packetsLost: 0
        });
      }

      this.connections.clear();
      connections.forEach(conn => {
        this.connections.set(conn.carNumber, conn);
      });

      return connections;
    } catch (error) {
      console.error('❌ Failed to get connected cars:', error);
      throw error;
    }
  }

  async getRecentPackets(limit: number = 20): Promise<LiveTelemetryData[]> {
    try {
      const response = await fetch(`${this.config.apiUrl}/live-data/recent?limit=${limit}`);
      if (!response.ok) {
        throw new Error(`Failed to get recent packets: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.packets.map((packet: any) => ({
        timestamp: packet.timestamp,
        carNumber: packet.car_number,
        speed: packet.telemetry.speed || 0,
        throttle: packet.telemetry.throttle || 0,
        brake: packet.telemetry.brake || 0,
        gear: packet.telemetry.gear || 1,
        rpm: packet.telemetry.rpm || 0,
        steering: packet.telemetry.steering || 0,
        gForceX: packet.telemetry.gForceX || 0,
        gForceY: packet.telemetry.gForceY || 0,
        lapNumber: packet.telemetry.lap || 1,
        sector: packet.telemetry.sector || 1
      }));
    } catch (error) {
      console.error('❌ Failed to get recent packets:', error);
      throw error;
    }
  }

  private startPolling(): void {
    if (this.pollInterval) return;

    this.pollInterval = setInterval(async () => {
      try {
        const packets = await this.getRecentPackets(5);
        packets.forEach(packet => {
          this.notifySubscribers(packet);
        });

        const status = await this.getStatus();
        this.notifyStatusSubscribers(status);

        await this.getConnectedCars();

      } catch (error) {
        console.warn('⚠️ Polling error:', error);
      }
    }, this.config.pollInterval);
  }

  private stopPolling(): void {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  subscribeToTelemetry(callback: (data: LiveTelemetryData) => void): () => void {
    this.subscribers.push(callback);
    
    return () => {
      const index = this.subscribers.indexOf(callback);
      if (index > -1) {
        this.subscribers.splice(index, 1);
      }
    };
  }

  subscribeToStatus(callback: (status: LiveDataStatus) => void): () => void {
    this.statusSubscribers.push(callback);
    
    return () => {
      const index = this.statusSubscribers.indexOf(callback);
      if (index > -1) {
        this.statusSubscribers.splice(index, 1);
      }
    };
  }

  private notifySubscribers(data: LiveTelemetryData): void {
    this.subscribers.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error('❌ Subscriber callback error:', error);
      }
    });
  }

  private notifyStatusSubscribers(status: LiveDataStatus): void {
    this.statusSubscribers.forEach(callback => {
      try {
        callback(status);
      } catch (error) {
        console.error('❌ Status subscriber callback error:', error);
      }
    });
  }

  async simulateTelemetry(carNumber: number, driverName: string): Promise<void> {
    const telemetryPacket = {
      car_number: carNumber,
      driver_name: driverName,
      timestamp: new Date().toISOString(),
      telemetry: {
        speed: Math.random() * 200 + 50,
        throttle: Math.random(),
        brake: Math.random() * 0.3,
        gear: Math.floor(Math.random() * 6) + 1,
        rpm: Math.random() * 3000 + 5000,
        steering: (Math.random() - 0.5) * 90,
        gForceX: (Math.random() - 0.5) * 4,
        gForceY: Math.random() * 3 - 1,
        lap: Math.floor(Math.random() * 50) + 1,
        sector: Math.floor(Math.random() * 3) + 1
      },
      session_info: {
        session_type: 'race',
        track: 'cota'
      }
    };

    try {
      const response = await fetch(`${this.config.apiUrl}/live-data/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(telemetryPacket)
      });

      if (!response.ok) {
        throw new Error(`Failed to send telemetry: ${response.statusText}`);
      }

      console.log(`📡 Simulated telemetry sent for Car #${carNumber}`);
    } catch (error) {
      console.error(`❌ Failed to simulate telemetry for Car #${carNumber}:`, error);
    }
  }

  get enabled(): boolean {
    return this.isEnabled;
  }

  get activeConnections(): LiveCarConnection[] {
    return Array.from(this.connections.values());
  }

  dispose(): void {
    this.stopPolling();
    this.subscribers.length = 0;
    this.statusSubscribers.length = 0;
    this.connections.clear();
    console.log('🧹 Live Data Service disposed');
  }
}

export const liveDataService = new LiveDataService();
export default LiveDataService;