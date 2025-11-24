// Live Data Types
export interface LiveDataConfig {
  apiUrl: string;
  pollInterval: number;
  maxRetries: number;
}

export interface LiveCarConnection {
  id: string;
  carNumber: number;
  driverName: string;
  status: 'connected' | 'disconnected' | 'error';
  lastUpdate: Date;
  dataRate: number;
  packetsReceived: number;
  packetsLost: number;
}

export interface LiveTelemetryData {
  timestamp: string;
  carNumber: number;
  speed: number;
  throttle: number;
  brake: number;
  gear: number;
  rpm: number;
  steering: number;
  gForceX: number;
  gForceY: number;
  lapNumber: number;
  sector: number;
}

export interface LiveDataStatus {
  enabled: boolean;
  connectedCars: number;
  totalPackets: number;
  packetsPerSecond: number;
  lastUpdate: string;
}