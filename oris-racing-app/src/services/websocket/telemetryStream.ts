import { getTelemetryDB } from '../database/influxdb';

interface WebSocketConfig {
  url: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface TelemetrySubscription {
  trackId: string;
  sessionId: string;
  fields?: string[];
  onData: (data: any) => void;
  onError?: (error: Error) => void;
}

export class TelemetryWebSocket {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private subscriptions: Map<string, TelemetrySubscription> = new Map();
  private reconnectAttempts = 0;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private isConnecting = false;

  constructor(config: WebSocketConfig) {
    this.config = {
      url: config.url,
      reconnectInterval: config.reconnectInterval || 5000,
      maxReconnectAttempts: config.maxReconnectAttempts || 10
    };
  }

  connect(): void {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      return;
    }

    this.isConnecting = true;

    try {
      this.ws = new WebSocket(this.config.url);

      this.ws.onopen = () => {
        console.log('Telemetry WebSocket connected');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        
        // Resubscribe to all active subscriptions
        this.subscriptions.forEach((sub, id) => {
          this.sendSubscription(id, sub);
        });
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.isConnecting = false;
      };

      this.ws.onclose = () => {
        console.log('WebSocket connection closed');
        this.isConnecting = false;
        this.ws = null;
        this.attemptReconnect();
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      this.isConnecting = false;
      this.attemptReconnect();
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.notifyAllSubscribersError(new Error('Connection lost'));
      return;
    }

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    this.reconnectAttempts++;
    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.config.maxReconnectAttempts})...`);

    this.reconnectTimeout = setTimeout(() => {
      this.connect();
    }, this.config.reconnectInterval);
  }

  private handleMessage(message: any): void {
    if (message.type === 'telemetry') {
      const subscriptionId = `${message.trackId}-${message.sessionId}`;
      const subscription = this.subscriptions.get(subscriptionId);
      
      if (subscription) {
        subscription.onData(message.data);
        
        // Also write to InfluxDB
        if (message.persist !== false) {
          this.persistTelemetry(message);
        }
      }
    } else if (message.type === 'error') {
      const subscriptionId = `${message.trackId}-${message.sessionId}`;
      const subscription = this.subscriptions.get(subscriptionId);
      
      if (subscription && subscription.onError) {
        subscription.onError(new Error(message.error));
      }
    }
  }

  private async persistTelemetry(message: any): Promise<void> {
    try {
      const telemetryDB = getTelemetryDB();
      await telemetryDB.writeTelemetryPoint({
        timestamp: new Date(message.timestamp),
        trackId: message.trackId,
        sessionId: message.sessionId,
        driverId: message.driverId,
        lapNumber: message.lapNumber,
        data: message.data
      });
    } catch (error) {
      console.error('Error persisting telemetry:', error);
    }
  }

  subscribe(subscription: TelemetrySubscription): string {
    const subscriptionId = `${subscription.trackId}-${subscription.sessionId}`;
    
    this.subscriptions.set(subscriptionId, subscription);
    
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.sendSubscription(subscriptionId, subscription);
    }
    
    return subscriptionId;
  }

  private sendSubscription(id: string, subscription: TelemetrySubscription): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    this.ws.send(JSON.stringify({
      type: 'subscribe',
      trackId: subscription.trackId,
      sessionId: subscription.sessionId,
      fields: subscription.fields
    }));
  }

  unsubscribe(subscriptionId: string): void {
    const subscription = this.subscriptions.get(subscriptionId);
    
    if (subscription && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'unsubscribe',
        trackId: subscription.trackId,
        sessionId: subscription.sessionId
      }));
    }
    
    this.subscriptions.delete(subscriptionId);
  }

  private notifyAllSubscribersError(error: Error): void {
    this.subscriptions.forEach(subscription => {
      if (subscription.onError) {
        subscription.onError(error);
      }
    });
  }

  disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.subscriptions.clear();
    this.reconnectAttempts = 0;
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
let telemetryWS: TelemetryWebSocket | null = null;

export function initializeTelemetryWebSocket(config: WebSocketConfig): TelemetryWebSocket {
  if (!telemetryWS) {
    telemetryWS = new TelemetryWebSocket(config);
    telemetryWS.connect();
  }
  return telemetryWS;
}

export function getTelemetryWebSocket(): TelemetryWebSocket {
  if (!telemetryWS) {
    throw new Error('TelemetryWebSocket not initialized. Call initializeTelemetryWebSocket first.');
  }
  return telemetryWS;
}