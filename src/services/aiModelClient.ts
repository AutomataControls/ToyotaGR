/**
 * AI Model Client - TypeScript service for calling ORIS AI models
 * Connects React components to Python AI model API
 */

export interface TelemetryData {
  timestamp: string;
  sessionId: string;
  trackId: string;
  carNumber: number;
  currentLap: number;
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
      tires: { frontLeft: number; frontRight: number; rearLeft: number; rearRight: number; };
      brakes: { frontLeft: number; frontRight: number; rearLeft: number; rearRight: number; };
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
}

export interface StrategyRequest {
  telemetry_sequence: number[][];
  race_context?: Record<string, any>;
}

export interface MinervaResponse {
  model: string;
  predictions: {
    pit_strategy: number[]; // [now, 1lap, 2lap, 3lap, no_pit]
    pace_strategy: number[]; // [push, maintain, conserve]
    tire_strategy: number[]; // [soft, medium, hard, current]
    traffic_strategy: number[]; // [overtake, follow, defend, let_pass]
    tire_degradation: number;
    fuel_strategy: number;
    confidence: number;
  };
  confidence: number;
  timestamp: string;
  recommendations: {
    pit_strategy: string;
    pace_strategy: string;
    tire_warning: string;
    traffic_strategy?: string;
  };
}

export interface AtlasResponse {
  model: string;
  predictions: {
    optimal_racing_line: number[][];
    track_position_score: number;
    sector_analysis: {
      sector1: { optimal_speed: number; current_speed: number; delta: number; };
      sector2: { optimal_speed: number; current_speed: number; delta: number; };
      sector3: { optimal_speed: number; current_speed: number; delta: number; };
    };
    confidence: number;
  };
  timestamp: string;
}

export interface IrisResponse {
  model: string;
  predictions: {
    vehicle_balance: number;
    handling_analysis: {
      understeer_tendency: number;
      oversteer_tendency: number;
      optimal_balance: number;
    };
    setup_recommendations: {
      front_wing: string;
      rear_wing: string;
      tire_pressure: string;
    };
    confidence: number;
  };
  timestamp: string;
}

export interface ChronosResponse {
  model: string;
  predictions: {
    predicted_lap_time: string;
    sector_predictions: {
      sector1: string;
      sector2: string;
      sector3: string;
    };
    time_delta_to_optimal: string;
    improvement_potential: number;
    confidence: number;
  };
  timestamp: string;
}

export interface PrometheusResponse {
  model: string;
  predictions: {
    position_forecast: {
      lap_25: { position: number; probability: number; };
      lap_35: { position: number; probability: number; };
      lap_45: { position: number; probability: number; };
    };
    incident_probability: number;
    weather_forecast: {
      rain_probability: number;
      track_temperature_trend: string;
    };
    strategic_opportunities: Array<{
      event: string;
      lap?: number;
      lap_range?: number[];
      probability: number;
    }>;
    confidence: number;
  };
  timestamp: string;
}

export interface EnsembleResponse {
  ensemble_results: {
    minerva: MinervaResponse;
    atlas: AtlasResponse;
    iris: IrisResponse;
    chronos: ChronosResponse;
    prometheus: PrometheusResponse;
  };
  consensus_score: number;
  primary_recommendation: string;
  timestamp: string;
}

class AIModelClient {
  private baseUrl: string;
  private isConnected: boolean = false;

  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.checkConnection();
  }

  private async checkConnection(): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/`);
      if (response.ok) {
        this.isConnected = true;
        console.log('🤖 Connected to ORIS AI Model API');
      }
    } catch (error) {
      this.isConnected = false;
      console.warn('⚠️ AI Model API not available, using mock data');
    }
  }

  /**
   * Convert telemetry data to model input format
   */
  private telemetryToModelInput(telemetryData: TelemetryData[]): StrategyRequest {
    const telemetry_sequence = telemetryData.map(data => [
      data.telemetry.speed,
      data.telemetry.throttle,
      data.telemetry.brake,
      data.telemetry.brake, // brake_r (assume same as brake for now)
      data.telemetry.gear,
      data.telemetry.steeringAngle,
      data.telemetry.gForce.lateral,
      data.telemetry.gForce.longitudinal
    ]);

    return {
      telemetry_sequence,
      race_context: {
        currentLap: telemetryData[telemetryData.length - 1]?.currentLap || 1,
        trackId: telemetryData[telemetryData.length - 1]?.trackId || 'cota',
        sessionId: telemetryData[telemetryData.length - 1]?.sessionId || 'test'
      }
    };
  }

  /**
   * Get strategic predictions from MINERVA
   */
  async getStrategyPredictions(telemetryData: TelemetryData[]): Promise<MinervaResponse> {
    if (!this.isConnected) {
      // Return mock data when API is not available
      return {
        model: 'minerva',
        predictions: {
          pit_strategy: [0.1, 0.3, 0.4, 0.15, 0.05],
          pace_strategy: [0.2, 0.6, 0.2],
          tire_strategy: [0.1, 0.7, 0.15, 0.05],
          traffic_strategy: [0.3, 0.4, 0.2, 0.1],
          tire_degradation: 0.65,
          fuel_strategy: 0.3,
          confidence: 0.87
        },
        confidence: 0.87,
        timestamp: new Date().toISOString(),
        recommendations: {
          pit_strategy: 'Pit in 2 Laps (confidence: 0.40)',
          pace_strategy: 'Maintain Pace (confidence: 0.60)',
          tire_warning: 'MEDIUM tire degradation: 65%'
        }
      };
    }

    try {
      const request = this.telemetryToModelInput(telemetryData);
      const response = await fetch(`${this.baseUrl}/predict/minerva`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error(`MINERVA API error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('MINERVA prediction error:', error);
      throw error;
    }
  }

  /**
   * Get spatial predictions from ATLAS
   */
  async getSpatialPredictions(telemetryData: TelemetryData[]): Promise<AtlasResponse> {
    if (!this.isConnected) {
      return {
        model: 'atlas',
        predictions: {
          optimal_racing_line: Array.from({length: 50}, (_, i) => [i/50, Math.sin(i/10)]),
          track_position_score: 0.82,
          sector_analysis: {
            sector1: { optimal_speed: 145.2, current_speed: 142.8, delta: -2.4 },
            sector2: { optimal_speed: 98.5, current_speed: 96.1, delta: -2.4 },
            sector3: { optimal_speed: 167.3, current_speed: 165.9, delta: -1.4 }
          },
          confidence: 0.78
        },
        timestamp: new Date().toISOString()
      };
    }

    const request = this.telemetryToModelInput(telemetryData);
    const response = await fetch(`${this.baseUrl}/predict/atlas`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    return await response.json();
  }

  /**
   * Get dynamics predictions from IRIS
   */
  async getDynamicsPredictions(telemetryData: TelemetryData[]): Promise<IrisResponse> {
    if (!this.isConnected) {
      return {
        model: 'iris',
        predictions: {
          vehicle_balance: 0.73,
          handling_analysis: {
            understeer_tendency: 0.25,
            oversteer_tendency: 0.15,
            optimal_balance: 0.60
          },
          setup_recommendations: {
            front_wing: '+2 clicks',
            rear_wing: 'maintain',
            tire_pressure: 'FL: -0.5 PSI, FR: -0.3 PSI'
          },
          confidence: 0.81
        },
        timestamp: new Date().toISOString()
      };
    }

    const request = this.telemetryToModelInput(telemetryData);
    const response = await fetch(`${this.baseUrl}/predict/iris`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    return await response.json();
  }

  /**
   * Get timing predictions from CHRONOS
   */
  async getTimingPredictions(telemetryData: TelemetryData[]): Promise<ChronosResponse> {
    if (!this.isConnected) {
      return {
        model: 'chronos',
        predictions: {
          predicted_lap_time: '1:23.456',
          sector_predictions: {
            sector1: '27.123',
            sector2: '28.891',
            sector3: '27.442'
          },
          time_delta_to_optimal: '+0.234',
          improvement_potential: 0.8,
          confidence: 0.85
        },
        timestamp: new Date().toISOString()
      };
    }

    const request = this.telemetryToModelInput(telemetryData);
    const response = await fetch(`${this.baseUrl}/predict/chronos`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    return await response.json();
  }

  /**
   * Get future predictions from PROMETHEUS
   */
  async getFuturePredictions(telemetryData: TelemetryData[]): Promise<PrometheusResponse> {
    if (!this.isConnected) {
      return {
        model: 'prometheus',
        predictions: {
          position_forecast: {
            lap_25: { position: 3, probability: 0.72 },
            lap_35: { position: 2, probability: 0.58 },
            lap_45: { position: 2, probability: 0.81 }
          },
          incident_probability: 0.12,
          weather_forecast: {
            rain_probability: 0.15,
            track_temperature_trend: 'stable'
          },
          strategic_opportunities: [
            { event: 'undercut_opportunity', lap: 42, probability: 0.67 },
            { event: 'safety_car_window', lap_range: [38, 44], probability: 0.23 }
          ],
          confidence: 0.79
        },
        timestamp: new Date().toISOString()
      };
    }

    const request = this.telemetryToModelInput(telemetryData);
    const response = await fetch(`${this.baseUrl}/predict/prometheus`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    return await response.json();
  }

  /**
   * Get ensemble predictions from all models
   */
  async getEnsemblePredictions(telemetryData: TelemetryData[]): Promise<EnsembleResponse> {
    if (!this.isConnected) {
      // Return mock ensemble data
      const [minerva, atlas, iris, chronos, prometheus] = await Promise.all([
        this.getStrategyPredictions(telemetryData),
        this.getSpatialPredictions(telemetryData),
        this.getDynamicsPredictions(telemetryData),
        this.getTimingPredictions(telemetryData),
        this.getFuturePredictions(telemetryData)
      ]);

      return {
        ensemble_results: { minerva, atlas, iris, chronos, prometheus },
        consensus_score: 0.87,
        primary_recommendation: 'Execute undercut strategy at lap 42',
        timestamp: new Date().toISOString()
      };
    }

    const request = this.telemetryToModelInput(telemetryData);
    const response = await fetch(`${this.baseUrl}/predict/ensemble`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    return await response.json();
  }

  /**
   * Check if AI models are available
   */
  async getModelStatus(): Promise<Record<string, boolean>> {
    try {
      const response = await fetch(`${this.baseUrl}/models/status`);
      if (!response.ok) {
        // API not reachable, return all models as offline
        return {
          minerva: false,
          atlas: false,
          iris: false,
          chronos: false,
          prometheus: false
        };
      }
      
      const status = await response.json();
      // Convert the API response format to boolean status
      return {
        minerva: status.minerva?.loaded || false,
        atlas: status.atlas?.loaded || false,
        iris: status.iris?.loaded || false,
        chronos: status.chronos?.loaded || false,
        prometheus: status.prometheus?.loaded || false
      };
    } catch (error) {
      console.warn('AI Model API not available:', error);
      // Return all models as offline if API is not reachable
      return {
        minerva: false,
        atlas: false,
        iris: false,
        chronos: false,
        prometheus: false
      };
    }
  }

  /**
   * Get connection status
   */
  getConnectionStatus(): boolean {
    return this.isConnected;
  }
}

// Export singleton instance
export const aiModelClient = new AIModelClient();
export default AIModelClient;