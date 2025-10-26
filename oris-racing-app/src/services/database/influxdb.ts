import { InfluxDBClient, Point } from '@influxdata/influxdb3-client';

interface InfluxDBConfig {
  host: string;
  token: string;
  database: string;
  org: string;
}

interface TelemetryPoint {
  timestamp: Date;
  trackId: string;
  sessionId: string;
  driverId: string;
  lapNumber: number;
  data: {
    speed: number;
    rpm: number;
    gear: number;
    throttle: number;
    brake: number;
    steeringAngle: number;
    gForceLateral: number;
    gForceLongitudinal: number;
    position: {
      lat: number;
      lng: number;
    };
    tireTemps: {
      fl: number;
      fr: number;
      rl: number;
      rr: number;
    };
    tirePressures: {
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
    oilTemp: number;
    waterTemp: number;
    fuelLevel: number;
    fuelFlow: number;
  };
}

export class TelemetryDatabase {
  private client: InfluxDBClient;
  private database: string;

  constructor(config: InfluxDBConfig) {
    this.client = new InfluxDBClient({
      host: config.host,
      token: config.token,
      database: config.database,
      org: config.org,
    });
    this.database = config.database;
  }

  async writeTelemetryPoint(telemetry: TelemetryPoint): Promise<void> {
    const point = Point.measurement('telemetry')
      .tag('track', telemetry.trackId)
      .tag('session', telemetry.sessionId)
      .tag('driver', telemetry.driverId)
      .tag('lap', telemetry.lapNumber.toString())
      .floatField('speed', telemetry.data.speed)
      .intField('rpm', telemetry.data.rpm)
      .intField('gear', telemetry.data.gear)
      .floatField('throttle', telemetry.data.throttle)
      .floatField('brake', telemetry.data.brake)
      .floatField('steering_angle', telemetry.data.steeringAngle)
      .floatField('g_force_lateral', telemetry.data.gForceLateral)
      .floatField('g_force_longitudinal', telemetry.data.gForceLongitudinal)
      .floatField('position_lat', telemetry.data.position.lat)
      .floatField('position_lng', telemetry.data.position.lng)
      .floatField('tire_temp_fl', telemetry.data.tireTemps.fl)
      .floatField('tire_temp_fr', telemetry.data.tireTemps.fr)
      .floatField('tire_temp_rl', telemetry.data.tireTemps.rl)
      .floatField('tire_temp_rr', telemetry.data.tireTemps.rr)
      .floatField('tire_pressure_fl', telemetry.data.tirePressures.fl)
      .floatField('tire_pressure_fr', telemetry.data.tirePressures.fr)
      .floatField('tire_pressure_rl', telemetry.data.tirePressures.rl)
      .floatField('tire_pressure_rr', telemetry.data.tirePressures.rr)
      .floatField('brake_temp_fl', telemetry.data.brakeTemps.fl)
      .floatField('brake_temp_fr', telemetry.data.brakeTemps.fr)
      .floatField('brake_temp_rl', telemetry.data.brakeTemps.rl)
      .floatField('brake_temp_rr', telemetry.data.brakeTemps.rr)
      .floatField('engine_temp', telemetry.data.engineTemp)
      .floatField('oil_pressure', telemetry.data.oilPressure)
      .floatField('oil_temp', telemetry.data.oilTemp)
      .floatField('water_temp', telemetry.data.waterTemp)
      .floatField('fuel_level', telemetry.data.fuelLevel)
      .floatField('fuel_flow', telemetry.data.fuelFlow)
      .timestamp(telemetry.timestamp);

    await this.client.write(point, this.database);
  }

  async queryTelemetry(
    trackId: string,
    sessionId: string,
    startTime: Date,
    endTime: Date,
    fields?: string[]
  ): Promise<any[]> {
    const fieldList = fields?.join(', ') || '*';
    const query = `
      SELECT ${fieldList}
      FROM telemetry
      WHERE track = '${trackId}'
        AND session = '${sessionId}'
        AND time >= '${startTime.toISOString()}'
        AND time <= '${endTime.toISOString()}'
      ORDER BY time ASC
    `;

    const queryApi = this.client.queryApi(this.database);
    const results: any[] = [];
    
    for await (const row of queryApi.iterateRows(query)) {
      results.push(row);
    }
    
    return results;
  }

  async queryLapTelemetry(
    trackId: string,
    sessionId: string,
    lapNumber: number
  ): Promise<any[]> {
    const query = `
      SELECT *
      FROM telemetry
      WHERE track = '${trackId}'
        AND session = '${sessionId}'
        AND lap = '${lapNumber}'
      ORDER BY time ASC
    `;

    const queryApi = this.client.queryApi(this.database);
    const results: any[] = [];
    
    for await (const row of queryApi.iterateRows(query)) {
      results.push(row);
    }
    
    return results;
  }

  async getLatestTelemetry(
    trackId: string,
    sessionId: string,
    limit: number = 1
  ): Promise<any[]> {
    const query = `
      SELECT *
      FROM telemetry
      WHERE track = '${trackId}'
        AND session = '${sessionId}'
      ORDER BY time DESC
      LIMIT ${limit}
    `;

    const queryApi = this.client.queryApi(this.database);
    const results: any[] = [];
    
    for await (const row of queryApi.iterateRows(query)) {
      results.push(row);
    }
    
    return results;
  }

  async close(): Promise<void> {
    await this.client.close();
  }
}

// Singleton instance
let telemetryDB: TelemetryDatabase | null = null;

export function initializeTelemetryDB(config: InfluxDBConfig): TelemetryDatabase {
  if (!telemetryDB) {
    telemetryDB = new TelemetryDatabase(config);
  }
  return telemetryDB;
}

export function getTelemetryDB(): TelemetryDatabase {
  if (!telemetryDB) {
    throw new Error('TelemetryDB not initialized. Call initializeTelemetryDB first.');
  }
  return telemetryDB;
}