// Data loader service for processing racing telemetry data

export interface TelemetryPoint {
  timestamp: string;
  lap: number;
  vehicleId: string;
  vehicleNumber: string;
  telemetryName: string;
  telemetryValue: number;
}

export interface LapTime {
  timestamp: string;
  vehicleId: string;
  lap: number;
  lapTimeMs: number;
  lapTimeFormatted?: string;
}

export interface WeatherData {
  timestamp: string;
  airTemp: number;
  trackTemp: number;
  humidity: number;
  pressure: number;
  windSpeed: number;
  windDirection: number;
  rain: number;
}

export interface RaceResult {
  position: number;
  vehicleNumber: string;
  driverName: string;
  bestLap: string;
  totalTime: string;
  laps: number;
  gap: string;
}

export class DataLoader {
  static tracks = [
    { id: 'cota', name: 'Circuit of the Americas', folder: 'COTA' },
    { id: 'roadamerica', name: 'Road America', folder: 'Road America' },
    { id: 'sebring', name: 'Sebring', folder: 'Sebring' },
    { id: 'sonoma', name: 'Sonoma', folder: 'Sonoma' },
    { id: 'vir', name: 'Virginia International Raceway', folder: 'VIR' },
    { id: 'barber', name: 'Barber Motorsports Park', folder: 'barber' }
  ];

  static formatLapTime(milliseconds: number): string {
    const totalSeconds = milliseconds / 1000;
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = (totalSeconds % 60).toFixed(3);
    return `${minutes}:${seconds.padStart(6, '0')}`;
  }

  static async loadTrackData(trackId: string, raceNumber: 1 | 2) {
    const track = this.tracks.find(t => t.id === trackId);
    if (!track) throw new Error(`Track ${trackId} not found`);

    // In a real application, these would be actual file imports
    // For now, we'll return mock data structured like the real data
    return {
      track: track.name,
      race: `Race ${raceNumber}`,
      telemetry: await this.loadTelemetryData(track.folder, raceNumber),
      lapTimes: await this.loadLapTimes(track.folder, raceNumber),
      weather: await this.loadWeatherData(track.folder, raceNumber),
      results: await this.loadRaceResults(track.folder, raceNumber)
    };
  }

  static async loadTelemetryData(trackFolder: string, raceNumber: number): Promise<TelemetryPoint[]> {
    // This would load from the CSV files
    // For now, return sample data
    return [];
  }

  static async loadLapTimes(trackFolder: string, raceNumber: number): Promise<LapTime[]> {
    // This would load from the lap time CSV files
    return [];
  }

  static async loadWeatherData(trackFolder: string, raceNumber: number): Promise<WeatherData[]> {
    // This would load from the weather CSV files
    return [];
  }

  static async loadRaceResults(trackFolder: string, raceNumber: number): Promise<RaceResult[]> {
    // This would load from the results CSV files
    return [];
  }

  // Process telemetry data for real-time display
  static processTelemetryStream(telemetry: TelemetryPoint[]): Map<string, any> {
    const vehicleData = new Map();
    
    telemetry.forEach(point => {
      if (!vehicleData.has(point.vehicleId)) {
        vehicleData.set(point.vehicleId, {
          vehicleNumber: point.vehicleNumber,
          latestLap: point.lap,
          telemetry: new Map()
        });
      }
      
      const vehicle = vehicleData.get(point.vehicleId);
      vehicle.telemetry.set(point.telemetryName, {
        value: point.telemetryValue,
        timestamp: point.timestamp
      });
    });
    
    return vehicleData;
  }

  // Calculate tire degradation from telemetry
  static calculateTireDegradation(telemetry: TelemetryPoint[], currentLap: number): number {
    // Simplified calculation based on lap count and temperature
    const degradationPerLap = 2.5; // 2.5% per lap
    const baseDegradation = currentLap * degradationPerLap;
    return Math.max(0, 100 - baseDegradation);
  }

  // Predict optimal pit window
  static predictPitWindow(
    currentLap: number, 
    tireDegradation: number, 
    fuelLevel: number,
    totalLaps: number
  ): { start: number, end: number, reason: string } {
    const tireCliffLap = Math.floor(currentLap + (tireDegradation / 5));
    const fuelEmptyLap = Math.floor(currentLap + (fuelLevel / 2.5));
    
    const criticalLap = Math.min(tireCliffLap, fuelEmptyLap);
    const optimalStart = Math.max(criticalLap - 3, currentLap + 1);
    const optimalEnd = criticalLap;
    
    const reason = tireCliffLap < fuelEmptyLap ? 'Tire degradation' : 'Fuel level';
    
    return { start: optimalStart, end: optimalEnd, reason };
  }
}