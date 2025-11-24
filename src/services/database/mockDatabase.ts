// Mock database implementation for browser environment

interface MockDB {
  tracks: Map<string, any>;
  sessions: Map<string, any>;
  results: Map<string, any>;
  lapTimes: Map<string, any>;
  drivers: Map<string, any>;
  settings: Map<string, any>;
}

class MockDatabase {
  private data: MockDB = {
    tracks: new Map(),
    sessions: new Map(),
    results: new Map(),
    lapTimes: new Map(),
    drivers: new Map(),
    settings: new Map()
  };

  constructor() {
    this.initializeMockData();
  }

  private initializeMockData() {
    // Add mock tracks
    const tracks = [
      {
        id: 'cota',
        name: 'Circuit of the Americas',
        location: 'Austin, TX',
        length: 5.513,
        turns: 20,
        sectors: 3,
        lapRecord: '1:32.029',
        lapRecordHolder: 'Valtteri Bottas',
        lapRecordYear: 2019
      },
      {
        id: 'roadamerica',
        name: 'Road America',
        location: 'Elkhart Lake, WI',
        length: 6.515,
        turns: 14,
        sectors: 3,
        lapRecord: '1:39.866',
        lapRecordHolder: 'IndyCar',
        lapRecordYear: 2022
      }
    ];

    tracks.forEach(track => this.data.tracks.set(track.id, track));

    // Add mock settings
    this.data.settings.set('theme', 'light');
    this.data.settings.set('units', 'metric');
  }

  getTrack(trackId: string): any {
    return this.data.tracks.get(trackId);
  }

  getAllTracks(): any[] {
    return Array.from(this.data.tracks.values());
  }

  insertTrack(track: any): void {
    this.data.tracks.set(track.id, track);
  }

  getSession(sessionId: string): any {
    return this.data.sessions.get(sessionId);
  }

  getTrackSessions(trackId: string): any[] {
    return Array.from(this.data.sessions.values())
      .filter(session => session.trackId === trackId);
  }

  insertSession(session: any): void {
    this.data.sessions.set(session.id, session);
  }

  getSessionResults(sessionId: string): any[] {
    return Array.from(this.data.results.values())
      .filter(result => result.sessionId === sessionId);
  }

  insertRaceResult(result: any): void {
    this.data.results.set(result.id, result);
  }

  getLapTimes(sessionId: string, driverId?: string): any[] {
    return Array.from(this.data.lapTimes.values())
      .filter(lap => lap.sessionId === sessionId && (!driverId || lap.driverId === driverId));
  }

  insertLapTime(lapTime: any): void {
    this.data.lapTimes.set(lapTime.id, lapTime);
  }

  getSetting(key: string): string | null {
    return this.data.settings.get(key) || null;
  }

  setSetting(key: string, value: string): void {
    this.data.settings.set(key, value);
  }

  close(): void {
    // No-op for mock
  }

  transaction<T>(callback: () => T): T {
    return callback();
  }
}

export { MockDatabase };