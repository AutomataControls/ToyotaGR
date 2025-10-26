// SQLite is only available in Node.js environment
let Database: any;
try {
  Database = require('better-sqlite3');
} catch (e) {
  console.warn('better-sqlite3 not available - SQLite features disabled');
}

import { MockDatabase } from './mockDatabase';

interface RaceSession {
  id: string;
  trackId: string;
  date: string;
  raceNumber: number;
  weatherConditions: string;
  trackTemp: number;
  airTemp: number;
  duration: number;
  totalLaps: number;
  status: 'scheduled' | 'live' | 'completed';
}

interface RaceResult {
  id: string;
  sessionId: string;
  driverId: string;
  position: number;
  startPosition: number;
  bestLapTime: string;
  totalTime: string;
  totalLaps: number;
  points: number;
  incidents: number;
  status: 'finished' | 'dnf' | 'dns';
}

interface LapTime {
  id: string;
  sessionId: string;
  driverId: string;
  lapNumber: number;
  lapTime: string;
  sector1: string;
  sector2: string;
  sector3: string;
  isValid: boolean;
  tireCompound: 'S' | 'M' | 'H';
  fuelLoad: number;
}

interface TrackInfo {
  id: string;
  name: string;
  location: string;
  length: number;
  turns: number;
  sectors: number;
  lapRecord: string;
  lapRecordHolder: string;
  lapRecordYear: number;
}

interface Driver {
  id: string;
  name: string;
  number: number;
  team: string;
  nationality: string;
  totalPoints: number;
  championshipPosition: number;
}

interface Settings {
  key: string;
  value: string;
  type: 'string' | 'number' | 'boolean' | 'json';
}

export class RaceDatabase {
  private db: any;
  private mockDb?: MockDatabase;

  constructor(dbPath: string = './data/race.db') {
    if (!Database) {
      console.warn('SQLite database not available in browser environment - using mock database');
      this.mockDb = new MockDatabase();
      return;
    }
    try {
      this.db = new Database(dbPath);
      this.initialize();
    } catch (error) {
      console.warn('Failed to initialize SQLite database, falling back to mock:', error);
      this.mockDb = new MockDatabase();
    }
  }

  private initialize(): void {
    if (!this.db) return;
    
    // Create tables
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS tracks (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        location TEXT NOT NULL,
        length REAL NOT NULL,
        turns INTEGER NOT NULL,
        sectors INTEGER NOT NULL,
        lap_record TEXT,
        lap_record_holder TEXT,
        lap_record_year INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );

      CREATE TABLE IF NOT EXISTS drivers (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        number INTEGER UNIQUE NOT NULL,
        team TEXT NOT NULL,
        nationality TEXT NOT NULL,
        total_points INTEGER DEFAULT 0,
        championship_position INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );

      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        track_id TEXT NOT NULL,
        date DATETIME NOT NULL,
        race_number INTEGER NOT NULL,
        weather_conditions TEXT,
        track_temp REAL,
        air_temp REAL,
        duration INTEGER,
        total_laps INTEGER,
        status TEXT DEFAULT 'scheduled',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (track_id) REFERENCES tracks(id)
      );

      CREATE TABLE IF NOT EXISTS race_results (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        driver_id TEXT NOT NULL,
        position INTEGER NOT NULL,
        start_position INTEGER NOT NULL,
        best_lap_time TEXT,
        total_time TEXT,
        total_laps INTEGER,
        points INTEGER DEFAULT 0,
        incidents INTEGER DEFAULT 0,
        status TEXT DEFAULT 'finished',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(id),
        FOREIGN KEY (driver_id) REFERENCES drivers(id)
      );

      CREATE TABLE IF NOT EXISTS lap_times (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        driver_id TEXT NOT NULL,
        lap_number INTEGER NOT NULL,
        lap_time TEXT NOT NULL,
        sector_1 TEXT,
        sector_2 TEXT,
        sector_3 TEXT,
        is_valid BOOLEAN DEFAULT 1,
        tire_compound TEXT,
        fuel_load REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(id),
        FOREIGN KEY (driver_id) REFERENCES drivers(id)
      );

      CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        type TEXT NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );

      CREATE INDEX IF NOT EXISTS idx_sessions_track ON sessions(track_id);
      CREATE INDEX IF NOT EXISTS idx_results_session ON race_results(session_id);
      CREATE INDEX IF NOT EXISTS idx_results_driver ON race_results(driver_id);
      CREATE INDEX IF NOT EXISTS idx_laps_session ON lap_times(session_id);
      CREATE INDEX IF NOT EXISTS idx_laps_driver ON lap_times(driver_id);
    `);

    // Add triggers for updated_at
    this.db.exec(`
      CREATE TRIGGER IF NOT EXISTS update_tracks_timestamp 
        AFTER UPDATE ON tracks
        FOR EACH ROW
        BEGIN
          UPDATE tracks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;

      CREATE TRIGGER IF NOT EXISTS update_drivers_timestamp 
        AFTER UPDATE ON drivers
        FOR EACH ROW
        BEGIN
          UPDATE drivers SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;

      CREATE TRIGGER IF NOT EXISTS update_sessions_timestamp 
        AFTER UPDATE ON sessions
        FOR EACH ROW
        BEGIN
          UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;

      CREATE TRIGGER IF NOT EXISTS update_settings_timestamp 
        AFTER UPDATE ON settings
        FOR EACH ROW
        BEGIN
          UPDATE settings SET updated_at = CURRENT_TIMESTAMP WHERE key = NEW.key;
        END;
    `);
  }

  // Track methods
  getTrack(trackId: string): TrackInfo | undefined {
    if (this.mockDb) {
      return this.mockDb.getTrack(trackId);
    }
    if (!this.db) return undefined;
    const stmt = this.db.prepare('SELECT * FROM tracks WHERE id = ?');
    return stmt.get(trackId) as TrackInfo | undefined;
  }

  getAllTracks(): TrackInfo[] {
    if (this.mockDb) {
      return this.mockDb.getAllTracks();
    }
    if (!this.db) return [];
    const stmt = this.db.prepare('SELECT * FROM tracks ORDER BY name');
    return stmt.all() as TrackInfo[];
  }

  insertTrack(track: TrackInfo): void {
    const stmt = this.db.prepare(`
      INSERT INTO tracks (id, name, location, length, turns, sectors, lap_record, lap_record_holder, lap_record_year)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    stmt.run(
      track.id,
      track.name,
      track.location,
      track.length,
      track.turns,
      track.sectors,
      track.lapRecord,
      track.lapRecordHolder,
      track.lapRecordYear
    );
  }

  // Session methods
  getSession(sessionId: string): RaceSession | undefined {
    const stmt = this.db.prepare('SELECT * FROM sessions WHERE id = ?');
    return stmt.get(sessionId) as RaceSession | undefined;
  }

  getTrackSessions(trackId: string): RaceSession[] {
    const stmt = this.db.prepare('SELECT * FROM sessions WHERE track_id = ? ORDER BY date DESC');
    return stmt.all(trackId) as RaceSession[];
  }

  insertSession(session: RaceSession): void {
    const stmt = this.db.prepare(`
      INSERT INTO sessions (id, track_id, date, race_number, weather_conditions, track_temp, air_temp, duration, total_laps, status)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    stmt.run(
      session.id,
      session.trackId,
      session.date,
      session.raceNumber,
      session.weatherConditions,
      session.trackTemp,
      session.airTemp,
      session.duration,
      session.totalLaps,
      session.status
    );
  }

  // Race result methods
  getSessionResults(sessionId: string): RaceResult[] {
    const stmt = this.db.prepare(`
      SELECT r.*, d.name as driver_name, d.number as driver_number
      FROM race_results r
      JOIN drivers d ON r.driver_id = d.id
      WHERE r.session_id = ?
      ORDER BY r.position
    `);
    return stmt.all(sessionId) as RaceResult[];
  }

  insertRaceResult(result: RaceResult): void {
    const stmt = this.db.prepare(`
      INSERT INTO race_results (id, session_id, driver_id, position, start_position, best_lap_time, total_time, total_laps, points, incidents, status)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    stmt.run(
      result.id,
      result.sessionId,
      result.driverId,
      result.position,
      result.startPosition,
      result.bestLapTime,
      result.totalTime,
      result.totalLaps,
      result.points,
      result.incidents,
      result.status
    );
  }

  // Lap time methods
  getLapTimes(sessionId: string, driverId?: string): LapTime[] {
    let query = 'SELECT * FROM lap_times WHERE session_id = ?';
    const params: any[] = [sessionId];
    
    if (driverId) {
      query += ' AND driver_id = ?';
      params.push(driverId);
    }
    
    query += ' ORDER BY lap_number';
    const stmt = this.db.prepare(query);
    return stmt.all(...params) as LapTime[];
  }

  insertLapTime(lapTime: LapTime): void {
    const stmt = this.db.prepare(`
      INSERT INTO lap_times (id, session_id, driver_id, lap_number, lap_time, sector_1, sector_2, sector_3, is_valid, tire_compound, fuel_load)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    stmt.run(
      lapTime.id,
      lapTime.sessionId,
      lapTime.driverId,
      lapTime.lapNumber,
      lapTime.lapTime,
      lapTime.sector1,
      lapTime.sector2,
      lapTime.sector3,
      lapTime.isValid ? 1 : 0,
      lapTime.tireCompound,
      lapTime.fuelLoad
    );
  }

  // Settings methods
  getSetting(key: string): string | null {
    const stmt = this.db.prepare('SELECT value FROM settings WHERE key = ?');
    const result = stmt.get(key) as { value: string } | undefined;
    return result?.value || null;
  }

  setSetting(key: string, value: string, type: Settings['type'] = 'string'): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO settings (key, value, type)
      VALUES (?, ?, ?)
    `);
    stmt.run(key, value, type);
  }

  // Utility methods
  close(): void {
    this.db.close();
  }

  // Transaction support
  transaction<T>(callback: () => T): T {
    const transaction = this.db.transaction(callback);
    return transaction();
  }
}

// Singleton instance
let raceDB: RaceDatabase | null = null;

export function initializeRaceDB(dbPath?: string): RaceDatabase {
  if (!raceDB) {
    raceDB = new RaceDatabase(dbPath);
  }
  return raceDB;
}

export function getRaceDB(): RaceDatabase {
  if (!raceDB) {
    throw new Error('RaceDB not initialized. Call initializeRaceDB first.');
  }
  return raceDB;
}