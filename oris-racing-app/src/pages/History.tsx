import React, { useState } from 'react';
import { Card } from '../components/common';
import { History as HistoryIcon, Trophy, Calendar, Filter } from 'lucide-react';
import styles from './History.module.css';

interface RaceResult {
  id: string;
  date: string;
  track: string;
  race: number;
  position: number;
  startPosition: number;
  bestLap: string;
  totalTime: string;
  points: number;
  incidents: number;
}

export const History: React.FC = () => {
  const [selectedTrack, setSelectedTrack] = useState<string>('all');
  const [selectedYear, setSelectedYear] = useState<number>(2024);

  const raceResults: RaceResult[] = [
    {
      id: '1',
      date: '2024-10-20',
      track: 'Circuit of the Americas',
      race: 2,
      position: 4,
      startPosition: 6,
      bestLap: '1:23.456',
      totalTime: '42:15.789',
      points: 12,
      incidents: 0
    },
    {
      id: '2',
      date: '2024-10-19',
      track: 'Circuit of the Americas',
      race: 1,
      position: 7,
      startPosition: 8,
      bestLap: '1:23.890',
      totalTime: '42:45.123',
      points: 6,
      incidents: 1
    },
    {
      id: '3',
      date: '2024-09-15',
      track: 'Road America',
      race: 2,
      position: 2,
      startPosition: 3,
      bestLap: '2:05.234',
      totalTime: '48:32.456',
      points: 18,
      incidents: 0
    },
    {
      id: '4',
      date: '2024-09-14',
      track: 'Road America',
      race: 1,
      position: 5,
      startPosition: 4,
      bestLap: '2:05.789',
      totalTime: '48:45.789',
      points: 10,
      incidents: 0
    },
  ];

  const seasonStats = {
    races: 18,
    wins: 2,
    podiums: 6,
    poles: 3,
    fastestLaps: 4,
    points: 156,
    championshipPosition: 5,
    avgFinish: 6.2,
    avgStart: 7.1,
    dnfs: 1
  };

  const tracks = ['all', 'cota', 'roadamerica', 'sebring', 'sonoma', 'vir', 'barber'];

  return (
    <div className={styles.historyPage}>
      <div className={styles.header}>
        <h1 className={styles.title}>
          <HistoryIcon size={24} />
          Race History
        </h1>
        <div className={styles.filters}>
          <select 
            className={styles.filter}
            value={selectedYear}
            onChange={(e) => setSelectedYear(Number(e.target.value))}
          >
            <option value={2024}>2024 Season</option>
            <option value={2023}>2023 Season</option>
          </select>
          <select 
            className={styles.filter}
            value={selectedTrack}
            onChange={(e) => setSelectedTrack(e.target.value)}
          >
            <option value="all">All Tracks</option>
            <option value="cota">Circuit of the Americas</option>
            <option value="roadamerica">Road America</option>
            <option value="sebring">Sebring</option>
            <option value="sonoma">Sonoma</option>
            <option value="vir">VIR</option>
            <option value="barber">Barber</option>
          </select>
        </div>
      </div>

      <div className={styles.content}>
        <div className={styles.statsSection}>
          <Card className={styles.seasonOverview}>
            <h2><Trophy size={18} /> {selectedYear} Season Overview</h2>
            <div className={styles.statsGrid}>
              <div className={styles.stat}>
                <span className={styles.statLabel}>Championship Position</span>
                <span className={styles.statValue}>P{seasonStats.championshipPosition}</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statLabel}>Points</span>
                <span className={styles.statValue}>{seasonStats.points}</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statLabel}>Races</span>
                <span className={styles.statValue}>{seasonStats.races}</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statLabel}>Wins</span>
                <span className={styles.statValue}>{seasonStats.wins}</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statLabel}>Podiums</span>
                <span className={styles.statValue}>{seasonStats.podiums}</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statLabel}>Poles</span>
                <span className={styles.statValue}>{seasonStats.poles}</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statLabel}>Fastest Laps</span>
                <span className={styles.statValue}>{seasonStats.fastestLaps}</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statLabel}>Avg Finish</span>
                <span className={styles.statValue}>{seasonStats.avgFinish.toFixed(1)}</span>
              </div>
            </div>
          </Card>
        </div>

        <Card className={styles.resultsSection}>
          <h2><Calendar size={18} /> Race Results</h2>
          <div className={styles.resultsTable}>
            <div className={styles.tableHeader}>
              <span>Date</span>
              <span>Track</span>
              <span>Race</span>
              <span>Start</span>
              <span>Finish</span>
              <span>Best Lap</span>
              <span>Time</span>
              <span>Points</span>
              <span>Inc</span>
            </div>
            {raceResults.map((result) => (
              <div key={result.id} className={styles.tableRow}>
                <span className={styles.date}>
                  {new Date(result.date).toLocaleDateString('en-US', { 
                    month: 'short', 
                    day: 'numeric' 
                  })}
                </span>
                <span className={styles.track}>{result.track}</span>
                <span className={styles.race}>R{result.race}</span>
                <span className={styles.position}>P{result.startPosition}</span>
                <span className={`${styles.position} ${styles.finish}`}>
                  P{result.position}
                  {result.position < result.startPosition && 
                    <span className={styles.gained}>▲{result.startPosition - result.position}</span>
                  }
                  {result.position > result.startPosition && 
                    <span className={styles.lost}>▼{result.position - result.startPosition}</span>
                  }
                </span>
                <span className={styles.lapTime}>{result.bestLap}</span>
                <span className={styles.totalTime}>{result.totalTime}</span>
                <span className={styles.points}>{result.points}</span>
                <span className={styles.incidents}>{result.incidents}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
};