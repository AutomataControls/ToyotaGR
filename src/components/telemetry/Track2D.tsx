import React from 'react';
import { getTrackConfig } from '../../data/trackMaps';
import styles from './Track2D.module.css';

interface Track2DProps {
  trackId: string;
  driverProgress: number;
  currentLap: number;
  totalLaps: number;
}

export const Track2D: React.FC<Track2DProps> = ({
  trackId,
  driverProgress,
  currentLap,
  totalLaps
}) => {
  const track = getTrackConfig(trackId);
  if (!track) return null;

  const currentSector = track.sectors.find(sector => 
    driverProgress >= sector.start && driverProgress < sector.end
  ) || track.sectors[0];

  // Calculate car position along actual SVG path
  const getCarPosition = (progress: number, svgPath: string) => {
    if (!svgPath) {
      // Fallback to simple circular positioning
      const angle = (progress / 100) * Math.PI * 2;
      return {
        x: 150 + Math.cos(angle) * 80,
        y: 100 + Math.sin(angle) * 60
      };
    }

    // Parse SVG path and calculate position based on progress
    // This is a simplified version - extract key points from the path
    const pathPoints = extractPathPoints(svgPath);
    if (pathPoints.length === 0) {
      return { x: 150, y: 100 };
    }

    const targetIndex = Math.floor((progress / 100) * (pathPoints.length - 1));
    const nextIndex = Math.min(targetIndex + 1, pathPoints.length - 1);
    
    if (targetIndex === nextIndex) {
      return pathPoints[targetIndex];
    }

    // Interpolate between points for smooth movement
    const t = ((progress / 100) * (pathPoints.length - 1)) - targetIndex;
    const current = pathPoints[targetIndex];
    const next = pathPoints[nextIndex];
    
    return {
      x: current.x + (next.x - current.x) * t,
      y: current.y + (next.y - current.y) * t
    };
  };

  // Extract points from SVG path string
  const extractPathPoints = (pathStr: string) => {
    const points: { x: number; y: number }[] = [];
    const commands = pathStr.match(/[MmLlHhVvCcSsQqTtAaZz][^MmLlHhVvCcSsQqTtAaZz]*/g) || [];
    
    let currentX = 0;
    let currentY = 0;
    
    commands.forEach(cmd => {
      const command = cmd[0];
      const coords = cmd.slice(1).trim().split(/[\s,]+/).map(Number).filter(n => !isNaN(n));
      
      switch (command.toUpperCase()) {
        case 'M': // Move to
          if (coords.length >= 2) {
            currentX = command === 'M' ? coords[0] : currentX + coords[0];
            currentY = command === 'M' ? coords[1] : currentY + coords[1];
            points.push({ x: currentX, y: currentY });
          }
          break;
        case 'L': // Line to
          for (let i = 0; i < coords.length; i += 2) {
            if (i + 1 < coords.length) {
              currentX = command === 'L' ? coords[i] : currentX + coords[i];
              currentY = command === 'L' ? coords[i + 1] : currentY + coords[i + 1];
              points.push({ x: currentX, y: currentY });
            }
          }
          break;
        case 'Q': // Quadratic curve
          for (let i = 0; i < coords.length; i += 4) {
            if (i + 3 < coords.length) {
              // Simplified: just add the end point
              currentX = command === 'Q' ? coords[i + 2] : currentX + coords[i + 2];
              currentY = command === 'Q' ? coords[i + 3] : currentY + coords[i + 3];
              points.push({ x: currentX, y: currentY });
            }
          }
          break;
      }
    });
    
    return points;
  };

  const carPos = getCarPosition(driverProgress, track.svgPath || '');

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h3>{track.name}</h3>
        <div className={styles.info}>
          <span>Lap {currentLap}/{totalLaps}</span>
          <span className={`${styles.sector} ${styles[currentSector.difficulty]}`}>
            {currentSector.name}
          </span>
        </div>
      </div>
      
      <div className={styles.trackContainer}>
        <svg width="300" height="200" viewBox="0 0 300 200" className={styles.track}>
          {/* Track background */}
          <rect width="300" height="200" fill="#1e293b" rx="8" />
          
          {/* Track layout using SVG path */}
          {track.svgPath && (
            <path
              d={track.svgPath}
              fill="none"
              stroke="#64748b"
              strokeWidth="8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
          
          {/* Track surface */}
          {track.svgPath && (
            <path
              d={track.svgPath}
              fill="none"
              stroke="#334155"
              strokeWidth="6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
          
          {/* Track center line */}
          {track.svgPath && (
            <path
              d={track.svgPath}
              fill="none"
              stroke="#475569"
              strokeWidth="1"
              strokeDasharray="4 4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
          
          {/* Sector markers */}
          {track.sectors.map((sector, index) => {
            const sectorPos = getCarPosition(sector.start, track.svgPath || '');
            return (
              <g key={sector.id}>
                <circle
                  cx={sectorPos.x}
                  cy={sectorPos.y}
                  r="3"
                  fill={sector.difficulty === 'hard' ? '#ef4444' : 
                        sector.difficulty === 'medium' ? '#f59e0b' : '#22c55e'}
                />
                <text
                  x={sectorPos.x}
                  y={sectorPos.y - 8}
                  textAnchor="middle"
                  fontSize="8"
                  fill="#e2e8f0"
                  className={styles.sectorLabel}
                >
                  S{index + 1}
                </text>
              </g>
            );
          })}
          
          {/* Car position */}
          <g transform={`translate(${carPos.x}, ${carPos.y})`}>
            <circle r="4" fill="#ef4444" />
            <circle r="6" fill="none" stroke="#ef4444" strokeWidth="1" opacity="0.5" />
            <circle r="8" fill="none" stroke="#ef4444" strokeWidth="1" opacity="0.3" />
          </g>
          
          {/* Start/Finish line */}
          <g>
            {(() => {
              const finishPos = getCarPosition(0, track.svgPath || '');
              return (
                <>
                  <line
                    x1={finishPos.x - 4}
                    y1={finishPos.y - 4}
                    x2={finishPos.x + 4}
                    y2={finishPos.y + 4}
                    stroke="#ffffff"
                    strokeWidth="2"
                  />
                  <text
                    x={finishPos.x + 10}
                    y={finishPos.y}
                    fontSize="8"
                    fill="#ffffff"
                    className={styles.finishLabel}
                  >
                    START/FINISH
                  </text>
                </>
              );
            })()}
          </g>
        </svg>
      </div>
      
      <div className={styles.stats}>
        <div className={styles.progressBar}>
          <div 
            className={styles.progressFill}
            style={{ width: `${driverProgress}%` }}
          />
        </div>
        <div className={styles.trackStats}>
          <span>{track.length} miles</span>
          <span>{track.turns} turns</span>
          <span>{driverProgress.toFixed(1)}% complete</span>
        </div>
      </div>
    </div>
  );
};