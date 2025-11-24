# ORIS Live Data Integration Specification
**For Toyota GR Cup Series Real-Time Race Analytics**

## Overview
This document specifies how real-time race data should be fed into ORIS (OLYMPUS Racing Intelligence System) for each page component. Race engineers can use this guide to understand data formats, endpoints, and integration points.

---

## 🏁 **1. DASHBOARD PAGE**
**Real-time overview with all components**

### Data Requirements:
```json
{
  "timestamp": "2024-11-22T23:15:42.123Z",
  "sessionId": "COTA_R2_2024_11_22",
  "trackId": "cota",
  "raceNumber": 2,
  "currentLap": 23,
  "totalLaps": 50,
  "position": {
    "current": 4,
    "starting": 7,
    "classPosition": 2
  },
  "timing": {
    "currentLapTime": "1:23.456",
    "lastLapTime": "1:23.234",
    "bestLapTime": "1:22.987",
    "sectorTimes": {
      "sector1": "27.123",
      "sector2": "28.789", 
      "sector3": "27.544"
    },
    "deltaToLeader": "+12.456",
    "deltaToAhead": "+2.345"
  },
  "strategy": {
    "currentStrategy": "two_stop",
    "nextStopLap": 42,
    "tireCompound": "MEDIUM",
    "fuelRemaining": 62.5
  }
}
```

### WebSocket Endpoint:
- **URL**: `ws://race-data-server:8080/live/dashboard`
- **Update Frequency**: 1Hz (every second)
- **Components Fed**: TrackPosition, LapComparison, StrategyAdvisor, PredictionsPanel

---

## 📊 **2. TELEMETRY PAGE**
**Live vehicle sensor data**

### Data Requirements:
```json
{
  "timestamp": "2024-11-22T23:15:42.123Z",
  "sessionId": "COTA_R2_2024_11_22",
  "carNumber": 7,
  "performance": {
    "speed": 156.7,
    "rpm": 7200,
    "gear": 4,
    "throttle": 85.2,
    "brake": 0.0,
    "steeringAngle": -15.3,
    "gForce": {
      "lateral": 1.24,
      "longitudinal": -0.32,
      "vertical": 0.98
    }
  },
  "temperatures": {
    "tires": {
      "frontLeft": 92.5,
      "frontRight": 94.2,
      "rearLeft": 88.1,
      "rearRight": 89.7
    },
    "brakes": {
      "frontLeft": 380.5,
      "frontRight": 390.2,
      "rearLeft": 350.1,
      "rearRight": 360.7
    },
    "engine": 98.5,
    "oil": 102.3,
    "coolant": 85.7
  },
  "pressures": {
    "tirePressure": {
      "frontLeft": 32.1,
      "frontRight": 32.3,
      "rearLeft": 30.8,
      "rearRight": 31.1
    },
    "oilPressure": 55.2,
    "fuelPressure": 3.8
  },
  "fuel": {
    "level": 62.5,
    "consumption": 2.3,
    "lapsRemaining": 27
  }
}
```

### WebSocket Endpoint:
- **URL**: `ws://race-data-server:8080/live/telemetry`
- **Update Frequency**: 10Hz (10 times per second)
- **InfluxDB Integration**: Yes - stores historical data
- **Components Fed**: TelemetryGraphs, Performance gauges, Temperature displays

---

## 🎯 **3. STRATEGY PAGE**
**AI-powered race strategy and pit optimization**

### Data Requirements:
```json
{
  "timestamp": "2024-11-22T23:15:42.123Z",
  "sessionId": "COTA_R2_2024_11_22",
  "strategy": {
    "currentStrategy": "two_stop",
    "alternatives": [
      {
        "name": "aggressive_undercut",
        "description": "Early stop on lap 15, overcut competitors",
        "probability": 72,
        "riskLevel": "medium",
        "expectedPosition": 3,
        "stopLaps": [15, 35]
      }
    ],
    "pitWindow": {
      "optimal": {
        "start": 12,
        "end": 18,
        "recommended": 15
      },
      "factors": {
        "tireDegradation": 65,
        "fuelRemaining": 42,
        "trackPositionRisk": 78,
        "weatherRisk": 12
      }
    },
    "predictions": {
      "P2ByLap45": {
        "probability": 78,
        "confidence": 82,
        "trend": "up"
      },
      "tireCliff": {
        "estimatedLap": 8,
        "confidence": 85,
        "degradationRate": 2.3
      },
      "safetyCarProbability": 15,
      "rainProbability": 12
    }
  },
  "competitors": [
    {
      "position": 1,
      "carNumber": 12,
      "gap": "+12.456",
      "strategy": "one_stop",
      "nextStop": 25,
      "tireAge": 15
    }
  ]
}
```

### AI Model Integration Points:
- **MINERVA**: Strategy recommendations
- **ATLAS**: Spatial track analysis  
- **CHRONOS**: Timing predictions
- **PROMETHEUS**: Weather/incident forecasting

### WebSocket Endpoint:
- **URL**: `ws://race-data-server:8080/live/strategy`
- **Update Frequency**: 0.5Hz (every 2 seconds)

---

## ⏱️ **4. TIMING PAGE**
**Lap times and sector analysis**

### Data Requirements:
```json
{
  "timestamp": "2024-11-22T23:15:42.123Z",
  "sessionId": "COTA_R2_2024_11_22",
  "currentLap": {
    "number": 23,
    "startTime": "2024-11-22T23:14:18.456Z",
    "sectors": {
      "sector1": {
        "time": "27.123",
        "delta": "-0.234",
        "isComplete": true
      },
      "sector2": {
        "time": "28.789",
        "delta": "+0.123", 
        "isComplete": true
      },
      "sector3": {
        "time": null,
        "delta": null,
        "isComplete": false
      }
    }
  },
  "lapHistory": [
    {
      "lap": 22,
      "time": "1:23.456",
      "sector1": "27.234",
      "sector2": "28.901", 
      "sector3": "27.321",
      "delta": "+0.234",
      "tireCompound": "MEDIUM",
      "fuelUsed": 2.3,
      "avgSpeed": 156.7
    }
  ],
  "sessionBests": {
    "fastestLap": {
      "time": "1:22.987",
      "lap": 18,
      "driver": "Car #7"
    },
    "personalBest": {
      "time": "1:23.234",
      "lap": 15,
      "sectors": ["27.012", "28.678", "27.544"]
    }
  },
  "leaderboard": [
    {
      "position": 1,
      "carNumber": 12,
      "lastLap": "1:23.123",
      "bestLap": "1:22.987",
      "gap": "LEADER"
    }
  ]
}
```

### WebSocket Endpoint:
- **URL**: `ws://race-data-server:8080/live/timing`
- **Update Frequency**: 1Hz (every second)

---

## 🎓 **5. TRAINING PAGE**
**Driver performance analysis and recommendations**

### Data Requirements:
```json
{
  "timestamp": "2024-11-22T23:15:42.123Z",
  "sessionId": "COTA_R2_2024_11_22",
  "driverAnalysis": {
    "performanceMetrics": [
      {
        "category": "Braking Consistency",
        "score": 87,
        "trend": "up",
        "benchmark": 85,
        "improvement": "+2 points since last session"
      },
      {
        "category": "Cornering Speed", 
        "score": 82,
        "trend": "stable",
        "benchmark": 88,
        "gapToBenchmark": "-6 points"
      },
      {
        "category": "Racing Line",
        "score": 79,
        "trend": "down", 
        "benchmark": 85,
        "criticalCorners": ["T4", "T7", "T11"]
      }
    ],
    "recommendations": [
      {
        "priority": "high",
        "area": "Racing Line Optimization",
        "description": "Focus on late apex corners (T4, T7, T11). You're turning in too early, losing exit speed.",
        "improvement": "+0.3s per lap",
        "technique": "Practice trail braking into corners, delay turn-in point by 2-3 meters"
      }
    ],
    "dataPoints": {
      "brakingPoints": [
        {
          "corner": "T1",
          "actual": 95.2,
          "optimal": 92.8,
          "delta": "+2.4m"
        }
      ],
      "corneringSpeeds": [
        {
          "corner": "T4", 
          "actual": 78.5,
          "optimal": 82.1,
          "delta": "-3.6 km/h"
        }
      ]
    }
  }
}
```

### AI Integration:
- **IRIS**: Driver behavior analysis
- **ATLAS**: Racing line optimization

### WebSocket Endpoint:
- **URL**: `ws://race-data-server:8080/live/training`
- **Update Frequency**: 0.2Hz (every 5 seconds)

---

## 📜 **6. HISTORY PAGE**
**Historical race data and trends**

### Data Requirements:
```json
{
  "sessionId": "COTA_R2_2024_11_22",
  "raceResults": [
    {
      "id": "unique-race-id",
      "date": "2024-10-20",
      "track": "Circuit of the Americas",
      "race": 2,
      "weather": "dry",
      "position": {
        "final": 4,
        "starting": 6,
        "best": 2
      },
      "lapTimes": {
        "best": "1:23.456",
        "average": "1:24.123",
        "consistency": 0.234
      },
      "performance": {
        "totalTime": "42:15.789",
        "points": 12,
        "incidents": 0,
        "overtakes": 3,
        "positions_lost": 1
      },
      "telemetrySummary": {
        "avgSpeed": 154.2,
        "topSpeed": 187.6,
        "fuelEfficiency": 2.1,
        "tireStrategy": "two_stop"
      }
    }
  ],
  "seasonStats": {
    "totalRaces": 12,
    "podiums": 3,
    "wins": 1,
    "points": 156,
    "avgPosition": 5.2,
    "bestLapTimes": {
      "COTA": "1:23.234",
      "Road_America": "2:05.123"
    }
  }
}
```

### Database Integration:
- **SQLite**: Historical race results
- **InfluxDB**: Telemetry history queries

---

## 🔌 **INTEGRATION SETUP FOR RACE ENGINEERS**

### 1. WebSocket Server Configuration
```bash
# Race data server endpoints
ws://localhost:8080/live/dashboard    # Dashboard updates
ws://localhost:8080/live/telemetry    # Telemetry stream
ws://localhost:8080/live/strategy     # Strategy updates  
ws://localhost:8080/live/timing       # Timing data
ws://localhost:8080/live/training     # Driver analysis
```

### 2. Data Pipeline Setup
```javascript
// Example integration in race control system
const raceDataClient = {
  // Send dashboard data
  sendDashboardUpdate: (data) => {
    websocket.send(JSON.stringify({
      type: 'dashboard_update',
      data: data
    }));
  },
  
  // Send telemetry stream
  sendTelemetry: (telemetryData) => {
    websocket.send(JSON.stringify({
      type: 'telemetry',
      data: telemetryData
    }));
  }
};
```

### 3. Required Environment Variables
```bash
# ORIS Configuration
VITE_WS_URL=ws://race-data-server:8080
VITE_INFLUX_HOST=http://influxdb:8086
VITE_INFLUX_TOKEN=your-influxdb-token
VITE_INFLUX_DATABASE=toyota_gr_telemetry
VITE_INFLUX_ORG=toyota_racing

# Race Session Config  
RACE_SESSION_ID=COTA_R2_2024_11_22
TRACK_ID=cota
CAR_NUMBER=7
```

### 4. Data Frequency Requirements
- **Telemetry**: 10Hz (critical for real-time analysis)
- **Dashboard**: 1Hz (live position updates)
- **Strategy**: 0.5Hz (AI model predictions)  
- **Timing**: 1Hz (lap time updates)
- **Training**: 0.2Hz (driver analysis)

### 5. AI Model Data Processing
The system feeds live data to 5 specialized AI models:
- **MINERVA** (Strategy): Uses position, timing, fuel data
- **ATLAS** (Spatial): Uses track position, racing line data
- **IRIS** (Dynamics): Uses telemetry sensors, driver inputs
- **CHRONOS** (Timing): Uses lap times, sector analysis
- **PROMETHEUS** (Prediction): Uses weather, incident data

---

## 📋 **TESTING DATA EXAMPLE**

Save this as `sample_race_data.json` for testing:

```json
{
  "testData": {
    "dashboard": {
      "timestamp": "2024-11-22T23:15:42.123Z",
      "sessionId": "COTA_R2_TEST",
      "trackId": "cota",
      "currentLap": 23,
      "position": {"current": 4, "starting": 7},
      "timing": {
        "currentLapTime": "1:23.456",
        "lastLapTime": "1:23.234", 
        "bestLapTime": "1:22.987"
      }
    },
    "telemetry": {
      "speed": 156.7,
      "rpm": 7200,
      "gear": 4,
      "throttle": 85.2,
      "brake": 0.0,
      "tireTemps": {"fl": 92.5, "fr": 94.2, "rl": 88.1, "rr": 89.7}
    }
  }
}
```

This specification provides race engineers with everything needed to integrate live Toyota GR Cup data into ORIS for real-time race analytics and strategy optimization.