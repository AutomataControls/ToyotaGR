import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from './components/layout'
import { Dashboard } from './components/Dashboard'
import { Telemetry } from './pages/Telemetry'
import { Strategy } from './pages/Strategy'
import { Timing } from './pages/Timing'
import { Training } from './pages/Training'
import { History } from './pages/History'
import { Settings } from './pages/Settings'
import { Help } from './pages/Help'
import { Database } from './pages/Database'
// import { Garage } from './pages/Garage' // REMOVED
// import { initializeTelemetryDB } from './services/database/influxdb'
// import { initializeRaceDB } from './services/database/sqlite'
// import { initializeTelemetryWebSocket } from './services/websocket/telemetryStream'
import './App.css'

// Database configuration
const INFLUX_CONFIG = {
  host: import.meta.env.VITE_INFLUX_HOST || 'http://localhost:8086',
  token: import.meta.env.VITE_INFLUX_TOKEN || 'your-token-here',
  database: import.meta.env.VITE_INFLUX_DATABASE || 'telemetry',
  org: import.meta.env.VITE_INFLUX_ORG || 'oris'
};

const WEBSOCKET_CONFIG = {
  url: import.meta.env.VITE_WS_URL || 'ws://localhost:8080/telemetry'
};

function App() {
  const [activeNav, setActiveNav] = useState('dashboard')
  const [isLive, setIsLive] = useState(true)

  useEffect(() => {
    // Initialize databases - but don't fail if they're not available
    const initializeDatabases = async () => {
      try {
        // Try to initialize SQLite database
        // Database initialization disabled for demo
        console.log('📊 Database initialization skipped for demo mode');
      } catch (error) {
        console.warn('SQLite initialization skipped:', error.message);
      }

      // InfluxDB and WebSocket initialization disabled for demo
      console.log('📊 Demo mode active - using Toyota data loader instead');
    };

    initializeDatabases();
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="/" element={
          <Layout 
            isLive={isLive} 
            sessionType="Race"
            activeNav={activeNav}
            onNavChange={setActiveNav}
          >
            <Navigate to="/dashboard" replace />
          </Layout>
        } />
        
        <Route path="/dashboard" element={
          <Layout 
            isLive={isLive} 
            sessionType="Race"
            activeNav="dashboard"
            onNavChange={setActiveNav}
          >
            <Dashboard />
          </Layout>
        } />
        
        <Route path="/telemetry" element={
          <Layout 
            isLive={isLive} 
            sessionType="Race"
            activeNav="telemetry"
            onNavChange={setActiveNav}
          >
            <Telemetry />
          </Layout>
        } />
        
        <Route path="/strategy" element={
          <Layout 
            isLive={isLive} 
            sessionType="Race"
            activeNav="strategy"
            onNavChange={setActiveNav}
          >
            <Strategy />
          </Layout>
        } />
        
        <Route path="/timing" element={
          <Layout 
            isLive={isLive} 
            sessionType="Race"
            activeNav="timing"
            onNavChange={setActiveNav}
          >
            <Timing />
          </Layout>
        } />
        
        <Route path="/training" element={
          <Layout 
            isLive={isLive} 
            sessionType="Race"
            activeNav="training"
            onNavChange={setActiveNav}
          >
            <Training />
          </Layout>
        } />
        
        <Route path="/history" element={
          <Layout 
            isLive={isLive} 
            sessionType="Race"
            activeNav="history"
            onNavChange={setActiveNav}
          >
            <History />
          </Layout>
        } />
        
        <Route path="/database" element={
          <Layout 
            isLive={isLive} 
            sessionType="Race"
            activeNav="database"
            onNavChange={setActiveNav}
          >
            <Database />
          </Layout>
        } />
        
        <Route path="/tracks" element={
          <Layout 
            isLive={isLive} 
            sessionType="Race"
            activeNav="tracks"
            onNavChange={setActiveNav}
          >
            <div style={{ padding: '2rem', textAlign: 'center' }}>
              <h2>Tracks</h2>
              <p style={{ color: '#6b7280' }}>Track management coming soon</p>
            </div>
          </Layout>
        } />

        <Route path="/settings" element={
          <Layout 
            isLive={isLive} 
            sessionType="Race"
            activeNav="settings"
            onNavChange={setActiveNav}
          >
            <Settings />
          </Layout>
        } />
        
        <Route path="/help" element={
          <Layout 
            isLive={isLive} 
            sessionType="Race"
            activeNav="help"
            onNavChange={setActiveNav}
          >
            <Help />
          </Layout>
        } />
      </Routes>
    </Router>
  )
}

export default App