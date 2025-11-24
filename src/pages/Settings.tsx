import React, { useState, useEffect } from 'react';
import { Card } from '../components/common';
import styles from './Settings.module.css';
import { aiModelClient } from '../services/aiModelClient';

interface SettingsData {
  general: {
    driverName: string;
    carNumber: number;
    team: string;
    units: 'metric' | 'imperial';
    theme: 'dark' | 'light';
  };
  aiModels: {
    enabled: boolean;
    apiUrl: string;
    updateFrequency: number;
    modelConfidence: number;
  };
  telemetry: {
    sampleRate: number;
    influxUrl: string;
    influxToken: string;
    database: string;
    username: string;
  };
  alerts: {
    fuelWarning: number;
    temperatureThreshold: number;
    tireWarning: number;
    enableAudio: boolean;
  };
}

export const Settings: React.FC = () => {
  const [settings, setSettings] = useState<SettingsData>({
    general: {
      driverName: 'Car #7',
      carNumber: 7,
      team: 'Toyota GR Cup',
      units: 'imperial',
      theme: 'dark'
    },
    aiModels: {
      enabled: true,
      apiUrl: 'http://localhost:8000',
      updateFrequency: 5000,
      modelConfidence: 85
    },
    telemetry: {
      sampleRate: 60,
      influxUrl: 'http://localhost:8181',
      influxToken: 'Invertedskynet2$',
      database: 'toyota_gr_telemetry',
      username: 'AutomataNexus'
    },
    alerts: {
      fuelWarning: 10,
      temperatureThreshold: 110,
      tireWarning: 85,
      enableAudio: true
    }
  });

  const [aiStatus, setAiStatus] = useState<Record<string, boolean>>({});
  const [isSaving, setIsSaving] = useState(false);

  // Check AI model status
  useEffect(() => {
    const checkAIStatus = async () => {
      try {
        const status = await aiModelClient.getModelStatus();
        setAiStatus(status);
      } catch (error) {
        console.error('Failed to get AI status:', error);
      }
    };

    checkAIStatus();
    const interval = setInterval(checkAIStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSettingChange = (section: keyof SettingsData, key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
  };

  const handleSaveSettings = async () => {
    setIsSaving(true);
    try {
      // Save to localStorage for now
      localStorage.setItem('oris-settings', JSON.stringify(settings));
      console.log('✅ Settings saved successfully');
      
      // Here you would typically save to a backend API
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
      
    } catch (error) {
      console.error('❌ Failed to save settings:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleResetSettings = () => {
    if (confirm('Are you sure you want to reset all settings to default?')) {
      localStorage.removeItem('oris-settings');
      window.location.reload();
    }
  };

  const handleCreateUser = async () => {
    const username = prompt('Enter username for new InfluxDB user:');
    const password = prompt('Enter password for new user:');
    const permissions = prompt('Enter permissions (read/write/admin):', 'read');
    
    if (username && password) {
      try {
        // Call InfluxDB 3 Core API to create user
        const response = await fetch(`${settings.telemetry.influxUrl}/api/v3/users`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${settings.telemetry.influxToken}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            username,
            password,
            permissions: permissions.split(',').map(p => p.trim())
          })
        });
        
        if (response.ok) {
          alert(`✅ User "${username}" created successfully`);
        } else {
          const error = await response.text();
          alert(`❌ Failed to create user: ${error}`);
        }
      } catch (error) {
        alert(`❌ Error creating user: ${error.message}`);
      }
    }
  };

  const handleManageTokens = async () => {
    try {
      // Call InfluxDB 3 Core API to list tokens
      const response = await fetch(`${settings.telemetry.influxUrl}/api/v3/tokens`, {
        headers: {
          'Authorization': `Bearer ${settings.telemetry.influxToken}`
        }
      });
      
      if (response.ok) {
        const tokens = await response.json();
        const tokenList = tokens.map(t => `${t.description || 'Unnamed'}: ${t.permissions.join(', ')}`).join('\n');
        alert(`Current Tokens:\n${tokenList}`);
      } else {
        alert('❌ Failed to fetch tokens');
      }
    } catch (error) {
      alert(`❌ Error fetching tokens: ${error.message}`);
    }
  };

  const handleTestConnection = async () => {
    try {
      const response = await fetch(`${settings.telemetry.influxUrl}/health`, {
        headers: {
          'Authorization': `Bearer ${settings.telemetry.influxToken}`
        }
      });
      
      if (response.ok) {
        const health = await response.json();
        alert(`✅ Connection successful!\nStatus: ${health.status}\nVersion: ${health.version || 'InfluxDB 3 Core'}`);
      } else {
        alert(`❌ Connection failed: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      alert(`❌ Connection error: ${error.message}`);
    }
  };

  // Load settings from localStorage on mount
  useEffect(() => {
    try {
      const savedSettings = localStorage.getItem('oris-settings');
      if (savedSettings) {
        setSettings(JSON.parse(savedSettings));
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  }, []);

  return (
      <div className={styles.settingsContainer}>
        <h1 className={styles.pageTitle}>System Settings</h1>
        
        <div className={styles.cardsGrid}>
          {/* General Settings */}
          <Card title="GENERAL SETTINGS">
          <div className={styles.settingsGrid}>
            <div className={styles.settingRow}>
              <label className={styles.label}>Driver Name</label>
              <input
                type="text"
                className={styles.input}
                value={settings.general.driverName}
                onChange={(e) => handleSettingChange('general', 'driverName', e.target.value)}
              />
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>Car Number</label>
              <input
                type="number"
                className={styles.input}
                value={settings.general.carNumber}
                onChange={(e) => handleSettingChange('general', 'carNumber', parseInt(e.target.value))}
              />
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>Team</label>
              <input
                type="text"
                className={styles.input}
                value={settings.general.team}
                onChange={(e) => handleSettingChange('general', 'team', e.target.value)}
              />
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>Units</label>
              <select
                className={styles.select}
                value={settings.general.units}
                onChange={(e) => handleSettingChange('general', 'units', e.target.value)}
              >
                <option value="imperial">Imperial (mph, °F)</option>
                <option value="metric">Metric (km/h, °C)</option>
              </select>
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>Theme</label>
              <select
                className={styles.select}
                value={settings.general.theme}
                onChange={(e) => handleSettingChange('general', 'theme', e.target.value)}
              >
                <option value="dark">Dark</option>
                <option value="light">Light</option>
              </select>
            </div>
          </div>
        </Card>

        {/* AI Models Settings */}
        <Card title="AI MODELS">
          <div className={styles.settingsGrid}>
            <div className={styles.settingRow}>
              <label className={styles.label}>Enable AI Models</label>
              <label className={styles.toggle}>
                <input
                  type="checkbox"
                  checked={settings.aiModels.enabled}
                  onChange={(e) => handleSettingChange('aiModels', 'enabled', e.target.checked)}
                />
                <span className={styles.slider}></span>
              </label>
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>API URL</label>
              <input
                type="url"
                className={styles.input}
                value={settings.aiModels.apiUrl}
                onChange={(e) => handleSettingChange('aiModels', 'apiUrl', e.target.value)}
              />
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>Update Frequency (ms)</label>
              <input
                type="number"
                className={styles.input}
                value={settings.aiModels.updateFrequency}
                onChange={(e) => handleSettingChange('aiModels', 'updateFrequency', parseInt(e.target.value))}
              />
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>Min Confidence (%)</label>
              <input
                type="range"
                className={styles.slider}
                min="0"
                max="100"
                value={settings.aiModels.modelConfidence}
                onChange={(e) => handleSettingChange('aiModels', 'modelConfidence', parseInt(e.target.value))}
              />
              <span className={styles.rangeValue}>{settings.aiModels.modelConfidence}%</span>
            </div>
          </div>
          
          {/* AI Model Status */}
          <div className={styles.modelStatus}>
            <h4>Model Status</h4>
            <div className={styles.statusGrid}>
              {['minerva', 'atlas', 'iris', 'chronos', 'prometheus'].map(model => (
                <div key={model} className={styles.modelItem}>
                  <span className={styles.modelName}>{model.toUpperCase()}</span>
                  <span className={`${styles.statusDot} ${aiStatus[model] ? styles.online : styles.offline}`}></span>
                  <span className={styles.statusText}>
                    {aiStatus[model] ? 'Online' : 'Offline'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Telemetry Settings */}
        <Card title="TELEMETRY DATABASE">
          <div className={styles.settingsGrid}>
            <div className={styles.settingRow}>
              <label className={styles.label}>Sample Rate (Hz)</label>
              <select
                className={styles.select}
                value={settings.telemetry.sampleRate}
                onChange={(e) => handleSettingChange('telemetry', 'sampleRate', parseInt(e.target.value))}
              >
                <option value={10}>10 Hz</option>
                <option value={30}>30 Hz</option>
                <option value={60}>60 Hz</option>
                <option value={100}>100 Hz</option>
              </select>
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>InfluxDB 3 OSS URL</label>
              <input
                type="url"
                className={styles.input}
                value={settings.telemetry.influxUrl}
                onChange={(e) => handleSettingChange('telemetry', 'influxUrl', e.target.value)}
                placeholder="http://localhost:8181"
              />
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>Username</label>
              <input
                type="text"
                className={styles.input}
                value={settings.telemetry.username}
                onChange={(e) => handleSettingChange('telemetry', 'username', e.target.value)}
                placeholder="AutomataNexus"
              />
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>Admin Token</label>
              <input
                type="password"
                className={styles.input}
                value={settings.telemetry.influxToken}
                onChange={(e) => handleSettingChange('telemetry', 'influxToken', e.target.value)}
                placeholder="InfluxDB 3 Core admin token"
              />
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>Database</label>
              <input
                type="text"
                className={styles.input}
                value={settings.telemetry.database}
                onChange={(e) => handleSettingChange('telemetry', 'database', e.target.value)}
                placeholder="toyota_gr_telemetry"
              />
            </div>
          </div>
          
          {/* Database User Management */}
          <div className={styles.userManagement}>
            <h4>Database User Management</h4>
            <div className={styles.actionButtons}>
              <button
                className={`${styles.button} ${styles.secondary}`}
                onClick={() => handleCreateUser()}
              >
                Create New User
              </button>
              <button
                className={`${styles.button} ${styles.secondary}`}
                onClick={() => handleManageTokens()}
              >
                Manage Tokens
              </button>
              <button
                className={`${styles.button} ${styles.secondary}`}
                onClick={() => handleTestConnection()}
              >
                Test Connection
              </button>
            </div>
          </div>
        </Card>

        {/* Alert Settings */}
        <Card title="ALERTS & WARNINGS">
          <div className={styles.settingsGrid}>
            <div className={styles.settingRow}>
              <label className={styles.label}>Fuel Warning (%)</label>
              <input
                type="number"
                className={styles.input}
                min="0"
                max="50"
                value={settings.alerts.fuelWarning}
                onChange={(e) => handleSettingChange('alerts', 'fuelWarning', parseInt(e.target.value))}
              />
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>Temperature Threshold (°C)</label>
              <input
                type="number"
                className={styles.input}
                min="80"
                max="150"
                value={settings.alerts.temperatureThreshold}
                onChange={(e) => handleSettingChange('alerts', 'temperatureThreshold', parseInt(e.target.value))}
              />
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>Tire Warning (%)</label>
              <input
                type="number"
                className={styles.input}
                min="50"
                max="95"
                value={settings.alerts.tireWarning}
                onChange={(e) => handleSettingChange('alerts', 'tireWarning', parseInt(e.target.value))}
              />
            </div>
            
            <div className={styles.settingRow}>
              <label className={styles.label}>Audio Alerts</label>
              <label className={styles.toggle}>
                <input
                  type="checkbox"
                  checked={settings.alerts.enableAudio}
                  onChange={(e) => handleSettingChange('alerts', 'enableAudio', e.target.checked)}
                />
                <span className={styles.slider}></span>
              </label>
            </div>
          </div>
        </Card>
        </div>

        {/* Action Buttons */}
        <div className={styles.actionButtons}>
          <button
            className={`${styles.button} ${styles.primary}`}
            onClick={handleSaveSettings}
            disabled={isSaving}
          >
            {isSaving ? 'Saving...' : 'Save Settings'}
          </button>
          
          <button
            className={`${styles.button} ${styles.secondary}`}
            onClick={handleResetSettings}
          >
            Reset to Defaults
          </button>
        </div>
      </div>
  );
};