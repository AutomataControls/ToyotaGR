import React, { useState } from 'react';
import { Card } from '../components/common';
import styles from './Help.module.css';
import { 
  HelpCircle, 
  Target, 
  Activity, 
  Timer, 
  Settings,
  Database,
  Brain,
  Zap,
  X,
  Layers,
  Network,
  GitBranch,
  Cpu
} from 'lucide-react';

interface ModelDetails {
  name: string;
  title: string;
  description: string;
  architecture: {
    type: string;
    layers: string[];
    inputSize: string;
    outputSize: string;
    parameters: string;
  };
  trainingData: string[];
  performance: {
    accuracy: string;
    latency: string;
    memoryUsage: string;
  };
  technicalDetails: string[];
}

const modelDetailsData: Record<string, ModelDetails> = {
  minerva: {
    name: 'MINERVA',
    title: 'Strategic Racing Intelligence Model',
    description: 'MINERVA processes Toyota GR Cup telemetry sequences to provide strategic racing decisions including pit strategy optimization, race pace management, traffic management, and fuel conservation strategies.',
    architecture: {
      type: 'Multi-Layer Attention + Strategic Analysis Network',
      layers: [
        'TelemetryEncoder: 8→256 dimensional encoding with LayerNorm',
        'StrategicAttention: 8-head MultiheadAttention with residual connections',
        'StrategyPredictor: Multiple specialized prediction heads',
        'Pit Strategy Head: 5 outputs (pit timing decisions)',
        'Pace Strategy Head: 3 outputs (push/maintain/conserve)',
        'Traffic Strategy Head: 4 outputs (overtake/follow/defend/let_pass)',
        'RaceStateAnalyzer: Tire degradation and gap analysis'
      ],
      inputSize: '(batch, 300, 8) - 5 minutes at 60Hz telemetry',
      outputSize: 'Multi-head strategic predictions + confidence scores',
      parameters: '~800K trainable parameters (256 hidden dim, 4 layers)'
    },
    trainingData: [
      'Toyota GR Cup telemetry: [speed, throttle, brake_f, brake_r, gear, steering, accx, accy]',
      'Strategic decision outcomes and race context',
      'Pit stop timing and tire strategy data',
      'Traffic management scenarios',
      'Race pace optimization patterns'
    ],
    performance: {
      accuracy: 'Real-time strategic decision making',
      latency: 'Optimized for 60Hz telemetry processing',
      memoryUsage: '256 hidden dimensions with dropout regularization'
    },
    technicalDetails: [
      'Multi-head attention captures strategic patterns across telemetry sequence',
      'Separate prediction heads for pit, pace, tire, fuel, and traffic strategies',
      'Confidence estimation with sigmoid activation for decision reliability',
      'Race state analysis includes tire degradation and gap monitoring',
      'Strategic priority determination and risk assessment algorithms'
    ]
  },
  atlas: {
    name: 'ATLAS',
    title: 'Spatial Track Intelligence Model',
    description: 'ATLAS specializes in spatial track intelligence for Toyota GR Cup Series, providing optimal racing line analysis, track position optimization, overtaking opportunity detection, and track limits monitoring.',
    architecture: {
      type: 'Multi-Module Spatial Analysis Network + Track Memory',
      layers: [
        'SpatialTelemetryEncoder: 8→256 dimensional spatial encoding',
        'SpatialAttention: 8-head attention with position bias',
        'RacingLineAnalyzer: Racing line quality and corner analysis',
        'OvertakingAnalyzer: Opportunity detection and defensive positioning',
        'TrackLimitsMonitor: Boundary detection and safety margins',
        'Track Memory: 100×256 parameter matrix for track learning',
        'Memory Attention: 4-head attention for track context integration'
      ],
      inputSize: '(batch, 300, 8) - Toyota GR Cup telemetry sequence',
      outputSize: 'Racing line + overtaking + track limits + spatial quality',
      parameters: '~1.2M trainable parameters (256 hidden dim, 4 layers + track memory)'
    },
    trainingData: [
      'Toyota GR Cup telemetry: [speed, throttle, brake_f, brake_r, gear, steering, accx, accy]',
      'Spatial positioning and track boundary data',
      'Racing line optimization patterns',
      'Overtaking scenario analysis',
      'Track limits and safety margin calculations'
    ],
    performance: {
      accuracy: 'Real-time spatial intelligence with confidence scoring',
      latency: 'Optimized for 60Hz spatial analysis',
      memoryUsage: '256 spatial features + track memory integration'
    },
    technicalDetails: [
      'Position-aware attention with bias parameters for spatial relationships',
      'Multi-head racing line analysis: quality, deviation, corner entry/apex/exit',
      'Overtaking analysis: probability, DRS advantage, slipstream, track position',
      'Track limits monitoring: risk assessment, kerb usage, off-track probability',
      'Spatial memory system learns track-specific patterns and characteristics'
    ]
  },
  iris: {
    name: 'IRIS',
    title: 'Vehicle Dynamics Intelligence Model',
    description: 'IRIS specializes in vehicle dynamics analysis for Toyota GR Cup Series, focusing on throttle/brake optimization, vehicle balance assessment, gear change strategy, steering analysis, and stability monitoring.',
    architecture: {
      type: 'Multi-Module Dynamics Analysis + Vehicle Memory',
      layers: [
        'DynamicsTelemetryEncoder: 8→256 dimensional dynamics encoding',
        'DynamicsAttention: 8-head attention for dynamics pattern recognition',
        'ThrottleBrakeAnalyzer: Throttle efficiency + brake modulation analysis',
        'VehicleBalanceAnalyzer: Balance, stability, and aerodynamic efficiency',
        'GearSteeringAnalyzer: Gear timing and steering smoothness analysis',
        'Vehicle Memory: 80×256 parameter matrix for dynamics pattern learning',
        'Performance Metrics: 6 efficiency and consistency indicators'
      ],
      inputSize: '(batch, 300, 8) - Toyota GR Cup telemetry sequence',
      outputSize: 'Throttle + brake + balance + gear + steering + performance metrics',
      parameters: '~1M trainable parameters (256 hidden dim, 4 layers + vehicle memory)'
    },
    trainingData: [
      'Toyota GR Cup telemetry: [speed, throttle, brake_f, brake_r, gear, steering, accx, accy]',
      'Vehicle dynamics patterns and efficiency metrics',
      'Throttle and brake coordination data',
      'Vehicle balance and stability measurements',
      'Gear change timing and steering input optimization'
    ],
    performance: {
      accuracy: 'Real-time dynamics analysis with performance scoring',
      latency: 'Optimized for 60Hz vehicle dynamics monitoring',
      memoryUsage: '256 dynamics features + vehicle memory integration'
    },
    technicalDetails: [
      'Multi-head attention captures dynamics patterns across telemetry sequence',
      'Throttle analysis: smoothness, efficiency, timing, peak usage, consistency',
      'Brake analysis: pressure, balance, modulation, timing, trail braking, efficiency',
      'Vehicle balance: front/rear balance, stability margin, grip utilization',
      'Performance metrics: acceleration/braking/cornering efficiency + optimization potential'
    ]
  },
  chronos: {
    name: 'CHRONOS',
    title: 'Timing Intelligence Model',
    description: 'CHRONOS specializes in timing intelligence for Toyota GR Cup Series, providing lap time prediction, sector analysis, race pace monitoring, timing consistency assessment, and performance progression tracking.',
    architecture: {
      type: 'Multi-Module Timing Analysis + LSTM + Timing Memory',
      layers: [
        'TimingTelemetryEncoder: 8→256 + positional encoding (1000 positions)',
        'TimingAttention: 8-head attention with temporal bias',
        'LapTimeAnalyzer: Lap prediction + sector analysis + consistency',
        'PaceAnalyzer: Race pace + stint analysis + position tracking',
        'TimingTrendAnalyzer: 2-layer LSTM (256→128) + trend classification',
        'Timing Memory: 120×256 parameter matrix for track timing patterns',
        'Benchmark Estimator: 8 performance benchmark metrics'
      ],
      inputSize: '(batch, 300, 8) - Toyota GR Cup telemetry sequence',
      outputSize: 'Lap times + sectors + pace + trends + benchmarks + progression',
      parameters: '~1.3M trainable parameters (256 hidden dim, 4 layers + LSTM + memory)'
    },
    trainingData: [
      'Toyota GR Cup telemetry: [speed, throttle, brake_f, brake_r, gear, steering, accx, accy]',
      'Lap time sequences and sector timing data',
      'Race pace patterns and stint analysis',
      'Timing consistency and performance progression',
      'Competitive benchmarks and position tracking'
    ],
    performance: {
      accuracy: 'Real-time timing analysis with confidence scoring',
      latency: 'Optimized for 60Hz timing intelligence',
      memoryUsage: '256 timing features + LSTM + track memory integration'
    },
    technicalDetails: [
      'Positional encoding for precise timing sequence relationships',
      'Multi-head timing attention with temporal bias for sequence patterns',
      'LSTM-based trend analysis: improving, declining, stable, variable, optimal',
      'Comprehensive sector analysis: 3 sectors + deltas + personal bests',
      '8-metric benchmarking: personal/session/theoretical/competitive + rankings'
    ]
  },
  prometheus: {
    name: 'PROMETHEUS',
    title: 'Predictive Analytics Model',
    description: 'PROMETHEUS specializes in predictive analytics for Toyota GR Cup Series, providing future lap time forecasting, tire degradation prediction, race outcome prediction, weather/fuel impact analysis, and performance trend forecasting.',
    architecture: {
      type: 'Multi-Module Predictive Analysis + LSTM + Predictive Memory',
      layers: [
        'PredictiveTelemetryEncoder: 8→256 dimensional forecasting encoding',
        'PredictiveAttention: 8-head attention for predictive pattern recognition',
        'LapTimePredictor: 10 future lap time forecasting outputs',
        'TireDegradationPredictor: Degradation + compound + pitstop optimization',
        'RaceOutcomePredictor: 2-layer LSTM (256→128) + position forecasting',
        'WeatherFuelPredictor: Weather impact + fuel consumption analysis',
        'Predictive Memory: 100×256 parameter matrix for forecasting patterns'
      ],
      inputSize: '(batch, 300, 8) - Toyota GR Cup telemetry sequence',
      outputSize: 'Lap forecasts + tire predictions + race outcomes + confidence metrics',
      parameters: '~1.4M trainable parameters (256 hidden dim, 4 layers + LSTM + memory)'
    },
    trainingData: [
      'Toyota GR Cup telemetry: [speed, throttle, brake_f, brake_r, gear, steering, accx, accy]',
      'Future lap time patterns and performance trajectories',
      'Tire degradation curves and compound performance data',
      'Race position progression and competitive analysis',
      'Weather impact patterns and fuel consumption rates'
    ],
    performance: {
      accuracy: 'Multi-horizon prediction with confidence intervals',
      latency: 'Optimized for 60Hz predictive analytics',
      memoryUsage: '256 predictive features + LSTM + forecasting memory'
    },
    technicalDetails: [
      'Multi-horizon forecasting: next 5 laps + best/worst/average case scenarios',
      'Tire degradation prediction: 8 metrics including critical thresholds + pit windows',
      'LSTM-based race outcome modeling: position, podium probability, overtake opportunities',
      '6-metric confidence estimation: short/medium/long-term + accuracy + uncertainty',
      'Weather and fuel impact analysis with strategic adjustment recommendations'
    ]
  }
};

export const Help: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const openModelModal = (modelKey: string) => {
    setSelectedModel(modelKey);
  };

  const closeModelModal = () => {
    setSelectedModel(null);
  };

  return (
    <div className={styles.helpContainer}>
      <h1 className={styles.pageTitle}>Help & Documentation</h1>
      
      {/* Getting Started */}
      <Card title="Getting Started">
        <div className={styles.section}>
          <h3><Target size={18} /> Welcome to ORIS</h3>
          <p>
            ORIS (OLYMPUS Racing Intelligence System) is an advanced AI-powered racing intelligence platform 
            designed for Toyota GR Cup drivers. The system provides real-time strategic insights, telemetry 
            analysis, and AI-driven recommendations to optimize race performance.
          </p>
          
          <div className={styles.quickStart}>
            <h4>Quick Start Guide:</h4>
            <ol>
              <li>Navigate to the <strong>Dashboard</strong> to view live race data</li>
              <li>Use the <strong>Strategy</strong> page for AI-powered pit stop recommendations</li>
              <li>Monitor <strong>Telemetry</strong> for real-time vehicle data analysis</li>
              <li>Manage live data connections in the <strong>Database</strong> page</li>
              <li>Configure system settings in the <strong>Settings</strong> page</li>
              <li>Reference this <strong>Help</strong> section for AI model details and troubleshooting</li>
            </ol>
          </div>
        </div>
      </Card>

      {/* AI Models */}
      <Card title="AI Specialist Models">
        <div className={styles.section}>
          <h3><Brain size={18} /> Understanding the AI Specialists</h3>
          <p>ORIS employs five specialized AI models, each focused on different aspects of racing intelligence:</p>
          
          <div className={styles.modelGrid}>
            <div className={`${styles.model} ${styles.clickable}`} onClick={() => openModelModal('minerva')}>
              <h4>MINERVA</h4>
              <p><strong>Strategic Analysis</strong> - Provides pit strategy recommendations, pace management, and tactical decision making based on race conditions.</p>
              <div className={styles.clickHint}>Click for detailed architecture</div>
            </div>
            
            <div className={`${styles.model} ${styles.clickable}`} onClick={() => openModelModal('atlas')}>
              <h4>ATLAS</h4>
              <p><strong>Spatial Reasoning</strong> - Analyzes track position, optimal racing lines, and sector-by-sector performance optimization.</p>
              <div className={styles.clickHint}>Click for detailed architecture</div>
            </div>
            
            <div className={`${styles.model} ${styles.clickable}`} onClick={() => openModelModal('iris')}>
              <h4>IRIS</h4>
              <p><strong>Vehicle Dynamics</strong> - Monitors vehicle balance, handling characteristics, and setup recommendations for optimal performance.</p>
              <div className={styles.clickHint}>Click for detailed architecture</div>
            </div>
            
            <div className={`${styles.model} ${styles.clickable}`} onClick={() => openModelModal('chronos')}>
              <h4>CHRONOS</h4>
              <p><strong>Timing Analysis</strong> - Provides lap time predictions, sector analysis, and improvement potential calculations.</p>
              <div className={styles.clickHint}>Click for detailed architecture</div>
            </div>
            
            <div className={`${styles.model} ${styles.clickable}`} onClick={() => openModelModal('prometheus')}>
              <h4>PROMETHEUS</h4>
              <p><strong>Predictive Modeling</strong> - Forecasts race outcomes, weather impacts, and strategic opportunities throughout the race.</p>
              <div className={styles.clickHint}>Click for detailed architecture</div>
            </div>
          </div>
        </div>
      </Card>

      {/* Features Guide */}
      <Card title="Features Guide">
        <div className={styles.section}>
          <h3><Activity size={18} /> Main Features</h3>
          
          <div className={styles.featureList}>
            <div className={styles.feature}>
              <h4><Target size={16} /> Strategy Advisor</h4>
              <p>AI-powered recommendations for pit strategy, tire management, and race tactics. Toggle between Standard, Push, and Conserve modes for different race situations.</p>
            </div>
            
            <div className={styles.feature}>
              <h4><Activity size={16} /> Live Telemetry</h4>
              <p>Real-time monitoring of vehicle data including speed, throttle, brake, steering, and sensor readings with interactive graphs and analysis.</p>
            </div>
            
            <div className={styles.feature}>
              <h4><Timer size={16} /> Timing Analysis</h4>
              <p>Comprehensive lap time analysis with sector breakdowns, personal best comparisons, and improvement recommendations.</p>
            </div>
            
            <div className={styles.feature}>
              <h4><Database size={16} /> Live Data Feed</h4>
              <p>Real-time telemetry streaming from field cars with toggle control, connection monitoring, and live packet analysis. Integrates with InfluxDB 3 Core for storage.</p>
            </div>
            
            <div className={styles.feature}>
              <h4><Settings size={16} /> System Configuration</h4>
              <p>Comprehensive settings management including general preferences, AI model configuration, database setup, and alert customization with full user access control.</p>
            </div>
          </div>
        </div>
      </Card>

      {/* Settings Help */}
      <Card title="System Configuration">
        <div className={styles.section}>
          <h3><Settings size={18} /> Settings Overview</h3>
          
          <div className={styles.settingsHelp}>
            <h4>General Settings</h4>
            <ul>
              <li><strong>Driver Name:</strong> Set your driver identification</li>
              <li><strong>Car Number:</strong> Specify your vehicle number</li>
              <li><strong>Team:</strong> Set your team affiliation</li>
              <li><strong>Units:</strong> Choose between Imperial (mph, °F) or Metric (km/h, °C)</li>
            </ul>
            
            <h4>AI Models Configuration</h4>
            <ul>
              <li><strong>API URL:</strong> Set the AI model server endpoint (default: http://localhost:8000)</li>
              <li><strong>Update Frequency:</strong> Configure how often AI predictions are updated</li>
              <li><strong>Confidence Threshold:</strong> Set minimum confidence level for AI recommendations</li>
            </ul>
            
            <h4>Database & Live Data Setup</h4>
            <ul>
              <li><strong>InfluxDB URL:</strong> Configure your InfluxDB 3 Core server (default: http://localhost:8181)</li>
              <li><strong>Username/Token:</strong> Set authentication credentials for database access</li>
              <li><strong>Database Name:</strong> Specify the telemetry database name</li>
              <li><strong>Live Data Feed:</strong> Toggle real-time data streaming from field cars</li>
              <li><strong>Connection Monitoring:</strong> View and manage active field car connections</li>
            </ul>
          </div>
        </div>
      </Card>

      {/* Troubleshooting */}
      <Card title="Troubleshooting">
        <div className={styles.section}>
          <h3><Zap size={18} /> Common Issues</h3>
          
          <div className={styles.troubleshooting}>
            <div className={styles.issue}>
              <h4>AI Models Showing Offline</h4>
              <p><strong>Solution:</strong> Ensure the AI API server is running on port 8000. Check the terminal for any Python errors and verify all dependencies are installed.</p>
            </div>
            
            <div className={styles.issue}>
              <h4>Database Connection Failed</h4>
              <p><strong>Solution:</strong> Verify InfluxDB 3 Core is running and accessible. Check your connection settings in the Settings page and ensure proper authentication tokens.</p>
            </div>
            
            <div className={styles.issue}>
              <h4>Telemetry Data Not Loading</h4>
              <p><strong>Solution:</strong> Check that Toyota GR Cup data files are present in the /data/tracks/ directory and properly formatted.</p>
            </div>
            
            <div className={styles.issue}>
              <h4>Live Data Feed Not Working</h4>
              <p><strong>Solution:</strong> Ensure the toggle is enabled in the Database page and that field cars are properly connected to the API endpoint. Check network connectivity.</p>
            </div>
            
            <div className={styles.issue}>
              <h4>Strategy Buttons Not Visible</h4>
              <p><strong>Solution:</strong> Try adjusting browser zoom level or maximize the window. The interface is optimized for desktop viewing.</p>
            </div>
          </div>
        </div>
      </Card>

      {/* Contact */}
      <Card title="Support">
        <div className={styles.section}>
          <h3><HelpCircle size={18} /> Need More Help?</h3>
          <p>
            ORIS is built for the Toyota GR "Hack the Track" hackathon. For technical support 
            or feature requests, please refer to the project documentation or contact the development team.
          </p>
          
          <div className={styles.supportInfo}>
            <h4>System Information</h4>
            <ul>
              <li>Version: 1.0.0</li>
              <li>Platform: Web Application</li>
              <li>AI Models: 5 Specialist Models (MINERVA, ATLAS, IRIS, CHRONOS, PROMETHEUS)</li>
              <li>Data Source: Toyota GR Cup Telemetry + Live Field Car Streams</li>
              <li>Database: InfluxDB 3 Core OSS</li>
              <li>Features: Real-time Analytics, Live Data Feed, AI Strategy Recommendations</li>
            </ul>
          </div>
        </div>
      </Card>

      {/* Model Details Modal */}
      {selectedModel && (
        <div className={styles.modalOverlay} onClick={closeModelModal}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h2>{modelDetailsData[selectedModel].name} - {modelDetailsData[selectedModel].title}</h2>
              <button className={styles.closeButton} onClick={closeModelModal}>
                <X size={24} />
              </button>
            </div>
            
            <div className={styles.modalContent}>
              <div className={styles.modalSection}>
                <h3>Overview</h3>
                <p>{modelDetailsData[selectedModel].description}</p>
              </div>

              <div className={styles.modalSection}>
                <h3><Cpu size={18} /> Architecture</h3>
                <div className={styles.architectureDetails}>
                  <div className={styles.archType}>
                    <strong>Type:</strong> {modelDetailsData[selectedModel].architecture.type}
                  </div>
                  <div className={styles.archSpecs}>
                    <div><strong>Input:</strong> {modelDetailsData[selectedModel].architecture.inputSize}</div>
                    <div><strong>Output:</strong> {modelDetailsData[selectedModel].architecture.outputSize}</div>
                    <div><strong>Parameters:</strong> {modelDetailsData[selectedModel].architecture.parameters}</div>
                  </div>
                  <div className={styles.layerList}>
                    <h4><Layers size={16} /> Network Layers:</h4>
                    <ul>
                      {modelDetailsData[selectedModel].architecture.layers.map((layer, index) => (
                        <li key={index}>{layer}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>

              <div className={styles.modalSection}>
                <h3><Database size={18} /> Training Data</h3>
                <ul>
                  {modelDetailsData[selectedModel].trainingData.map((data, index) => (
                    <li key={index}>{data}</li>
                  ))}
                </ul>
              </div>

              <div className={styles.modalSection}>
                <h3><Zap size={18} /> Performance Metrics</h3>
                <div className={styles.performanceGrid}>
                  <div className={styles.metric}>
                    <strong>Accuracy:</strong> {modelDetailsData[selectedModel].performance.accuracy}
                  </div>
                  <div className={styles.metric}>
                    <strong>Latency:</strong> {modelDetailsData[selectedModel].performance.latency}
                  </div>
                  <div className={styles.metric}>
                    <strong>Memory:</strong> {modelDetailsData[selectedModel].performance.memoryUsage}
                  </div>
                </div>
              </div>

              <div className={styles.modalSection}>
                <h3><Network size={18} /> Technical Details</h3>
                <ul>
                  {modelDetailsData[selectedModel].technicalDetails.map((detail, index) => (
                    <li key={index}>{detail}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};