import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, useGLTF, PerspectiveCamera, Box, Plane, Html, Loader } from '@react-three/drei';
import { getTrackConfig } from '../../data/trackMaps';
import styles from './TrackVisualization3D.module.css';

// Preload all track models
useGLTF.preload('/track-models/cota.glb');
useGLTF.preload('/track-models/roadamerica.glb');
useGLTF.preload('/track-models/sebring.glb');
useGLTF.preload('/track-models/sonoma.glb');
useGLTF.preload('/track-models/barber.glb');
useGLTF.preload('/track-models/vir.glb');

// Track model component
function TrackModel({ modelPath, trackId }: { modelPath: string; trackId: string }) {
  try {
    const { scene } = useGLTF(modelPath);
    
    // Adjust scale based on track - these values may need tweaking based on actual model sizes
    const trackScales: Record<string, number> = {
      cota: 0.05,
      roadamerica: 0.04,
      sebring: 0.05,
      sonoma: 0.06,
      barber: 0.07,
      vir: 0.05
    };
    
    const scale = trackScales[trackId] || 0.05;
    
    return (
      <primitive 
        object={scene} 
        scale={[scale, scale, scale]}
        position={[0, 0.01, 0]}
        rotation={[0, 0, 0]}
        castShadow
        receiveShadow
      />
    );
  } catch (error) {
    console.error('Error loading track model:', modelPath, error);
    return (
      <Box args={[2, 0.1, 1.5]} position={[0, 0.05, 0]}>
        <meshStandardMaterial color="#ff0000" opacity={0.5} transparent />
      </Box>
    );
  }
}

// Racing car indicator
function CarIndicator({ progress }: { progress: number }) {
  // Simple car representation - positioned based on progress
  const angle = (progress / 100) * Math.PI * 2;
  const radius = 0.8;
  const x = Math.sin(angle) * radius;
  const z = Math.cos(angle) * radius;
  
  return (
    <group position={[x, 0.03, z]}>
      <Box args={[0.06, 0.02, 0.1]} castShadow>
        <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.2} />
      </Box>
      {/* Car light effect */}
      <pointLight position={[0, 0.03, 0]} intensity={0.3} color="#ef4444" distance={0.5} />
    </group>
  );
}

interface TrackVisualization3DProps {
  trackId: string;
  driverProgress: number;
  currentLap: number;
  totalLaps: number;
}

export const TrackVisualization3D: React.FC<TrackVisualization3DProps> = ({
  trackId,
  driverProgress,
  currentLap,
  totalLaps
}) => {
  const track = getTrackConfig(trackId);
  if (!track || !track.model3D) {
    console.error('Track not found or no 3D model:', trackId, track);
    return null;
  }
  
  console.log('Loading track model:', track.model3D);

  const currentSector = track.sectors.find(sector => 
    driverProgress >= sector.start && driverProgress < sector.end
  ) || track.sectors[0];

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h4 className={styles.trackName}>{track.name}</h4>
        <div className={styles.stats}>
          <span className={styles.lapInfo}>Lap {currentLap}/{totalLaps}</span>
          <span className={`${styles.sector} ${styles[currentSector.difficulty]}`}>
            {currentSector.name}
          </span>
        </div>
      </div>
      
      <div className={styles.canvas}>
        <Canvas shadows>
          <PerspectiveCamera 
            makeDefault 
            position={[2, 1.5, 2]} 
            fov={45}
          />
          
          {/* Lighting */}
          <ambientLight intensity={0.6} />
          <directionalLight 
            position={[5, 10, 5]} 
            intensity={1} 
            castShadow
            shadow-mapSize={[2048, 2048]}
          />
          <pointLight position={[-5, 5, -5]} intensity={0.5} />
          
          {/* Track platform/board */}
          <group position={[0, -0.2, 0]}>
            {/* Main platform - thick board */}
            <Box args={[2.2, 0.4, 1.8]} receiveShadow castShadow>
              <meshStandardMaterial 
                color="#e5e7eb" 
                roughness={0.6} 
                metalness={0.05} 
              />
            </Box>
            
            {/* Subtle edge bevel effect */}
            <Box args={[2.15, 0.39, 1.75]} position={[0, 0, 0]}>
              <meshStandardMaterial 
                color="#f3f4f6" 
                roughness={0.7} 
                metalness={0} 
              />
            </Box>
          </group>
          
          {/* Track model */}
          <Suspense fallback={
            <Html center>
              <div style={{ 
                background: 'rgba(255, 255, 255, 0.9)', 
                padding: '10px', 
                borderRadius: '5px',
                fontSize: '14px',
                fontFamily: 'monospace'
              }}>
                Loading track model...
              </div>
            </Html>
          }>
            <TrackModel modelPath={track.model3D} trackId={trackId} />
            <CarIndicator progress={driverProgress} />
          </Suspense>
          
          {/* Controls */}
          <OrbitControls 
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={1}
            maxDistance={5}
            maxPolarAngle={Math.PI / 2.2}
            autoRotate={false}
            target={[0, 0, 0]}
          />
        </Canvas>
        <Loader />
      </div>
      
      <div className={styles.info}>
        <div className={styles.progressBar}>
          <div 
            className={styles.progressFill}
            style={{ width: `${driverProgress}%` }}
          />
        </div>
        <div className={styles.trackInfo}>
          <span>{track.length} miles</span>
          <span>{track.turns} turns</span>
          <span>{driverProgress.toFixed(1)}% complete</span>
        </div>
      </div>
    </div>
  );
};