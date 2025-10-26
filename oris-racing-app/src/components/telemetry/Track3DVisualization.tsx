import React, { Suspense, useRef, useMemo } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, Grid, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import { Card } from '../common';
import { getTrackConfig } from '../../data/trackMaps';
import styles from './Track3DVisualization.module.css';

interface Track3DVisualizationProps {
  trackId: string;
  carPositions: Array<{
    id: string;
    position: number;
    progress: number; // 0-1 along track
    color: string;
  }>;
  showGrid?: boolean;
  cameraFollow?: boolean;
}

function TrackModel({ trackId }: { trackId: string }) {
  const trackConfig = getTrackConfig(trackId);
  const modelPath = trackConfig?.model3D || null;
  
  // Load 3D model if available
  const { scene } = useGLTF(modelPath || '/track-models/placeholder.glb', true);
  
  // If we have a real model, use it
  if (modelPath && scene) {
    // Different scales for different tracks based on their size
    const trackScales: Record<string, number[]> = {
      barber: [0.01, 0.01, 0.01],      // 2.28 miles
      sonoma: [0.009, 0.009, 0.009],   // 2.505 miles
      vir: [0.0075, 0.0075, 0.0075],   // 3.27 miles
      cota: [0.008, 0.008, 0.008],     // 3.416 miles
      sebring: [0.007, 0.007, 0.007],  // 3.74 miles
      roadamerica: [0.006, 0.006, 0.006] // 4.014 miles - largest track
    };
    
    return (
      <primitive 
        object={scene} 
        scale={trackScales[trackId] || [0.01, 0.01, 0.01]}
        position={[0, 0, 0]}
        rotation={[0, 0, 0]}
      />
    );
  }
  
  // Otherwise create a placeholder track
  const track = useMemo(() => {
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-10, 0, -10),
      new THREE.Vector3(10, 0, -10),
      new THREE.Vector3(10, 0, 10),
      new THREE.Vector3(-10, 0, 10),
      new THREE.Vector3(-10, 0, -10),
    ]);
    
    const geometry = new THREE.TubeGeometry(curve, 100, 0.5, 8, true);
    const material = new THREE.MeshStandardMaterial({ 
      color: '#374151',
      roughness: 0.8 
    });
    
    return new THREE.Mesh(geometry, material);
  }, []);
  
  return <primitive object={track} />;
}

function Car({ position, color, progress }: { position: THREE.Vector3; color: string; progress: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame(() => {
    if (meshRef.current) {
      // Animate car movement
      meshRef.current.rotation.y = progress * Math.PI * 2;
    }
  });
  
  return (
    <mesh ref={meshRef} position={position}>
      <boxGeometry args={[0.8, 0.4, 1.6]} />
      <meshStandardMaterial color={color} metalness={0.6} roughness={0.3} />
    </mesh>
  );
}

function Scene({ trackId, carPositions, showGrid, cameraFollow }: Track3DVisualizationProps) {
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);
  
  // Calculate car positions on track
  const carMeshes = carPositions.map((car) => {
    // Simple circular track for demo - replace with actual track spline data
    const angle = car.progress * Math.PI * 2;
    const radius = 10;
    const position = new THREE.Vector3(
      Math.cos(angle) * radius,
      0.2,
      Math.sin(angle) * radius
    );
    
    return (
      <Car 
        key={car.id}
        position={position}
        color={car.color}
        progress={car.progress}
      />
    );
  });
  
  return (
    <>
      <PerspectiveCamera
        ref={cameraRef}
        makeDefault
        position={[0, 20, 20]}
        fov={60}
      />
      <OrbitControls 
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={10}
        maxDistance={50}
      />
      
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
      
      <TrackModel trackId={trackId} />
      {carMeshes}
      
      {showGrid && (
        <Grid 
          args={[50, 50]} 
          cellSize={1} 
          sectionSize={5}
          fadeDistance={50}
          cellColor="#e5e7eb"
          sectionColor="#9ca3af"
        />
      )}
      
      <Environment preset="park" />
    </>
  );
}

export const Track3DVisualization: React.FC<Track3DVisualizationProps> = ({
  trackId,
  carPositions,
  showGrid = true,
  cameraFollow = false
}) => {
  return (
    <Card title="3D TRACK VIEW">
      <div className={styles.container}>
        <Canvas shadows>
          <Suspense fallback={null}>
            <Scene 
              trackId={trackId}
              carPositions={carPositions}
              showGrid={showGrid}
              cameraFollow={cameraFollow}
            />
          </Suspense>
        </Canvas>
        
        <div className={styles.controls}>
          <div className={styles.legend}>
            {carPositions.slice(0, 5).map((car, index) => (
              <div key={car.id} className={styles.legendItem}>
                <div 
                  className={styles.colorDot} 
                  style={{ backgroundColor: car.color }}
                />
                <span className={styles.position}>P{car.position}</span>
              </div>
            ))}
          </div>
          <div className={styles.hint}>
            Click and drag to rotate • Scroll to zoom
          </div>
        </div>
      </div>
    </Card>
  );
};