import React, { useState, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, useGLTF, Environment, PresentationControls } from '@react-three/drei';
import { Card } from '../components/common';
import { Wrench, ChevronLeft, ChevronRight, RotateCw } from 'lucide-react';
import styles from './Garage.module.css';

// Vehicle model component
function VehicleModel({ modelPath }: { modelPath: string }) {
  const { scene } = useGLTF(modelPath);
  
  return (
    <primitive 
      object={scene} 
      scale={[3, 3, 3]} 
      position={[0, -0.5, 0]}
    />
  );
}

interface Vehicle {
  id: string;
  name: string;
  model: string;
  specs: {
    engine: string;
    power: string;
    weight: string;
    transmission: string;
  };
}

export const Garage: React.FC = () => {
  const [autoRotate, setAutoRotate] = useState(true);
  const [currentVehicleIndex, setCurrentVehicleIndex] = useState(0);

  // Vehicle data with actual GLB file names
  const vehicles: Vehicle[] = [
    {
      id: 'racing-hybrid',
      name: 'Racing Hybrid Model',
      model: '/Vehicles/Racing_Hybrid_Model__1025233304_texture.glb',
      specs: {
        engine: '2.4L Hybrid',
        power: '300 hp @ 7,000 rpm',
        weight: '2,950 lbs',
        transmission: '8-speed automatic'
      }
    },
    {
      id: 'race-car-champion',
      name: 'Race Car Champion',
      model: '/Vehicles/Race_Car_Champion_1025233323_texture.glb',
      specs: {
        engine: '3.0L Turbo V6',
        power: '450 hp @ 6,500 rpm',
        weight: '3,100 lbs',
        transmission: '7-speed sequential'
      }
    },
    {
      id: 'racing-champion',
      name: 'Racing Champion Car',
      model: '/Vehicles/Racing_Champion_Car_1025231759_texture.glb',
      specs: {
        engine: '2.0L Turbo I4',
        power: '350 hp @ 7,500 rpm',
        weight: '2,800 lbs',
        transmission: '6-speed sequential'
      }
    },
    {
      id: 'rc-car-chassis',
      name: 'RC Car Chassis Model',
      model: '/Vehicles/RC_Car_Chassis_Model_1025231453_texture.glb',
      specs: {
        engine: '1.6L Turbo I4',
        power: '280 hp @ 6,800 rpm',
        weight: '2,600 lbs',
        transmission: '6-speed manual'
      }
    }
  ];

  const currentVehicle = vehicles[currentVehicleIndex];

  const nextVehicle = () => {
    setCurrentVehicleIndex((prev) => (prev + 1) % vehicles.length);
  };

  const previousVehicle = () => {
    setCurrentVehicleIndex((prev) => (prev - 1 + vehicles.length) % vehicles.length);
  };

  return (
    <div className={styles.garagePage}>
      <div className={styles.header}>
        <h1 className={styles.title}>
          <Wrench size={24} />
          Garage
        </h1>
        <div className={styles.controls}>
          <button 
            className={styles.rotateButton}
            onClick={() => setAutoRotate(!autoRotate)}
            title={autoRotate ? "Stop rotation" : "Start rotation"}
          >
            <RotateCw size={18} className={autoRotate ? styles.rotating : ''} />
            {autoRotate ? 'Auto-Rotate On' : 'Auto-Rotate Off'}
          </button>
        </div>
      </div>

      <div className={styles.content}>
        <Card className={styles.viewerSection}>
            <div className={styles.vehicleHeader}>
              <button 
                className={styles.navButton} 
                onClick={previousVehicle}
                aria-label="Previous vehicle"
              >
                <ChevronLeft size={24} />
              </button>
              
              <h2 className={styles.vehicleName}>{currentVehicle.name}</h2>
              
              <button 
                className={styles.navButton} 
                onClick={nextVehicle}
                aria-label="Next vehicle"
              >
                <ChevronRight size={24} />
              </button>
            </div>

            <div className={styles.canvas}>
              <Canvas
                camera={{ 
                  position: [2.5, 1.5, 2.5], 
                  fov: 50,
                  near: 0.1,
                  far: 100
                }}
                shadows
              >
                <ambientLight intensity={0.7} />
                <directionalLight 
                  position={[5, 8, 5]} 
                  intensity={1.2} 
                  castShadow 
                  shadow-mapSize-width={2048}
                  shadow-mapSize-height={2048}
                  shadow-camera-near={0.1}
                  shadow-camera-far={50}
                  shadow-camera-left={-10}
                  shadow-camera-right={10}
                  shadow-camera-top={10}
                  shadow-camera-bottom={-10}
                />
                <spotLight 
                  position={[-5, 5, 0]} 
                  intensity={0.5} 
                  angle={0.3} 
                />
                <pointLight position={[0, 5, 0]} intensity={0.3} />
                
                <Suspense fallback={null}>
                  <VehicleModel modelPath={currentVehicle.model} />
                </Suspense>
                
                <OrbitControls 
                  enablePan={true}
                  autoRotate={autoRotate}
                  autoRotateSpeed={1}
                  maxPolarAngle={Math.PI / 2}
                  minDistance={1}
                  maxDistance={8}
                  target={[0, 0, 0]}
                />
                
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]} receiveShadow>
                  <planeGeometry args={[20, 20]} />
                  <meshStandardMaterial color="#f0f0f0" />
                </mesh>
              </Canvas>
            </div>

            <div className={styles.vehicleIndicators}>
              {vehicles.map((_, index) => (
                <button
                  key={index}
                  className={`${styles.indicator} ${index === currentVehicleIndex ? styles.active : ''}`}
                  onClick={() => setCurrentVehicleIndex(index)}
                  aria-label={`Select vehicle ${index + 1}`}
                />
              ))}
            </div>
        </Card>

        <div className={styles.infoSection}>
          <Card className={styles.specsCard}>
            <h3>Vehicle Specifications</h3>
            <div className={styles.specs}>
              <div className={styles.specItem}>
                <span className={styles.specLabel}>Engine</span>
                <span className={styles.specValue}>{currentVehicle.specs.engine}</span>
              </div>
              <div className={styles.specItem}>
                <span className={styles.specLabel}>Power</span>
                <span className={styles.specValue}>{currentVehicle.specs.power}</span>
              </div>
              <div className={styles.specItem}>
                <span className={styles.specLabel}>Weight</span>
                <span className={styles.specValue}>{currentVehicle.specs.weight}</span>
              </div>
              <div className={styles.specItem}>
                <span className={styles.specLabel}>Transmission</span>
                <span className={styles.specValue}>{currentVehicle.specs.transmission}</span>
              </div>
            </div>
          </Card>

          <Card className={styles.setupCard}>
            <h3>Race Setup</h3>
            <div className={styles.setupOptions}>
              <div className={styles.setupGroup}>
                <h4>Suspension</h4>
                <div className={styles.setupItem}>
                  <label>Front Ride Height</label>
                  <input type="range" min="0" max="100" defaultValue="50" />
                </div>
                <div className={styles.setupItem}>
                  <label>Rear Ride Height</label>
                  <input type="range" min="0" max="100" defaultValue="50" />
                </div>
              </div>

              <div className={styles.setupGroup}>
                <h4>Aerodynamics</h4>
                <div className={styles.setupItem}>
                  <label>Front Wing</label>
                  <input type="range" min="0" max="100" defaultValue="30" />
                </div>
                <div className={styles.setupItem}>
                  <label>Rear Wing</label>
                  <input type="range" min="0" max="100" defaultValue="70" />
                </div>
              </div>

              <div className={styles.setupGroup}>
                <h4>Tire Pressure (PSI)</h4>
                <div className={styles.tireGrid}>
                  <div className={styles.tireInput}>
                    <label>FL</label>
                    <input type="number" defaultValue="32" min="25" max="40" />
                  </div>
                  <div className={styles.tireInput}>
                    <label>FR</label>
                    <input type="number" defaultValue="32" min="25" max="40" />
                  </div>
                  <div className={styles.tireInput}>
                    <label>RL</label>
                    <input type="number" defaultValue="30" min="25" max="40" />
                  </div>
                  <div className={styles.tireInput}>
                    <label>RR</label>
                    <input type="number" defaultValue="30" min="25" max="40" />
                  </div>
                </div>
              </div>
            </div>

            <button className={styles.saveButton}>Save Setup</button>
          </Card>
        </div>
      </div>
    </div>
  );
};