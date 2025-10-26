// Track map configurations and data

export interface TrackSector {
  id: number;
  name: string;
  start: number; // percentage of track length
  end: number;   // percentage of track length
  difficulty: 'easy' | 'medium' | 'hard';
}

export interface TrackConfig {
  id: string;
  name: string;
  location: string;
  length: number; // in miles
  turns: number;
  sectors: TrackSector[];
  mapPdf: string;
  svgPath?: string; // Simplified SVG path for visualization
  model3D?: string; // Path to GLB 3D model
  characteristics: {
    type: string;
    elevation: string;
    surface: string;
  };
  gpsCoordinates?: {
    finishLine: { lat: number; lng: number };
    pitIn: { lat: number; lng: number };
    pitOut: { lat: number; lng: number };
  };
  pitLaneTime?: number; // seconds
}

export const trackConfigs: TrackConfig[] = [
  {
    id: 'cota',
    name: 'Circuit of the Americas',
    location: 'Austin, Texas',
    length: 3.416, // Accurate from spec sheet
    turns: 20,
    sectors: [
      { id: 1, name: 'Sector 1', start: 0, end: 24, difficulty: 'hard' }, // 51528/216468 = 24%
      { id: 2, name: 'Sector 2', start: 24, end: 64, difficulty: 'medium' }, // 88188/216468 = 40%
      { id: 3, name: 'Sector 3', start: 64, end: 100, difficulty: 'hard' } // 76752/216468 = 36%
    ],
    mapPdf: 'COTA_Circuit_Map.pdf',
    // COTA has a dramatic elevation change at Turn 1 and characteristic esses section
    svgPath: 'M 80 180 L 140 100 L 160 80 Q 170 70 180 70 L 220 70 L 240 80 L 250 100 L 250 120 L 240 140 L 220 150 L 200 150 L 180 140 L 160 140 L 140 150 L 130 170 L 120 190 L 100 200 L 80 200 Q 60 200 50 190 L 40 170 L 40 150 L 50 130 L 70 120 L 90 120 L 110 130 L 120 150 L 110 170 L 90 180 Z',
    model3D: '/track-models/cota.glb',
    characteristics: {
      type: 'Permanent Road Course',
      elevation: '508 ft',
      surface: 'Asphalt'
    },
    gpsCoordinates: {
      finishLine: { lat: 30.1335278, lng: -97.6422583 },
      pitIn: { lat: 30.1343371, lng: -97.6340257 },
      pitOut: { lat: 30.1314446, lng: -97.6389209 }
    },
    pitLaneTime: 36 // seconds at 50 kph
  },
  {
    id: 'roadamerica',
    name: 'Road America',
    location: 'Elkhart Lake, Wisconsin',
    length: 4.014, // Confirmed from spec sheet
    turns: 14,
    sectors: [
      { id: 1, name: 'Sector 1', start: 0, end: 32, difficulty: 'medium' }, // 81048/254316 = 32%
      { id: 2, name: 'Sector 2', start: 32, end: 66, difficulty: 'easy' }, // 86928/254316 = 34%
      { id: 3, name: 'Sector 3', start: 66, end: 100, difficulty: 'hard' } // 86340/254316 = 34%
    ],
    mapPdf: 'Road_America_Map.pdf',
    // Road America is known for its long straights and fast sweeping corners
    svgPath: 'M 40 160 L 40 100 L 50 80 L 80 60 L 120 50 L 180 50 L 220 60 L 250 80 L 260 100 L 260 140 L 250 160 L 240 170 L 220 180 L 200 180 L 180 170 L 160 170 L 140 180 L 120 190 L 80 190 L 50 180 L 40 160 Z',
    model3D: '/track-models/roadamerica.glb',
    characteristics: {
      type: 'Permanent Road Course',
      elevation: '1058 ft',
      surface: 'Asphalt'
    },
    gpsCoordinates: {
      finishLine: { lat: 43.7979056, lng: -87.9896333 },
      pitIn: { lat: 43.80057, lng: -87.98992 },
      pitOut: { lat: 43.7948061, lng: -87.9897494 }
    },
    pitLaneTime: 52 // seconds
  },
  {
    id: 'sebring',
    name: 'Sebring International Raceway',
    location: 'Sebring, Florida',
    length: 3.74, // Confirmed from spec sheet
    turns: 17,
    sectors: [
      { id: 1, name: 'Sector 1', start: 0, end: 30, difficulty: 'hard' }, // 71813/236966 = 30%
      { id: 2, name: 'Sector 2', start: 30, end: 61, difficulty: 'hard' }, // 73374/236966 = 31%
      { id: 3, name: 'Sector 3', start: 61, end: 100, difficulty: 'medium' } // 91773/236966 = 39%
    ],
    mapPdf: 'Sebring_Track_Sector_Map.pdf',
    // Sebring is famous for its bumpy surface and technical turns
    svgPath: 'M 60 120 L 80 100 L 120 90 L 160 90 L 200 100 L 220 120 L 230 140 L 230 160 L 220 170 L 200 180 L 160 180 L 120 170 L 90 160 L 70 140 L 60 120 Z',
    model3D: '/track-models/sebring.glb',
    characteristics: {
      type: 'Permanent Road Course',
      elevation: '61 ft',
      surface: 'Asphalt/Concrete'
    },
    gpsCoordinates: {
      finishLine: { lat: 27.4502340, lng: -81.3536980 },
      pitIn: { lat: 27.45012, lng: -81.35547 },
      pitOut: { lat: 27.45011, lng: -81.35051 }
    },
    pitLaneTime: 39 // seconds at 50 kph
  },
  {
    id: 'sonoma',
    name: 'Sonoma Raceway',
    location: 'Sonoma, California',
    length: 2.505, // Accurate from spec sheet
    turns: 12,
    sectors: [
      { id: 1, name: 'Sector 1', start: 0, end: 34, difficulty: 'medium' }, // 54520/158716 = 34%
      { id: 2, name: 'Sector 2', start: 34, end: 70, difficulty: 'hard' }, // 55976/158716 = 35%
      { id: 3, name: 'Sector 3', start: 70, end: 100, difficulty: 'medium' } // 48220/158716 = 31%
    ],
    mapPdf: 'Sonoma_Map.pdf',
    // Sonoma features dramatic elevation changes and a carousel turn
    svgPath: 'M 80 140 L 90 120 L 110 100 L 140 90 L 170 90 L 190 100 L 200 120 L 200 140 L 190 160 L 170 170 L 140 170 L 110 160 L 90 150 L 80 140 Z',
    model3D: '/track-models/sonoma.glb',
    characteristics: {
      type: 'Permanent Road Course',
      elevation: '20 ft',
      surface: 'Asphalt'
    },
    gpsCoordinates: {
      finishLine: { lat: 38.1615139, lng: -122.4547166 },
      pitIn: { lat: null, lng: null }, // Negative offset indicates before finish line
      pitOut: { lat: null, lng: null }
    },
    pitLaneTime: 45 // seconds at 50 kph
  },
  {
    id: 'barber',
    name: 'Barber Motorsports Park',
    location: 'Birmingham, Alabama',
    length: 2.28, // Accurate from spec sheet
    turns: 15,
    sectors: [
      { id: 1, name: 'Sector 1', start: 0, end: 28, difficulty: 'easy' }, // 40512/144672 = 28%
      { id: 2, name: 'Sector 2', start: 28, end: 71, difficulty: 'medium' }, // 62220/144672 = 43%
      { id: 3, name: 'Sector 3', start: 71, end: 100, difficulty: 'hard' } // 41940/144672 = 29%
    ],
    mapPdf: 'Barber_Circuit_Map.pdf',
    // Barber is known for its flowing nature and museum setting
    svgPath: 'M 80 130 L 90 110 L 110 90 L 140 80 L 170 80 L 190 90 L 200 110 L 200 130 L 190 150 L 170 160 L 140 160 L 110 150 L 90 140 L 80 130 Z',
    model3D: '/track-models/barber.glb',
    characteristics: {
      type: 'Permanent Road Course', 
      elevation: '646 ft',
      surface: 'Asphalt'
    },
    gpsCoordinates: {
      finishLine: { lat: 33.5326722, lng: -86.6196083 },
      pitIn: { lat: 33.531077, lng: -86.622592 },
      pitOut: { lat: 33.531111, lng: -86.622526 }
    },
    pitLaneTime: 34 // seconds
  },
  {
    id: 'vir',
    name: 'Virginia International Raceway',
    location: 'Alton, Virginia',
    length: 3.27, // From spec sheet
    turns: 17,
    sectors: [
      { id: 1, name: 'Sector 1', start: 0, end: 31, difficulty: 'medium' }, // 65064/207189 = 31%
      { id: 2, name: 'Sector 2', start: 31, end: 72, difficulty: 'hard' }, // 84960/207189 = 41%
      { id: 3, name: 'Sector 3', start: 72, end: 100, difficulty: 'medium' } // 57165/207189 = 28%
    ],
    mapPdf: 'VIR_Map.pdf',
    // VIR features the famous Oak Tree turn and uphill esses
    svgPath: 'M 50 150 L 60 130 L 80 110 L 110 90 L 150 80 L 190 80 L 220 90 L 240 110 L 250 130 L 250 150 L 240 170 L 220 180 L 190 190 L 150 190 L 110 180 L 80 170 L 60 160 L 50 150 Z',
    model3D: '/track-models/vir.glb',
    characteristics: {
      type: 'Permanent Road Course',
      elevation: '375 ft',
      surface: 'Asphalt'
    },
    gpsCoordinates: {
      finishLine: { lat: 36.5688167, lng: -79.2086639 },
      pitIn: { lat: 36.587581, lng: -79.210428 },
      pitOut: { lat: 36.568867, lng: -79.206797 }
    },
    pitLaneTime: 25 // seconds - notably the fastest pit lane!
  }
];

export const getTrackConfig = (trackId: string): TrackConfig | undefined => {
  return trackConfigs.find(track => track.id === trackId);
};

export const getCurrentSector = (progressPercentage: number, sectors: TrackSector[]): TrackSector => {
  return sectors.find(sector => 
    progressPercentage >= sector.start && progressPercentage < sector.end
  ) || sectors[0];
};