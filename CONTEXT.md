# ORIS Racing App Development Context

## Current Task
Building the OLYMPUS Racing Intelligence System (ORIS) - a real-time strategy and analytics platform for Toyota GR racing teams.

## UI Design Approach
- Using NexusController's light-themed design patterns from D:\opt\remote-access-portal\src\components
- Key design elements to implement:
  - Light theme with white backgrounds (#ffffff) and light gray accents
  - Teal primary color (#14b8a6) for active states
  - Card-based dashboard layout with subtle shadows
  - Real-time telemetry displays using Recharts
  - Three-tier layout: status bar, sidebar navigation, main content
  - Status colors: Green (connected), Orange (warning), Red (critical), Blue (info)

## Completed Tasks
1. Created project plan with 12 todos
2. Analyzed PRD requirements for ORIS system
3. Examined NexusController UI patterns

## Next Steps (Todo List)
1. ✅ Set up React/TypeScript project structure for ORIS racing app
2. Create base layout components (Header, Sidebar, MainContent)
3. Implement theme system with NexusController-inspired light theme
4. Build real-time telemetry dashboard with track position view
5. Create lap comparison and timing components
6. Implement strategy advisor panel with AI recommendations
7. Build telemetry graphs for speed, brake, steering data
8. Create predictions panel with confidence scores
9. Implement specialist consensus view for AI insights
10. Add WebSocket support for real-time data streaming
11. Create driver training module interface
12. Implement responsive design for mobile/tablet views

## Key Features to Implement (from PRD)
- Real-time telemetry view (10Hz refresh, track position, sector times, tire degradation)
- Strategy optimizer (pit window calculation, fuel saving modes, tire strategy)
- Competitor analysis (pace comparison, overtaking probability)
- Predictive capabilities (tire cliff prediction, safety car probability)
- Driver training module (corner analysis, braking optimization)
- OLYMPUS ensemble specialists (MINERVA, ATLAS, IRIS, CHRONOS, PROMETHEUS)

## Technical Stack
- React with TypeScript
- Vite for build tooling
- Recharts for data visualization
- WebSocket for real-time data
- CSS Modules for styling
- React Router for navigation

## UI Layout Structure
```
┌─────────────────────────────────────────────────────────────┐
│ ORIS - OLYMPUS Racing Intelligence System         [LIVE] 🔴 │
├─────────────────┬───────────────────┬───────────────────────┤
│ TRACK POSITION  │ LAP COMPARISON    │ STRATEGY ADVISOR      │
│ [3D Track View] │ Current: 1:23.456 │ ⚡ PIT WINDOW: 5-8    │
│ P3 ↑1          │ Best:    1:23.123 │ 🛞 Tire Life: 73%     │
│                 │ Delta:   +0.333   │ ⛽ Fuel to End: YES   │
├─────────────────┼───────────────────┼───────────────────────┤
│ TELEMETRY       │ PREDICTIONS       │ SPECIALIST CONSENSUS  │
│ [Live Graphs]   │ P2 by Lap 25: 78% │ MINERVA:  PIT LAP 42 │
│ Speed ████████  │ Tire Cliff: ~8L   │ ATLAS:    T4 WIDER    │
│ Brake ██        │ Next SC: Low      │ IRIS:     TEMPS OK    │
│ Steer ████      │ Rain Risk: 12%    │ CHRONOS:  PUSH NOW    │
│                 │                   │ PROMETHEUS: UNDERCUT  │
└─────────────────┴───────────────────┴───────────────────────┘
```

## Continue Development
When starting the new chat session in D:\opt\ToyotaGR:
1. The React/TypeScript project is already initialized
2. Dependencies should be installed
3. Start with creating the base layout components
4. Follow the NexusController design patterns
5. Implement the dashboard layout as shown above