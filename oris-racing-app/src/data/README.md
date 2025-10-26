# ORIS Data Directory

This directory contains the racing datasets and track information for the OLYMPUS Racing Intelligence System.

## Directory Structure

- `/datasets` - Historical race data, telemetry logs, and performance metrics
- `/tracks` - Track maps, sector definitions, and circuit characteristics
- `/telemetry` - Real-time telemetry data samples and schemas

## Data Sources

Data downloaded from: https://trddev.com/hackathon-2025/

## File Formats

- Track maps: SVG/JSON format with sector boundaries
- Telemetry data: CSV/JSON with timestamps
- Historical data: Structured JSON with race results, lap times, weather conditions

## Usage

Import data files directly into components:
```typescript
import trackData from '@/data/tracks/suzuka.json';
import telemetrySchema from '@/data/telemetry/schema.json';
```