#!/usr/bin/env python3
"""
Generate consensus track from multiple GPS recordings.

This script processes multiple GPS recordings for each "pasada" (pass) and generates
a consensus track by calculating the mean position at each time point, filtering
outliers, and computing quality metrics.
"""

import os
import csv
import glob
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, NamedTuple, Optional, Tuple
import warnings

import numpy as np
import gpxpy
import gpxpy.gpx
from tqdm import tqdm


class Position(NamedTuple):
    """Represents a GPS position with coordinates and elevation."""
    latitude: float
    longitude: float
    elevation: Optional[float]
    timestamp: datetime


class ConsensusPoint(NamedTuple):
    """Represents a consensus point with statistics."""
    position: Position
    lat_std: float
    lon_std: float
    alt_std: float
    quality_score: float
    num_points: int


class TrackProcessor:
    """Handles GPS track processing and consensus generation."""
    
    def __init__(self, base_data_path: str = "data"):
        """Initialize the track processor.
        
        Args:
            base_data_path: Base path to the data directory
        """
        self.base_data_path = Path(base_data_path)
        self.raw_path = self.base_data_path / "raw"
        self.preprocessed_path = self.base_data_path / "preprocessed"
        
    def validate_paths(self) -> bool:
        """Validate that required paths exist.
        
        Returns:
            True if all paths exist, False otherwise
        """
        if not self.raw_path.exists():
            print(f"Error: Raw data path not found: {self.raw_path}")
            return False
        if not self.preprocessed_path.exists():
            print(f"Error: Preprocessed data path not found: {self.preprocessed_path}")
            return False
        return True
    
    def load_gpx_file(self, file_path: Path) -> Optional[List[Position]]:
        """Load GPX file and extract positions.
        
        Args:
            file_path: Path to the GPX file
            
        Returns:
            List of positions or None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            positions = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        # Convert to UTC if timezone info is missing
                        timestamp = point.time
                        if timestamp and timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                        elif timestamp is None:
                            continue  # Skip points without timestamp
                            
                        pos = Position(
                            latitude=point.latitude,
                            longitude=point.longitude,
                            elevation=point.elevation,
                            timestamp=timestamp
                        )
                        positions.append(pos)
            
            return positions if positions else None
            
        except Exception as e:
            print(f"Error loading GPX file {file_path}: {e}")
            return None
    
    def find_recording_files(self, pasada: str) -> List[Path]:
        """Find all recording files for a given pasada.
        
        Args:
            pasada: The pasada identifier
            
        Returns:
            List of paths to recording files
        """
        preprocessed_dir = self.preprocessed_path / pasada
        if not preprocessed_dir.exists():
            return []
        
        # Find all GPX files that are not pattern files and not consensus files
        gpx_files = list(preprocessed_dir.glob("*.gpx"))
        recording_files = [f for f in gpx_files if not f.name.endswith("_pattern.gpx") 
                          and not f.name.endswith("_consensus.gpx")
                          and "pattern" not in f.name.lower()]
        
        return recording_files
    
    def find_pattern_file(self, pasada: str) -> Optional[Path]:
        """Find the pattern file for a given pasada.
        
        Args:
            pasada: The pasada identifier
            
        Returns:
            Path to pattern file or None if not found
        """
        raw_dir = self.raw_path / pasada
        if not raw_dir.exists():
            return None
        
        pattern_files = list(raw_dir.glob("*_pattern.gpx"))
        return pattern_files[0] if pattern_files else None
    
    def get_time_bounds(self, recordings: Dict[str, List[Position]]) -> Tuple[datetime, datetime]:
        """Get the time bounds for all recordings.
        
        Args:
            recordings: Dictionary of recording name to positions
            
        Returns:
            Tuple of (start_time, end_time)
        """
        all_times = []
        for positions in recordings.values():
            if positions:
                times = [pos.timestamp for pos in positions if pos.timestamp]
                all_times.extend(times)
        
        if not all_times:
            raise ValueError("No valid timestamps found in recordings")
        
        return min(all_times), max(all_times)
    
    def filter_outliers(self, positions: List[Position], z_threshold: float = 2.0) -> List[Position]:
        """Filter outliers using Z-score method.
        
        Args:
            positions: List of positions to filter
            z_threshold: Z-score threshold for outlier detection
            
        Returns:
            Filtered list of positions
        """
        if len(positions) < 3:
            return positions
        
        # Extract coordinates
        lats = np.array([pos.latitude for pos in positions])
        lons = np.array([pos.longitude for pos in positions])
        
        # Calculate Z-scores
        lat_mean, lat_std = np.mean(lats), np.std(lats)
        lon_mean, lon_std = np.mean(lons), np.std(lons)
        
        if lat_std == 0 or lon_std == 0:
            return positions
        
        lat_z_scores = np.abs((lats - lat_mean) / lat_std)
        lon_z_scores = np.abs((lons - lon_mean) / lon_std)
        
        # Keep positions within threshold
        valid_indices = (lat_z_scores < z_threshold) & (lon_z_scores < z_threshold)
        
        return [pos for i, pos in enumerate(positions) if valid_indices[i]]
    
    def calculate_consensus_point(self, positions: List[Position], timestamp: datetime) -> ConsensusPoint:
        """Calculate consensus point from multiple positions.
        
        Args:
            positions: List of positions at the same time
            timestamp: The timestamp for this consensus point
            
        Returns:
            Consensus point with statistics
        """
        if not positions:
            raise ValueError("No positions provided for consensus calculation")
        
        # Filter outliers
        filtered_positions = self.filter_outliers(positions)
        
        if not filtered_positions:
            # If all points are outliers, use original positions
            filtered_positions = positions
        
        # Calculate statistics
        lats = np.array([pos.latitude for pos in filtered_positions])
        lons = np.array([pos.longitude for pos in filtered_positions])
        alts = np.array([pos.elevation for pos in filtered_positions if pos.elevation is not None])
        
        # Mean position
        mean_lat = np.mean(lats)
        mean_lon = np.mean(lons)
        mean_alt = np.mean(alts) if len(alts) > 0 else None
        
        # Standard deviations
        lat_std = np.std(lats) if len(lats) > 1 else 0.0
        lon_std = np.std(lons) if len(lons) > 1 else 0.0
        alt_std = np.std(alts) if len(alts) > 1 else 0.0
        
        # Convert position uncertainty to meters
        # At latitude, 1 degree longitude = cos(lat) * 111,319.9 meters
        # 1 degree latitude = 111,319.9 meters (approximately constant)
        lat_rad = np.radians(mean_lat)
        meters_per_deg_lat = 111319.9
        meters_per_deg_lon = 111319.9 * np.cos(lat_rad)
        
        # Convert standard deviations to meters
        lat_std_meters = lat_std * meters_per_deg_lat
        lon_std_meters = lon_std * meters_per_deg_lon
        
        # Position uncertainty in meters (2D distance)
        position_uncertainty_meters = np.sqrt(lat_std_meters**2 + lon_std_meters**2)
        
        # Quality score: exponential decay based on uncertainty in meters
        # Score = exp(-uncertainty/characteristic_scale)
        # characteristic_scale = 10 meters (good GPS accuracy threshold)
        characteristic_scale = 10.0  # meters
        quality_score = np.exp(-position_uncertainty_meters / characteristic_scale)
        
        consensus_pos = Position(
            latitude=mean_lat,
            longitude=mean_lon,
            elevation=mean_alt,
            timestamp=timestamp
        )
        
        return ConsensusPoint(
            position=consensus_pos,
            lat_std=lat_std,
            lon_std=lon_std,
            alt_std=alt_std,
            quality_score=quality_score,
            num_points=len(filtered_positions)
        )
    
    def get_positions_at_time(self, recordings: Dict[str, List[Position]], target_time: datetime, 
                             tolerance_seconds: int = 1) -> List[Position]:
        """Get all positions from recordings at a specific time.
        
        Args:
            recordings: Dictionary of recording name to positions
            target_time: Target timestamp
            tolerance_seconds: Tolerance in seconds for time matching
            
        Returns:
            List of positions at the target time
        """
        positions_at_time = []
        
        for recording_name, positions in recordings.items():
            for pos in positions:
                if pos.timestamp:
                    time_diff = abs((pos.timestamp - target_time).total_seconds())
                    if time_diff <= tolerance_seconds:
                        positions_at_time.append(pos)
                        break  # Take the first match for this recording
        
        return positions_at_time
    
    def find_common_time_base(self, recordings: Dict[str, List[Position]]) -> Tuple[datetime, datetime, int]:
        """Find the common time base for all recordings.
        
        Args:
            recordings: Dictionary of recording name to positions
            
        Returns:
            Tuple of (common_start_time, common_end_time, total_seconds)
        """
        # Get start and end times for each recording
        recording_bounds = {}
        for name, positions in recordings.items():
            if positions:
                times = [pos.timestamp for pos in positions if pos.timestamp]
                if times:
                    recording_bounds[name] = (min(times), max(times))
        
        if not recording_bounds:
            raise ValueError("No valid timestamps found in recordings")
        
        # Find the latest start time (when all recordings have started)
        common_start = max(start for start, _ in recording_bounds.values())
        
        # Find the lastest end time (when last recording ends)
        common_end = max(end for _, end in recording_bounds.values())
        
        if common_start >= common_end:
            raise ValueError("No overlapping time period found between recordings")
        
        total_seconds = int((common_end - common_start).total_seconds())
        
        print(f"Common time base: {common_start} to {common_end} ({total_seconds} seconds)")
        
        return common_start, common_end, total_seconds

    def create_time_indexed_recordings(self, recordings: Dict[str, List[Position]], 
                                     common_start: datetime) -> Dict[str, Dict[int, Position]]:
        """Create time-indexed recordings for O(1) lookup.
        
        Args:
            recordings: Dictionary of recording name to positions
            common_start: Common start time to use as time base
            
        Returns:
            Dictionary of recording name to time-indexed positions
        """
        indexed_recordings = {}
        
        for name, positions in recordings.items():
            time_index = {}
            for pos in positions:
                if pos.timestamp:
                    # Calculate seconds offset from common start time
                    seconds_offset = int((pos.timestamp - common_start).total_seconds())
                    if seconds_offset >= 0:  # Only include positions after common start
                        time_index[seconds_offset] = pos
            
            indexed_recordings[name] = time_index
            print(f"Indexed {name}: {len(time_index)} positions")
        
        return indexed_recordings

    def get_positions_at_second(self, indexed_recordings: Dict[str, Dict[int, Position]], 
                              second_offset: int) -> List[Position]:
        """Get all positions at a specific second offset (O(1) lookup per recording).
        
        Args:
            indexed_recordings: Time-indexed recordings
            second_offset: Second offset from common start time
            
        Returns:
            List of positions at the specified second
        """
        positions_at_time = []
        
        for recording_name, time_index in indexed_recordings.items():
            if second_offset in time_index:
                positions_at_time.append(time_index[second_offset])
        
        return positions_at_time
    
    def generate_consensus_track(self, pasada: str) -> bool:
        """Generate consensus track for a given pasada.
        
        Args:
            pasada: The pasada identifier
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\nProcessing pasada: {pasada}")
        
        # Find recording files
        recording_files = self.find_recording_files(pasada)
        if not recording_files:
            print(f"No recording files found for pasada {pasada}")
            return False
        
        print(f"Found {len(recording_files)} recording files")
        
        # Load all recordings
        recordings = {}
        for file_path in recording_files:
            positions = self.load_gpx_file(file_path)
            if positions:
                recordings[file_path.stem] = positions
                print(f"Loaded {len(positions)} points from {file_path.name}")
            else:
                print(f"Failed to load {file_path.name}")
        
        if not recordings:
            print(f"No valid recordings loaded for pasada {pasada}")
            return False
        
        # Get time bounds
        try:
            start_time, end_time = self.get_time_bounds(recordings)
        except ValueError as e:
            print(f"Error getting time bounds: {e}")
            return False
        
        print(f"Time range: {start_time} to {end_time}")
        
        # Find common time base
        try:
            common_start, common_end, total_seconds = self.find_common_time_base(recordings)
        except ValueError as e:
            print(f"Error finding common time base: {e}")
            return False
        
        # Create time-indexed recordings
        indexed_recordings = self.create_time_indexed_recordings(recordings, common_start)
        
        # Generate consensus points at 1Hz
        consensus_points = []
        current_time = common_start
        total_seconds = int((common_end - common_start).total_seconds())
        
        print("Generating consensus points...")
        for second in tqdm(range(total_seconds + 1)):
            positions_at_time = self.get_positions_at_second(indexed_recordings, second)
            
            if len(positions_at_time) >= 2:  # Need at least 2 points for consensus
                try:
                    consensus_point = self.calculate_consensus_point(positions_at_time, current_time)
                    consensus_points.append(consensus_point)
                except Exception as e:
                    print(f"Error calculating consensus at {current_time}: {e}")
            
            current_time = common_start + timedelta(seconds=second + 1)
        
        if not consensus_points:
            print(f"No consensus points generated for pasada {pasada}")
            return False
        
        print(f"Generated {len(consensus_points)} consensus points")
        
        # Save results
        output_dir = self.preprocessed_path / pasada
        output_dir.mkdir(exist_ok=True)
        
        # Save GPX file
        gpx_path = output_dir / f"{pasada}_consensus.gpx"
        self.save_consensus_gpx(consensus_points, gpx_path)
        
        # Save CSV file
        csv_path = output_dir / f"{pasada}_consensus.csv"
        self.save_consensus_csv(consensus_points, csv_path)
        
        print(f"Saved consensus track to {gpx_path}")
        print(f"Saved consensus data to {csv_path}")
        
        return True
    
    def save_consensus_gpx(self, consensus_points: List[ConsensusPoint], output_path: Path):
        """Save consensus points to GPX file.
        
        Args:
            consensus_points: List of consensus points
            output_path: Output file path
        """
        gpx = gpxpy.gpx.GPX()
        gpx.creator = "generate_consensus_track.py"
        
        # Create track
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx_track.name = "Consensus Track"
        gpx.tracks.append(gpx_track)
        
        # Create segment
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)
        
        # Add points
        for point in consensus_points:
            gpx_point = gpxpy.gpx.GPXTrackPoint(
                latitude=point.position.latitude,
                longitude=point.position.longitude,
                elevation=point.position.elevation,
                time=point.position.timestamp
            )
            gpx_segment.points.append(gpx_point)
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(gpx.to_xml())
    
    def save_consensus_csv(self, consensus_points: List[ConsensusPoint], output_path: Path):
        """Save consensus points to CSV file.
        
        Args:
            consensus_points: List of consensus points
            output_path: Output file path
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            header = [
                'timestamp', 'latitude', 'longitude', 'elevation',
                'lat_std', 'lon_std', 'alt_std', 'quality_score', 'num_points'
            ]
            writer.writerow(header)
            
            # Write data
            for point in consensus_points:
                row = [
                    point.position.timestamp.isoformat(),
                    point.position.latitude,
                    point.position.longitude,
                    point.position.elevation,
                    point.lat_std,
                    point.lon_std,
                    point.alt_std,
                    point.quality_score,
                    point.num_points
                ]
                writer.writerow(row)
    
    def find_all_pasadas(self) -> List[str]:
        """Find all available pasadas.
        
        Returns:
            List of pasada identifiers
        """
        pasadas = []
        
        if self.preprocessed_path.exists():
            for item in self.preprocessed_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Check if there are recording files
                    recording_files = self.find_recording_files(item.name)
                    if recording_files:
                        pasadas.append(item.name)
        
        return sorted(pasadas)
    
    def process_all_pasadas(self):
        """Process all available pasadas."""
        if not self.validate_paths():
            return
        
        pasadas = self.find_all_pasadas()
        if not pasadas:
            print("No pasadas found with recording files")
            return
        
        print(f"Found {len(pasadas)} pasadas to process: {pasadas}")
        
        successful = 0
        failed = 0
        
        for pasada in pasadas:
            try:
                if self.generate_consensus_track(pasada):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing pasada {pasada}: {e}")
                failed += 1
        
        print(f"\n=== SUMMARY ===")
        print(f"Successfully processed: {successful} pasadas")
        print(f"Failed: {failed} pasadas")
        print(f"Total: {len(pasadas)} pasadas")


def main():
    """Main function."""
    print("GPS Consensus Track Generator")
    print("============================")
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Initialize processor
    processor = TrackProcessor()
    
    # Process all pasadas
    processor.process_all_pasadas()


if __name__ == "__main__":
    main()