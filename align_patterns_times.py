#!/usr/bin/env python3
"""
Align pattern tracks with consensus timestamps.

This script aligns pattern tracks (spatially accurate) with consensus tracks 
(temporally accurate) to create time-stamped pattern tracks for each pasada.
"""

import os
import csv
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, NamedTuple, Optional, Tuple
from dataclasses import dataclass
import math
import warnings

import numpy as np
import pandas as pd
import gpxpy
import gpxpy.gpx
from tqdm import tqdm


@dataclass
class ConsensusPoint:
    """Represents a point from the consensus track."""
    timestamp: datetime
    latitude: float
    longitude: float
    elevation: Optional[float]
    lat_std: float
    lon_std: float
    alt_std: float
    quality_score: float
    num_points: int


@dataclass
class PatternPoint:
    """Represents a point from the pattern track."""
    latitude: float
    longitude: float
    elevation: Optional[float]
    distance_from_start: float  # Cumulative distance along pattern


@dataclass
class AlignedPoint:
    """Represents an aligned pattern point with timestamp."""
    latitude: float
    longitude: float
    elevation: Optional[float]
    timestamp: datetime
    quality_score: float
    assignment_method: str  # 'direct', 'interpolated', 'extrapolated'
    distance_to_consensus: Optional[float]  # Distance to closest consensus point


@dataclass
class InitialMatch:
    """Represents the initial synchronization match between pattern and consensus."""
    pattern_idx: int
    consensus_idx: int
    distance: float


@dataclass
class SearchWindow:
    """Represents a search window for consensus points."""
    start_idx: int
    end_idx: int
    

@dataclass
class SlidingWindow:
    """Represents a sliding window for efficient searching."""
    start_index: int
    end_index: int
    center_distance: float  # Distance from start of track


class PatternAligner:
    """Handles alignment of pattern tracks with consensus timestamps."""
    
    def __init__(self, base_data_path: str = "data"):
        """Initialize the pattern aligner.
        
        Args:
            base_data_path: Base path to the data directory
        """
        self.base_data_path = Path(base_data_path)
        self.raw_path = self.base_data_path / "raw"
        self.preprocessed_path = self.base_data_path / "preprocessed"
        
        # Algorithm parameters
        self.z_threshold = 2.0  # Z-score threshold
        self.quality_threshold = 0.35  # Minimum quality score
        
    def validate_paths(self) -> bool:
        """Validate that required paths exist."""
        if not self.raw_path.exists():
            print(f"Error: Raw data path not found: {self.raw_path}")
            return False
        if not self.preprocessed_path.exists():
            print(f"Error: Preprocessed data path not found: {self.preprocessed_path}")
            return False
        return True
    
    def haversine_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate the Haversine distance between two points in meters."""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def calculate_cumulative_distances(self, points: List[PatternPoint]) -> List[float]:
        """Calculate cumulative distances along the pattern track."""
        if not points:
            return []
        
        distances = [0.0]
        for i in range(1, len(points)):
            dist = self.haversine_distance(
                points[i-1].latitude, points[i-1].longitude,
                points[i].latitude, points[i].longitude
            )
            distances.append(distances[-1] + dist)
        
        # Update the distance_from_start in the points
        for i, point in enumerate(points):
            point.distance_from_start = distances[i]
        
        return distances
    
    def load_consensus_csv(self, file_path: Path) -> List[ConsensusPoint]:
        """Load consensus data from CSV file."""
        consensus_points = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(row['timestamp'])
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    
                    point = ConsensusPoint(
                        timestamp=timestamp,
                        latitude=float(row['latitude']),
                        longitude=float(row['longitude']),
                        elevation=float(row['elevation']) if row['elevation'] else None,
                        lat_std=float(row['lat_std']),
                        lon_std=float(row['lon_std']),
                        alt_std=float(row['alt_std']),
                        quality_score=float(row['quality_score']),
                        num_points=int(row['num_points'])
                    )
                    consensus_points.append(point)
            
            return consensus_points
            
        except Exception as e:
            print(f"Error loading consensus CSV {file_path}: {e}")
            return []
    
    def load_pattern_gpx(self, file_path: Path) -> List[PatternPoint]:
        """Load pattern track from GPX file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        pattern_point = PatternPoint(
                            latitude=point.latitude,
                            longitude=point.longitude,
                            elevation=point.elevation,
                            distance_from_start=0.0  # Will be calculated later
                        )
                        points.append(pattern_point)
            
            # Calculate cumulative distances
            if points:
                self.calculate_cumulative_distances(points)
            
            return points
            
        except Exception as e:
            print(f"Error loading pattern GPX {file_path}: {e}")
            return []
    
    def find_closest_consensus_point(self, pattern_point: PatternPoint,
                                   consensus_points: List[ConsensusPoint],
                                   consensus_windows: List[SlidingWindow],
                                   consensus_distances: List[float],
                                   pattern_distance: float) -> Optional[Tuple[int, float]]:
        """Find the closest consensus point to a pattern point within a window."""
        # Find the consensus window that best matches the pattern distance
        best_window_idx = 0
        min_distance_diff = float('inf')
        
        for i, distance in enumerate(consensus_distances):
            distance_diff = abs(distance - pattern_distance)
            if distance_diff < min_distance_diff:
                min_distance_diff = distance_diff
                best_window_idx = i
        
        # Search within the selected window
        if best_window_idx >= len(consensus_windows):
            return None
        
        window = consensus_windows[best_window_idx]
        best_consensus_idx = None
        min_distance = float('inf')
        best_quality = 0.0
        
        for i in range(window.start_index, min(window.end_index + 1, len(consensus_points))):
            consensus_point = consensus_points[i]
            
            # Calculate distance
            distance = self.haversine_distance(
                pattern_point.latitude, pattern_point.longitude,
                consensus_point.latitude, consensus_point.longitude
            )
            
            # Check if this is a better match
            is_better = False
            if distance < min_distance:
                is_better = True
            elif abs(distance - min_distance) < 1.0: # Very similar distances
                if consensus_point.quality_score > best_quality:
                    is_better = True
            
            if is_better:
                min_distance = distance
                best_consensus_idx = i
                best_quality = consensus_point.quality_score
        
        if best_consensus_idx is not None:
            return best_consensus_idx, min_distance
        return None
    
    def filter_by_quality_and_distance(self, matches: List[Tuple[int, int, float]],
                                     consensus_points: List[ConsensusPoint]) -> List[Tuple[int, int, float]]:
        """Filter matches by quality score and distance using Z-score."""
        if len(matches) < 3:  # Need at least 3 points for Z-score
            return [m for m in matches if consensus_points[m[1]].quality_score >= self.quality_threshold]
        
        # Extract distances and calculate Z-scores
        distances = np.array([m[2] for m in matches])
        distance_mean = np.mean(distances)
        distance_std = np.std(distances)
        
        filtered_matches = []
        for match in matches:
            pattern_idx, consensus_idx, distance = match
            consensus_point = consensus_points[consensus_idx]
            
            # Check quality score
            if consensus_point.quality_score < self.quality_threshold:
                continue
            
            # Check distance Z-score
            if distance_std > 0:
                z_score = abs((distance - distance_mean) / distance_std)
                if z_score > self.z_threshold:
                    continue
            
            filtered_matches.append(match)
        
        return filtered_matches
    
    def interpolate_timestamps(self, pattern_points: List[PatternPoint],
                             matches: List[Tuple[int, int, float]],
                             consensus_points: List[ConsensusPoint]) -> List[AlignedPoint]:
        """Interpolate timestamps for all pattern points."""
        aligned_points = [None] * len(pattern_points)
        
        # First, assign direct matches
        for pattern_idx, consensus_idx, distance in matches:
            consensus_point = consensus_points[consensus_idx]
            aligned_points[pattern_idx] = AlignedPoint(
                latitude=pattern_points[pattern_idx].latitude,
                longitude=pattern_points[pattern_idx].longitude,
                elevation=pattern_points[pattern_idx].elevation,
                timestamp=consensus_point.timestamp,
                quality_score=consensus_point.quality_score,
                assignment_method='direct',
                distance_to_consensus=distance
            )
        
        # Now interpolate/extrapolate for missing points
        for i in range(len(pattern_points)):
            if aligned_points[i] is not None:
                continue  # Already assigned
            
            pattern_point = pattern_points[i]
            
            # Find nearest assigned points
            left_idx = None
            right_idx = None
            
            # Look for assigned point to the left
            for j in range(i - 1, -1, -1):
                if aligned_points[j] is not None:
                    left_idx = j
                    break
            
            # Look for assigned point to the right
            for j in range(i + 1, len(aligned_points)):
                if aligned_points[j] is not None:
                    right_idx = j
                    break
            
            # Determine interpolation/extrapolation method
            if left_idx is not None and right_idx is not None:
                # Interpolation
                timestamp = self.interpolate_between_points(
                    pattern_points, aligned_points, i, left_idx, right_idx
                )
                method = 'interpolated'
                quality = min(aligned_points[left_idx].quality_score, 
                            aligned_points[right_idx].quality_score)
            
            elif left_idx is not None:
                # Extrapolation to the right
                timestamp = self.extrapolate_from_point(
                    pattern_points, aligned_points, i, left_idx, forward=True
                )
                method = 'extrapolated'
                quality = aligned_points[left_idx].quality_score * 0.5
            
            elif right_idx is not None:
                # Extrapolation to the left
                timestamp = self.extrapolate_from_point(
                    pattern_points, aligned_points, i, right_idx, forward=False
                )
                method = 'extrapolated'
                quality = aligned_points[right_idx].quality_score * 0.5
            
            else:
                # No reference points available
                timestamp = datetime.now(timezone.utc)
                method = 'default'
                quality = 0.0
            
            aligned_points[i] = AlignedPoint(
                latitude=pattern_point.latitude,
                longitude=pattern_point.longitude,
                elevation=pattern_point.elevation,
                timestamp=timestamp,
                quality_score=quality,
                assignment_method=method,
                distance_to_consensus=None
            )
        
        return aligned_points
    
    def interpolate_between_points(self, pattern_points: List[PatternPoint],
                                 aligned_points: List[AlignedPoint],
                                 target_idx: int, left_idx: int, right_idx: int) -> datetime:
        """Interpolate timestamp between two assigned points."""
        # Calculate distance ratios
        left_distance = pattern_points[left_idx].distance_from_start
        target_distance = pattern_points[target_idx].distance_from_start
        right_distance = pattern_points[right_idx].distance_from_start
        
        # Linear interpolation by distance
        total_distance = right_distance - left_distance
        if total_distance == 0:
            ratio = 0.5
        else:
            ratio = (target_distance - left_distance) / total_distance
        
        # Interpolate timestamps
        left_timestamp = aligned_points[left_idx].timestamp
        right_timestamp = aligned_points[right_idx].timestamp
        
        time_diff = (right_timestamp - left_timestamp).total_seconds()
        interpolated_seconds = time_diff * ratio
        
        return left_timestamp + timedelta(seconds=interpolated_seconds)
    
    def extrapolate_from_point(self, pattern_points: List[PatternPoint],
                             aligned_points: List[AlignedPoint],
                             target_idx: int, ref_idx: int, forward: bool) -> datetime:
        """Extrapolate timestamp from a reference point."""
        # Find a second reference point to calculate speed
        second_ref_idx = None
        
        if forward:
            # Look for another assigned point before ref_idx
            for j in range(ref_idx - 1, -1, -1):
                if aligned_points[j] is not None:
                    second_ref_idx = j
                    break
        else:
            # Look for another assigned point after ref_idx
            for j in range(ref_idx + 1, len(aligned_points)):
                if aligned_points[j] is not None:
                    second_ref_idx = j
                    break
        
        # Calculate speed if we have two reference points
        if second_ref_idx is not None:
            distance_diff = abs(pattern_points[ref_idx].distance_from_start - 
                              pattern_points[second_ref_idx].distance_from_start)
            time_diff = abs((aligned_points[ref_idx].timestamp - 
                           aligned_points[second_ref_idx].timestamp).total_seconds())
            
            if time_diff > 0:
                speed = distance_diff / time_diff  # meters per second
            else:
                speed = 1.0  # Default speed: 1 m/s
        else:
            speed = 1.0  # Default speed: 1 m/s
        
        # Calculate distance from reference to target
        distance_to_target = abs(pattern_points[target_idx].distance_from_start - 
                               pattern_points[ref_idx].distance_from_start)
        
        # Calculate time offset
        time_offset = distance_to_target / speed
        
        # Apply time offset
        ref_timestamp = aligned_points[ref_idx].timestamp
        if forward:
            return ref_timestamp + timedelta(seconds=time_offset)
        else:
            return ref_timestamp - timedelta(seconds=time_offset)
    
    def save_aligned_pattern(self, aligned_points: List[AlignedPoint], 
                           output_path: Path, pasada: str):
        """Save aligned pattern to GPX file."""
        gpx = gpxpy.gpx.GPX()
        gpx.creator = "align_patterns_times.py"
        
        # Create track
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx_track.name = f"Aligned Pattern Track - Pasada {pasada}"
        gpx.tracks.append(gpx_track)
        
        # Create segment
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)
        
        # Add points
        for point in aligned_points:
            gpx_point = gpxpy.gpx.GPXTrackPoint(
                latitude=point.latitude,
                longitude=point.longitude,
                elevation=point.elevation,
                time=point.timestamp
            )
            gpx_segment.points.append(gpx_point)
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(gpx.to_xml())
    
    def align_pattern_for_pasada(self, pasada: str) -> bool:
        """Align pattern track with consensus timestamps for a single pasada."""
        print(f"\nAligning pattern for pasada: {pasada}")
        
        # Define file paths
        consensus_csv_path = self.preprocessed_path / pasada / f"{pasada}_consensus.csv"
        pattern_gpx_path = self.raw_path / pasada / f"{pasada}_pattern.gpx"
        output_gpx_path = self.preprocessed_path / pasada / f"{pasada}_aligned_pattern.gpx"
        
        # Check if input files exist
        if not consensus_csv_path.exists():
            print(f"Consensus CSV not found: {consensus_csv_path}")
            return False
        
        if not pattern_gpx_path.exists():
            print(f"Pattern GPX not found: {pattern_gpx_path}")
            return False
        
        # Load data
        print("Loading consensus data...")
        consensus_points = self.load_consensus_csv(consensus_csv_path)
        if not consensus_points:
            print("Failed to load consensus data")
            return False
        
        print("Loading pattern data...")
        pattern_points = self.load_pattern_gpx(pattern_gpx_path)
        if not pattern_points:
            print("Failed to load pattern data")
            return False
        
        print(f"Loaded {len(consensus_points)} consensus points and {len(pattern_points)} pattern points")
        
        # Use improved algorithm for finding matches
        print("Finding matches with improved algorithm...")
        matches = self.find_matches_with_improved_algorithm(pattern_points, consensus_points)
        
        if not matches:
            print("No matches found with improved algorithm")
            return False
        
        print(f"Found {len(matches)} initial matches")
        print(f"Average distance: {np.mean([m[2] for m in matches]):.2f}m")
        
        # Remove duplicate consensus point matches
        print("Removing duplicate consensus matches...")
        matches = self.remove_duplicate_consensus_matches(matches)
        print(f"After removing duplicates: {len(matches)} matches")
        
        # Filter matches by quality and distance
        print("Filtering matches by quality and distance...")
        filtered_matches = self.filter_by_quality_and_distance(matches, consensus_points)
        print(f"Retained {len(filtered_matches)} matches after filtering")
        
        if not filtered_matches:
            print("No valid matches found after filtering")
            return False
        
        # Interpolate timestamps for all pattern points
        print("Interpolating timestamps...")
        aligned_points = self.interpolate_timestamps(
            pattern_points, filtered_matches, consensus_points
        )
        
        # Count assignment methods
        method_counts = {}
        for point in aligned_points:
            method = point.assignment_method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print(f"Assignment methods: {method_counts}")
        
        # Save results
        print("Saving aligned pattern...")
        output_gpx_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_aligned_pattern(aligned_points, output_gpx_path, pasada)
        
        print(f"Saved aligned pattern to {output_gpx_path}")
        
        return True
    
    def find_all_pasadas(self) -> List[str]:
        """Find all available pasadas with consensus data."""
        pasadas = []
        
        if self.preprocessed_path.exists():
            for item in self.preprocessed_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    consensus_file = item / f"{item.name}_consensus.csv"
                    if consensus_file.exists():
                        pasadas.append(item.name)
        
        return sorted(pasadas)
    
    def process_all_pasadas(self):
        """Process all available pasadas."""
        if not self.validate_paths():
            return
        
        pasadas = self.find_all_pasadas()
        if not pasadas:
            print("No pasadas found with consensus data")
            return
        
        print(f"Found {len(pasadas)} pasadas to process: {pasadas}")
        
        successful = 0
        failed = 0
        
        for pasada in pasadas:
            try:
                if self.align_pattern_for_pasada(pasada):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing pasada {pasada}: {e}")
                failed += 1
        
        print(f"\n=== ALIGNMENT SUMMARY ===")
        print(f"Successfully processed: {successful} pasadas")
        print(f"Failed: {failed} pasadas")
        print(f"Total: {len(pasadas)} pasadas")
    
    def find_initial_synchronization(self, pattern_points: List[PatternPoint], 
                                   consensus_points: List[ConsensusPoint]) -> InitialMatch:
        """Find the best initial synchronization between pattern and consensus tracks."""
        print("Finding initial synchronization...")
        
        MAX_SEARCH_POINTS = 50  # Limit search to first 50 points
        
        # Method 1: Start from first pattern point, find closest consensus point within first 50 consensus points
        first_pattern = pattern_points[0]
        min_distance_1 = float('inf')
        best_consensus_idx_1 = 0
        
        search_limit_1 = min(MAX_SEARCH_POINTS, len(consensus_points))
        for i in range(search_limit_1):
            consensus_point = consensus_points[i]
            distance = self.haversine_distance(
                first_pattern.latitude, first_pattern.longitude,
                consensus_point.latitude, consensus_point.longitude
            )
            if distance < min_distance_1:
                min_distance_1 = distance
                best_consensus_idx_1 = i
        
        match_1 = InitialMatch(
            pattern_idx=0,
            consensus_idx=best_consensus_idx_1,
            distance=min_distance_1
        )
        
        # Method 2: Start from first consensus point, find closest pattern point within first 50 pattern points
        first_consensus = consensus_points[0]
        min_distance_2 = float('inf')
        best_pattern_idx_2 = 0
        
        search_limit_2 = min(MAX_SEARCH_POINTS, len(pattern_points))
        for i in range(search_limit_2):
            pattern_point = pattern_points[i]
            distance = self.haversine_distance(
                first_consensus.latitude, first_consensus.longitude,
                pattern_point.latitude, pattern_point.longitude
            )
            if distance < min_distance_2:
                min_distance_2 = distance
                best_pattern_idx_2 = i
        
        match_2 = InitialMatch(
            pattern_idx=best_pattern_idx_2,
            consensus_idx=0,
            distance=min_distance_2
        )
        
        # Choose the best match (smallest distance)
        if match_1.distance <= match_2.distance:
            best_match = match_1
        else:
            best_match = match_2
        
        print(f"Initial synchronization: pattern[{best_match.pattern_idx}] <-> consensus[{best_match.consensus_idx}], distance: {best_match.distance:.2f}m")
        
        return best_match
    
    def calculate_consensus_distances(self, consensus_points: List[ConsensusPoint]) -> List[float]:
        """Calculate cumulative distances along the consensus track."""
        if not consensus_points:
            return []
        
        distances = [0.0]
        for i in range(1, len(consensus_points)):
            dist = self.haversine_distance(
                consensus_points[i-1].latitude, consensus_points[i-1].longitude,
                consensus_points[i].latitude, consensus_points[i].longitude
            )
            distances.append(distances[-1] + dist)
        
        return distances
    
    def get_search_window(self, consensus_points: List[ConsensusPoint],
                         consensus_distances: List[float],
                         last_consensus_idx: int,
                         pattern_distance_delta: float) -> SearchWindow:
        """Get search window for consensus points based on pattern distance delta."""
        # Minimum window size of 10 meters
        window_distance = max(10.0, 1.5 * pattern_distance_delta)
        
        start_idx = last_consensus_idx
        end_idx = len(consensus_points) - 1
        
        # Find end of window based on curvilinear distance
        start_distance = consensus_distances[last_consensus_idx]
        target_distance = start_distance + window_distance
        
        for i in range(last_consensus_idx, len(consensus_distances)):
            if consensus_distances[i] >= target_distance:
                end_idx = i
                break
        
        return SearchWindow(start_idx=start_idx, end_idx=end_idx)
    
    def find_matches_with_improved_algorithm(self, pattern_points: List[PatternPoint],
                                           consensus_points: List[ConsensusPoint]) -> List[Tuple[int, int, float]]:
        """Find matches using the improved synchronization and search algorithm."""
        # Calculate consensus distances
        consensus_distances = self.calculate_consensus_distances(consensus_points)
        
        # Find initial synchronization
        initial_match = self.find_initial_synchronization(pattern_points, consensus_points)
        
        matches = []
        last_consensus_idx = initial_match.consensus_idx
        
        # Add the initial match
        matches.append((initial_match.pattern_idx, initial_match.consensus_idx, initial_match.distance))
        
        # Process remaining pattern points starting from the synchronized position
        for i in range(initial_match.pattern_idx + 1, len(pattern_points)):
            # Calculate distance delta from previous pattern point
            prev_pattern_point = pattern_points[i - 1]
            current_pattern_point = pattern_points[i]
            
            pattern_distance_delta = self.haversine_distance(
                prev_pattern_point.latitude, prev_pattern_point.longitude,
                current_pattern_point.latitude, current_pattern_point.longitude
            )
            
            # Get search window
            window = self.get_search_window(
                consensus_points, consensus_distances, 
                last_consensus_idx, pattern_distance_delta
            )
            
            # Find closest consensus point within the window
            best_consensus_idx = None
            min_distance = float('inf')
            best_quality = 0.0
            
            for j in range(window.start_idx, min(window.end_idx + 1, len(consensus_points))):
                consensus_point = consensus_points[j]
                
                # Calculate distance
                distance = self.haversine_distance(
                    current_pattern_point.latitude, current_pattern_point.longitude,
                    consensus_point.latitude, consensus_point.longitude
                )
                
                # Check if this is a better match (prioritize distance, then quality)
                is_better = False
                if distance < min_distance:
                    is_better = True
                elif abs(distance - min_distance) < 1.0:  # Very similar distances
                    if consensus_point.quality_score > best_quality:
                        is_better = True
                
                if is_better:
                    min_distance = distance
                    best_consensus_idx = j
                    best_quality = consensus_point.quality_score
            
            if best_consensus_idx is not None:
                matches.append((i, best_consensus_idx, min_distance))
                last_consensus_idx = best_consensus_idx
        
        # Process pattern points before the initial synchronization point
        last_consensus_idx = initial_match.consensus_idx
        
        for i in range(initial_match.pattern_idx - 1, -1, -1):
            # Calculate distance delta from next pattern point
            next_pattern_point = pattern_points[i + 1]
            current_pattern_point = pattern_points[i]
            
            pattern_distance_delta = self.haversine_distance(
                current_pattern_point.latitude, current_pattern_point.longitude,
                next_pattern_point.latitude, next_pattern_point.longitude
            )
            
            # For backward search, we look before the last consensus index
            window_distance = max(10.0, 1.5 * pattern_distance_delta)
            
            start_idx = max(0, last_consensus_idx - int(window_distance / 10))  # Rough estimate
            end_idx = last_consensus_idx
            
            # Find closest consensus point within the window
            best_consensus_idx = None
            min_distance = float('inf')
            best_quality = 0.0
            
            for j in range(start_idx, end_idx + 1):
                if j >= len(consensus_points):
                    continue
                    
                consensus_point = consensus_points[j]
                
                # Calculate distance
                distance = self.haversine_distance(
                    current_pattern_point.latitude, current_pattern_point.longitude,
                    consensus_point.latitude, consensus_point.longitude
                )
                
                # Check if this is a better match
                is_better = False
                if distance < min_distance:
                    is_better = True
                elif abs(distance - min_distance) < 1.0:  # Very similar distances
                    if consensus_point.quality_score > best_quality:
                        is_better = True
                
                if is_better:
                    min_distance = distance
                    best_consensus_idx = j
                    best_quality = consensus_point.quality_score
            
            if best_consensus_idx is not None:
                matches.append((i, best_consensus_idx, min_distance))
                last_consensus_idx = best_consensus_idx
        
        return matches
    
    def remove_duplicate_consensus_matches(self, matches: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Remove duplicate consensus point matches, keeping the one with smallest distance."""
        # Group matches by consensus index
        consensus_groups = {}
        for pattern_idx, consensus_idx, distance in matches:
            if consensus_idx not in consensus_groups:
                consensus_groups[consensus_idx] = []
            consensus_groups[consensus_idx].append((pattern_idx, consensus_idx, distance))
        
        # For each consensus point, keep only the match with smallest distance
        filtered_matches = []
        for consensus_idx, group_matches in consensus_groups.items():
            # Sort by distance and keep the best one
            best_match = min(group_matches, key=lambda x: x[2])
            filtered_matches.append(best_match)
        
        # Sort by pattern index to maintain order
        filtered_matches.sort(key=lambda x: x[0])
        
        return filtered_matches
    


def main():
    """Main function."""
    print("Pattern Track Alignment Tool")
    print("===========================")
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Initialize aligner
    aligner = PatternAligner()
    
    # Process all pasadas
    aligner.process_all_pasadas()


if __name__ == "__main__":
    main()