"""
Advanced Track Re-linking System
Reconnects broken trajectories using spatial-temporal matching
"""

from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from collections import defaultdict

@dataclass
class TrackSegment:
    """Represents a single track segment"""
    track_id: int
    frames: List[int] = field(default_factory=list)
    x_coords: List[float] = field(default_factory=list)
    y_coords: List[float] = field(default_factory=list)
    in_leg_numbers: Set[int] = field(default_factory=set)
    out_leg_numbers: Set[int] = field(default_factory=set)
    
    def first_frame(self) -> int:
        return self.frames[0] if self.frames else -1
    
    def last_frame(self) -> int:
        return self.frames[-1] if self.frames else -1
    
    def first_position(self) -> Tuple[float, float]:
        if self.frames:
            return (self.x_coords[0], self.y_coords[0])
        return (0, 0)
    
    def last_position(self) -> Tuple[float, float]:
        if self.frames:
            return (self.x_coords[-1], self.y_coords[-1])
        return (0, 0)
    
    def to_dataframe(self, new_track_id: int = None) -> pd.DataFrame:
        """Convert segment to DataFrame rows"""
        track_id = new_track_id if new_track_id is not None else self.track_id
        data = {
            'track_id': [track_id] * len(self.frames),
            'frame': self.frames,
            'x': self.x_coords,
            'y': self.y_coords,
            'in_leg': [list(self.in_leg_numbers) if self.in_leg_numbers else [] for _ in self.frames],
            'out_leg': [list(self.out_leg_numbers) if self.out_leg_numbers else [] for _ in self.frames],
        }
        return pd.DataFrame(data)


class TrackRelinkSystem:
    """System to relink broken tracks"""
    
    def __init__(self, trajectories_df: pd.DataFrame, 
                 frame_threshold: int = 10,
                 distance_threshold: float = 100.0):
        """
        Initialize the relinking system
        
        Args:
            trajectories_df: DataFrame with columns [track_id, frame, x, y, in_leg, out_leg]
            frame_threshold: Maximum frame difference for matching
            distance_threshold: Maximum pixel distance for matching
        """
        self.df = trajectories_df.copy()
        self.frame_threshold = frame_threshold
        self.distance_threshold = distance_threshold
        self.new_tracks: List[pd.DataFrame] = []
        self.next_track_id = int(self.df['track_id'].max()) + 1
        
    def _parse_leg_numbers(self, leg_str):
        """Parse leg numbers from string or list"""
        if isinstance(leg_str, str):
            try:
                return set(map(int, leg_str.strip('[]').split(','))) if leg_str != '[]' else set()
            except:
                return set()
        elif isinstance(leg_str, list):
            return set(leg_str)
        return set()
    
    def _segment_tracks(self) -> Tuple[List[TrackSegment], List[TrackSegment], List[TrackSegment]]:
        """
        Phase A: Classify tracks into 3 categories
        """
        out_only = []
        in_only = []
        middle = []
        
        for track_id in self.df['track_id'].unique():
            track_data = self.df[self.df['track_id'] == track_id].sort_values('frame')
            
            # Collect all leg crossings
            all_in_legs = set()
            all_out_legs = set()
            
            for _, row in track_data.iterrows():
                in_legs = self._parse_leg_numbers(row.get('in_leg', '[]'))
                out_legs = self._parse_leg_numbers(row.get('out_leg', '[]'))
                all_in_legs.update(in_legs)
                all_out_legs.update(out_legs)
            
            # Create segment
            segment = TrackSegment(
                track_id=track_id,
                frames=track_data['frame'].tolist(),
                x_coords=track_data['x'].tolist(),
                y_coords=track_data['y'].tolist(),
                in_leg_numbers=all_in_legs,
                out_leg_numbers=all_out_legs
            )
            
            # Classification
            has_in = len(all_in_legs) > 0
            has_out = len(all_out_legs) > 0
            
            if not has_in and has_out:
                out_only.append(segment)
            elif has_in and not has_out:
                in_only.append(segment)
            else:  # neither or both (but we focus on neither)
                if not has_in and not has_out:
                    middle.append(segment)
        
        return out_only, in_only, middle
    
    def _matches(self, seg1_end: Tuple[int, float, float], 
                 seg2_start: Tuple[int, float, float]) -> bool:
        """
        Check if two segments match based on spatial-temporal criteria
        
        seg1_end: (frame, x, y) - last point of first segment
        seg2_start: (frame, x, y) - first point of second segment
        """
        frame1, x1, y1 = seg1_end
        frame2, x2, y2 = seg2_start
        
        frame_diff = abs(frame2 - frame1)
        x_diff = abs(x2 - x1)
        y_diff = abs(y2 - y1)
        
        return (frame_diff <= self.frame_threshold and 
                x_diff <= self.distance_threshold and 
                y_diff <= self.distance_threshold)
    
    def _merge_segments(self, seg1: TrackSegment, seg2: TrackSegment) -> TrackSegment:
        """Merge two segments into one"""
        merged = TrackSegment(
            track_id=self.next_track_id,
            frames=seg1.frames + seg2.frames,
            x_coords=seg1.x_coords + seg2.x_coords,
            y_coords=seg1.y_coords + seg2.y_coords,
            in_leg_numbers=seg1.in_leg_numbers | seg2.in_leg_numbers,
            out_leg_numbers=seg1.out_leg_numbers | seg2.out_leg_numbers
        )
        self.next_track_id += 1
        return merged
    
    def _phase_b_in_to_middle(self, in_only: List[TrackSegment], 
                              middle: List[TrackSegment]) -> Tuple[List[TrackSegment], Set[int]]:
        """
        Phase B: Link In-only to Middle objects
        """
        linked_tracks = []
        used_middle_ids = set()
        
        for in_track in in_only:
            last_frame_in = in_track.last_frame()
            x_in, y_in = in_track.last_position()
            
            for mid_idx, mid_track in enumerate(middle):
                if mid_idx in used_middle_ids:
                    continue
                
                first_frame_mid = mid_track.first_frame()
                x_mid, y_mid = mid_track.first_position()
                
                if self._matches((last_frame_in, x_in, y_in), 
                                (first_frame_mid, x_mid, y_mid)):
                    # Match found - merge
                    merged = self._merge_segments(in_track, mid_track)
                    linked_tracks.append(merged)
                    used_middle_ids.add(mid_idx)
                    break
        
        return linked_tracks, used_middle_ids
    
    def _phase_c_middle_to_out(self, out_only: List[TrackSegment],
                               middle: List[TrackSegment],
                               in_to_mid_tracks: List[TrackSegment],
                               used_middle_ids: Set[int]) -> List[TrackSegment]:
        """
        Phase C: Link Middle (or In→Middle results) to Out-only objects
        """
        linked_tracks = []
        
        for out_track in out_only:
            first_frame_out = out_track.first_frame()
            x_out, y_out = out_track.first_position()
            
            # Try to match with unused middle tracks
            for mid_idx, mid_track in enumerate(middle):
                if mid_idx in used_middle_ids:
                    continue
                
                last_frame_mid = mid_track.last_frame()
                x_mid, y_mid = mid_track.last_position()
                
                if self._matches((last_frame_mid, x_mid, y_mid),
                                (first_frame_out, x_out, y_out)):
                    merged = self._merge_segments(mid_track, out_track)
                    linked_tracks.append(merged)
                    used_middle_ids.add(mid_idx)
                    break
            
            # Try to match with In→Middle results
            for in_mid_track in in_to_mid_tracks:
                last_frame = in_mid_track.last_frame()
                x_last, y_last = in_mid_track.last_position()
                
                if self._matches((last_frame, x_last, y_last),
                                (first_frame_out, x_out, y_out)):
                    merged = self._merge_segments(in_mid_track, out_track)
                    linked_tracks.append(merged)
                    in_to_mid_tracks.remove(in_mid_track)
                    break
        
        return linked_tracks
    
    def relink_tracks(self) -> pd.DataFrame:
        """
        Main method: Execute all 4 phases to relink broken tracks
        
        Returns:
            DataFrame with relinked tracks
        """
        print("🔗 Starting Track Relinking System...")
        
        # Phase A: Segment tracks
        print("📊 Phase A: Classifying tracks...")
        out_only, in_only, middle = self._segment_tracks()
        print(f"  - Out-only: {len(out_only)} tracks")
        print(f"  - In-only: {len(in_only)} tracks")
        print(f"  - Middle: {len(middle)} tracks")
        
        # Phase B: Link In → Middle
        print("🔗 Phase B: Linking In-only to Middle...")
        in_to_mid_tracks, used_middle = self._phase_b_in_to_middle(in_only, middle)
        print(f"  - Created {len(in_to_mid_tracks)} new links")
        
        # Phase C: Link Middle → Out
        print("🔗 Phase C: Linking Middle to Out-only...")
        mid_to_out_tracks = self._phase_c_middle_to_out(out_only, middle, 
                                                         in_to_mid_tracks, used_middle)
        print(f"  - Created {len(mid_to_out_tracks)} new links")
        
        # Phase D: Compile results
        print("📦 Phase D: Compiling final results...")
        
        # Add original complete tracks
        complete_tracks = self.df.copy()
        
        # Add newly linked tracks
        new_track_dfs = []
        
        for track in in_to_mid_tracks:
            new_track_dfs.append(track.to_dataframe())
        
        for track in mid_to_out_tracks:
            new_track_dfs.append(track.to_dataframe())
        
        if new_track_dfs:
            new_tracks_df = pd.concat(new_track_dfs, ignore_index=True)
            complete_tracks = pd.concat([complete_tracks, new_tracks_df], ignore_index=True)
        
        print(f"✅ Relinking complete!")
        print(f"  - Original unique tracks: {self.df['track_id'].nunique()}")
        print(f"  - Final unique tracks: {complete_tracks['track_id'].nunique()}")
        
        return complete_tracks.sort_values(['track_id', 'frame']).reset_index(drop=True)


def relink_trajectories(trajectories_df: pd.DataFrame,
                       frame_threshold: int = 10,
                       distance_threshold: float = 100.0) -> pd.DataFrame:
    """
    Convenience function to relink trajectories
    
    Args:
        trajectories_df: DataFrame with trajectory data
        frame_threshold: Max frame difference for matching
        distance_threshold: Max pixel distance for matching
    
    Returns:
        DataFrame with relinked trajectories
    """
    system = TrackRelinkSystem(trajectories_df, frame_threshold, distance_threshold)
    return system.relink_tracks()
