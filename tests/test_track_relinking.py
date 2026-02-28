import pandas as pd
from track_relinking import relink_trajectories


def test_merge_middle_tracks():
    # two segments representing same vehicle with small gap
    df = pd.DataFrame({
        'track_id': [1, 1, 2, 2],
        'frame': [1, 2, 3, 4],
        'x': [10, 20, 21, 30],
        'y': [10, 20, 21, 30],
        'vehicle_type': ['car', 'car', 'car', 'car'],
    })
    out = relink_trajectories(df, frame_threshold=5, distance_threshold=5, angle_threshold=45)
    # after relink there should be exactly one unique track id
    assert out['track_id'].nunique() == 1


def test_do_not_merge_different_type():
    df = pd.DataFrame({
        'track_id': [1, 1, 2, 2],
        'frame': [1, 2, 3, 4],
        'x': [10, 20, 21, 30],
        'y': [10, 20, 21, 30],
        'vehicle_type': ['car', 'car', 'truck', 'truck'],
    })
    out = relink_trajectories(df, frame_threshold=5, distance_threshold=5, angle_threshold=45)
    # different vehicle types should not merge
    assert out['track_id'].nunique() == 2
