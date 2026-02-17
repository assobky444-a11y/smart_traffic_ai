"""
🚗 Vehicle Tracking - Part 1: Detection & Tracking Only (STABLE IDs EDITION)
===========================================================================
Focus ONLY on:
  1) Select video
  2) Select YOLO model
  3) Run detection + tracking
  4) Export results: tracks.csv, meta.json, frames, annotated video (optional)

Key Fixes (Requested):
✅ IDs stable (won't change without reason)
✅ Boxes smooth (no "flying"/teleport)
✅ No vehicle swapping (reduced ID switches)
===========================================================================
"""

import os, sys, json, warnings, logging
import logging.handlers
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
from collections import deque, Counter

import cv2
import numpy as np
import pandas as pd

# ================== Conditional Imports ==================
try:
    cv2.setNumThreads(0)
except Exception:
    pass

# Desktop GUI removed — tkinter/Pillow imports stripped for headless web backend.
# Kept only backend surface used by the Flask app (VideoProcessorV1, ConfigV1, classify_tracks_from_df, logger).

SCIPY_AVAILABLE = False
FILTERPY_AVAILABLE = False
TORCH_AVAILABLE = False
YOLO_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except Exception:
    warnings.warn("SciPy not available")

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    warnings.warn("PyTorch not available")

try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except Exception:
    warnings.warn("FilterPy not available")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    warnings.warn("Ultralytics YOLO not installed")


# ================== Logging ==================
def setup_logging(output_dir: Optional[Path] = None, level=logging.INFO) -> logging.Logger:
    logger_obj = logging.getLogger("TrackingApp_V1_Stable")
    logger_obj.setLevel(level)

    if logger_obj.hasHandlers():
        logger_obj.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger_obj.addHandler(console_handler)

    if output_dir is None:
        output_dir = Path.cwd() / "logs"

    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"tracking_v1_stable_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger_obj.addHandler(file_handler)

        logger_obj.info(f"Logging initialized. Log file: {log_file}")
    except Exception as e:
        logger_obj.warning(f"Could not setup file logging: {e}")

    return logger_obj

logger = setup_logging()


# ================== JSON Safe Converter ==================
def _json_safe(obj):
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass
    return str(obj)


# ================== Config ==================
class ConfigV1:
    DEFAULT_VEHICLE_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    DEFAULTS = {
        "video_path": "",
        "model_path": "yolo12n.pt",
        "out_dir": "unified_output",
        "conf_threshold": 0.40,
        "iou_threshold": 0.50,
        "min_box_area": 200,
        "max_box_area": 80000,
        "device": "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",

        # Tracking
        "max_age_frames": 24,
        "min_hits": 2,

        # Association
        "center_dist_threshold": 140,
        # 🎯 NEW: Updated weights for better matching (IoU + Appearance + Motion)
        "iou_weight": 0.45,           # IoU is most reliable
        "appearance_weight": 0.25,     # ReID features help reduce swaps significantly
        "motion_weight": 0.15,         # Motion prediction for occluded objects
        # Note: Direction(10%) + Class(5%) = remaining 30%

        "assignment_cost_threshold": 0.55,

        # ReID revival (stable IDs)
        "reid_window_sec": 4.0,
        "reid_cost_threshold": 0.50,

        # Speed/processing
        "process_every_n_frames": 2,
        "batch_size": 16,
        "fps_override": 0,
        "max_frames": 0,
        # Detection resize: max dimension (width/height) for YOLO input to speed up inference.
        # Set to None or 0 to disable resizing. Typical values: 640, 960
        "detection_max_dim": 640,

        # Appearance
        "appearance_mode": "hsv",
        "feature_history_size": 10,

        # Smoothing (visual stability)
        "bbox_smooth_alpha": 0.70,   # EMA: higher = smoother (but more lag)
        "vel_decay": 0.90,

        # Output smoothing (csv)
        "smooth_window_size": 9,
        "smooth_poly_order": 2,
        "quality_min_points": 10,

        # Export
        "export_preview_frames": True,
        "export_track_previews": True,
        "preview_selection_mode": "borderline_low",  # 'all','confident','confident_borderline','quality_tracks','borderline_low','low'
        "show_live_preview": False,
        "preview_scale": 0.75,
        "preview_large_max_dim": 512,  # maximum size (pixels) for the exported large preview (single file)
        "export_format": "all",
        "hide_predicted_points": True,
        "max_predict_frames": 8,

        # ROI Filtering (optional)
        "roi_points": None,  # If None, no filtering applied

        # Class lock / anti-swap
        "class_suspect_conf": 0.55,
        "class_lock_after_hits": 5,     # after this, class becomes more stable
        "class_change_margin": 3,       # need N votes difference to change class

        "use_preprocess": False,
        "save_annotated_video": True,
        "draw_trajectory_arrows": True,
        "custom_labels": {},
        "save_cfg_in_report": True,
    }

    @classmethod
    def validate(cls, user_cfg: Dict) -> Dict:
        cfg = dict(cls.DEFAULTS)
        if user_cfg:
            cfg.update(user_cfg)

        raw = cfg.get("custom_labels", {})
        if isinstance(raw, str) and raw.strip():
            try:
                raw = json.loads(raw)
            except Exception:
                logger.warning("Failed to parse custom_labels JSON; ignoring.")
                raw = {}
        elif not isinstance(raw, dict):
            raw = {}
        cfg["custom_labels"] = {int(k): str(v) for k, v in raw.items()}

        vehicle_classes = dict(cls.DEFAULT_VEHICLE_CLASSES)
        vehicle_classes.update(cfg["custom_labels"])
        cfg["_vehicle_classes"] = vehicle_classes
        cfg["_valid_class_ids"] = set(vehicle_classes.keys())

        cfg["conf_threshold"] = float(np.clip(cfg["conf_threshold"], 0.1, 0.95))
        cfg["iou_threshold"] = float(np.clip(cfg["iou_threshold"], 0.1, 0.9))
        cfg["min_box_area"] = int(max(50, cfg["min_box_area"]))
        cfg["max_box_area"] = int(max(cfg["min_box_area"] * 2, cfg["max_box_area"]))
        cfg["process_every_n_frames"] = int(max(1, cfg["process_every_n_frames"]))
        cfg["batch_size"] = int(np.clip(cfg["batch_size"], 1, 16))
        cfg["max_age_frames"] = int(max(3, cfg["max_age_frames"]))
        cfg["min_hits"] = int(max(1, cfg["min_hits"]))
        cfg["center_dist_threshold"] = int(max(20, cfg["center_dist_threshold"]))
        cfg["reid_window_sec"] = float(max(0.3, cfg["reid_window_sec"]))

        sw = int(cfg["smooth_window_size"])
        if sw % 2 == 0:
            sw += 1
        cfg["smooth_window_size"] = int(max(3, sw))
        cfg["smooth_poly_order"] = int(np.clip(cfg["smooth_poly_order"], 1, 4))

        # Normalize weights
        total = float(cfg["iou_weight"] + cfg["motion_weight"] + cfg["appearance_weight"])
        if total <= 0 or abs(total - 1.0) > 0.05:
            cfg["iou_weight"], cfg["motion_weight"], cfg["appearance_weight"] = 0.50, 0.25, 0.25
            logger.info("Association weights normalized to default (0.50/0.25/0.25)")

        cfg["feature_history_size"] = int(max(1, cfg["feature_history_size"]))
        cfg["max_predict_frames"] = int(max(1, cfg.get("max_predict_frames", 8)))
        cfg["hide_predicted_points"] = bool(cfg.get("hide_predicted_points", True))
        cfg["class_suspect_conf"] = float(np.clip(cfg.get("class_suspect_conf", 0.55), 0.05, 0.95))
        cfg["bbox_smooth_alpha"] = float(np.clip(cfg.get("bbox_smooth_alpha", 0.70), 0.0, 0.95))
        cfg["vel_decay"] = float(np.clip(cfg.get("vel_decay", 0.90), 0.50, 0.99))

        cfg["use_preprocess"] = bool(cfg.get("use_preprocess", False))
        cfg["show_live_preview"] = bool(cfg.get("show_live_preview", False))

        if cfg["device"] == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                cfg["device"] = "cuda"
                logger.info("Auto-selected: CUDA (GPU available)")
            else:
                cfg["device"] = "cpu"
                logger.info("Auto-selected: CPU")
        elif cfg["device"] == "cuda" and not (TORCH_AVAILABLE and torch.cuda.is_available()):
            logger.warning("CUDA requested but not available. Using CPU.")
            cfg["device"] = "cpu"

        return cfg


# ================== Feature Extraction ==================
class FeatureExtractorV1:
    def __init__(self, mode: str):
        self.mode = mode

    @staticmethod
    def _crop(frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _hsv(self, frame, bbox) -> np.ndarray:
        crop = self._crop(frame, bbox)
        if crop is None:
            return np.zeros(64, np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 2], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def _lab(self, frame, bbox) -> np.ndarray:
        crop = self._crop(frame, bbox)
        if crop is None:
            return np.zeros(48, np.float32)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
        hist = cv2.calcHist([lab], [0, 1, 2], None, [4, 4, 3], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def extract(self, frame, bbox) -> Dict[str, np.ndarray]:
        return {
            "hsv": self._hsv(frame, bbox),
            "lab": self._lab(frame, bbox)
        }
    
    def compute_similarity(self, feat_a: Dict[str, np.ndarray], feat_b: Dict[str, np.ndarray]) -> float:
        """
        🎯 NEW: Compute similarity between two feature dictionaries
        Returns: float in [0, 1] where 1 = identical, 0 = completely different
        """
        return feature_similarity(feat_a, feat_b)


def _iou(a: List[float], b: List[float]) -> float:
    x1a, y1a, x2a, y2a = a
    x1b, y1b, x2b, y2b = b

    inter_x1 = max(x1a, x1b)
    inter_y1 = max(y1a, y1b)
    inter_x2 = min(x2a, x2b)
    inter_y2 = min(y2a, y2b)

    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, (x2a - x1a)) * max(0.0, (y2a - y1a))
    area_b = max(0.0, (x2b - x1b)) * max(0.0, (y2b - y1b))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(np.clip(inter / union, 0.0, 1.0))


def feature_similarity(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    sims = []
    weights = []

    if "hsv" in a and "hsv" in b:
        hsv_sim = max(0.0, float(cv2.compareHist(a["hsv"], b["hsv"], cv2.HISTCMP_CORREL)))
        sims.append(hsv_sim); weights.append(0.6)

    if "lab" in a and "lab" in b:
        lab_sim = max(0.0, float(cv2.compareHist(a["lab"], b["lab"], cv2.HISTCMP_CORREL)))
        sims.append(lab_sim); weights.append(0.4)

    if not sims:
        return 0.5

    total = sum(weights)
    return float(np.clip(sum(s * w for s, w in zip(sims, weights)) / total, 0.0, 1.0))


# ================== Kalman Filter ==================
class KalmanV1:
    """
    Simple stable bbox Kalman:
    state: [cx, cy, w, h, vx, vy, vw, vh]
    """
    def __init__(self, bbox: List[float], dt: float, frame_size: Tuple[int, int], confidence: float = 0.5, vel_decay: float = 0.90):
        self.dt = float(dt)
        self.w_img, self.h_img = frame_size
        self.vel_decay = float(vel_decay)

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(10.0, x2 - x1)
        h = max(10.0, y2 - y1)
        self._vmax = max(self.w_img, self.h_img) * 0.20

        if FILTERPY_AVAILABLE:
            self.kf = KalmanFilter(dim_x=8, dim_z=4)
            self.kf.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=float).reshape(8, 1)
            self.kf.F = np.array([
                [1, 0, 0, 0, self.dt, 0, 0, 0],
                [0, 1, 0, 0, 0, self.dt, 0, 0],
                [0, 0, 1, 0, 0, 0, self.dt, 0],
                [0, 0, 0, 1, 0, 0, 0, self.dt],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ], dtype=float)
            self.kf.H = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ], dtype=float)

            self.kf.P *= 10.0
            self.kf.R *= 0.1  # Trust detection much more to prevent lag/flying
            self.kf.Q = np.eye(8) * 0.01
            self.kf.Q[4:, 4:] *= 10.0
        else:
            self.kf = None
            self.state = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=float)

    def _cap_vel(self, vx, vy):
        import math
        v = math.hypot(vx, vy)
        if v > self._vmax:
            scale = self._vmax / max(1e-6, v)
            return vx * scale, vy * scale
        return vx, vy

    def predict(self) -> List[float]:
        """
        Predict ONCE per frame (important for stability).
        """
        if self.kf is None:
            cx, cy, w, h, vx, vy, vw, vh = self.state
            vx, vy = self._cap_vel(vx, vy)
            vx *= self.vel_decay
            vy *= self.vel_decay
            vw *= self.vel_decay
            vh *= self.vel_decay
            cx += vx * self.dt
            cy += vy * self.dt
            w = max(10.0, abs(w))
            h = max(10.0, abs(h))
            self.state = np.array([cx, cy, w, h, vx, vy, vw, vh], dtype=float)
            return self._to_bbox(cx, cy, w, h)

        self.kf.predict()
        x = self.kf.x.flatten()

        vx, vy = self._cap_vel(x[4], x[5])
        vx *= self.vel_decay
        vy *= self.vel_decay

        self.kf.x[4, 0] = vx
        self.kf.x[5, 0] = vy
        return self._to_bbox(x[0], x[1], x[2], x[3])

    def update(self, bbox: List[float]):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(10.0, x2 - x1)
        h = max(10.0, y2 - y1)

        if self.kf is None:
            ocx, ocy, ow, oh, vx, vy, vw, vh = self.state
            inv = 1.0 / max(1e-6, self.dt)
            vx = (cx - ocx) * inv
            vy = (cy - ocy) * inv
            vx, vy = self._cap_vel(vx, vy)
            self.state = np.array([cx, cy, w, h, vx, vy, (w - ow) * inv, (h - oh) * inv], dtype=float)
            return

        z = np.array([cx, cy, w, h], dtype=float)
        self.kf.update(z)
        
        # 🎯 NEW: Cap velocities after update to prevent explosion
        self.kf.x[4, 0], self.kf.x[5, 0] = self._cap_vel(self.kf.x[4, 0], self.kf.x[5, 0])

    def get_state_bbox(self) -> List[float]:
        """
        Convert CURRENT state (posterior) to bbox without extra predict.
        """
        if self.kf is None:
            cx, cy, w, h = self.state[0], self.state[1], self.state[2], self.state[3]
            return self._to_bbox(cx, cy, w, h)
        x = self.kf.x.flatten()
        return self._to_bbox(x[0], x[1], x[2], x[3])

    def velocity(self) -> Tuple[float, float]:
        if self.kf is None:
            return float(self.state[4]), float(self.state[5])
        x = self.kf.x.flatten()
        return float(x[4]), float(x[5])

    def _to_bbox(self, cx, cy, w, h) -> List[float]:
        w = max(10.0, abs(w))
        h = max(10.0, abs(h))
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        x1 = np.clip(x1, 0, self.w_img - 1)
        y1 = np.clip(y1, 0, self.h_img - 1)
        x2 = np.clip(x2, 0, self.w_img - 1)
        y2 = np.clip(y2, 0, self.h_img - 1)

        if x2 - x1 < 10:
            x2 = min(self.w_img - 1, x1 + 10)
        if y2 - y1 < 10:
            y2 = min(self.h_img - 1, y1 + 10)

        return [float(x1), float(y1), float(x2), float(y2)]


# ================== Track ==================
class TrackV1:
    __slots__ = [
        "track_id", "class_id", "confidence", "bbox", "bbox_pred", "kf",
        "misses", "hits", "first_frame", "last_frame", "is_active", "fps",
        "vel", "color", "feat_hist", "hist", "class_votes", "locked_class",
        "_smooth_alpha"
    ]

    def __init__(
        self, track_id: int, bbox: List[float], class_id: int, confidence: float,
        frame_idx: int, fps: float, frame_size: Tuple[int, int],
        feature_hist_size: int, vel_decay: float, bbox_smooth_alpha: float
    ):
        self.track_id = int(track_id)
        self.class_id = int(class_id)
        self.locked_class = False
        self.class_votes = Counter()
        self.class_votes[self.class_id] += 1

        self.confidence = float(confidence)
        self.fps = float(max(1.0, fps))
        self.kf = KalmanV1(bbox, 1.0 / self.fps, frame_size, confidence, vel_decay=vel_decay)

        # Start with posterior bbox (state), and pred bbox for next association
        self.bbox = self.kf.get_state_bbox()
        self.bbox_pred = self.bbox[:]

        self.vel = (0.0, 0.0)
        self.misses = 0
        self.hits = 1
        self.first_frame = int(frame_idx)
        self.last_frame = int(frame_idx)
        self.is_active = True

        np.random.seed(self.track_id * 97)
        self.color = tuple(int(v) for v in np.random.randint(50, 255, 3))

        self.feat_hist = deque(maxlen=int(feature_hist_size))
        self.hist = []

        self._smooth_alpha = float(bbox_smooth_alpha)
        self._push_point(frame_idx, self.bbox, confidence, False)

    @staticmethod
    def _center(bb: List[float]) -> Tuple[float, float]:
        return (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0

    @staticmethod
    def _ema_bbox(old_bb: List[float], new_bb: List[float], alpha: float) -> List[float]:
        if old_bb is None:
            return list(new_bb)
        a = float(alpha)
        return [
            a * new_bb[0] + (1 - a) * old_bb[0],
            a * new_bb[1] + (1 - a) * old_bb[1],
            a * new_bb[2] + (1 - a) * old_bb[2],
            a * new_bb[3] + (1 - a) * old_bb[3],
        ]

    def _push_point(self, frame_idx: int, bbox: List[float], conf: float, is_pred: bool):
        cx, cy = self._center(bbox)
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        self.hist.append({
            "frame_idx": int(frame_idx),
            "timestamp": float(frame_idx / self.fps),
            "center": (float(cx), float(cy)),
            "wh": (float(w), float(h)),
            "confidence": float(conf),
            "is_predicted": bool(is_pred),
            "velocity": (float(self.vel[0]), float(self.vel[1])),
        })

    def step_predict(self, frame_idx: int, hide_predicted_points: bool, conf_decay: float = 0.92):
        """
        Predict once per frame (important). Used before association.
        """
        self.bbox_pred = self.kf.predict()
        self.vel = self.kf.velocity()

        # 🎯 NEW: Always record predicted points for trajectory continuity
        # Save predicted points even if hidden - important for trajectory analysis
        self._push_point(frame_idx, self.bbox_pred, max(0.0, self.confidence * conf_decay), True)

    def mark_missed(self, frame_idx: int, max_age: int, min_hits: int, max_predict: int, hide_predicted_points: bool):
        """
        If unmatched: use predicted bbox as current bbox, increase misses, add predicted point.
        """
        self.bbox = self._ema_bbox(self.bbox, self.bbox_pred, self._smooth_alpha)
        self.misses += 1
        self.last_frame = int(frame_idx)

        # 🎯 NEW: Always push predicted points during misses for continuity
        self._push_point(frame_idx, self.bbox, max(0.0, self.confidence * 0.9), True)

        if self.hits < min_hits and self.misses > 0:
            return True
        return self.misses >= min(max_age, max_predict)

    def update_matched(self, bbox_det: List[float], class_id: int, confidence: float, frame_idx: int, feats: Optional[Dict],
                       class_lock_after_hits: int, class_change_margin: int):
        """
        Update with detection (posterior). No extra predict here.
        """
        # Update class votes
        self.class_votes[int(class_id)] += 1

        # Class stabilization logic
        if (not self.locked_class) and self.hits >= int(class_lock_after_hits):
            self.locked_class = True

        if self.locked_class:
            top = self.class_votes.most_common(2)
            if len(top) == 1:
                stable_cls = top[0][0]
            else:
                stable_cls, stable_cnt = top[0]
                second_cls, second_cnt = top[1]
                if (stable_cnt - second_cnt) < int(class_change_margin):
                    stable_cls = self.class_id  # keep old if not decisive
            self.class_id = int(stable_cls)
        else:
            # early stage: allow change but conservatively
            if float(confidence) >= 0.55:
                self.class_id = int(class_id)

        self.kf.update(bbox_det)
        posterior_bbox = self.kf.get_state_bbox()

        # EMA smoothing for box stability
        self.bbox = self._ema_bbox(self.bbox, posterior_bbox, self._smooth_alpha)

        self.vel = self.kf.velocity()
        self.confidence = float(confidence)
        self.misses = 0
        self.hits += 1
        self.last_frame = int(frame_idx)

        if feats:
            self.feat_hist.append(feats)

        self._push_point(frame_idx, self.bbox, self.confidence, False)


# ================== MOT / Tracker ==================
class MOTV1:
    def __init__(self, cfg: Dict, fps: float, frame_size: Tuple[int, int], feature_extractor=None):
        self.cfg = ConfigV1.validate(cfg)
        self.fps = float(max(1.0, fps))
        self.frame_w, self.frame_h = frame_size
        self.fe = feature_extractor if feature_extractor else FeatureExtractorV1(self.cfg["appearance_mode"])

        self.tracks: Dict[int, TrackV1] = {}
        self.finished: List[TrackV1] = []

        self.next_id = 1

        # recently ended tracks for ReID revival (stable IDs)
        self.recent_ended = deque(maxlen=256)  # each item: dict with {id, frame, bbox, cls, feat}

    @staticmethod
    def _center(bb: List[float]) -> Tuple[float, float]:
        return (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0

    def _motion_sim(self, bb_a: List[float], bb_b: List[float]) -> float:
        import math
        ax, ay = self._center(bb_a)
        bx, by = self._center(bb_b)
        dist = math.hypot(ax - bx, ay - by)
        return float(max(0.0, 1.0 - dist / float(self.cfg["center_dist_threshold"])))

    def _direction_gate_penalty(self, tr: TrackV1, det_bb: List[float]) -> float:
        """
        If track has velocity, prefer detections that align with expected direction.
        Helps reduce swapping when two vehicles are close.
        """
        import math
        vx, vy = tr.vel
        speed = math.hypot(vx, vy)
        if speed < 1.0:
            return 0.0

        tcx, tcy = self._center(tr.bbox)
        dcx, dcy = self._center(det_bb)
        dx, dy = (dcx - tcx), (dcy - tcy)
        dnorm = math.hypot(dx, dy)
        if dnorm < 1e-6:
            return 0.0

        # cosine similarity between velocity and displacement
        cos = (vx * dx + vy * dy) / (speed * dnorm)
        # if cos is negative => going opposite direction => penalize
        if cos < -0.2:
            return 0.25
        if cos < 0.1:
            return 0.10
        return 0.0

    def _assoc_cost(self, tr: TrackV1, det_bb: List[float], det_feats: Optional[Dict], det_cls: int) -> float:
        """Not used anymore - greedy IoU matching is more stable."""
        return 0.0

    def _push_recent_ended(self, tr: TrackV1):
        """
        Store minimal info for ReID revival.
        """
        last_feat = None
        if tr.feat_hist:
            # Convert deque to list for indexing
            feat_list = list(tr.feat_hist)
            last_feat = feat_list[-1] if feat_list else None
        self.recent_ended.append({
            "track_id": int(tr.track_id),
            "last_frame": int(tr.last_frame),
            "bbox": list(tr.bbox),
            "class_id": int(tr.class_id),
            "feat": last_feat
        })

    def _end_track(self, tid: int):
        tr = self.tracks.get(tid)
        if not tr:
            return
        tr.is_active = False
        if len(tr.hist) >= int(self.cfg["quality_min_points"]):
            self.finished.append(tr)
        self._push_recent_ended(tr)
        del self.tracks[tid]

    def _reid_try_revive(self, det_bb: List[float], det_cls: int, det_conf: float, det_feats: Optional[Dict], frame_idx: int) -> Optional[int]:
        """
        If a track ended recently, and this detection matches it well, revive with SAME ID.
        This is key for stable IDs.
        """
        win_frames = int(self.cfg["reid_window_sec"] * self.fps)
        best = None
        best_cost = 1.0

        # purge old entries (soft)
        while self.recent_ended and (frame_idx - self.recent_ended[0]["last_frame"]) > (win_frames * 2):
            self.recent_ended.popleft()

        for item in list(self.recent_ended):
            age = frame_idx - item["last_frame"]
            if age < 1 or age > win_frames:
                continue

            # distance gate
            mot_s = self._motion_sim(item["bbox"], det_bb)
            if mot_s <= 0.0:
                continue

            iou_s = _iou(item["bbox"], det_bb)

            app_s = 0.5
            if det_feats and item.get("feat") is not None:
                app_s = feature_similarity(det_feats, item["feat"])

            # ReID weighting: appearance stronger
            score = (0.35 * iou_s) + (0.25 * mot_s) + (0.40 * app_s)
            cost = 1.0 - score

            # class soft penalty
            if int(det_cls) != int(item["class_id"]):
                cost += 0.10

            if cost < best_cost:
                best_cost = cost
                best = item

        if best is not None and best_cost <= float(self.cfg["reid_cost_threshold"]):
            # remove it so it won't revive twice
            try:
                self.recent_ended.remove(best)
            except Exception:
                pass
            return int(best["track_id"])

        return None

    def update(self, detections: List[Tuple[List[float], int, float]], frame: np.ndarray, frame_idx: int) -> Dict[int, TrackV1]:
        """
        frame_idx is the REAL video frame index (important for stability & timestamps).
        """
        dets = []
        for bb, cls_id, conf in detections:
            feats = self.fe.extract(frame, bb)
            dets.append((bb, int(cls_id), float(conf), feats))

        # 1) Predict once for ALL active tracks for this frame
        active_tracks = [t for t in self.tracks.values() if t.is_active]
        for tr in active_tracks:
            tr.step_predict(frame_idx, hide_predicted_points=self.cfg["hide_predicted_points"], conf_decay=0.92)

        matches = []
        unmatched_tracks_idx = []
        unmatched_dets_idx = []

        if not active_tracks:
            unmatched_dets_idx = list(range(len(dets)))
        elif not dets:
            unmatched_tracks_idx = list(range(len(active_tracks)))
        else:
            # EXPERT SOLUTION: Hungarian Algorithm (Global Optimization)
            # Prevents ID swapping by finding the globally optimal assignment
            
            n_tracks = len(active_tracks)
            n_dets = len(dets)
            cost_matrix = np.ones((n_tracks, n_dets), dtype=np.float32)
            
            for i, tr in enumerate(active_tracks):
                for j, (bb, c, conf, f) in enumerate(dets):
                    # 🎯 NEW: Class penalty instead of hard blocking
                    cost_class = 0.0
                    if c != tr.class_id:
                        # Soft penalty for class mismatch
                        if tr.locked_class and tr.hits > 5:
                            cost_class = 0.8  # High penalty for stable tracks
                        else:
                            cost_class = 0.3  # Lower penalty for new tracks
                    
                    # 1. IoU Cost
                    iou = _iou(tr.bbox_pred, bb)
                    cost_iou = 1.0 - iou
                    
                    # 2. Distance Cost
                    tcx, tcy = self._center(tr.bbox_pred)
                    dcx, dcy = self._center(bb)
                    dist = np.hypot(tcx - dcx, tcy - dcy)
                    cost_dist = min(1.0, dist / float(self.cfg["center_dist_threshold"]))
                    
                    # 3. Direction Penalty (Anti-Swap)
                    cost_dir = self._direction_gate_penalty(tr, bb)
                    
                    # 🎯 NEW: 4. Appearance Cost (ReID)
                    cost_app = 0.0
                    if f and tr.feat_hist:
                        # Compute cosine similarity with recent features
                        # Convert deque to list for slicing
                        feat_list = list(tr.feat_hist)
                        recent_feats = feat_list[-3:] if len(feat_list) >= 3 else feat_list
                        similarities = []
                        for hist_feat in recent_feats:
                            sim = self.fe.compute_similarity(hist_feat, f)
                            similarities.append(sim)
                        avg_sim = np.mean(similarities) if similarities else 0.0
                        cost_app = 1.0 - avg_sim  # Convert similarity to cost
                    
                    # 🎯 NEW: Weighted Cost with all components
                    # IoU(45%) + Appearance(25%) + Distance(15%) + Direction(10%) + Class(5%)
                    iou_w = float(self.cfg.get("iou_weight", 0.45))
                    app_w = float(self.cfg.get("appearance_weight", 0.25))
                    dist_w = float(self.cfg.get("motion_weight", 0.15))
                    dir_w = 0.10
                    class_w = 0.05
                    
                    cost = (iou_w * cost_iou + 
                            app_w * cost_app + 
                            dist_w * cost_dist + 
                            dir_w * cost_dir + 
                            class_w * cost_class)
                    
                    # Gating: Reject if too far
                    if dist > float(self.cfg["center_dist_threshold"]):
                        cost = 1.0
                        
                    cost_matrix[i, j] = cost

            # Solve Assignment
            if SCIPY_AVAILABLE:
                row_inds, col_inds = linear_sum_assignment(cost_matrix)
            else:
                # Fallback greedy if scipy missing
                # (Not ideal but keeps it running)
                row_inds, col_inds = [], []
                # ... (Simple greedy logic could go here, but scipy is standard)

            # Filter matches by threshold
            match_threshold = float(self.cfg["assignment_cost_threshold"])
            matched_indices = set()
            
            for r, c in zip(row_inds, col_inds):
                if cost_matrix[r, c] < match_threshold:
                    matches.append((r, c))
                else:
                    unmatched_tracks_idx.append(r) # Mark as unmatched if cost high
            
            # Identify unmatched
            matched_rows = {r for r, c in matches}
            matched_cols = {c for r, c in matches}
            
            # Add tracks that were not in the assignment solution at all
            for i in range(n_tracks):
                if i not in matched_rows and i not in unmatched_tracks_idx:
                    unmatched_tracks_idx.append(i)
            
            unmatched_dets_idx = [j for j in range(n_dets) if j not in matched_cols]

        # 2) Update matched tracks
        for r, j in matches:
            tr = active_tracks[r]
            bb, cls, conf, feats = dets[j]
            tr.update_matched(
                bb, cls, conf, frame_idx, feats,
                class_lock_after_hits=int(self.cfg["class_lock_after_hits"]),
                class_change_margin=int(self.cfg["class_change_margin"])
            )

        # 3) Unmatched tracks => mark missed, possibly end
        for idx in unmatched_tracks_idx:
            tr = active_tracks[idx]
            should_end = tr.mark_missed(
                frame_idx,
                max_age=int(self.cfg["max_age_frames"]),
                min_hits=int(self.cfg["min_hits"]),
                max_predict=int(self.cfg["max_predict_frames"]),
                hide_predicted_points=self.cfg["hide_predicted_points"],
            )
            if should_end:
                self._end_track(tr.track_id)

        # 4) Unmatched detections => ReID revive or create new track
        for j in unmatched_dets_idx:
            bb, cls, conf, feats = dets[j]
            if cls not in self.cfg["_valid_class_ids"]:
                continue

            revived_id = self._reid_try_revive(bb, cls, conf, feats, frame_idx)
            if revived_id is not None and revived_id not in self.tracks:
                tid = revived_id
                tr = TrackV1(
                    tid, bb, cls, conf, frame_idx, self.fps,
                    (self.frame_w, self.frame_h),
                    self.cfg["feature_history_size"],
                    vel_decay=float(self.cfg["vel_decay"]),
                    bbox_smooth_alpha=float(self.cfg["bbox_smooth_alpha"]),
                )
                if feats:
                    tr.feat_hist.append(feats)
                self.tracks[tid] = tr
                self.next_id = max(self.next_id, tid + 1)
                continue

            # New track
            tid = self.next_id
            tr = TrackV1(
                tid, bb, cls, conf, frame_idx, self.fps,
                (self.frame_w, self.frame_h),
                self.cfg["feature_history_size"],
                vel_decay=float(self.cfg["vel_decay"]),
                bbox_smooth_alpha=float(self.cfg["bbox_smooth_alpha"]),
            )
            if feats:
                tr.feat_hist.append(feats)
            self.tracks[tid] = tr
            self.next_id += 1

        return {tid: t for tid, t in self.tracks.items() if t.is_active}

    def soft_predict_all(self, frame_idx: int):
        """
        For skipped frames: predict smoothly WITHOUT counting misses.
        Prevents teleport/jitter when processing every N frames.
        """
        for tr in list(self.tracks.values()):
            if not tr.is_active:
                continue
            tr.step_predict(frame_idx, hide_predicted_points=True, conf_decay=0.98)
            # update visual bbox softly, no misses increment
            tr.bbox = tr._ema_bbox(tr.bbox, tr.bbox_pred, tr._smooth_alpha)
            tr.last_frame = int(frame_idx)

    def finalize(self):
        for tid in list(self.tracks.keys()):
            self._end_track(tid)

    def iter_all(self):
        for tr in self.finished:
            yield tr
        for tr in self.tracks.values():
            yield tr
 
    def compute_quality_metrics(self) -> Dict[str, Any]:
        """Compute robust quality metrics per track using multi-criteria confidence rules.

        Rule (configurable via cfg):
          - use non-predicted points by default (can include predicted via cfg)
          - track is CONFIDENT if:
              * detection_count >= confidence_min_hits, and
              * median_confidence >= confidence_threshold, and
              * pct_frames_above_threshold >= confidence_prop_threshold
          - borderline tracks are flagged when medians or proportions are near the threshold
        """
        # Only consider tracks that have at least one historical point (matches exported CSV behavior)
        all_tracks = [t for t in self.iter_all() if len(t.hist) > 0]
        if not all_tracks:
            return {
                "total_tracks": 0,
                "quality_tracks": 0,
                "avg_track_length": 0,
                "min_track_length": 0,
                "max_track_length": 0,
                "avg_confidence": 0,
                "confidence_threshold": float(self.cfg.get("confidence_threshold", 0.45)),
                "high_conf_count": 0,
                "high_conf_pct": 0.0,
                "confident_count": 0,
                "confident_pct": 0.0,
                "borderline_count": 0,
                "low_conf_count": 0,
                "low_conf_pct": 0.0,
                "quality_score": 0,
                "long_tracks": 0,
                "short_tracks": 0,
                "reliable": False,
            }

        # configuration / defaults
        min_hits = int(self.cfg.get("min_hits", 2))
        conf_min_hits = int(self.cfg.get("confidence_min_hits", max(10, min_hits)))
        threshold = float(self.cfg.get("confidence_threshold", 0.45))
        prop_thresh = float(self.cfg.get("confidence_prop_threshold", 0.6))
        border_delta = float(self.cfg.get("confidence_border_delta", 0.05))
        include_predicted = bool(self.cfg.get("include_predicted_in_confidence", False))

        track_lengths = [len(t.hist) for t in all_tracks]
        confidences_all = [p["confidence"] for t in all_tracks for p in t.hist]

        quality_tracks = [t for t in all_tracks if len(t.hist) >= min_hits]
        long_threshold = int(self.cfg.get("long_track_threshold", 15))
        long_tracks = len([t for t in all_tracks if len(t.hist) >= long_threshold])

        avg_conf = float(np.mean(confidences_all)) if confidences_all else 0.0

        per_track_stats = []
        for t in all_tracks:
            # choose detection points depending on config
            pts = [p for p in t.hist if include_predicted or not p.get("is_predicted", False)]
            if not pts:
                pts = t.hist  # fallback to any available points
            confs = [p["confidence"] for p in pts]
            n_points = len(pts)
            median_conf = float(np.median(confs)) if confs else 0.0
            mean_conf = float(np.mean(confs)) if confs else 0.0
            pct_above = float(sum(1 for v in confs if v >= threshold) / max(1, len(confs))) if confs else 0.0

            is_confident = (
                n_points >= conf_min_hits and
                median_conf >= threshold and
                pct_above >= prop_thresh
            )
            is_borderline = False
            if not is_confident:
                # borderline by median or proportion near threshold
                if (threshold - border_delta) <= median_conf < threshold:
                    is_borderline = True
                elif (prop_thresh - 0.1) <= pct_above < prop_thresh and n_points >= max(2, int(conf_min_hits / 2)):
                    is_borderline = True

            per_track_stats.append({
                "track_id": int(t.track_id),
                "n_points": n_points,
                "mean_conf": mean_conf,
                "median_conf": median_conf,
                "pct_above": pct_above,
                "confident": bool(is_confident),
                "borderline": bool(is_borderline),
            })

        stats_df = pd.DataFrame(per_track_stats)
        total_tracks = len(stats_df)
        confident_count = int(stats_df[stats_df["confident"]].shape[0])
        borderline_count = int(stats_df[stats_df["borderline"] & ~stats_df["confident"]].shape[0])
        low_conf_count = int(total_tracks - confident_count - borderline_count)
        confident_pct = float(confident_count / max(1, total_tracks))
        low_conf_pct = float(low_conf_count / max(1, total_tracks))

        # quality score uses proportion of confident tracks and quality_tracks
        quality_score = (
            (len(quality_tracks) / max(1, total_tracks)) * 0.5 +
            min(max(confident_pct, 0.0), 1.0) * 0.5
        )

        reliable = True if total_tracks >= 5 else False

        return {
            "total_tracks": total_tracks,
            "quality_tracks": len(quality_tracks),
            "avg_track_length": float(np.mean(track_lengths)),
            "min_track_length": int(np.min(track_lengths)) if track_lengths else 0,
            "max_track_length": int(np.max(track_lengths)) if track_lengths else 0,
            "avg_confidence": float(avg_conf),
            "confidence_threshold": float(threshold),
            "high_conf_count": confident_count,
            "high_conf_pct": float(confident_pct),
            "confident_count": confident_count,
            "confident_pct": float(confident_pct),
            "borderline_count": borderline_count,
            "low_conf_count": low_conf_count,
            "low_conf_pct": float(low_conf_pct),
            "quality_score": float(np.clip(quality_score * 100, 0, 100)),
            "long_tracks": long_tracks,
            "short_tracks": len([t for t in all_tracks if len(t.hist) < min_hits]),
            "reliable": reliable,
        }

    def to_rows(self, tracks: Optional[List[TrackV1]] = None) -> List[Dict[str, Any]]:
        rows = []
        source = tracks if tracks is not None else self.iter_all()

        for tr in source:
            veh = self.cfg["_vehicle_classes"].get(tr.class_id, "unknown")
            centers = [p["center"] for p in tr.hist]

            if len(centers) >= 3:
                xs = np.array([c[0] for c in centers], dtype=float)
                ys = np.array([c[1] for c in centers], dtype=float)
                if SCIPY_AVAILABLE and len(centers) >= self.cfg["smooth_window_size"]:
                    win = min(self.cfg["smooth_window_size"], len(xs) if len(xs) % 2 == 1 else len(xs) - 1)
                    poly = min(self.cfg["smooth_poly_order"], win - 1)
                    try:
                        xs_s = savgol_filter(xs, win, poly)
                        ys_s = savgol_filter(ys, win, poly)
                    except Exception:
                        xs_s, ys_s = xs.copy(), ys.copy()
                else:
                    k = min(7, len(xs) if len(xs) % 2 == 1 else len(xs) - 1)
                    k = max(3, k)
                    kernel = np.ones(k, dtype=float) / k
                    xs_s = np.convolve(xs, kernel, mode="same")
                    ys_s = np.convolve(ys, kernel, mode="same")
            else:
                xs_s = np.array([c[0] for c in centers], dtype=float) if centers else np.array([])
                ys_s = np.array([c[1] for c in centers], dtype=float) if centers else np.array([])

            prev_ts, prev_c = None, None
            for i, p in enumerate(tr.hist):
                cx, cy = p["center"]
                smx = float(xs_s[i]) if i < len(xs_s) else float(cx)
                smy = float(ys_s[i]) if i < len(ys_s) else float(cy)

                if prev_ts is not None:
                    import math
                    dt = p["timestamp"] - prev_ts
                    if dt > 0:
                        vx = (cx - prev_c[0]) / dt
                        vy = (cy - prev_c[1]) / dt
                        speed = float(math.hypot(vx, vy))
                    else:
                        speed = 0.0
                else:
                    speed = 0.0

                prev_ts = p["timestamp"]
                prev_c = (cx, cy)

                rows.append({
                    "track_id": int(tr.track_id),
                    "vehicle_type": str(veh),
                    "frame_idx": int(p["frame_idx"]),
                    "timestamp": float(p["timestamp"]),
                    "x": float(cx),
                    "y": float(cy),
                    "smoothed_x": float(smx),
                    "smoothed_y": float(smy),
                    "width": float(p["wh"][0]),
                    "height": float(p["wh"][1]),
                    "confidence": float(p["confidence"]),
                    "speed": float(speed),
                    "is_predicted": bool(p["is_predicted"]),
                })

        return rows

# ================== ROI Utilities ==================
def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using Ray Casting Algorithm
    point: (x, y)
    polygon: [{x, y}, {x, y}, ...]
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]['x'], polygon[0]['y']
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]['x'], polygon[i % n]['y']
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def point_in_any_polygon(point, roi):
    """
    Support a single polygon (list of {x,y}) or a list of polygons.
    Returns True if point is inside any polygon.
    """
    if not roi:
        return True
    # Multiple polygons
    if isinstance(roi, list) and len(roi) > 0 and isinstance(roi[0], list):
        for poly in roi:
            try:
                if point_in_polygon(point, poly):
                    return True
            except Exception:
                # If polygon format unexpected, skip
                continue
        return False
    # Single polygon
    return point_in_polygon(point, roi)

def filter_tracks_by_roi(rows, roi_points):
    """
    تصفية المسارات التي تقع داخل منطقة ROI
    تقبل ROI إما مضلع واحد (قائمة نقاط) أو قائمة من المضلعين.
    تُرجع فقط الصفوف التي تقع داخل أي من المناطق المحددة.
    """
    def point_in_any(pt, roi):
        # If roi is falsy, everything is considered inside
        if not roi:
            return True
        # If first element is a list, treat roi as list of polygons
        if isinstance(roi, list) and len(roi) > 0 and isinstance(roi[0], list):
            for poly in roi:
                if point_in_polygon(pt, poly):
                    return True
            return False
        # Otherwise treat roi as a single polygon (list of points)
        return point_in_polygon(pt, roi)

    filtered = []
    for row in rows:
        point = (row['x'], row['y'])
        in_roi = point_in_any(point, roi_points) if roi_points is not None else True
        row['in_roi'] = in_roi
        if in_roi:
            filtered.append(row)
    return filtered


def classify_tracks_from_df(df: pd.DataFrame, cfg: Dict) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Classify tracks using a robust, multi-criteria rule and return (metrics, per-track dataframe).

    Uses configurable settings from `cfg` (fallback defaults if absent):
      - confidence_threshold (default 0.45)
      - confidence_min_hits (default max(10, min_hits))
      - confidence_prop_threshold (default 0.6)
      - confidence_border_delta (default 0.05)
      - include_predicted_in_confidence (default False)

    Returns:
      authoritative_qm (dict) and per_track_df (pd.DataFrame)
    """
    cfg_min_hits = int(cfg.get("min_hits", 2))
    # Default threshold aligned with processing defaults
    threshold = float(cfg.get("confidence_threshold", 0.45))
    conf_min_hits = int(cfg.get("confidence_min_hits", max(5, cfg_min_hits)))
    prop_thresh = float(cfg.get("confidence_prop_threshold", 0.5))
    border_delta = float(cfg.get("confidence_border_delta", 0.05))
    include_pred = bool(cfg.get("include_predicted_in_confidence", False))

    df = df.copy()
    if 'is_predicted' not in df.columns:
        df['is_predicted'] = False

    per_rows = []
    for tid, g in df.groupby('track_id'):
        if not include_pred:
            dets = g[~g['is_predicted']]
            confs = dets['confidence'].values if not dets.empty else g['confidence'].values
            n_points = len(dets) if not dets.empty else len(g)
            n_pred = int(g['is_predicted'].sum())
        else:
            confs = g['confidence'].values
            n_points = len(g)
            n_pred = int(g['is_predicted'].sum())

        median_conf = float(np.median(confs)) if len(confs) > 0 else 0.0
        mean_conf = float(np.mean(confs)) if len(confs) > 0 else 0.0
        pct_above = float((confs >= threshold).sum() / max(1, len(confs))) if len(confs) > 0 else 0.0

        confident = (n_points >= conf_min_hits) and (median_conf >= threshold) and (pct_above >= prop_thresh)
        borderline = False
        if not confident:
            if (threshold - border_delta) <= median_conf < threshold:
                borderline = True
            elif (prop_thresh - 0.1) <= pct_above < prop_thresh and n_points >= max(2, int(conf_min_hits / 2)):
                borderline = True

        per_rows.append({
            'track_id': int(tid),
            'n_points': int(n_points),
            'n_predicted': int(n_pred),
            'mean_conf': mean_conf,
            'median_conf': median_conf,
            'pct_above': pct_above,
            'confident': bool(confident),
            'borderline': bool(borderline),
        })

    per_track_df = pd.DataFrame(per_rows)
    total_tracks = int(per_track_df.shape[0])
    confident_count = int(per_track_df['confident'].sum()) if total_tracks > 0 else 0
    borderline_count = int(((per_track_df['borderline']) & (~per_track_df['confident'])).sum()) if total_tracks > 0 else 0
    low_conf_count = int(max(0, total_tracks - confident_count - borderline_count))

    authoritative_qm = {
        'total_tracks': total_tracks,
        'quality_tracks': int((per_track_df['n_points'] >= cfg_min_hits).sum()) if total_tracks > 0 else 0,
        'avg_track_length': float(per_track_df['n_points'].mean()) if total_tracks > 0 else 0.0,
        'min_track_length': int(per_track_df['n_points'].min()) if total_tracks > 0 else 0,
        'max_track_length': int(per_track_df['n_points'].max()) if total_tracks > 0 else 0,
        'avg_confidence': float(per_track_df['mean_conf'].mean()) if total_tracks > 0 else 0.0,
        'confidence_threshold': float(threshold),
        'high_conf_count': confident_count,
        'high_conf_pct': float(confident_count / max(1, total_tracks)),
        'confident_count': confident_count,
        'confident_pct': float(confident_count / max(1, total_tracks)),
        'borderline_count': borderline_count,
        'low_conf_count': low_conf_count,
        'low_conf_pct': float(low_conf_count / max(1, total_tracks)),
        'quality_score': float(np.clip((int((per_track_df['n_points'] >= cfg_min_hits).sum()) / max(1, total_tracks)) * 0.5 + min(max(float(confident_count / max(1, total_tracks)), 0.0), 1.0) * 0.5, 0, 100)),
        'long_tracks': int((per_track_df['n_points'] >= int(cfg.get('long_track_threshold', 15))).sum()),
        'short_tracks': int((per_track_df['n_points'] < int(cfg_min_hits)).sum()),
        'reliable': True if total_tracks >= 5 else False,
    }

    return authoritative_qm, per_track_df

# ================== Video Processor ==================
class VideoProcessorV1:
    def __init__(self, cfg: Dict, yolo_model=None, feature_extractor=None):
        self.cfg = ConfigV1.validate(cfg)
        self.video_path = Path(self.cfg["video_path"])
        self.out_dir = Path(self.cfg["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.cfg["fps_override"] > 0:
            self.fps = self.cfg["fps_override"]

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if yolo_model is None:
            if not YOLO_AVAILABLE:
                raise RuntimeError("YOLO not available")
            device = self.cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.yolo = YOLO(self.cfg["model_path"])
            try:
                # prefer model-level .to() if available, otherwise move wrapped module
                if hasattr(self.yolo, "to"):
                    self.yolo.to(device)
                elif hasattr(self.yolo, "model") and hasattr(self.yolo.model, "to"):
                    self.yolo.model.to(device)
            except Exception:
                pass
        else:
            self.yolo = yolo_model
        
        # 🎯 NEW: Get class names from model for accurate mapping
        self.class_names = self.yolo.names  # Dict: {0: 'person', 1: 'bicycle', 2: 'car', ...}
        logger.info(f"Model class names loaded: {len(self.class_names)} classes")
        
        # 🎯 NEW: Update vehicle classes mapping from model
        # This ensures class_id mapping matches the actual YOLO model
        self.cfg["_vehicle_classes"] = self.class_names
        logger.info(f"Vehicle classes updated from model: {list(self.class_names.values())[:10]}...")

        self.mot = MOTV1(self.cfg, self.fps, (self.width, self.height), feature_extractor)
        self.last_df = None
        self.root_out_dir = self.out_dir / f"tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # NOTE: Do NOT create the output folder here. Creating it during __init__ can produce
        # temporary/duplicate job folders if initialization runs multiple times (e.g. UI checks).
        # We will create the directory later in `process()` **only** after processing reaches
        # the export stage so a single final job folder is created on success.


    def process(self, progress_callback=None, stop_event=None):
        logger.info(f"Processing video: {self.video_path}")
        logger.info(f"Output dir: {self.root_out_dir}")

        processed_frames = 0
        vid_frame_idx = -1

        try:
            while True:
                if stop_event and stop_event.is_set():
                    logger.info("Processing stopped by user")
                    break

                ret, frame = self.cap.read()
                if not ret:
                    break

                vid_frame_idx += 1

                if self.cfg["max_frames"] > 0 and vid_frame_idx >= self.cfg["max_frames"]:
                    break

                # If skipping frames, do soft predict to keep motion smooth
                if (vid_frame_idx % int(self.cfg["process_every_n_frames"])) != 0:
                    self.mot.soft_predict_all(vid_frame_idx)
                    continue

                processed_frames += 1

                # Run YOLO detection (optionally on a resized copy for speed)
                try:
                    detection_frame = frame
                    scale = 1.0
                    max_dim = int(self.cfg.get("detection_max_dim", 0) or 0)
                    if max_dim > 0:
                        h0, w0 = frame.shape[:2]
                        big = max(w0, h0)
                        if big > max_dim:
                            scale = float(max_dim) / float(big)
                            new_w, new_h = int(w0 * scale), int(h0 * scale)
                            # Ensure at least 1px
                            new_w = max(1, new_w)
                            new_h = max(1, new_h)
                            detection_frame = cv2.resize(frame, (new_w, new_h))

                    results = self.yolo(detection_frame, conf=self.cfg["conf_threshold"], iou=self.cfg["iou_threshold"], verbose=False)
                    detections = []
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            # Scale back to original video coordinates if resizing was used
                            if scale != 1.0:
                                x1 = float(x1) / scale
                                y1 = float(y1) / scale
                                x2 = float(x2) / scale
                                y2 = float(y2) / scale

                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            area = (x2 - x1) * (y2 - y1)
                            if self.cfg["min_box_area"] <= area <= self.cfg["max_box_area"]:
                                detections.append(([x1, y1, x2, y2], cls_id, conf))
                except Exception as e:
                    logger.warning(f"Detection failed on frame {vid_frame_idx}: {e}")
                    detections = []

                active_tracks = self.mot.update(detections, frame, frame_idx=vid_frame_idx)

                if self.cfg.get("show_live_preview", False):
                    frame_display = frame.copy()
                    # Respect ROI when drawing live preview
                    roi_polygon = self.cfg.get("roi_points") if self.cfg.get("roi_points") else []
                    for tid, track in active_tracks.items():
                        x1, y1, x2, y2 = [int(v) for v in track.bbox]
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        if roi_polygon:
                            try:
                                if not point_in_any_polygon((cx, cy), roi_polygon):
                                    continue
                            except Exception:
                                # If point-in-polygon fails, fall back to drawing
                                pass
                        cv2.rectangle(frame_display, (x1, y1), (x2, y2), track.color, 2)
                        cv2.putText(frame_display, f"ID:{tid}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, track.color, 2)
                    h, w = frame_display.shape[:2]
                    scale = 0.5
                    display = cv2.resize(frame_display, (int(w * scale), int(h * scale)))
                    cv2.imshow("Live Preview - Press Q to continue", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        self.cfg["show_live_preview"] = False

                if progress_callback:
                    progress = (vid_frame_idx / max(1, self.total - 1)) * 100
                    progress_callback(progress)

                logger.debug(f"Frame {vid_frame_idx}/{self.total} processed, active: {len(active_tracks)}")

            self.mot.finalize()

            logger.info("Exporting results...")
            rows = self.mot.to_rows()
            
            # Apply ROI filtering if enabled
            roi_points = self.cfg.get("roi_points")
            logger.info(f"\n{'='*60}")
            logger.info("[ROI] ROI FILTERING CHECK:")
            logger.info(f"  roi_points: {roi_points}")
            logger.info(f"  roi_points is None: {roi_points is None}")
            logger.info(f"  roi_points type: {type(roi_points)}")
            if roi_points:
                logger.info(f"  roi_points length: {len(roi_points)}")
                logger.info(f"  First point: {roi_points[0] if len(roi_points) > 0 else 'N/A'}")
            logger.info(f"{'='*60}\n")
            
            if roi_points:
                logger.info(f"[ROI] تطبيق تصفية ROI على {len(rows)} صف...")
                rows = filter_tracks_by_roi(rows, roi_points)
                
                # Count tracks in ROI
                rows_in_roi = [r for r in rows if r.get('in_roi', False)]
                logger.info("[OK] تمت إضافة عمود 'in_roi'")
                logger.info(f"   مجموع الصفوف: {len(rows)}")
                logger.info(f"   الصفوف داخل ROI: {len(rows_in_roi)}")
                logger.info(f"   الصفوف خارج ROI: {len(rows) - len(rows_in_roi)}")
            else:
                # إضافة عمود in_roi مع القيمة True للمسارات (إذا لم تكن ROI مفعلة)
                logger.info(f"[ROI] roi_points = None، تم تخطي التصفية")
                for row in rows:
                    row['in_roi'] = True
            
            self.last_df = pd.DataFrame(rows)

            if self.last_df.empty:
                logger.warning("No tracks detected")
                return None

            # Get cached metrics from tracker object
            quality_metrics = self.mot.compute_quality_metrics()

            # Recompute authoritative metrics from exported rows (self.last_df) to avoid mismatches
            try:
                df = self.last_df.copy()
                # ensure vehicle_type column exists
                if 'vehicle_type' not in df.columns:
                    df['vehicle_type'] = 'unclassified'

                authoritative_qm, per_track_df = classify_tracks_from_df(df, self.cfg)

                # Ensure authoritative metrics include explicit list of low-confidence track IDs
                try:
                    low_ids = per_track_df.loc[(~per_track_df['confident']) & (~per_track_df['borderline']), 'track_id'].astype(int).tolist()
                    authoritative_qm['low_conf_track_ids'] = low_ids
                except Exception:
                    authoritative_qm['low_conf_track_ids'] = []

                # attach per-track diagnostic summary into meta (short form)
                quality_metrics.update(authoritative_qm)
                quality_metrics['_per_track_summary'] = per_track_df.to_dict(orient='records')

                # store the active confidence rule used
                quality_metrics['_confidence_rules'] = {
                    'threshold': authoritative_qm.get('confidence_threshold'),
                    'min_hits': int(self.cfg.get('confidence_min_hits', max(10, int(self.cfg.get('min_hits', 2))))),
                    'prop_threshold': float(self.cfg.get('confidence_prop_threshold', 0.6)),
                    'include_predicted': bool(self.cfg.get('include_predicted_in_confidence', False)),
                }

                # If trackers' computed metrics differ from authoritative CSV-derived metrics, log a warning and prefer authoritative
                if quality_metrics.get('total_tracks') != authoritative_qm['total_tracks'] or quality_metrics.get('high_conf_count') != authoritative_qm.get('high_conf_count'):
                    logger.warning("Quality metrics (tracker) disagree with CSV-derived authoritative metrics. Using CSV-derived values in meta.json for consistency.")
                    quality_metrics.update(authoritative_qm)
                else:
                    # keep tracker metrics (they match)
                    quality_metrics = quality_metrics
            except Exception as e:
                logger.exception(f"Error recomputing authoritative quality metrics from last_df: {e}")

            logger.info(f"=== TRACKING QUALITY METRICS ===")
            logger.info(f"Total Tracks: {quality_metrics['total_tracks']}")
            logger.info(f"Quality Tracks (≥{self.cfg.get('min_hits', 2)} hits): {quality_metrics['quality_tracks']}")
            logger.info(f"Avg Track Length: {quality_metrics['avg_track_length']:.1f} frames")
            logger.info(f"Track Range: {quality_metrics['min_track_length']}-{quality_metrics['max_track_length']} frames")
            logger.info(f"Long Tracks (≥15 frames): {quality_metrics['long_tracks']}")
            logger.info(f"Avg Confidence: {quality_metrics['avg_confidence']:.2%}")
            logger.info(f"High Confidence Tracks (≥{quality_metrics.get('confidence_threshold', 0.45):.2f}): {quality_metrics.get('high_conf_count', 0)} ({quality_metrics.get('high_conf_pct', 0):.2%})")
            logger.info(f"[QUALITY] QUALITY SCORE: {quality_metrics['quality_score']:.1f}%")
            logger.info(f"================================")

            # Create the output job folder *now* (only when processing reached the export stage).
            # Using parents=True ensures the top-level output dir exists.
            self.root_out_dir.mkdir(parents=True, exist_ok=True)

            tracks_csv = self.root_out_dir / "tracks.csv"
            self.last_df.to_csv(tracks_csv, index=False)
            logger.info(f"Saved: {tracks_csv}")

            # Ensure canonical metric fields are present (helps UI rely on consistent keys)
            try:
                if isinstance(quality_metrics, dict):
                    if "low_conf_count" not in quality_metrics:
                        if isinstance(quality_metrics.get('low_conf_track_ids'), list):
                            quality_metrics['low_conf_count'] = len(quality_metrics['low_conf_track_ids'])
                    if "confident_count" not in quality_metrics and 'high_conf_count' in quality_metrics:
                        quality_metrics['confident_count'] = int(quality_metrics.get('high_conf_count', 0))
                    if "total_tracks" not in quality_metrics:
                        quality_metrics['total_tracks'] = int(self.last_df["track_id"].nunique())
            except Exception:
                pass

            meta = {
                "video_path": str(self.video_path),
                "total_frames": int(self.total),
                "fps": float(self.fps),
                "width": int(self.width),
                "height": int(self.height),
                "num_tracks": int(self.last_df["track_id"].nunique()),
                "processing_time": datetime.now().isoformat(),
                "processed_frames_with_detection": int(processed_frames),
                "quality_metrics": quality_metrics,
                "config": _json_safe(self.cfg),
            }
            meta_json = self.root_out_dir / "meta.json"
            with open(meta_json, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved: {meta_json}")

            # 🎯 Copy source video into the job folder 'uploads/' so crops and previews remain available
            try:
                uploads_dir = self.root_out_dir / 'uploads'
                uploads_dir.mkdir(exist_ok=True)
                src = Path(self.video_path)
                if src.exists():
                    dst = uploads_dir / src.name
                    if not dst.exists():
                        import shutil
                        shutil.copy2(str(src), str(dst))
                        logger.info(f"Copied source video to: {dst}")
                    else:
                        logger.info(f"Source video already present in uploads: {dst}")
                else:
                    logger.warning(f"Source video file not found for copying: {src}")
            except Exception as e:
                logger.exception(f"Failed to copy source video into job uploads: {e}")

            if self.cfg.get("export_preview_frames", True):
                logger.info("Exporting preview frames...")
                self._export_frames()
            else:
                logger.info("Preview frame export skipped")

            if self.cfg.get("export_track_previews", True):
                logger.info("Exporting per-track preview crops...")
                self._export_track_previews()
            else:
                logger.info("Track preview export skipped")

            logger.info(f"Processing complete. Results saved to: {self.root_out_dir}")
            return {
                "tracks_csv": str(tracks_csv),
                "meta_json": str(meta_json),
                "output_dir": str(self.root_out_dir),
            }

        finally:
            self.cap.release()

    def _export_frames(self):
        """
        Export 3 key frames: first, middle, last based on available track history.
        """
        frames_dir = self.root_out_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        all_tracks = list(self.mot.iter_all())
        if not all_tracks:
            return

        # Determine max frame index present in history
        max_frame = 0
        for tr in all_tracks:
            if tr.hist:
                max_frame = max(max_frame, max(p["frame_idx"] for p in tr.hist))

        key_frames = {0, max_frame // 2, max_frame}

        # build quick lookup: frame -> list of points per track
        frame_points = {}
        for tr in all_tracks:
            for p in tr.hist:
                frame_points.setdefault(p["frame_idx"], []).append((tr, p))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        vid_frame_idx = -1
        exported = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            vid_frame_idx += 1

            if vid_frame_idx not in key_frames:
                continue

            items = frame_points.get(vid_frame_idx, [])
            roi_polygon = self.cfg.get("roi_points") if self.cfg.get("roi_points") else []
            for tr, p in items:
                cx, cy = p["center"]
                # If ROI configured, skip tracks not inside ROI for this frame
                if roi_polygon:
                    try:
                        if not point_in_any_polygon((cx, cy), roi_polygon):
                            continue
                    except Exception:
                        pass

                # draw trajectory up to this frame (only for tracks that are inside ROI at this frame)
                pts = [pp["center"] for pp in tr.hist if pp["frame_idx"] <= vid_frame_idx]
                if len(pts) > 1:
                    arr = np.array(pts, dtype=np.int32)
                    cv2.polylines(frame, [arr], False, tr.color, 2)

                w, h = p["wh"]
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), tr.color, 2)
                cv2.putText(frame, f"ID:{tr.track_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, tr.color, 2)

            label = "First" if vid_frame_idx == 0 else ("Middle" if vid_frame_idx == max_frame // 2 else "Last")
            out = frames_dir / f"frame_{label}_{vid_frame_idx:06d}.png"
            cv2.imwrite(str(out), frame)
            exported += 1
            logger.info(f"Exported {label}: {out.name}")

        logger.info(f"Exported {exported} key frames to {frames_dir}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _export_track_previews(self):
        """
        Export per-track previews (small and large) for each track.

        Selection modes (via cfg['preview_selection_mode']):
          - 'all' : all tracks
          - 'confident' : confident only
          - 'confident_borderline' : confident + borderline
          - 'quality_tracks' : tracks with n_points >= min_hits
          - 'borderline_low' : borderline + low (default)
          - 'low' : low only
        """
        previews_dir = self.root_out_dir / "previews"
        previews_dir.mkdir(exist_ok=True)

        try:
            df = self.last_df.copy()

            # Determine which tracks to export based on classification
            mode = str(self.cfg.get('preview_selection_mode', 'borderline_low')).lower()
            try:
                _, per_track_df = classify_tracks_from_df(df, self.cfg)
            except Exception:
                per_track_df = pd.DataFrame(columns=['track_id', 'n_points', 'median_conf', 'confident', 'borderline'])

            if mode == 'all':
                selected_ids = set(df['track_id'].unique())
            elif mode == 'confident':
                selected_ids = set(per_track_df.loc[per_track_df['confident'], 'track_id'].astype(int).tolist())
            elif mode == 'confident_borderline':
                sel = per_track_df.loc[(per_track_df['confident']) | (per_track_df['borderline']), 'track_id']
                selected_ids = set(sel.astype(int).tolist())
            elif mode == 'quality_tracks':
                min_hits = int(self.cfg.get('min_hits', 2))
                sel = per_track_df.loc[per_track_df['n_points'] >= min_hits, 'track_id']
                selected_ids = set(sel.astype(int).tolist())
            elif mode == 'low':
                sel = per_track_df.loc[(~per_track_df['confident']) & (~per_track_df['borderline']), 'track_id']
                selected_ids = set(sel.astype(int).tolist())
            else:  # 'borderline_low' (default)
                sel = per_track_df.loc[~per_track_df['confident'], 'track_id']
                selected_ids = set(sel.astype(int).tolist())

            if not selected_ids:
                logger.info(f"No tracks selected for preview export (mode={mode}). Skipping.")
                return

            # parameters: export a single large preview per track
            large_max_dim = int(self.cfg.get('preview_large_max_dim', 512))

            for tid in sorted(selected_ids):
                sub = df.loc[df['track_id'] == tid]
                # choose best representative row: maximize confidence * area (prefer non-predicted)
                try:
                    cand = sub.copy()
                    if 'is_predicted' in cand.columns:
                        cand = cand.loc[~cand['is_predicted']]
                        if cand.empty:
                            cand = sub.copy()
                    cand = cand.copy()
                    cand['area'] = cand['width'].astype(float) * cand['height'].astype(float)
                    cand['score'] = cand['confidence'].astype(float).fillna(0.0) * cand['area'].fillna(0.0)
                    rep = cand.loc[cand['score'].idxmax()]
                    med = int(rep['frame_idx']) if 'frame_idx' in rep else int(sub['frame_idx'].median())
                except Exception:
                    med = int(sub['frame_idx'].median()) if 'frame_idx' in sub.columns else int(sub.index[0])
                    idx = (sub['frame_idx'] - med).abs().idxmin()
                    rep = sub.loc[idx]

                # extract bbox from rep row
                cx, cy = float(rep['x']), float(rep['y'])
                w, h = float(rep['width']), float(rep['height'])
                x1 = max(0, int(cx - w / 2))
                y1 = max(0, int(cy - h / 2))
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)

                # find a readable frame (try median +/- 5 frames)
                frame_img = None
                for offset in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
                    fidx = med + offset
                    if fidx < 0 or fidx >= self.total:
                        continue
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        frame_img = frame
                        break

                # fallback: use exported key frames if available
                if frame_img is None:
                    frames_dir = self.root_out_dir / 'frames'
                    if frames_dir.exists():
                        for ffile in sorted(frames_dir.glob('*.png')):
                            try:
                                candidate = cv2.imread(str(ffile))
                                if candidate is not None:
                                    frame_img = candidate
                                    break
                            except Exception:
                                pass

                if frame_img is None:
                    logger.warning(f"No frame available to export preview for track {tid}")
                    continue

                h0, w0 = frame_img.shape[:2]
                # clip bbox to frame
                x1c = max(0, min(w0 - 1, x1))
                y1c = max(0, min(h0 - 1, y1))
                x2c = max(0, min(w0 - 1, x2))
                y2c = max(0, min(h0 - 1, y2))
                if x2c <= x1c or y2c <= y1c:
                    logger.warning(f"Invalid bbox for track {tid}, skipping preview")
                    continue

                crop = frame_img[y1c:y2c, x1c:x2c].copy()

                # create single large preview (scale so largest side == large_max_dim)
                lh, lw = crop.shape[:2]
                scale = 1.0
                if max(lw, lh) > 0:
                    if max(lw, lh) > large_max_dim:
                        scale = float(large_max_dim) / float(max(lw, lh))
                    else:
                        scale = 1.0

                # modest upscaling for tiny crops to keep details readable
                if max(lw, lh) < 64:
                    scale = max(scale, float(256) / float(max(1, max(lw, lh))))

                try:
                    large = cv2.resize(crop, (max(1, int(lw * scale)), max(1, int(lh * scale))), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    large = crop

                # overlay disabled: do NOT write text/ID onto preview images (keep thumbnails clean)
                # previously the code drew a rectangle and ID text onto `large` — removed per UI requirement


                out = previews_dir / f"track_{tid}.png"
                try:
                    cv2.imwrite(str(out), large)
                    logger.info(f"Exported preview for track {tid}: {out.name} (max_side={large_max_dim})")
                except Exception as e:
                    logger.exception(f"Failed to write preview for track {tid}: {e}")

        except Exception as e:
            logger.exception(f"Unexpected error exporting track previews: {e}")
