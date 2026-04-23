"""
State Machine Definitions for Active Perception System.

States:
- MonitorState: Watch the scene and perform smooth tracking (Nudge).
- ExploreExposureState: Sweep exposure levels to adapt to lighting changes.
- ExploreZoomState: Sweep zoom levels to adapt to target size changes.
- SniperRecoveryState: Fallback to 1.0x to quickly locate a lost target.
"""

import time

import cv2
import numpy as np

def _aruco_centroid_pixel(corners) -> tuple:
    """Mean (x, y) of the first marker's 4 corners in pixel coords."""
    if corners is None or len(corners) < 1:
        return None
    pts = corners[0][0]
    mx = float(np.mean(pts[:, 0]))
    my = float(np.mean(pts[:, 1]))
    return mx, my


def _use_face_exposure_scoring(context) -> bool:
    """Face-style exposure sweep scoring (LBPH + ROI) when tracking a face."""
    if getattr(context, "perception_mode", "") == "face":
        return True
    if getattr(context, "perception_mode", "") != "mixed":
        return False
    return getattr(getattr(context, "perception", None), "active_backend", None) == "face"


def _face_roi_brightness_penalty(frame, corners) -> float:
    """Penalty 0..~0.25 when primary face ROI is very bright (Session 26 exposure sweep)."""
    if frame is None or corners is None or len(corners) < 1:
        return 0.0
    poly = corners[0][0].astype(np.int32)
    x, y, bw, bh = cv2.boundingRect(poly)
    h0, w0 = frame.shape[:2]
    x = max(0, min(x, w0 - 1))
    y = max(0, min(y, h0 - 1))
    bw = max(1, min(bw, w0 - x))
    bh = max(1, min(bh, h0 - y))
    roi = frame[y : y + bh, x : x + bw]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = float(np.mean(hsv[:, :, 2]))
    if v <= 170.0:
        return 0.0
    return min(0.35, (v - 170.0) / 85.0 * 0.28)


def _face_exposure_composite_score(context, raw_u, frame, corners, detected) -> float:
    """
    Face mode: add LBPH distance + ROI highlight terms on top of uncertainty `raw_u`
    so exposure winners track recognition-friendly lighting, not only geometry.
    """
    if not _use_face_exposure_scoring(context):
        return float(raw_u)
    score = float(raw_u)
    w_b = float(getattr(context, "face_exposure_brightness_weight", 0.0))
    w_l = float(getattr(context, "face_exposure_lbph_weight", 0.0))
    if detected and corners is not None and len(corners) > 0 and w_b > 0.0:
        score += w_b * _face_roi_brightness_penalty(frame, corners)
    if w_l > 0.0:
        pc = getattr(context.perception, "primary_confidence", None)
        th = float(getattr(context.perception, "match_threshold", 85.0) or 85.0)
        if pc is not None and th > 1e-6:
            score += w_l * min(1.0, float(pc) / th)
    return min(1.5, score)


def _log_event(context, event_type: str, frame=None, screenshot_name=None, **payload) -> None:
    logger = getattr(context, "blackbox", None)
    if logger is not None:
        logger.log_event(event_type, frame=frame, screenshot_name=screenshot_name, **payload)


def _gesture_quiet_auto_explore(context) -> bool:
    """
    True while MediaPipe sees a hand (``gesture_hands_for_explore_quiet``).

    Suppresses entering auto exposure / zoom exploration from MONITOR and
    aborts in-progress sweeps so gesture demos are not visually interrupted.
    """
    return bool(getattr(context, "gesture_hands_for_explore_quiet", False))


class State:
    """Base State Class."""
    def __init__(self):
        self.name = "BASE"
        
    def on_enter(self, context):
        pass
        
    def update(self, context, frame, detected, corners, ids, smooth_u, raw_u, metrics, current_brightness, size_value):
        """
        Execute state logic for the current frame.
        Must return the next State object (or self if staying in the same state).
        """
        return self
        
    def on_exit(self, context):
        pass


class MonitorState(State):
    """
    Normal monitoring and smooth visual servoing (Nudge).
    Decides when to trigger explorations or recoveries.
    """
    def __init__(self):
        super().__init__()
        self.name = "MONITOR"
        self.roi_lost_frames = 0
        self.roi_lost_threshold = 8
        self.explore_enter_threshold = 0.60
        self.explore_exit_threshold = 0.50
        self.zoom_enter_threshold = 0.55
        self.zoom_exit_threshold = 0.45
        self.explore_trigger_frames = 2
        self.zoom_trigger_frames = 2
        self._explore_trigger_count = 0
        self._zoom_trigger_count = 0
        self._pt_lost_frames = 0

    def on_enter(self, context):
        self.roi_lost_threshold = context.monitor_roi_lost_threshold
        self.explore_enter_threshold = context.monitor_explore_enter_threshold
        self.explore_exit_threshold = context.monitor_explore_exit_threshold
        self.zoom_enter_threshold = context.monitor_zoom_enter_threshold
        self.zoom_exit_threshold = context.monitor_zoom_exit_threshold
        self.explore_trigger_frames = context.monitor_explore_trigger_frames
        self.zoom_trigger_frames = context.monitor_zoom_trigger_frames

    def update(self, context, frame, detected, corners, ids, smooth_u, raw_u, metrics, current_brightness, size_value):
        # 1. Stabilization Check (Cooldown)
        if context.frame_count < context.ignore_until_frame:
            return self
            
        # For monitor logic, we use the smoothed uncertainty to avoid jitter
        uncertainty = smooth_u

        # 2. Update Baseline Brightness
        if context.baseline_brightness is None:
            context.baseline_brightness = current_brightness
            print(f"[i] Baseline Brightness Set: {context.baseline_brightness:.1f}")

        # 3. Check Lighting Change
        env_changed = False
        if context.baseline_brightness is not None:
            diff = abs(current_brightness - context.baseline_brightness)
            dynamic_threshold = max(context.baseline_brightness * context.brightness_change_ratio, 5.0)
            if diff > dynamic_threshold:
                env_changed = True
                print(f"[!] Lighting Changed! Diff: {diff:.1f} baseline: {context.baseline_brightness:.1f} (Thresh: {dynamic_threshold:.1f})")

        # 4. Trigger EXPOSURE EXPLORE with hysteresis + consecutive-frame gating
        if context.enable_exposure_control and context.policy.exposure_supported:
            if _gesture_quiet_auto_explore(context):
                self._explore_trigger_count = 0
            elif env_changed and uncertainty >= self.explore_enter_threshold:
                self._explore_trigger_count += 1
            elif (not env_changed) or (uncertainty <= self.explore_exit_threshold):
                self._explore_trigger_count = 0

            if (
                self._explore_trigger_count >= self.explore_trigger_frames
                and not _gesture_quiet_auto_explore(context)
            ):
                print(f"[!] Triggering EXPLORE (Score: {uncertainty:.2f})")
                context.baseline_brightness = None
                self._explore_trigger_count = 0
                _log_event(
                    context,
                    "explore_exposure_triggered",
                    frame_idx=context.frame_count,
                    reason="env_changed_and_high_uncertainty",
                    smooth_u=uncertainty,
                )
                return ExploreExposureState()
        else:
            self._explore_trigger_count = 0

        # 5. Check Target Size Change and Quality
        if _gesture_quiet_auto_explore(context):
            self._zoom_trigger_count = 0
        if context.enable_zoom_control and context.frame_count >= context.zoom_ignore_until_frame:
            if context.baseline_size is None and context.confirmed_detected:
                context.baseline_size = size_value
                print(f"[i] Baseline Size Set: {context.baseline_size:.1f}")
            
            size_changed = False
            if context.baseline_size is not None:
                if context.confirmed_detected:
                    size_diff = abs(size_value - context.baseline_size)
                    size_threshold = max(context.baseline_size * context.size_change_ratio, context.size_change_floor)
                    if size_diff > size_threshold:
                        size_changed = True
                        print(f"[!] Size Changed! Diff: {size_diff:.1f} baseline: {context.baseline_size:.1f} (Thresh: {size_threshold:.1f})")
                else:
                    if context.confirmed_lost and context.policy.current_zoom_level <= 1.0:
                        size_changed = True
                        print("[!] Target lost at 1.0x! Triggering Zoom Search.")
            
            poor_quality_at_base = (
                context.confirmed_detected
                and (uncertainty >= self.zoom_enter_threshold)
                and (context.policy.current_zoom_level <= 1.0)
                and (not context.zoom_initialized)
            )
            if poor_quality_at_base:
                self._zoom_trigger_count += 1
            elif uncertainty <= self.zoom_exit_threshold or context.confirmed_lost or context.policy.current_zoom_level > 1.0:
                self._zoom_trigger_count = 0

            if self._zoom_trigger_count >= self.zoom_trigger_frames:
                print(f"[!] Poor quality at base (Score: {uncertainty:.2f}). Forcing Zoom Search.")
                self._zoom_trigger_count = 0
                poor_quality_at_base = True
            else:
                poor_quality_at_base = False
            
            # 6. Trigger ZOOM EXPLORE
            if (
                (not context.zoom_initialized) or size_changed or poor_quality_at_base
            ) and not _gesture_quiet_auto_explore(context):
                print(f"[!] Triggering ZOOM EXPLORE (Score: {uncertainty:.2f})")
                context.baseline_size = None
                _log_event(
                    context,
                    "explore_zoom_triggered",
                    frame_idx=context.frame_count,
                    reason=(
                        "zoom_uninitialized" if not context.zoom_initialized
                        else "size_changed" if size_changed
                        else "poor_quality_at_base"
                    ),
                    smooth_u=uncertainty,
                    size_value=size_value,
                )
                return ExploreZoomState()
        else:
            self._zoom_trigger_count = 0

        # 7. Zoomed-in Logic: Nudge Tracking or Sniper Recovery
        if context.enable_zoom_control and context.policy.current_zoom_level > 1.0:
            if context.confirmed_lost:
                self.roi_lost_frames += 1
            elif context.confirmed_detected:
                self.roi_lost_frames = 0
                
            # Trigger Sniper Recovery if lost for too long
            if self.roi_lost_frames >= self.roi_lost_threshold:
                if context.frame_count >= context.ignore_until_frame and context.frame_count >= context.zoom_ignore_until_frame:
                    _log_event(
                        context,
                        "sniper_recovery_triggered",
                        frame_idx=context.frame_count,
                        reason="confirmed_lost_while_zoomed",
                        zoom=float(context.policy.current_zoom_level),
                    )
                    return SniperRecoveryState()
                    
            # Smooth Digital Visual Servoing (Nudge ROI)
            if (
                detected
                and corners is not None
                and context.frame_count >= context.ignore_until_frame
                and context.frame_count >= context.zoom_ignore_until_frame
                and not getattr(context, "gesture_pt_suppress", False)
            ):
                cent = _aruco_centroid_pixel(corners)
                if cent is not None:
                    mx, my = cent
                    hh, ww = frame.shape[:2]
                    tnx, tny = context.policy.marker_center_to_full_norm(mx, my, ww, hh)
                    context.policy.nudge_roi_towards(tnx, tny, gain=context.monitor_nudge_gain)

        # 8. Physical Visual Servoing (Pan-Tilt)
        if context.enable_pan_tilt and context.pan_tilt is not None and not getattr(
            context, "gesture_pt_suppress", False
        ):
            if detected and corners is not None:
                self._pt_lost_frames = 0
                pose = context.pan_tilt.current_pose
                context.last_seen_pan = pose.pan
                context.last_seen_tilt = pose.tilt
                cent = _aruco_centroid_pixel(corners)
                if cent is not None:
                    mx, my = cent
                    hh, ww = frame.shape[:2]
                    err_x = (mx / ww) - 0.5
                    err_y = (my / hh) - 0.5
                    deadzone = context.pan_tilt_deadzone
                    if abs(err_x) > deadzone or abs(err_y) > deadzone:
                        delta_pan = -err_x * context.pan_tilt_gain_pan
                        delta_tilt = -err_y * context.pan_tilt_gain_tilt
                        context.pan_tilt.nudge(delta_pan, delta_tilt, settle_s=0.02)
            else:
                self._pt_lost_frames += 1

            # 9. Trigger Physical Search when target lost at base zoom
            if self._pt_lost_frames >= context.pan_tilt_search_lost_threshold:
                if context.policy.current_zoom_level <= 1.0:
                    print(f"[!] Target lost for {self._pt_lost_frames} frames with pan-tilt. Starting Physical Search.")
                    _log_event(
                        context,
                        "physical_search_triggered",
                        frame_idx=context.frame_count,
                        reason="lost_at_base_zoom_with_pan_tilt",
                    )
                    return PhysicalSearchState()
                    
        return self


class ExploreExposureState(State):
    """
    Sweeps through all available exposure levels to find the best one.
    """
    def __init__(self):
        super().__init__()
        self.name = "EXPLORE"
        self.explore_step = 0
        self.exploration_results = {}
        self.settle_frames = 2
        self.sample_frames = 3
        self._settle_count = 0
        self._sample_scores = []
        self._exp_sequence = []

    def on_enter(self, context):
        print(f"[i] {self.name}: Initializing sweep, setting index 0.")
        self.settle_frames = context.exposure_settle_frames
        self.sample_frames = context.exposure_sample_frames
        n = len(context.policy.exposure_levels)
        seq = getattr(context, "face_exposure_indices", None)
        if seq is not None and _use_face_exposure_scoring(context):
            self._exp_sequence = [i for i in sorted(set(seq)) if 0 <= i < n]
            if not self._exp_sequence:
                self._exp_sequence = list(range(n))
        else:
            self._exp_sequence = list(range(n))
        self.explore_step = 0
        context.policy.execute_exposure(self._exp_sequence[0])
        self._settle_count = 0
        self._sample_scores = []

    def update(self, context, frame, detected, corners, ids, smooth_u, raw_u, metrics, current_brightness, size_value):
        if _gesture_quiet_auto_explore(context):
            try:
                context.policy.execute_exposure(int(context.current_exposure_idx))
            except Exception:
                pass
            context.ignore_until_frame = context.frame_count + 8
            _log_event(
                context,
                "explore_exposure_aborted",
                frame_idx=context.frame_count,
                reason="gesture_hands",
            )
            return MonitorState()

        if self._settle_count < self.settle_frames:
            self._settle_count += 1
            return self

        current_idx = self._exp_sequence[self.explore_step]
        sample_u = _face_exposure_composite_score(
            context, raw_u, frame, corners, detected
        )
        self._sample_scores.append(sample_u)

        if len(self._sample_scores) < self.sample_frames:
            return self

        avg_score = float(np.mean(self._sample_scores))
        self.exploration_results[current_idx] = avg_score
        print(
            f"   -> Testing Exp Level {current_idx}: "
            f"Avg Score {avg_score:.2f} from {self.sample_frames} frames"
        )
        
        self.explore_step += 1
        
        if self.explore_step >= len(self._exp_sequence):
            self._apply_best_action(context)
            return MonitorState()
        else:
            context.policy.execute_exposure(self._exp_sequence[self.explore_step])
            self._settle_count = 0
            self._sample_scores = []
            
        return self
        
    def _apply_best_action(self, context):
        if not self.exploration_results: return
        min_score = min(self.exploration_results.values())
        candidates = [idx for idx, score in self.exploration_results.items() if abs(score - min_score) < 0.01]
        
        # Among tied minimum-uncertainty exposures, bias toward a "reasonable" hardware value.
        # Markers: middle of the list (-4) worked well in tuning.
        # Faces: bias slightly shorter exposure (more negative OpenCV log scale on this camera)
        # to reduce skin blow-out, which wrecks Haar/LBPH vs enrollment lighting.
        preferred_exposure_val = float(
            getattr(context, "exposure_tiebreak_preferred_val", -4.0)
        )
        best_idx = min(
            candidates, 
            key=lambda idx: (
                abs(context.policy.exposure_levels[idx] - preferred_exposure_val), 
                context.policy.exposure_levels[idx]
            )
        )
        
        best_score = self.exploration_results[best_idx]
        print(f"\n[V] Exploration Done. Winner: Level {best_idx} (Score {best_score:.2f})")
        _log_event(
            context,
            "exposure_winner_selected",
            frame_idx=context.frame_count,
            winner_index=best_idx,
            winner_exposure=context.policy.exposure_levels[best_idx],
            score=best_score,
            results=self.exploration_results,
        )
        
        context.policy.execute_exposure(best_idx)
        context.current_exposure_idx = best_idx
        
        context.baseline_brightness = None
        context.baseline_size = None
        context.ignore_until_frame = context.frame_count + 10


class ExploreZoomState(State):
    """
    Sweeps through digital zoom levels to find the one with lowest uncertainty.
    Usually triggers ExploreExposureState afterwards to re-optimize lighting.
    """
    def __init__(self):
        super().__init__()
        self.name = "EXPLORE_ZOOM"
        self.zoom_step = 0
        self.zoom_exploration_results = {}
        self.settle_frames = 1
        self.sample_frames = 2
        self._settle_count = 0
        self._sample_scores = []
        self._target_center = None
        
    def on_enter(self, context):
        print(f"[i] {self.name}: Initializing sweep, setting zoom index 0.")
        self.settle_frames = context.zoom_settle_frames
        self.sample_frames = context.zoom_sample_frames
        context.policy.set_zoom(context.policy.zoom_levels[0])
        self._settle_count = 0
        self._sample_scores = []
        self._target_center = None

    def update(self, context, frame, detected, corners, ids, smooth_u, raw_u, metrics, current_brightness, size_value):
        if _gesture_quiet_auto_explore(context):
            lvls = context.policy.zoom_levels
            cur = float(context.policy.current_zoom_level)
            try:
                context.current_zoom_idx = lvls.index(cur)
            except ValueError:
                context.current_zoom_idx = 0
                context.policy.set_zoom(lvls[0])
            context.zoom_initialized = True
            context.zoom_ignore_until_frame = context.frame_count + 15
            _log_event(
                context,
                "explore_zoom_aborted",
                frame_idx=context.frame_count,
                reason="gesture_hands",
                zoom=float(context.policy.current_zoom_level),
            )
            return MonitorState()

        if context.confirmed_detected and detected and corners is not None:
            cent = _aruco_centroid_pixel(corners)
            if cent is not None:
                mx, my = cent
                hh, ww = frame.shape[:2]
                self._target_center = context.policy.marker_center_to_full_norm(mx, my, ww, hh)

        if self._settle_count < self.settle_frames:
            self._settle_count += 1
            return self

        current_idx = self.zoom_step
        self._sample_scores.append(raw_u)

        if len(self._sample_scores) < self.sample_frames:
            return self

        avg_score = float(np.mean(self._sample_scores))
        self.zoom_exploration_results[current_idx] = avg_score
        print(
            f"   -> Testing Zoom Level {current_idx}: "
            f"Avg Score {avg_score:.2f} from {self.sample_frames} frames"
        )
        
        self.zoom_step += 1
        
        if self.zoom_step >= len(context.policy.zoom_levels):
            # Calculate the winner
            self._apply_best_zoom(context)
            
            if context.enable_exposure_control and context.policy.exposure_supported:
                print("[!] Triggering EXPLORE (Exposure) after ZOOM selection")
                return ExploreExposureState()
            else:
                return MonitorState()
        else:
            next_zoom = context.policy.zoom_levels[self.zoom_step]
            if self._target_center is not None:
                context.policy.set_roi_center(*self._target_center, zoom_level=next_zoom)
            context.policy.set_zoom(next_zoom)
            self._settle_count = 0
            self._sample_scores = []
            
        return self
        
    def _apply_best_zoom(self, context):
        if not self.zoom_exploration_results: return
        min_score = min(self.zoom_exploration_results.values())
        candidates = [idx for idx, score in self.zoom_exploration_results.items() if abs(score - min_score) < 0.01]
        
        best_idx = min(candidates)
        best_score = self.zoom_exploration_results[best_idx]
        
        print(f"\n[V] Zoom Exploration Done. Winner: Level {best_idx} (Score {best_score:.2f})")
        _log_event(
            context,
            "zoom_winner_selected",
            frame_idx=context.frame_count,
            winner_index=best_idx,
            winner_zoom=context.policy.zoom_levels[best_idx],
            score=best_score,
            results=self.zoom_exploration_results,
        )
        
        best_zoom = context.policy.zoom_levels[best_idx]
        if self._target_center is not None:
            context.policy.set_roi_center(*self._target_center, zoom_level=best_zoom)
        
        # We MUST call set_zoom here! If we transition to MonitorState, MonitorState
        # doesn't set zoom on entry, it just reads it. 
        # If we transition to ExploreExposureState, it changes exposure but leaves zoom alone.
        # So we have to apply the winning zoom right now.
        context.policy.set_zoom(best_zoom)
        context.current_zoom_idx = best_idx
        context.zoom_initialized = True
        
        context.baseline_size = None
        context.zoom_ignore_until_frame = context.frame_count + 5


class SniperRecoveryState(State):
    """
    Quickly zooms out to 1.0x to find a lost target.
    If found, re-centers and triggers zoom exploration.
    If timeout, falls back to PhysicalSearchState.
    """
    def __init__(self):
        super().__init__()
        self.name = "SNIPER"
        self.sniper_frame_count = 0
        self.sniper_timeout_frames = 60
        
    def on_enter(self, context):
        self.sniper_timeout_frames = context.sniper_timeout_frames
        print(f"[i] Sniper Recovery started! Target lost at {context.policy.current_zoom_level}x zoom.")
        _log_event(
            context,
            "sniper_recovery_started",
            frame_idx=context.frame_count,
            zoom=float(context.policy.current_zoom_level),
        )
        context.policy.set_zoom(1.0)

    def update(self, context, frame, detected, corners, ids, smooth_u, raw_u, metrics, current_brightness, size_value):
        self.sniper_frame_count += 1
        
        if detected and corners is not None:
            cent = _aruco_centroid_pixel(corners)
            if cent is not None:
                mx, my = cent
                hh, ww = frame.shape[:2]
                tnx, tny = context.policy.marker_center_to_full_norm(mx, my, ww, hh)
                context.policy.set_roi_center(tnx, tny)
                
                print("[V] Sniper Locked! Target found at 1.0x. Triggering re-optimization.")
                _log_event(
                    context,
                    "sniper_recovery_locked",
                    frame=frame,
                    screenshot_name=f"sniper_lock_f{context.frame_count}",
                    frame_idx=context.frame_count,
                )
                context.zoom_initialized = False
                context.baseline_size = None
                context.current_zoom_idx = 0
                context.ignore_until_frame = context.frame_count + 5
                context.zoom_ignore_until_frame = context.frame_count + 5
                
                return MonitorState()
                
        if self.sniper_frame_count >= self.sniper_timeout_frames:
            print("[!] Sniper Timeout! Target completely lost. Entering Physical Search.")
            _log_event(
                context,
                "sniper_timeout",
                frame=frame,
                screenshot_name=f"sniper_timeout_f{context.frame_count}",
                frame_idx=context.frame_count,
                timeout_frames=self.sniper_timeout_frames,
            )
            context.zoom_initialized = False
            context.baseline_size = None
            context.current_zoom_idx = 0
            return PhysicalSearchState()
            
        return self


class PhysicalSearchState(State):
    """
    Two-phase physical search:

    - Phase A (Local): 3x3 grid around the last known target position.
      Exploits the fact that a just-lost target is most likely very close to
      where it was seen.
    - Phase B (Wide): Sparse 5x3 grid covering the full workspace.
      Steps are sized to roughly match the camera FOV (with overlap), so the
      entire reachable area is covered in few positions.
    - Phase C (Wait): If nothing found, park at home and wait passively.

    If no last known position is available (cold start), Phase A is skipped.
    """

    _LOCAL_OFFSETS_PAN = [-15, 0, 15]
    _LOCAL_OFFSETS_TILT = [-15, 0, 15]
    _FACE_LOCAL_OFFSETS_TILT = [-15, 0, 15]
    _WIDE_OFFSETS_PAN = [-50, -25, 0, 25, 50]
    _WIDE_OFFSETS_TILT = [0, 25,-25]
    _FACE_WIDE_OFFSETS_TILT = [0, 25,-25]

    _LOCAL_FRAMES_PER_POS = 6
    _LOCAL_SETTLE_S = 0.6
    _WIDE_FRAMES_PER_POS = 12
    _WIDE_SETTLE_S = 1.2

    def __init__(self):
        super().__init__()
        self.name = "PHYSICAL_SEARCH"
        self._positions = []
        self._pos_idx = 0
        self._frames_at_pos = 0
        self._done = False

    @staticmethod
    def _dedupe(positions):
        """Remove consecutive duplicate positions (compare pan/tilt only)."""
        result = []
        for p in positions:
            if not result or (result[-1][0], result[-1][1]) != (p[0], p[1]):
                result.append(p)
        return result

    @staticmethod
    def _build_grid(center_pan, center_tilt, pan_offsets, tilt_offsets,
                    pan_limits, tilt_limits, settle_s, frames_per_pos):
        """Build a zigzag grid of clamped (pan, tilt, settle_s, frames) entries."""
        pan_min, pan_max = pan_limits
        tilt_min, tilt_max = tilt_limits
        positions = []
        for i, t_off in enumerate(tilt_offsets):
            row_pans = pan_offsets if i % 2 == 0 else list(reversed(pan_offsets))
            for p_off in row_pans:
                p = max(pan_min, min(pan_max, center_pan + p_off))
                t = max(tilt_min, min(tilt_max, center_tilt + t_off))
                positions.append((p, t, settle_s, frames_per_pos))
        return positions

    def on_enter(self, context):
        if not (context.enable_pan_tilt and context.pan_tilt is not None):
            return

        pan_limits = context.pan_tilt.pan_limits
        tilt_limits = context.pan_tilt.tilt_limits
        home = context.pan_tilt.home_pose

        local_positions = []
        last_pan = context.last_seen_pan
        last_tilt = context.last_seen_tilt
        tilt_local = (
            self._FACE_LOCAL_OFFSETS_TILT
            if getattr(context, "perception_mode", "") == "face"
            else self._LOCAL_OFFSETS_TILT
        )
        tilt_wide = (
            self._FACE_WIDE_OFFSETS_TILT
            if getattr(context, "perception_mode", "") == "face"
            else self._WIDE_OFFSETS_TILT
        )
        if last_pan is not None and last_tilt is not None:
            local_positions = self._build_grid(
                last_pan, last_tilt,
                self._LOCAL_OFFSETS_PAN, tilt_local,
                pan_limits, tilt_limits,
                self._LOCAL_SETTLE_S, self._LOCAL_FRAMES_PER_POS,
            )
            local_positions = self._dedupe(local_positions)
            print(f"[i] Physical Search Phase A: {len(local_positions)} local positions "
                  f"around last known ({last_pan}°, {last_tilt}°).")
        else:
            print("[i] Physical Search: no last known position, skipping Phase A.")

        wide_positions = self._build_grid(
            home.pan, home.tilt,
            self._WIDE_OFFSETS_PAN, tilt_wide,
            pan_limits, tilt_limits,
            self._WIDE_SETTLE_S, self._WIDE_FRAMES_PER_POS,
        )
        wide_positions = self._dedupe(wide_positions)
        print(f"[i] Physical Search Phase B: {len(wide_positions)} wide grid positions "
              f"(dwell {self._WIDE_FRAMES_PER_POS} frames, settle {self._WIDE_SETTLE_S}s).")

        self._positions = local_positions + wide_positions
        self._pos_idx = 0
        self._frames_at_pos = 0
        self._done = False

        if self._positions:
            pan, tilt, settle_s, _ = self._positions[0]
            context.pan_tilt.move_to(pan, tilt, smooth=False, settle_s=settle_s)

    def update(self, context, frame, detected, corners, ids, smooth_u, raw_u, metrics, current_brightness, size_value):
        if context.confirmed_detected:
            print("[i] Target found during Physical Search. Resuming normal operations.")
            _log_event(
                context,
                "physical_search_exit",
                frame_idx=context.frame_count,
                reason="target_reentered",
            )
            context.ignore_until_frame = context.frame_count + 5
            context.zoom_ignore_until_frame = context.frame_count + 5
            return MonitorState()

        if not (context.enable_pan_tilt and context.pan_tilt is not None):
            return self

        if self._done:
            return self

        _, _, _, frames_per_pos = self._positions[self._pos_idx]
        self._frames_at_pos += 1
        if self._frames_at_pos >= frames_per_pos:
            self._frames_at_pos = 0
            self._pos_idx += 1
            if self._pos_idx >= len(self._positions):
                print("[i] Physical Search: all positions scanned. Parking at home, waiting passively.")
                context.pan_tilt.home(smooth=True)
                self._done = True
            else:
                pan, tilt, settle_s, _ = self._positions[self._pos_idx]
                context.pan_tilt.move_to(pan, tilt, smooth=False, settle_s=settle_s)

        return self

