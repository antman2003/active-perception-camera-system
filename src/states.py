"""
State Machine Definitions for Active Perception System.

States:
- MonitorState: Watch the scene and perform smooth tracking (Nudge).
- ExploreExposureState: Sweep exposure levels to adapt to lighting changes.
- ExploreZoomState: Sweep zoom levels to adapt to target size changes.
- SniperRecoveryState: Fallback to 1.0x to quickly locate a lost target.
"""

import time
import numpy as np

def _aruco_centroid_pixel(corners) -> tuple:
    """Mean (x, y) of the first marker's 4 corners in pixel coords."""
    if corners is None or len(corners) < 1:
        return None
    pts = corners[0][0]
    mx = float(np.mean(pts[:, 0]))
    my = float(np.mean(pts[:, 1]))
    return mx, my


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
        if env_changed and uncertainty >= self.explore_enter_threshold:
            self._explore_trigger_count += 1
        elif (not env_changed) or (uncertainty <= self.explore_exit_threshold):
            self._explore_trigger_count = 0

        if self._explore_trigger_count >= self.explore_trigger_frames:
            print(f"[!] Triggering EXPLORE (Score: {uncertainty:.2f})")
            context.baseline_brightness = None
            self._explore_trigger_count = 0
            return ExploreExposureState()

        # 5. Check Target Size Change and Quality
        if context.frame_count >= context.zoom_ignore_until_frame:
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
            if (not context.zoom_initialized) or size_changed or poor_quality_at_base:
                print(f"[!] Triggering ZOOM EXPLORE (Score: {uncertainty:.2f})")
                context.baseline_size = None
                return ExploreZoomState()

        # 7. Zoomed-in Logic: Nudge Tracking or Sniper Recovery
        if context.policy.current_zoom_level > 1.0:
            if context.confirmed_lost:
                self.roi_lost_frames += 1
            elif context.confirmed_detected:
                self.roi_lost_frames = 0
                
            # Trigger Sniper Recovery if lost for too long
            if self.roi_lost_frames >= self.roi_lost_threshold:
                if context.frame_count >= context.ignore_until_frame and context.frame_count >= context.zoom_ignore_until_frame:
                    return SniperRecoveryState()
                    
            # Smooth Visual Servoing (Nudge)
            if detected and corners is not None and context.frame_count >= context.ignore_until_frame and context.frame_count >= context.zoom_ignore_until_frame:
                cent = _aruco_centroid_pixel(corners)
                if cent is not None:
                    mx, my = cent
                    hh, ww = frame.shape[:2]
                    tnx, tny = context.policy.marker_center_to_full_norm(mx, my, ww, hh)
                    context.policy.nudge_roi_towards(tnx, tny, gain=context.monitor_nudge_gain)
                    
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
        
    def on_enter(self, context):
        print(f"[i] {self.name}: Initializing sweep, setting index 0.")
        self.settle_frames = context.exposure_settle_frames
        self.sample_frames = context.exposure_sample_frames
        context.policy.execute_exposure(0)
        self._settle_count = 0
        self._sample_scores = []

    def update(self, context, frame, detected, corners, ids, smooth_u, raw_u, metrics, current_brightness, size_value):
        if self._settle_count < self.settle_frames:
            self._settle_count += 1
            return self

        current_idx = self.explore_step
        self._sample_scores.append(raw_u)

        if len(self._sample_scores) < self.sample_frames:
            return self

        avg_score = float(np.mean(self._sample_scores))
        self.exploration_results[current_idx] = avg_score
        print(
            f"   -> Testing Exp Level {current_idx}: "
            f"Avg Score {avg_score:.2f} from {self.sample_frames} frames"
        )
        
        self.explore_step += 1
        
        if self.explore_step >= len(context.policy.exposure_levels):
            self._apply_best_action(context)
            return MonitorState()
        else:
            context.policy.execute_exposure(self.explore_step)
            self._settle_count = 0
            self._sample_scores = []
            
        return self
        
    def _apply_best_action(self, context):
        if not self.exploration_results: return
        min_score = min(self.exploration_results.values())
        candidates = [idx for idx, score in self.exploration_results.items() if abs(score - min_score) < 0.01]
        
        preferred_exposure_val = -4
        best_idx = min(
            candidates, 
            key=lambda idx: (
                abs(context.policy.exposure_levels[idx] - preferred_exposure_val), 
                context.policy.exposure_levels[idx]
            )
        )
        
        best_score = self.exploration_results[best_idx]
        print(f"\n[V] Exploration Done. Winner: Level {best_idx} (Score {best_score:.2f})")
        
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
            
            if context.policy.exposure_supported:
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
                context.zoom_initialized = False
                context.baseline_size = None
                context.current_zoom_idx = 0
                context.ignore_until_frame = context.frame_count + 5
                context.zoom_ignore_until_frame = context.frame_count + 5
                
                return MonitorState()
                
        if self.sniper_frame_count >= self.sniper_timeout_frames:
            print("[!] Sniper Timeout! Target completely lost. Entering Physical Search.")
            context.zoom_initialized = False
            context.baseline_size = None
            context.current_zoom_idx = 0
            return PhysicalSearchState()
            
        return self


class PhysicalSearchState(State):
    """
    Final fallback state when Sniper Recovery times out.
    Stays passively at 1.0x wide-angle, waiting for user to move the target back.
    Once target is detected again, goes back to MonitorState (which will likely trigger zoom explore).
    """
    def __init__(self):
        super().__init__()
        self.name = "PHYSICAL_SEARCH"

    def update(self, context, frame, detected, corners, ids, smooth_u, raw_u, metrics, current_brightness, size_value):
        # We are at 1.0x zoom and waiting for target.
        if context.confirmed_detected:
            print("[i] Target re-entered scene. Resuming normal operations.")
            context.ignore_until_frame = context.frame_count + 5
            context.zoom_ignore_until_frame = context.frame_count + 5
            return MonitorState()
            
        return self

