"""
Main Loop for Active Perception System.

Connects:
Camera -> Perception -> Uncertainty -> Policy

Refactored to use State Machine (Session 11).
"""

import cv2
import time
import threading
import numpy as np
from collections import deque
from typing import Any, Optional
from src.camera import Camera
from src.controller import HardwareController
from src.logger import BlackboxLogger
from src.perception import create_perception
from src.perception.privacy_blur import blur_face_boxes_bgr, collect_face_boxes
from src.uncertainty import (
    ARUCO_UNCERTAINTY_PARAMS,
    FACE_UNCERTAINTY_PARAMS,
    UncertaintyEngine,
    TemporalSmoother,
)
from src.policy import ActionPolicy
from src.face_registry_resolve import resolve_face_registry_dir
from src.states import MonitorState
from src.ui_text import try_draw_text_bgr

# Display privacy: anonymized face box caption (LBPH identity hidden on HUD).
PRIVACY_FACE_HUD_LABEL = "Test Object One"
_VOICE_UI_UNSET = object()

# HUD schema line:
# When `voice_last_schema` is empty we render a stable SCHEMA anchor placeholder.
_VOICE_SCHEMA_HIDE_UNSET = object()
_VOICE_TEXT_HIDE_UNSET = object()

# Voice HUD layout (bottom-left)
# User preference: typically 1 line, occasionally 2 lines max.
_VOICE_HUD_MAX_VOICE_LINES = 2
_VOICE_HUD_MAX_SCHEMA_LINES = 2


class ActivePerceptionLoop:
    def __init__(
        self,
        camera_id: int = 1,
        debug: bool = False,
        enable_exposure_control: bool = True,
        enable_zoom_control: bool = True,
        enable_pan_tilt: bool = False,
        pan_tilt_port: str = "COM3",
        show_window: bool = True,
        perception_mode: str = "aruco",
        face_registry_dir: str | None = None,
        face_match_threshold: float = 85.0,
        primary_hysteresis_frames: int = 0,
        mixed_policy: str = "aruco_first",
        enable_gesture_actions: bool = True,
        privacy_blur_faces: bool = False,
        privacy_blur_kernel: int = 99,
        privacy_blur_pad: float = 0.10,
        privacy_blur_passes: int = 2,
        enable_voice: bool = False,
        voice_mode: str = "ptt",
        voice_wake_backend: str = "openwakeword",
        voice_wake_models: str | None = None,
        voice_wake_threshold: float = 0.5,
        voice_wake_confirm_chunks: int = 3,
        voice_wake_refractory_ms: int = 1200,
        voice_wake_mock_every_chunks: int = 50,
        voice_capture_silence_hangover_ms: int = 2000,
        voice_lang: str = "zh",
        voice_model: str = "base",
        voice_record_seconds: float = 8.0,
        voice_llm: bool = True,
        voice_llm_model: str = "qwen2.5:1.5b",
        voice_clarify: bool = False,
        voice_save_wav: bool = False,
    ):
        print("Initializing System Modules...")
        
        # 1. Hardware
        self.camera = Camera(camera_id)
        
        self.enable_pan_tilt = enable_pan_tilt
        self.pan_tilt: HardwareController | None = None
        if enable_pan_tilt:
            try:
                self.pan_tilt = HardwareController(port=pan_tilt_port)
                self.pan_tilt.connect()
                print(f"[i] Pan-Tilt stage connected on {pan_tilt_port}")
            except Exception as e:
                print(f"[!] Pan-Tilt init failed ({e}). Continuing without physical servoing.")
                self.pan_tilt = None
                self.enable_pan_tilt = False
        
        # 2. Perception & Brain
        self.perception_mode = (perception_mode or "aruco").lower().strip()
        if self.perception_mode == "auto":
            self.perception_mode = "mixed"
        self.mixed_policy = (mixed_policy or "aruco_first").lower().strip()
        self.face_registry_dir: str | None = None
        self.uncertainty_engine_aruco: Optional[UncertaintyEngine] = None
        self.uncertainty_engine_face: Optional[UncertaintyEngine] = None
        if self.perception_mode == "face":
            face_registry_dir = resolve_face_registry_dir(face_registry_dir)
            self.face_registry_dir = face_registry_dir
            print(f"[i] Face registry: {face_registry_dir}")
            self.uncertainty_engine = UncertaintyEngine(**FACE_UNCERTAINTY_PARAMS)
        elif self.perception_mode == "aruco":
            self.uncertainty_engine = UncertaintyEngine(**ARUCO_UNCERTAINTY_PARAMS)
        elif self.perception_mode == "mixed":
            face_registry_dir = resolve_face_registry_dir(face_registry_dir)
            self.face_registry_dir = face_registry_dir
            print(f"[i] Mixed perception (policy={self.mixed_policy}); face registry: {face_registry_dir}")
            self.uncertainty_engine_aruco = UncertaintyEngine(**ARUCO_UNCERTAINTY_PARAMS)
            self.uncertainty_engine_face = UncertaintyEngine(**FACE_UNCERTAINTY_PARAMS)
            self.uncertainty_engine = self.uncertainty_engine_aruco
        else:
            raise ValueError(
                "perception_mode must be 'aruco', 'face', 'mixed', or 'auto'"
            )
        self.perception = create_perception(
            self.perception_mode,
            face_registry_dir=face_registry_dir,
            face_match_threshold=face_match_threshold,
            primary_hysteresis_frames=primary_hysteresis_frames,
            mixed_policy=self.mixed_policy,
        )
        self.privacy_blur_faces = bool(privacy_blur_faces)
        self._privacy_blur_kernel = max(3, int(privacy_blur_kernel))
        self._privacy_blur_pad = max(0.0, float(privacy_blur_pad))
        self._privacy_blur_passes = max(1, int(privacy_blur_passes))
        if self.privacy_blur_faces and self.perception_mode == "aruco":
            print(
                "[i] privacy_blur_faces: no face detector in aruco mode; display blur disabled."
            )
            self.privacy_blur_faces = False
        elif self.privacy_blur_faces:
            print(
                "[i] Privacy: face regions blurred on display only "
                f"(kernel={self._privacy_blur_kernel}, pad={self._privacy_blur_pad}, "
                f"passes={self._privacy_blur_passes}); "
                "tracking uses the raw frame."
            )
        self.smoother = TemporalSmoother(window_size=5)

        self.enable_gesture_actions = bool(enable_gesture_actions)
        self._hand_detector = None
        self._gesture_engine = None
        self.gesture_label: str | None = None
        self.gesture_pt_suppress = False
        self.gesture_hands_for_explore_quiet: bool = False
        # (Removed) gesture_pointing_pan_deg / tilt_deg: pointing no longer triggers actions.
        if self.enable_gesture_actions:
            try:
                from src.gesture_actions import GestureActionEngine
                from src.perception.hand import HandGestureDetector

                self._hand_detector = HandGestureDetector()
                self._gesture_engine = GestureActionEngine()
                print(
                    "[i] Gesture actions: ON (fist/pointing/thumbs/victory/heart; mediapipe + pan-tilt)"
                )
            except ImportError as e:
                print(f"[!] Gesture actions disabled: {e}")
                self.enable_gesture_actions = False
        
        # 3. Action
        self.policy = ActionPolicy(self.camera)
        self.blackbox = BlackboxLogger(frame_logging_enabled=debug)
        self.policy.logger = self.blackbox
        self.enable_exposure_control = enable_exposure_control
        self.enable_zoom_control = enable_zoom_control
        self.show_window = show_window

        # Session 30: voice context flags (read by states / HUD).
        self.enable_voice = bool(enable_voice)
        self.voice_mode = (voice_mode or "ptt").lower().strip()
        self.voice_status: str = "off" if not self.enable_voice else "idle"
        self.voice_last_text: str = ""
        self.voice_last_schema: str = ""
        self.voice_clarify_prompt: str | None = None
        self._voice_ui_lock = threading.Lock()
        self._voice_ui = {
            "status": self.voice_status,
            "last_text": self.voice_last_text,
            "last_schema": self.voice_last_schema,
            "clarify_prompt": self.voice_clarify_prompt,
            # Always-on wake capture metering (mostly relevant for `wake_capturing`)
            "capture_rms_e": float("nan"),
            "capture_speech_started": False,
            # When non-None (monotonic seconds), HUD may hide concrete SCHEMA via timed expiry logic.
            "schema_hide_until": None,
            # When non-None (monotonic seconds), HUD may hide transient ASR text.
            "text_hide_until": None,
            # For observability: de-duplicate schema show events.
            "schema_last_emitted": "",
            # For observability: de-duplicate text show events.
            "text_last_emitted": "",
        }
        self.voice_request_search: bool = False
        self.voice_pt_suppress_until: float = 0.0
        self.voice_obs = None
        self._voice_worker = None
        if self.enable_voice:
            try:
                from src.voice.metrics import VoiceObsCounters
                from src.voice.worker import VoicePttWorker, VoiceWorkerConfig

                self.voice_obs = VoiceObsCounters()
                if self.voice_mode == "always":
                    from src.voice.always_worker import AlwaysOnVoiceConfig, AlwaysOnVoiceWorker

                    models: list[str] | None = None
                    if voice_wake_models:
                        models = [s.strip() for s in str(voice_wake_models).split(",") if s.strip()]

                    self._voice_worker = AlwaysOnVoiceWorker(
                        self,
                        AlwaysOnVoiceConfig(
                            lang_hint=voice_lang if voice_lang.lower() != "auto" else None,
                            model_size=str(voice_model),
                            use_llm=bool(voice_llm),
                            llm_model=str(voice_llm_model),
                            enable_clarify=bool(voice_clarify),
                            # keep capture max similar to PTT window by default
                            capture_max_s=float(voice_record_seconds),
                            capture_silence_hangover_ms=int(voice_capture_silence_hangover_ms),
                            wake_backend=str(voice_wake_backend),
                            wake_models=models,
                            wake_threshold=float(voice_wake_threshold),
                            wake_confirm_chunks=int(voice_wake_confirm_chunks),
                            wake_refractory_ms=int(voice_wake_refractory_ms),
                            mock_trigger_every_chunks=int(voice_wake_mock_every_chunks),
                        ),
                    )
                    self._voice_worker.start()
                    print("[i] Voice: ON (always-on wake word + VAD capture).")
                else:
                    self._voice_worker = VoicePttWorker(
                        self,
                        VoiceWorkerConfig(
                            lang_hint=voice_lang if voice_lang.lower() != "auto" else None,
                            record_seconds=float(voice_record_seconds),
                            model_size=str(voice_model),
                            use_llm=bool(voice_llm),
                            llm_model=str(voice_llm_model),
                            enable_clarify=bool(voice_clarify),
                            save_wav=bool(voice_save_wav),
                        ),
                    )
                    self._voice_worker.start()
                    print("[i] Voice: ON (PTT key: 'v').")
            except Exception as e:
                # Keep voice HUD visible even if init fails, so users can see the reason
                # (missing deps, mic device error, openwakeword not installed, etc.).
                err = str(e)
                print(f"[!] Voice init failed ({err}). Voice worker disabled (HUD shows init_failed).")
                try:
                    self.set_voice_ui(status="init_failed", last_text=err, last_schema="")
                except Exception:
                    self.voice_status = "init_failed"
                    self.voice_last_text = err
                    self.voice_last_schema = ""
                # Disable the worker, but keep `enable_voice=True` so HUD renders the failure state.
                self.voice_obs = None
                self._voice_worker = None
        
        # 4. Context Variables (accessed by states)
        self.current_exposure_idx = 3
        self.current_zoom_idx = 0
        if self.privacy_blur_faces and self.perception_mode in ("face", "mixed"):
            if self.enable_zoom_control:
                print(
                    "[i] Privacy: digital zoom auto-control disabled (display stays at 1.0x)."
                )
            self.enable_zoom_control = False
            self.current_zoom_idx = 0

        self.baseline_brightness = None
        self.brightness_change_ratio = 0.10
        self.frame_count = 0
        self.ignore_until_frame = 0
        self.zoom_ignore_until_frame = 0
        self.baseline_size = None
        self.size_change_ratio = 0.35
        self.size_change_floor = 200.0
        self.zoom_initialized = False

        # Stability and sweep parameters. Tune here instead of editing state logic.
        self.monitor_explore_enter_threshold = 0.60
        self.monitor_explore_exit_threshold = 0.50
        self.monitor_zoom_enter_threshold = 0.55
        self.monitor_zoom_exit_threshold = 0.45
        self.monitor_explore_trigger_frames = 2
        self.monitor_zoom_trigger_frames = 2
        self.monitor_roi_lost_threshold = 8
        self.monitor_nudge_gain = 0.15
        self.pan_tilt_gain_pan = 8.0
        self.pan_tilt_gain_tilt = 5.0
        self.pan_tilt_deadzone = 0.05
        self.pan_tilt_search_lost_threshold = 30
        self.last_seen_pan = None
        self.last_seen_tilt = None
        self.sniper_timeout_frames = 60

        # Session 26: face vs ArUco exposure tie-break + optional sweep subset / score shaping.
        self.exposure_tiebreak_preferred_val = (
            -6.0 if self.perception_mode == "face" else -4.0
        )
        self.face_exposure_lbph_weight = 0.12
        self.face_exposure_brightness_weight = 0.10
        self.face_exposure_indices = None  # None = full sweep; else list of exposure level indices

        if self.perception_mode == "face":
            # Larger faces + centroid stability: slightly softer PT gains than marker defaults.
            self.pan_tilt_gain_pan = 6.5
            self.pan_tilt_gain_tilt = 4.5
            self.pan_tilt_deadzone = 0.055
        self._pt_gain_pan_aruco = 8.0
        self._pt_gain_tilt_aruco = 5.0
        self._pt_deadzone_aruco = 0.05
        self._pt_gain_pan_face = 6.5
        self._pt_gain_tilt_face = 4.5
        self._pt_deadzone_face = 0.055

        self.exposure_settle_frames = 2
        self.exposure_sample_frames = 3
        self.zoom_settle_frames = 1
        self.zoom_sample_frames = 2

        # Detection debounce state.
        self.detect_confirm_frames = 2
        self.lost_confirm_frames = 3
        self.detected_streak = 0
        self.lost_streak = 0
        self.confirmed_detected = False
        self.confirmed_lost = True
        self.detection_history = deque(maxlen=30)
        self.benchmark_stable_window_frames = 10
        self.runtime_stats = {
            "frames": 0,
            "detected_frames": 0,
            "confirmed_detected_frames": 0,
            "monitor_frames": 0,
            "raw_u_sum": 0.0,
            "smooth_u_sum": 0.0,
            "size_sum": 0.0,
            "q_size_sum": 0.0,
            "q_sharpness_sum": 0.0,
            "detected_metric_frames": 0,
            "stable_samples": [],
        }
        
        # Initialize camera to default
        if self.policy.exposure_supported:
            self.policy.execute_exposure(self.current_exposure_idx)
            
        self.policy.set_zoom(self.policy.zoom_levels[self.current_zoom_idx])

        # 5. Initialize State Machine
        self.current_state = MonitorState()
        self.current_state.on_enter(self)
        self.blackbox.log_event(
            "session_started",
            camera_id=camera_id,
            initial_state=self.current_state.name,
            enable_gesture_actions=self.enable_gesture_actions,
        )

    def run(self, duration_s: float = None):
        print("\n=== Active Perception Loop Started ===")
        print("Press 'q' in the window to quit.")
        if self.enable_voice and self.voice_mode != "always":
            print("Press 'v' in the window to speak (PTT).")
        elif self.enable_voice and self.voice_mode == "always":
            print("Voice: always-on wake word mode (see HUD + blackbox events).")
        start_time = time.time()
        last_loop_t = time.perf_counter()
        
        try:
            while True:
                if duration_s is not None and (time.time() - start_time) >= duration_s:
                    break

                # Detect UI stalls (helps diagnose "ASR wrote but not shown" reports).
                now_loop_t = time.perf_counter()
                dt_loop_s = now_loop_t - last_loop_t
                last_loop_t = now_loop_t
                if dt_loop_s > 0.35 and self.enable_voice:
                    try:
                        st = str(getattr(self, "voice_status", "") or "")
                        if st in (
                            "wake_capturing",
                            "recording",
                            "asr",
                            "intent",
                            "execute",
                            "llm_loading",
                            "llm_textfix_loading",
                            "llm_bundle_loading",
                        ):
                            self.blackbox.log_event(
                                "hud_loop_stall",
                                dt_ms=int(round(dt_loop_s * 1000)),
                                voice_status=st,
                                frame_idx=int(getattr(self, "frame_count", 0)),
                            )
                    except Exception:
                        pass

                self.frame_count += 1
                
                # --- Step 1: Sense ---
                ret, frame = self.camera.read()
                if not ret: break
                
                # --- Step 2: Perceive ---
                frame = self.policy.apply_digital_zoom(frame)
                detected, ids, corners = self.perception.detect(frame)

                self.gesture_label = None
                self.gesture_pt_suppress = False
                self.gesture_hands_for_explore_quiet = False
                if self.enable_gesture_actions and self._hand_detector is not None and self._gesture_engine is not None:
                    glab, _conf = self._hand_detector.classify(frame)
                    self.gesture_label = glab
                    self._gesture_engine.tick(glab, self)
                    self.gesture_pt_suppress = self._gesture_engine.pt_suppress
                    self.gesture_hands_for_explore_quiet = bool(
                        getattr(self._hand_detector, "hands_visible", False)
                    )

                # Tracking target priority: **hand (MediaPipe) > ArUco > face** when gesture path is on.
                if (
                    self.enable_gesture_actions
                    and self._hand_detector is not None
                    and getattr(self._hand_detector, "hands_visible", False)
                    and self._hand_detector.primary_hand_landmarks is not None
                ):
                    from src.perception.hand import hand_landmarks_to_aruco_corners

                    lm = self._hand_detector.primary_hand_landmarks
                    corners = [hand_landmarks_to_aruco_corners(lm, frame.shape)]
                    ids = np.array([[-1]], dtype=np.int32)
                    detected = True
                    if hasattr(self.perception, "active_backend"):
                        self.perception.active_backend = "gesture"
                elif (
                    self.enable_gesture_actions
                    and self.perception_mode == "mixed"
                    and hasattr(self.perception, "pick_aruco_before_face")
                ):
                    detected, ids, corners = self.perception.pick_aruco_before_face()

                if detected:
                    self.detected_streak += 1
                    self.lost_streak = 0
                else:
                    self.lost_streak += 1
                    self.detected_streak = 0

                self.confirmed_detected = self.detected_streak >= self.detect_confirm_frames
                self.confirmed_lost = self.lost_streak >= self.lost_confirm_frames
                self.detection_history.append(1 if detected else 0)

                # --- Step 3: Evaluate (Brain) ---
                if self.perception_mode == "mixed":
                    ab = getattr(self.perception, "active_backend", None) or "aruco"
                    if ab == "face":
                        self.pan_tilt_gain_pan = self._pt_gain_pan_face
                        self.pan_tilt_gain_tilt = self._pt_gain_tilt_face
                        self.pan_tilt_deadzone = self._pt_deadzone_face
                        self.exposure_tiebreak_preferred_val = -6.0
                        unc_engine = self.uncertainty_engine_face
                    else:
                        self.pan_tilt_gain_pan = self._pt_gain_pan_aruco
                        self.pan_tilt_gain_tilt = self._pt_gain_tilt_aruco
                        self.pan_tilt_deadzone = self._pt_deadzone_aruco
                        self.exposure_tiebreak_preferred_val = -4.0
                        unc_engine = self.uncertainty_engine_aruco
                    raw_u, metrics = unc_engine.compute(frame, corners)
                else:
                    raw_u, metrics = self.uncertainty_engine.compute(frame, corners)
                smooth_u = self.smoother.update(raw_u)
                
                # --- Step 4: Act (State Machine Update) ---
                current_brightness = np.mean(frame)
                size_value = metrics.get("size_raw", 0.0)
                detection_rate = sum(self.detection_history) / len(self.detection_history)
                elapsed_s = time.time() - start_time
                self._update_runtime_stats(raw_u, smooth_u, metrics, detected, elapsed_s)

                self.blackbox.log_frame(
                    frame_idx=self.frame_count,
                    state=self.current_state.name,
                    raw_u=raw_u,
                    smooth_u=smooth_u,
                    detected=detected,
                    confirmed_detected=self.confirmed_detected,
                    confirmed_lost=self.confirmed_lost,
                    detection_rate=detection_rate,
                    brightness=float(current_brightness),
                    zoom=float(self.policy.current_zoom_level),
                    exposure_idx=self.current_exposure_idx,
                    metrics=metrics,
                )

                next_state = self.current_state.update(
                    self, frame, detected, corners, ids, 
                    smooth_u, raw_u, metrics, current_brightness, size_value
                )

                if next_state != self.current_state:
                    self.blackbox.log_event(
                        "state_transition",
                        frame_idx=self.frame_count,
                        from_state=self.current_state.name,
                        to_state=next_state.name,
                        raw_u=raw_u,
                        smooth_u=smooth_u,
                        detected=detected,
                        confirmed_detected=self.confirmed_detected,
                        confirmed_lost=self.confirmed_lost,
                        zoom=float(self.policy.current_zoom_level),
                    )
                    self.current_state.on_exit(self)
                    self.current_state = next_state
                    self.current_state.on_enter(self)

                # --- Step 5: Visualize ---
                if self.show_window:
                    vis_frame = self._draw_hud(frame, smooth_u, metrics, corners, ids)
                    self.camera.display(vis_frame, "Active Perception System")

                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if (
                        self.enable_voice
                        and self.voice_mode != "always"
                        and k == ord("v")
                        and self._voice_worker is not None
                    ):
                        # PTT: background voice thread will record + ASR + (optional) clarify + execute.
                        self._voice_worker.trigger_once()
                    
        finally:
            self.blackbox.log_event(
                "session_stopped",
                final_state=self.current_state.name,
                frame_idx=self.frame_count,
            )
            obs = getattr(self, "voice_obs", None)
            if obs is not None:
                try:
                    self.blackbox.log_event(
                        "voice_metrics_session_final",
                        metrics=obs.snapshot(),
                    )
                except Exception:
                    pass
            if self._voice_worker is not None:
                try:
                    self._voice_worker.stop()
                except Exception:
                    pass
            if self.pan_tilt is not None:
                try:
                    self.pan_tilt.home(smooth=True)
                except Exception:
                    pass
                self.pan_tilt.close()
            if self._hand_detector is not None:
                try:
                    self._hand_detector.close()
                except Exception:
                    pass
            self.camera.release()
            print("System Shutdown.")

        return self.build_summary(time.time() - start_time)

    def _maybe_expire_voice_schema_hud(self) -> None:
        """
        If a timed SCHEMA display window elapsed, revert to stable anchor lines.
        This runs on the main thread during HUD draws (cheap + thread-safe snapshot).
        """
        with self._voice_ui_lock:
            until = self._voice_ui.get("schema_hide_until", None)
            if until is None:
                return
            try:
                if time.monotonic() < float(until):
                    return
            except Exception:
                return

            old = str(self._voice_ui.get("last_schema", "") or "")
            self._voice_ui["last_schema"] = ""
            self.voice_last_schema = ""
            self._voice_ui["schema_hide_until"] = None

        # Emit event outside the lock.
        if old.strip():
            try:
                self.blackbox.log_event(
                    "voice_schema_hidden",
                    schema=old,
                    frame_idx=getattr(self, "frame_count", None),
                    status=str(getattr(self, "voice_status", "")),
                )
            except Exception:
                pass

    def _maybe_expire_voice_text_hud(self) -> None:
        """
        If a timed ASR text display window elapsed, clear the subtitle line.
        """
        # UX rule: don't clear transcript while we're still processing (ASR/LLM/intent/execute).
        # Only allow expiry once we're back to a stable listening/idle state.
        try:
            cur_status = str(getattr(self, "voice_status", "") or "")
        except Exception:
            cur_status = ""
        if cur_status not in ("wake_listening", "idle", "no_speech", "off", "init_failed"):
            return

        with self._voice_ui_lock:
            until = self._voice_ui.get("text_hide_until", None)
            if until is None:
                return
            try:
                if time.monotonic() < float(until):
                    return
            except Exception:
                return

            old = str(self._voice_ui.get("last_text", "") or "")
            self._voice_ui["last_text"] = ""
            self.voice_last_text = ""
            self._voice_ui["text_hide_until"] = None

        if old.strip():
            try:
                self.blackbox.log_event(
                    "voice_text_hidden",
                    text_preview=old[:160],
                    frame_idx=getattr(self, "frame_count", None),
                    status=str(getattr(self, "voice_status", "")),
                )
            except Exception:
                pass

    def set_voice_ui(
        self,
        *,
        status: str | None = None,
        last_text: str | None = None,
        last_schema: str | None = None,
        clarify_prompt: Any = _VOICE_UI_UNSET,
        capture_rms_e: Any = _VOICE_UI_UNSET,
        capture_speech_started: Any = _VOICE_UI_UNSET,
        schema_hide_after: Any = _VOICE_SCHEMA_HIDE_UNSET,
        schema_hold_s: Any = _VOICE_SCHEMA_HIDE_UNSET,
        text_hide_after: Any = _VOICE_TEXT_HIDE_UNSET,
        text_hold_s: Any = _VOICE_TEXT_HIDE_UNSET,
    ) -> None:
        """
        Thread-safe handshake for voice worker -> main-thread HUD state.
        """
        with self._voice_ui_lock:
            if status is not None:
                self._voice_ui["status"] = str(status)
                self.voice_status = str(status)
            if last_text is not None:
                new_text = str(last_text)
                self._voice_ui["last_text"] = new_text
                self.voice_last_text = new_text
                if text_hold_s is _VOICE_TEXT_HIDE_UNSET and text_hide_after is _VOICE_TEXT_HIDE_UNSET:
                    self._voice_ui["text_hide_until"] = None
            if last_schema is not None:
                new_schema = str(last_schema)
                self._voice_ui["last_schema"] = new_schema
                self.voice_last_schema = new_schema
                # Any explicit SCHEMA update cancels pending hide unless caller schedules a new hold window.
                if schema_hold_s is _VOICE_SCHEMA_HIDE_UNSET and schema_hide_after is _VOICE_SCHEMA_HIDE_UNSET:
                    self._voice_ui["schema_hide_until"] = None
            if clarify_prompt is not _VOICE_UI_UNSET:
                self._voice_ui["clarify_prompt"] = (
                    None if clarify_prompt is None else str(clarify_prompt)
                )
                self.voice_clarify_prompt = self._voice_ui["clarify_prompt"]

            if capture_rms_e is not _VOICE_UI_UNSET:
                try:
                    self._voice_ui["capture_rms_e"] = float(capture_rms_e)
                except Exception:
                    self._voice_ui["capture_rms_e"] = float("nan")

            if capture_speech_started is not _VOICE_UI_UNSET:
                self._voice_ui["capture_speech_started"] = bool(capture_speech_started)

            if schema_hide_after is not _VOICE_SCHEMA_HIDE_UNSET:
                self._voice_ui["schema_hide_until"] = schema_hide_after

            if schema_hold_s is not _VOICE_SCHEMA_HIDE_UNSET:
                if schema_hold_s is None:
                    self._voice_ui["schema_hide_until"] = None
                else:
                    try:
                        hs = float(schema_hold_s)
                    except Exception:
                        hs = 0.0
                    if hs > 0.0 and last_schema is None:
                        # Scheduling without providing a SCHEMA string doesn't make sense; ignore safely.
                        self._voice_ui["schema_hide_until"] = None
                    elif hs > 0.0:
                        self._voice_ui["schema_hide_until"] = time.monotonic() + hs
                    else:
                        self._voice_ui["schema_hide_until"] = None

            if text_hide_after is not _VOICE_TEXT_HIDE_UNSET:
                self._voice_ui["text_hide_until"] = text_hide_after

            if text_hold_s is not _VOICE_TEXT_HIDE_UNSET:
                if text_hold_s is None:
                    self._voice_ui["text_hide_until"] = None
                else:
                    try:
                        hs = float(text_hold_s)
                    except Exception:
                        hs = 0.0
                    if hs > 0.0 and last_text is None:
                        self._voice_ui["text_hide_until"] = None
                    elif hs > 0.0:
                        self._voice_ui["text_hide_until"] = time.monotonic() + hs
                    else:
                        self._voice_ui["text_hide_until"] = None

            # Emit text-shown events only on changes, and only for non-empty texts.
            emit_text = None
            try:
                if last_text is not None:
                    t = str(last_text or "")
                    if t.strip():
                        last_emitted = str(self._voice_ui.get("text_last_emitted", "") or "")
                        if t != last_emitted:
                            self._voice_ui["text_last_emitted"] = t
                            emit_text = t
            except Exception:
                emit_text = None

            # Emit schema-shown events only on changes, and only for non-empty schemas.
            emit_schema = None
            try:
                if last_schema is not None:
                    s = str(last_schema or "")
                    if s.strip():
                        last_emitted = str(self._voice_ui.get("schema_last_emitted", "") or "")
                        if s != last_emitted:
                            self._voice_ui["schema_last_emitted"] = s
                            emit_schema = s
            except Exception:
                emit_schema = None

        if emit_schema is not None:
            hide_until = None
            try:
                with self._voice_ui_lock:
                    hide_until = self._voice_ui.get("schema_hide_until", None)
            except Exception:
                hide_until = None
            try:
                self.blackbox.log_event(
                    "voice_schema_shown",
                    schema=str(emit_schema),
                    # Snapshot minimal state for diagnosing UI race/clears.
                    frame_idx=getattr(self, "frame_count", None),
                    status=str(getattr(self, "voice_status", "")),
                    hide_until=hide_until,
                )
            except Exception:
                pass

        if emit_text is not None:
            hide_until = None
            try:
                with self._voice_ui_lock:
                    hide_until = self._voice_ui.get("text_hide_until", None)
            except Exception:
                hide_until = None
            try:
                self.blackbox.log_event(
                    "voice_text_shown",
                    text_preview=str(emit_text)[:200],
                    frame_idx=getattr(self, "frame_count", None),
                    status=str(getattr(self, "voice_status", "")),
                    hide_until=hide_until,
                )
            except Exception:
                pass

    def get_voice_ui_snapshot(self) -> tuple[str, str, str, str | None, float, bool]:
        with self._voice_ui_lock:
            return (
                str(self._voice_ui["status"]),
                str(self._voice_ui["last_text"]),
                str(self._voice_ui["last_schema"]),
                self._voice_ui["clarify_prompt"],
                float(self._voice_ui.get("capture_rms_e", float("nan"))),
                bool(self._voice_ui.get("capture_speech_started", False)),
            )

    def _update_runtime_stats(self, raw_u, smooth_u, metrics, detected, elapsed_s: float):
        self.runtime_stats["frames"] += 1
        self.runtime_stats["raw_u_sum"] += float(raw_u)
        self.runtime_stats["smooth_u_sum"] += float(smooth_u)
        if detected:
            self.runtime_stats["detected_frames"] += 1
        if self.confirmed_detected:
            self.runtime_stats["confirmed_detected_frames"] += 1
        if self.current_state.name == "MONITOR":
            self.runtime_stats["monitor_frames"] += 1
        if metrics.get("detected", False):
            self.runtime_stats["detected_metric_frames"] += 1
            self.runtime_stats["size_sum"] += float(metrics.get("size_raw", 0.0))
            self.runtime_stats["q_size_sum"] += float(metrics.get("q_size", 0.0))
            self.runtime_stats["q_sharpness_sum"] += float(metrics.get("q_sharpness", 0.0))

        is_stable_monitor_frame = (
            self.current_state.name == "MONITOR"
            and self.confirmed_detected
            and self.frame_count >= self.ignore_until_frame
            and self.frame_count >= self.zoom_ignore_until_frame
        )
        if is_stable_monitor_frame:
            self.runtime_stats["stable_samples"].append(
                {
                    "frame_idx": int(self.frame_count),
                    "elapsed_s": float(elapsed_s),
                    "raw_u": float(raw_u),
                    "smooth_u": float(smooth_u),
                    "q_size": float(metrics.get("q_size", 0.0)),
                    "q_sharpness": float(metrics.get("q_sharpness", 0.0)),
                    "size_raw": float(metrics.get("size_raw", 0.0)),
                    "zoom": float(self.policy.current_zoom_level),
                    "exposure_idx": int(self.current_exposure_idx),
                }
            )

    def build_summary(self, duration_s: float) -> dict:
        frames = max(1, self.runtime_stats["frames"])
        detected_metric_frames = max(1, self.runtime_stats["detected_metric_frames"])
        stable_metrics = self._compute_stable_metrics()
        return {
            "duration_s": float(duration_s),
            "frames": self.runtime_stats["frames"],
            "detected_rate": self.runtime_stats["detected_frames"] / frames,
            "confirmed_detected_rate": self.runtime_stats["confirmed_detected_frames"] / frames,
            "search_frames_ratio": 1.0 - (self.runtime_stats["monitor_frames"] / frames),
            "avg_raw_uncertainty": self.runtime_stats["raw_u_sum"] / frames,
            "avg_smooth_uncertainty": self.runtime_stats["smooth_u_sum"] / frames,
            "avg_size_when_detected": self.runtime_stats["size_sum"] / detected_metric_frames,
            "avg_q_size_when_detected": self.runtime_stats["q_size_sum"] / detected_metric_frames,
            "avg_q_sharpness_when_detected": self.runtime_stats["q_sharpness_sum"] / detected_metric_frames,
            "final_zoom": float(self.policy.current_zoom_level),
            "final_exposure_idx": int(self.current_exposure_idx),
            **stable_metrics,
            "mode_flags": {
                "enable_exposure_control": self.enable_exposure_control,
                "enable_zoom_control": self.enable_zoom_control,
                "enable_gesture_actions": self.enable_gesture_actions,
                "privacy_blur_faces": self.privacy_blur_faces,
            },
        }

    def _compute_stable_metrics(self) -> dict:
        samples = self.runtime_stats["stable_samples"]
        window = self.benchmark_stable_window_frames
        if len(samples) < window:
            return {
                "stable_window_frames": window,
                "stable_samples": len(samples),
                "time_to_stable_s": None,
                "best_stable_uncertainty": None,
                "best_stable_smooth_uncertainty": None,
                "best_stable_q_size": None,
                "best_stable_q_sharpness": None,
                "best_stable_size_raw": None,
                "best_stable_zoom": None,
                "best_stable_exposure_idx": None,
                "final_stable_uncertainty": None,
                "final_stable_smooth_uncertainty": None,
                "final_stable_q_size": None,
                "final_stable_q_sharpness": None,
                "final_stable_size_raw": None,
                "final_stable_zoom": None,
                "final_stable_exposure_idx": None,
            }

        stable_runs = []
        current_run = [samples[0]]
        for sample in samples[1:]:
            if sample["frame_idx"] == current_run[-1]["frame_idx"] + 1:
                current_run.append(sample)
            else:
                stable_runs.append(current_run)
                current_run = [sample]
        stable_runs.append(current_run)

        window_summaries = []
        for run in stable_runs:
            if len(run) < window:
                continue
            for i in range(len(run) - window + 1):
                chunk = run[i : i + window]
                window_summaries.append(
                    {
                        "elapsed_s": chunk[-1]["elapsed_s"],
                        "raw_u": float(np.mean([s["raw_u"] for s in chunk])),
                        "smooth_u": float(np.mean([s["smooth_u"] for s in chunk])),
                        "q_size": float(np.mean([s["q_size"] for s in chunk])),
                        "q_sharpness": float(np.mean([s["q_sharpness"] for s in chunk])),
                        "size_raw": float(np.mean([s["size_raw"] for s in chunk])),
                        "zoom": float(np.mean([s["zoom"] for s in chunk])),
                        "exposure_idx": int(round(np.mean([s["exposure_idx"] for s in chunk]))),
                    }
                )

        if not window_summaries:
            return {
                "stable_window_frames": window,
                "stable_samples": len(samples),
                "time_to_stable_s": None,
                "best_stable_uncertainty": None,
                "best_stable_smooth_uncertainty": None,
                "best_stable_q_size": None,
                "best_stable_q_sharpness": None,
                "best_stable_size_raw": None,
                "best_stable_zoom": None,
                "best_stable_exposure_idx": None,
                "final_stable_uncertainty": None,
                "final_stable_smooth_uncertainty": None,
                "final_stable_q_size": None,
                "final_stable_q_sharpness": None,
                "final_stable_size_raw": None,
                "final_stable_zoom": None,
                "final_stable_exposure_idx": None,
            }

        best_window = min(window_summaries, key=lambda item: item["raw_u"])
        final_window = window_summaries[-1]

        return {
            "stable_window_frames": window,
            "stable_samples": len(samples),
            "time_to_stable_s": window_summaries[0]["elapsed_s"],
            "best_stable_uncertainty": best_window["raw_u"],
            "best_stable_smooth_uncertainty": best_window["smooth_u"],
            "best_stable_q_size": best_window["q_size"],
            "best_stable_q_sharpness": best_window["q_sharpness"],
            "best_stable_size_raw": best_window["size_raw"],
            "best_stable_zoom": best_window["zoom"],
            "best_stable_exposure_idx": best_window["exposure_idx"],
            "final_stable_uncertainty": final_window["raw_u"],
            "final_stable_smooth_uncertainty": final_window["smooth_u"],
            "final_stable_q_size": final_window["q_size"],
            "final_stable_q_sharpness": final_window["q_sharpness"],
            "final_stable_size_raw": final_window["size_raw"],
            "final_stable_zoom": final_window["zoom"],
            "final_stable_exposure_idx": final_window["exposure_idx"],
        }

    def _draw_hud(self, frame, uncertainty, metrics, corners, ids):
        """
        Draw status on screen.
        Returns: Annotated frame
        """
        # 1. Perception overlay (optional display-only blur + anonymized face labels)
        face_label_override = (
            PRIVACY_FACE_HUD_LABEL
            if self.privacy_blur_faces and self.perception_mode in ("face", "mixed")
            else None
        )
        if self.privacy_blur_faces and self.perception_mode in ("face", "mixed"):
            vis_base = np.copy(frame)
            blur_face_boxes_bgr(
                vis_base,
                collect_face_boxes(self.perception),
                kernel_size=self._privacy_blur_kernel,
                pad_ratio=self._privacy_blur_pad,
                passes=self._privacy_blur_passes,
            )
            annotated = self.perception.visualize(
                vis_base, corners, ids, face_label_override
            )
        else:
            annotated = self.perception.visualize(
                frame, corners, ids, face_label_override
            )
            
        # 2–3. Top-center / right column (avoid overlap with perception text at left)
        color = (0, 255, 0) if self.current_state.name == "MONITOR" else (0, 255, 255)
        rx = 220
        y_mode = 28
        cv2.putText(
            annotated,
            f"MODE: {self.current_state.name}",
            (rx, y_mode),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        y_gesture = 52
        y_voice = 78
        y_unc_text = 104
        if self.enable_gesture_actions:
            gtxt = self.gesture_label if self.gesture_label else "—"
            gline = f"GESTURE: {gtxt}".strip()
            cv2.putText(
                annotated,
                gline,
                (rx, y_gesture),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 255, 180),
                2,
            )
        else:
            y_voice = 52
            y_unc_text = 78

        if self.enable_voice:
            self._maybe_expire_voice_schema_hud()
            self._maybe_expire_voice_text_hud()
            (
                voice_status,
                voice_last_text,
                voice_last_schema,
                voice_clarify_prompt,
                voice_capture_rms_e,
                voice_capture_speech,
            ) = self.get_voice_ui_snapshot()

            rec_line = ""
            if voice_status == "wake_capturing":
                try:
                    rms_txt = (
                        "nan"
                        if voice_capture_rms_e is None or (voice_capture_rms_e != voice_capture_rms_e)
                        else f"{float(voice_capture_rms_e):.4f}"
                    )
                except Exception:
                    rms_txt = "nan"
                rec_line = f"ACTIVE REC • rms={rms_txt} • speech={'ON' if voice_capture_speech else 'WAITING'}"
            # Subtitle-style voice UI at bottom-left (more room for long text).
            h, w = annotated.shape[:2]
            vx = 12
            font_size = 18
            line_h = 22
            max_chars = 48  # heuristic for font_size=18 on typical 1280px width

            def _wrap(s: str, limit: int) -> list[str]:
                s = (s or "").strip()
                if not s:
                    return []
                out = []
                cur = ""
                for ch in s:
                    cur += ch
                    if len(cur) >= limit:
                        out.append(cur)
                        cur = ""
                if cur:
                    out.append(cur)
                return out

            def _wrap_cap(s: str, limit: int, *, max_lines: int) -> list[str]:
                lines = _wrap(s, limit)
                if not lines:
                    return []
                if len(lines) <= max_lines:
                    return lines
                capped = lines[:max_lines]
                # Add an ellipsis to indicate truncation.
                last = capped[-1]
                # Use ASCII so OpenCV fallback never shows "???".
                capped[-1] = (last[:-3] + "...") if len(last) >= 4 else (last + "...")
                return capped

            if voice_clarify_prompt:
                lines = _wrap_cap(
                    f"VOICE: {voice_clarify_prompt}",
                    max_chars,
                    max_lines=_VOICE_HUD_MAX_VOICE_LINES,
                )
                schema_line = ""
            elif voice_last_text:
                headline = voice_status
                if headline == "wake_capturing":
                    headline = "ACTIVE REC"
                lines = _wrap_cap(
                    f"VOICE({headline}): {voice_last_text}",
                    max_chars,
                    max_lines=_VOICE_HUD_MAX_VOICE_LINES,
                )
            elif voice_status == "wake_capturing":
                # Capture window: subtitle may still be blank; emphasize metering + actionable hint.
                if rec_line:
                    lines = _wrap_cap(
                        rec_line,
                        max_chars,
                        max_lines=_VOICE_HUD_MAX_VOICE_LINES,
                    )
                else:
                    lines = ["ACTIVE REC: speak your command..."]
            elif voice_status in (
                "recording",
                "wake_capturing",
                "asr",
                "intent",
                "execute",
                "llm_loading",
                "llm_textfix_loading",
                "llm_bundle_loading",
            ):
                # Always-on ASR/LLM can spend noticeable time with empty `voice_last_text`
                # (we only set transcript after ASR returns). Use explicit placeholders so users
                # don't misread this as a rendering failure.
                if voice_status == "asr":
                    lines = [f"VOICE({voice_status}): (transcribing...)"]
                elif voice_status in ("llm_loading", "llm_textfix_loading", "llm_bundle_loading"):
                    lines = [f"VOICE({voice_status}): (LLM working...)"]
                elif voice_status == "intent":
                    lines = [f"VOICE({voice_status}): (parsing intent...)"]
                elif voice_status == "execute":
                    lines = [f"VOICE({voice_status}): (executing...)"]
                elif voice_status == "recording":
                    lines = [f"VOICE({voice_status}): (recording...)"]
                else:
                    # wake_capturing handled earlier; keep a safe fallback.
                    lines = [f"VOICE({voice_status}): —"]
            else:
                lines = [f"VOICE({voice_status}): —"]

            # SCHEMA line stays stable as an anchor; concrete intent shows briefly then expires.
            if voice_last_schema:
                schema_lines = _wrap_cap(
                    f"SCHEMA: {voice_last_schema}",
                    max_chars,
                    max_lines=_VOICE_HUD_MAX_SCHEMA_LINES,
                )
            else:
                schema_lines = ["SCHEMA: ------"]

            def _has_non_ascii(s: str) -> bool:
                try:
                    return any(ord(ch) > 127 for ch in (s or ""))
                except Exception:
                    return False

            # Layout: keep SCHEMA anchored near bottom; VOICE sits above SCHEMA.
            # NOTE: Pillow `draw.text` uses top-left-ish coordinates; OpenCV `putText` uses baseline.
            margin = 12
            gap = 6
            step = int(line_h)

            schema_bottom_y = int(h - margin - step)  # top-ish y for bottom schema line
            schema_bottom_y = max(0, min(schema_bottom_y, h - step))
            schema_top_y = int(schema_bottom_y - (len(schema_lines) - 1) * step)
            schema_top_y = max(0, min(schema_top_y, h - step))

            voice_bottom_y = int(schema_top_y - gap - step * len(lines))
            voice_bottom_y = max(0, min(voice_bottom_y, h - step))
            voice_top_y = int(voice_bottom_y)

            # Subtle backing plate for readability (helps when background is busy).
            try:
                x0 = max(0, vx - 6)
                x1 = min(w - 1, vx + int(max_chars * 9) + 10)
                y_top = int(voice_top_y - 4)
                y_bot = int(schema_bottom_y + step + 6)
                y_top = max(0, y_top)
                y_bot = min(h - 1, y_bot)
                if x1 > x0 and y_bot > y_top:
                    overlay = annotated.copy()
                    cv2.rectangle(overlay, (x0, y_top), (x1, y_bot), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, dst=annotated)
            except Exception:
                pass

            # Draw each line with Pillow if possible; fall back per-line to ASCII if not.
            for i, ln in enumerate(lines):
                y = int(voice_top_y + i * step)
                ok = try_draw_text_bgr(
                    annotated,
                    ln,
                    x=vx,
                    y=y,
                    font_size=font_size,
                    color_bgr=(255, 220, 180),
                )
                if not ok:
                    if _has_non_ascii(ln):
                        try:
                            self.blackbox.log_event(
                                "hud_voice_draw_failed",
                                status=str(voice_status),
                                line_preview=str(ln)[:120],
                                y=int(y),
                                frame_h=int(h),
                            )
                        except Exception:
                            pass
                    (_tw, th), bl = cv2.getTextSize((ln or "")[:80], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    baseline_y = int(y + th - bl)
                    cv2.putText(
                        annotated,
                        (ln or "")[:80],
                        (vx, baseline_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 220, 180),
                        1,
                    )
            for j, sline in enumerate(schema_lines):
                y = int(schema_top_y + j * step)
                ok = try_draw_text_bgr(
                    annotated,
                    sline,
                    x=vx,
                    y=y,
                    font_size=font_size,
                    color_bgr=(180, 220, 255),
                )
                if not ok:
                    if _has_non_ascii(sline):
                        try:
                            self.blackbox.log_event(
                                "hud_schema_draw_failed",
                                status=str(voice_status),
                                line_preview=str(sline)[:120],
                                y=int(y),
                                frame_h=int(h),
                            )
                        except Exception:
                            pass
                    (_tw, th), bl = cv2.getTextSize((sline or "")[:80], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    baseline_y = int(y + th - bl)
                    cv2.putText(
                        annotated,
                        (sline or "")[:80],
                        (vx, baseline_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (180, 220, 255),
                        1,
                    )

            # Note: Pillow draws into a tight ROI now; avoid comparing full-frame arrays for fallback.

        # Uncertainty bar: below Faces / ACTIVE lines from visualize() (see combined.py y)
        bar_top, bar_bot = 62, 80
        bar_len = int(uncertainty * 200)
        u_color = (0, 0, 255) if uncertainty > 0.6 else (0, 255, 0)
        cv2.rectangle(annotated, (10, bar_top), (10 + bar_len, bar_bot), u_color, -1)
        cv2.putText(
            annotated,
            f"Uncertainty: {uncertainty:.2f}",
            (rx, y_unc_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        if self.privacy_blur_faces:
            cv2.putText(
                annotated,
                "PRIVACY: face blur",
                (rx, 102),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 255),
                1,
            )

        # 4–6. Left column below uncertainty bar
        y_zoom = 96
        zoom_text = f"ZOOM: {self.policy.current_zoom_level}x"
        cv2.putText(
            annotated,
            zoom_text,
            (10, y_zoom),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"Sharpness: {metrics.get('sharpness_raw', 0):.0f}",
            (10, y_zoom + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            annotated,
            f"Size: {metrics.get('size_raw', 0):.0f}",
            (10, y_zoom + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        if self.pan_tilt is not None:
            pose = self.pan_tilt.current_pose
            cv2.putText(
                annotated,
                f"PT: P{pose.pan} T{pose.tilt}",
                (10, y_zoom + 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 100),
                1,
            )
                   
        return annotated

if __name__ == "__main__":
    app = ActivePerceptionLoop()
    app.run()
