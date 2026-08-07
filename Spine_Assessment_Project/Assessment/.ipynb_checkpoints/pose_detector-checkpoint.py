import cv2
import os
import math
import mediapipe as mp
import numpy as np


class BiomechanicalAssessment:
    DEBUG_CALIBRATION = False  # set True to print raw/neutral/calibrated values each frame

    def __init__(self):
        self.reset_live_state()

    def reset_live_state(self):
        self._live_captured_angles = []
        self._live_rep_count = 0
        self._live_direction = 0
        self._live_prev_angle = None
        self._calibration_buffer = []
        self._neutral_offset = None

    def _get_calibrated_angle(self, raw_angle, calibration_frames=5):
        """
        Auto-calibrates a 'neutral' baseline from the first `calibration_frames`
        successfully-detected raw angle readings of a session, then returns
        the deviation from that baseline for every reading after. This lets
        "standing still and straight" define zero for a given recording,
        instead of relying on a fixed anatomical convention (e.g. an
        idealized straight-down arm) that may not match how a specific
        person is actually standing or how the camera is framed.

        Uses the median of the calibration window, not the mean -- pose
        models are typically least stable in their very first frames
        (before any temporal smoothing settles), so a single unstable
        reading could otherwise skew the whole baseline.

        Returns abs(raw_angle - neutral_offset): the physical quantity here
        (degrees away from the calibrated neutral pose) is always
        non-negative regardless of whether the calibrated baseline sits a
        few degrees off from "true" anatomical neutral. An earlier version
        used max(0.0, raw - neutral) instead, which meant a baseline that
        happened to calibrate slightly high would make every real reading
        clip to exactly 0 -- looking identical to "nothing is being
        computed" even though calibration had technically completed. If
        angles are still stuck at 0 after this change, set
        DEBUG_CALIBRATION = True and check the terminal for the actual
        raw/neutral/calibrated numbers rather than guessing again.

        Returns (calibrated_angle, is_calibrating). While is_calibrating is
        True, calibrated_angle is 0.0 and the caller should treat the frame
        as a "hold still" period rather than real movement data (skip rep
        counting / history append for those frames).
        """
        if self._neutral_offset is None:
            self._calibration_buffer.append(raw_angle)
            if len(self._calibration_buffer) < calibration_frames:
                return 0.0, True
            sorted_buf = sorted(self._calibration_buffer)
            n = len(sorted_buf)
            self._neutral_offset = (sorted_buf[n // 2] if n % 2 == 1
                                     else (sorted_buf[n // 2 - 1] + sorted_buf[n // 2]) / 2.0)
            if self.DEBUG_CALIBRATION:
                print(f"[DEBUG calibration] buffer={self._calibration_buffer} "
                      f"-> neutral_offset={self._neutral_offset:.1f}")
            return 0.0, False

        calibrated = abs(raw_angle - self._neutral_offset)
        if self.DEBUG_CALIBRATION:
            print(f"[DEBUG calibration] raw={raw_angle:.1f} neutral={self._neutral_offset:.1f} "
                  f"-> calibrated={calibrated:.1f}")
        return calibrated, False

    def calculate_2d_angle(self, a_pt, b_pt, c_pt):
        """Calculates 2D angle between three (x, y) point tuples."""
        a = np.array(a_pt)
        b = np.array(b_pt)
        c = np.array(c_pt)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360.0 - angle
        return angle

    def get_live_metrics_from_keypoints(self, keypoints, frame, side="left"):
        """
        Unified per-frame entry point, called by both app.py (Streamlit) and
        _run_frame_loop() below (CLI analyze_video). `keypoints` is the
        model-agnostic COCO-17 dict from any Estimators/ adapter. Draws the
        assessment-specific overlay onto `frame` in place when detection
        succeeds.
        """
        metrics = self._compute_live_metrics_from_kpts(keypoints, side=side)
        if metrics.get("success") and frame is not None:
            self._draw_overlay(frame, side, metrics)
        return metrics

    def get_session_summary(self):
        """
        Returns a summary dict for the just-completed recording (live app
        session or analyze_video() run), built from the same _live_* state
        both paths share. Used for left-vs-right symmetry comparisons in
        app.py. Subclasses can override to add assessment-specific fields
        (e.g. ShoulderPainfulArcAssessment adds painful_arc_triggered) --
        always call super().get_session_summary() first and extend the dict.
        Returns None if no frames were successfully analyzed.
        """
        if not self._live_captured_angles:
            return None
        return {
            "max_angle": max(self._live_captured_angles),
            "min_angle": min(self._live_captured_angles),
            "reps": self._live_rep_count,
        }

    @staticmethod
    def compute_symmetry_index(left_value, right_value):
        """
        Shared symmetry-index calculation so every assessment reports it the
        same way: 100% = identical, lower = more asymmetric. Returns None if
        both sides are 0 (nothing to compare).
        """
        denom = max(abs(left_value), abs(right_value))
        if denom == 0:
            return None
        return (1.0 - abs(left_value - right_value) / denom) * 100.0

    @staticmethod
    def analyze_motion_quality(time_history, angle_history,
                                stall_velocity_threshold=3.0, stall_min_duration=0.5,
                                jitter_velocity_threshold=250.0):
        """
        Shared motion-quality analysis for any (time_history, angle_history)
        trace. Deliberately generic -- it only needs a plain time/angle
        series, no assessment-specific logic -- so it's reusable both for
        per-assessment hesitation detection and for cross-model temporal-
        consistency comparison (Sara's comparison work), not shoulder- or
        even assessment-specific.

        - stall_velocity_threshold (deg/s): below this speed, motion counts
          as "paused" for stall detection.
        - stall_min_duration (s): a pause shorter than this is normal (e.g.
          a brief hold between reps) and isn't flagged as hesitation.
        - jitter_velocity_threshold (deg/s): above this speed, a single
          frame-to-frame jump is treated as implausible for genuine human
          movement and flagged as a likely tracking artifact rather than
          real motion.

        NOTE: none of these thresholds have been validated against real
        recordings -- they're reasonable starting points, not calibrated
        values (same caveat as the confidence-threshold attempt earlier).
        Treat flagged results as candidates worth a human look, not ground
        truth, until checked against footage with known-good/bad segments.

        Returns:
            {
                "velocities": [...],       # deg/s between consecutive samples
                "velocity_times": [...],   # midpoint time of each velocity sample
                "stalls": [(start_t, end_t), ...],   # hesitation/stagnation periods
                "jitter_events": [t1, t2, ...],       # timestamps of likely tracking artifacts
            }
        """
        if len(angle_history) < 2 or len(time_history) < 2:
            return {"velocities": [], "velocity_times": [], "stalls": [], "jitter_events": []}

        angles = np.array(angle_history, dtype=float)
        times = np.array(time_history, dtype=float)

        dt = np.diff(times)
        dt[dt == 0] = 1e-6  # guard against duplicate timestamps
        velocities = np.diff(angles) / dt
        velocity_times = (times[:-1] + times[1:]) / 2.0

        jitter_events = [float(t) for t, v in zip(velocity_times, velocities)
                          if abs(v) > jitter_velocity_threshold]

        # Stalls: contiguous runs where |velocity| stays below threshold for
        # at least stall_min_duration seconds.
        stalls = []
        run_start_idx = None
        for i, v in enumerate(velocities):
            if abs(v) < stall_velocity_threshold:
                if run_start_idx is None:
                    run_start_idx = i
            else:
                if run_start_idx is not None:
                    run_start_t = velocity_times[run_start_idx]
                    run_end_t = velocity_times[i - 1]
                    if run_end_t - run_start_t >= stall_min_duration:
                        stalls.append((float(run_start_t), float(run_end_t)))
                    run_start_idx = None
        if run_start_idx is not None:
            run_start_t = velocity_times[run_start_idx]
            run_end_t = velocity_times[-1]
            if run_end_t - run_start_t >= stall_min_duration:
                stalls.append((float(run_start_t), float(run_end_t)))

        return {
            "velocities": velocities.tolist(),
            "velocity_times": velocity_times.tolist(),
            "stalls": stalls,
            "jitter_events": jitter_events,
        }

    def _compute_live_metrics_from_kpts(self, keypoints, side="left"):
        """Default fallback method to be overridden by subclasses."""
        return {"angle": 0.0, "success": False}

    def _draw_overlay(self, frame, side, metrics):
        """Optional hook for subclasses to draw their custom dashboard."""
        pass

    def _run_frame_loop(self, video_path, side="left", estimator=None, window_title="Assessment"):
        """
        Shared CLI video-processing loop used by every subclass's
        analyze_video(). Runs the given pose estimator (any Estimators/
        adapter; defaults to MediaPipe if none is passed) frame-by-frame and
        feeds keypoints through get_live_metrics_from_keypoints, so the CLI
        report and the live Streamlit app always compute angles identically.
        Returns True if at least one frame was successfully analyzed.
        """
        if not os.path.exists(video_path):
            print(f"[ERROR] Video path invalid: {video_path}")
            return False

        self.reset_live_state()

        if estimator is None:
            from Estimators.mediapipe_adapter import MediaPipeAdapter
            estimator = MediaPipeAdapter()
        estimator.initialize_model()

        cap = cv2.VideoCapture(video_path)
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            output = estimator.predict(frame)
            if output["success"]:
                self.get_live_metrics_from_keypoints(output["keypoints"], frame, side=side)

            cv2.imshow(window_title, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(window_title)
        return bool(self._live_captured_angles)

import os
import cv2
import numpy as np

class AnkleAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Upper Ankle Joint Assessment.
    Calculates angles directly using model-agnostic keypoints.
    """
    def __init__(self):
        super().__init__()
        # NOTE: these were originally calibrated for a knee-ankle-toe interior
        # angle (~90 deg = neutral foot). Since the metric below is now a
        # shank-tilt-from-vertical (0 deg = neutral), these thresholds are
        # placeholders carried over in spirit, not clinically re-validated --
        # treat them as a starting point to check against real recordings.
        self.NEUTRAL_ANGLE = 0.0
        self.CLINICAL_NORM_PLANTAR = 20.0
        self.CLINICAL_NORM_DORSI = 20.0

    def draw_dashboard(self, frame, side, live_angle, max_rom, reps):
        """Draws a translucent HUD box in the Top Right corner with relative positioning."""
        h, w, _ = frame.shape
        box_width = int(w * 0.25) if w > 1000 else 340
        box_height = int(h * 0.15) if h > 1000 else 160
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.8
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"ANKLE COACH: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Angle: {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Peak ROM:   {int(max_rom)} Deg", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def _compute_live_metrics_from_kpts(self, keypoints, side="left"):
        """
        Computes ankle metrics using only knee+ankle keypoints.

        NOTE: COCO-17 -- the format shared by every adapter in Estimators/ --
        has no foot/toe keypoint, so true dorsiflexion/plantarflexion (which
        needs the foot's own direction) cannot be computed model-agnostically.
        This uses shank (knee->ankle) tilt from vertical as a real,
        comparable proxy (tibial inclination) instead of fabricating a fake
        toe point. It is NOT the same clinical quantity as true ankle ROM --
        document that distinction if this goes in the report.
        """
        is_left = side.lower() == "left"
        knee_idx = 13 if is_left else 14
        ankle_idx = 15 if is_left else 16

        if not all(idx in keypoints for idx in [knee_idx, ankle_idx]):
            return {
                "angle": 0.0,
                "max_angle": 0.0,
                "reps": self._live_rep_count,
                "feedback": "Detecting ankle keypoints...",
                "success": False
            }

        knee = keypoints[knee_idx][:2]
        ankle = keypoints[ankle_idx][:2]

        shank_x = ankle[0] - knee[0]
        shank_y = ankle[1] - knee[1]
        live_angle = math.degrees(math.atan2(abs(shank_x), shank_y if shank_y != 0 else 1e-6))
        if shank_x < 0:
            live_angle = -live_angle

        self._live_captured_angles.append(live_angle)

        # Repetition tracking
        if self._live_prev_angle is not None:
            if live_angle > self._live_prev_angle + 0.3 and self._live_direction == 0:
                if live_angle > 15:
                    self._live_direction = 1
            elif live_angle < self._live_prev_angle - 0.3 and self._live_direction == 1:
                if live_angle < 5:
                    self._live_rep_count += 1
                    self._live_direction = 0
        self._live_prev_angle = live_angle

        deviation = abs(live_angle - self.NEUTRAL_ANGLE)
        feedback_msg = "Keep moving your foot/ankle up and down."
        if 5.0 < deviation < self.CLINICAL_NORM_PLANTAR:
            feedback_msg = "Good start! Try to push a little further."
        elif deviation >= self.CLINICAL_NORM_PLANTAR:
            feedback_msg = "Excellent range achieved!"
        if self._live_rep_count == 3:
            feedback_msg = "🌟 STREAK ACTIVE! Patient level: PRO! 🌟"

        rom = max(self._live_captured_angles) - min(self._live_captured_angles)
        return {
            "angle": live_angle,
            "max_angle": rom,
            "reps": self._live_rep_count,
            "feedback": feedback_msg,
            "success": True
        }

    def _draw_overlay(self, frame, side, metrics):
        self.draw_dashboard(frame, side, metrics["angle"], metrics["max_angle"], metrics["reps"])

    def analyze_video(self, video_path, side="left", estimator=None):
        print(f"[INFO] Running Ankle Pipeline for {side.upper()} side...")
        if not self._run_frame_loop(video_path, side=side, estimator=estimator,
                                     window_title=f"Assessment Protocol - {side.upper()}"):
            return None

        max_angle = max(self._live_captured_angles)
        min_angle = min(self._live_captured_angles)

        return {
            "max_angle": max_angle,
            "min_angle": min_angle,
            "rom": max_angle - min_angle,
            "measured_plantar": max(0.0, max_angle - self.NEUTRAL_ANGLE),
            "measured_dorsi": max(0.0, self.NEUTRAL_ANGLE - min_angle),
            "repetitions": self._live_rep_count
        }

    def print_symmetry_report(self, left_metrics, right_metrics):
        print("\n" + "="*60)
        print("          BILATERAL SYMMETRY & CLINICAL REPORT          ")
        print("="*60)
        print(f"LEFT SIDE ROM:  {left_metrics['rom']:.1f}° | Completed Reps: {left_metrics['repetitions']}")
        print(f"RIGHT SIDE ROM: {right_metrics['rom']:.1f}° | Completed Reps: {right_metrics['repetitions']}")
        print("-"*60)
        symmetry_index = (1.0 - abs(left_metrics['rom'] - right_metrics['rom']) / max(left_metrics['rom'], right_metrics['rom'])) * 100
        print(f"Bilateral Joint Symmetry Index: {symmetry_index:.1f}%")
        print("-"*60)
        print("CLINICAL FEEDBACK:")
        if symmetry_index >= 85.0:
            print(" [PASS] Good symmetry between both sides. Balanced motor control.")
        else:
            print(" [FLAG] WARNING: Asymmetry detected! Significant mobility difference between limbs.")
        print("="*60 + "\n")


class KneeAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Knee Flexion Assessment (Sagittal Plane).
    Measures the hinge angle between Hip, Knee, and Ankle using COCO-17
    keypoints -- unlike AnkleAssessment, this needs no foot/toe keypoint, so
    it's genuinely computable the same way across every model in Estimators/.
    """
    def __init__(self):
        super().__init__()
        # Normal active knee flexion ROM is roughly 130-140 deg; treat this as
        # a reasonable starting threshold, not a clinically validated one.
        self.CLINICAL_NORM_FLEXION = 130.0

    def draw_dashboard(self, frame, side, live_angle, max_rom, reps):
        h, w, _ = frame.shape
        box_width = int(w * 0.25) if w > 1000 else 340
        box_height = int(h * 0.15) if h > 1000 else 160
        padding = int(w * 0.02)

        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), int(w * 0.002) + 1)

        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.8
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"KNEE COACH: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Flexion: {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Flexion:  {int(max_rom)} Deg", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def _compute_live_metrics_from_kpts(self, keypoints, side="left"):
        """Computes knee flexion angle and reps from COCO-17 hip/knee/ankle keypoints."""
        is_left = side.lower() == "left"
        hip_idx = 11 if is_left else 12
        knee_idx = 13 if is_left else 14
        ankle_idx = 15 if is_left else 16

        if not all(idx in keypoints for idx in [hip_idx, knee_idx, ankle_idx]):
            return {
                "angle": 0.0,
                "max_angle": 0.0,
                "reps": self._live_rep_count,
                "feedback": "Detecting knee keypoints...",
                "success": False
            }

        hip = keypoints[hip_idx][:2]
        knee = keypoints[knee_idx][:2]
        ankle = keypoints[ankle_idx][:2]

        # Interior angle at the knee: ~180 deg when the leg is straight.
        # Converted to a flexion-from-straight measure (0 deg = straight,
        # larger = more bent), same convention HipFlexionAssessment uses.
        interior_angle = self.calculate_2d_angle(hip, knee, ankle)
        live_angle = max(0.0, 180.0 - interior_angle)
        self._live_captured_angles.append(live_angle)

        # Repetition tracking
        if self._live_prev_angle is not None:
            if live_angle > self._live_prev_angle + 0.3 and self._live_direction == 0:
                if live_angle > 40:
                    self._live_direction = 1
            elif live_angle < self._live_prev_angle - 0.3 and self._live_direction == 1:
                if live_angle < 15:
                    self._live_rep_count += 1
                    self._live_direction = 0
        self._live_prev_angle = live_angle

        feedback_msg = "Bend your knee slowly."
        if 45.0 < live_angle < self.CLINICAL_NORM_FLEXION:
            feedback_msg = "Good bend! Keep going if comfortable."
        elif live_angle >= self.CLINICAL_NORM_FLEXION:
            feedback_msg = "Excellent knee flexion range! 🌟"
        if self._live_rep_count == 3:
            feedback_msg = "🌟 STREAK ACTIVE! Great consistency! 🌟"

        max_angle = max(self._live_captured_angles) if self._live_captured_angles else live_angle

        return {
            "angle": live_angle,
            "max_angle": max_angle,
            "reps": self._live_rep_count,
            "feedback": feedback_msg,
            "success": True
        }

    def _draw_overlay(self, frame, side, metrics):
        self.draw_dashboard(frame, side, metrics["angle"], metrics["max_angle"], metrics["reps"])

    def analyze_video(self, video_path, side="left", estimator=None):
        print(f"[INFO] Running Knee Pipeline for {side.upper()} side...")
        if not self._run_frame_loop(video_path, side=side, estimator=estimator,
                                     window_title=f"Knee Assessment - {side.upper()}"):
            return None

        max_flexion = max(self._live_captured_angles)
        return {
            "max_flexion": max_flexion,
            "repetitions": self._live_rep_count,
            "clinical_status": "PASS" if max_flexion >= self.CLINICAL_NORM_FLEXION else "FLAG"
        }

    def print_report(self, left_res, right_res):
        print("\n" + "="*60)
        print("              KNEE FLEXION CLINICAL REPORT               ")
        print("="*60)
        print(f"Left Knee Flexion:  {left_res['max_flexion']:.1f}° / {self.CLINICAL_NORM_FLEXION}° | Reps: {left_res['repetitions']} | [{left_res['clinical_status']}]")
        print(f"Right Knee Flexion: {right_res['max_flexion']:.1f}° / {self.CLINICAL_NORM_FLEXION}° | Reps: {right_res['repetitions']} | [{right_res['clinical_status']}]")
        print("="*60 + "\n")


class HipAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Hip Abduction Assessment.
    Calculates angles directly using model-agnostic COCO-17 keypoints.
    """
    def __init__(self):
        super().__init__()
        self.NEUTRAL_ANGLE = 180.0
        self.CLINICAL_NORM_ABDUCTION = 40.0 

    def get_session_summary(self):
        """
        Overridden because _live_captured_angles stores the RAW shoulder-hip-knee
        angle (~180 deg at rest, decreasing with abduction), not the abduction
        deflection shown live ("angle" = abs(NEUTRAL_ANGLE - raw)). The base
        class's generic max()/min() would report near-resting raw values
        (e.g. 176 deg) instead of true peak abduction (e.g. 24 deg).
        """
        if not self._live_captured_angles:
            return None
        all_deflections = [abs(self.NEUTRAL_ANGLE - a) for a in self._live_captured_angles]
        return {
            "max_angle": max(all_deflections),
            "min_angle": min(all_deflections),
            "reps": self._live_rep_count,
        }

    def draw_dashboard(self, frame, side, live_angle, max_rom, reps):
        h, w, _ = frame.shape
        box_width = int(w * 0.25) if w > 1000 else 340
        box_height = int(h * 0.15) if h > 1000 else 160
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.8
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"HIP COACH: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Hip Angle: {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Abduction: {int(max_rom)} Deg", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def _compute_live_metrics_from_kpts(self, keypoints, side="left"):
        """Computes hip abduction using standard COCO-17 keypoint mapping."""
        # COCO-17 Indices: 5: L_Shoulder, 6: R_Shoulder, 11: L_Hip, 12: R_Hip, 13: L_Knee, 14: R_Knee
        is_left = side.lower() == "left"
        shoulder_idx = 5 if is_left else 6
        hip_idx = 11 if is_left else 12
        knee_idx = 13 if is_left else 14

        # Validate that selected model retrieved necessary keypoints
        if not all(idx in keypoints for idx in [shoulder_idx, hip_idx, knee_idx]):
            return {"angle": 0.0, "max_angle": 0.0, "reps": self._live_rep_count, "feedback": "Detecting...", "success": False}

        shoulder = keypoints[shoulder_idx][:2]
        hip = keypoints[hip_idx][:2]
        knee = keypoints[knee_idx][:2]

        live_angle = self.calculate_2d_angle(shoulder, hip, knee)
        self._live_captured_angles.append(live_angle)
        current_abduction = abs(self.NEUTRAL_ANGLE - live_angle)

        # Repetition tracking algorithm
        if self._live_prev_angle is not None:
            if live_angle < self._live_prev_angle - 0.3 and self._live_direction == 0:
                if current_abduction > 15:
                    self._live_direction = 1
            elif live_angle > self._live_prev_angle + 0.3 and self._live_direction == 1:
                if current_abduction < 8:
                    self._live_rep_count += 1
                    self._live_direction = 0
        self._live_prev_angle = live_angle

        # Clinical Feedback Rules
        feedback_msg = "Lift your leg out to the side slowly."
        if 10.0 < current_abduction < self.CLINICAL_NORM_ABDUCTION:
            feedback_msg = "Good! Keep lifting higher if you can."
        elif current_abduction >= self.CLINICAL_NORM_ABDUCTION:
            feedback_msg = "Target reached! Excellent Hip Mobility! 🌟"
        if self._live_rep_count == 3:
            feedback_msg = "🌟 HIP FREQUENCY STREAK ACTIVE! Great stamina! 🌟"

        all_deflections = [abs(self.NEUTRAL_ANGLE - a) for a in self._live_captured_angles]
        return {
            "angle": current_abduction,
            "max_angle": max(all_deflections) if all_deflections else current_abduction,
            "reps": self._live_rep_count,
            "feedback": feedback_msg,
            "success": True
        }

    def _draw_overlay(self, frame, side, metrics):
        self.draw_dashboard(frame, side, metrics["angle"], metrics["max_angle"], metrics["reps"])

    def analyze_video(self, video_path, side="left", estimator=None):
        print(f"[INFO] Running Hip Pipeline for {side.upper()} side...")
        if not self._run_frame_loop(video_path, side=side, estimator=estimator,
                                     window_title=f"Assessment Protocol - {side.upper()}"):
            return None

        all_deflections = [abs(self.NEUTRAL_ANGLE - a) for a in self._live_captured_angles]
        max_abduction = max(all_deflections)

        return {
            "max_abduction": max_abduction,
            "repetitions": self._live_rep_count,
            "clinical_status": "PASS" if max_abduction >= self.CLINICAL_NORM_ABDUCTION else "FLAG"
        }

    def print_report(self, left_res, right_res):
        print("\n" + "="*60)
        print("         HIP ABDUCTION KINEMATIC REPORT                ")
        print("="*60)
        print(f"Left Hip Peak Abduction:  {left_res['max_abduction']:.1f}° | Reps: {left_res['repetitions']} | [{left_res['clinical_status']}]")
        print(f"Right Hip Peak Abduction: {right_res['max_abduction']:.1f}° | Reps: {right_res['repetitions']} | [{right_res['clinical_status']}]")
        print("="*60 + "\n")


import os
import cv2
import math

class SpineAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Spine Lateral Flexion Assessment.
    Measures the spine deviation from the vertical axis using mid-shoulder and mid-hip.
    """
    def __init__(self):
        super().__init__()
        self.CLINICAL_NORM_FLEXION = 30.0 

    def draw_dashboard(self, frame, side, live_angle, max_rom, reps):
        h, w, _ = frame.shape
        box_width = int(w * 0.25) if w > 1000 else 340
        box_height = int(h * 0.15) if h > 1000 else 160
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.8
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"SPINE COACH: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Bending: {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Bending:  {int(max_rom)} Deg", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def _compute_live_metrics_from_kpts(self, keypoints, side="left"):
        """Computes spine lateral flexion metrics using COCO-17 shoulder and hip keypoints."""
        # COCO-17 Indices: 5: L_Shoulder, 6: R_Shoulder, 11: L_Hip, 12: R_Hip
        l_shoulder_idx, r_shoulder_idx = 5, 6
        l_hip_idx, r_hip_idx = 11, 12

        # Validate that all required keypoints are present
        required_indices = [l_shoulder_idx, r_shoulder_idx, l_hip_idx, r_hip_idx]
        if not all(idx in keypoints for idx in required_indices):
            return {
                "angle": 0.0,
                "max_angle": 0.0,
                "reps": self._live_rep_count,
                "feedback": "Detecting torso keypoints...",
                "success": False
            }

        l_shoulder = keypoints[l_shoulder_idx][:2]
        r_shoulder = keypoints[r_shoulder_idx][:2]
        l_hip = keypoints[l_hip_idx][:2]
        r_hip = keypoints[r_hip_idx][:2]

        # Calculate Midpoints
        mid_shoulder_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
        mid_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
        mid_hip_x = (l_hip[0] + r_hip[0]) / 2.0
        mid_hip_y = (l_hip[1] + r_hip[1]) / 2.0

        spine_vector_x = mid_shoulder_x - mid_hip_x
        spine_vector_y = mid_shoulder_y - mid_hip_y

        # Angle relative to vertical axis
        live_angle = math.degrees(math.atan2(abs(spine_vector_x), -spine_vector_y))
        self._live_captured_angles.append(live_angle)

        # Repetition tracking
        if self._live_prev_angle is not None:
            if live_angle > self._live_prev_angle + 0.2 and self._live_direction == 0:
                if live_angle > 12:
                    self._live_direction = 1
            elif live_angle < self._live_prev_angle - 0.2 and self._live_direction == 1:
                if live_angle < 5:
                    self._live_rep_count += 1
                    self._live_direction = 0
        self._live_prev_angle = live_angle

        feedback_msg = f"Bend your torso slowly to your {side.upper()} side."
        if 5.0 < live_angle < self.CLINICAL_NORM_FLEXION:
            feedback_msg = "Good flex! Keep leaning without twisting."
        elif live_angle >= self.CLINICAL_NORM_FLEXION:
            feedback_msg = "Excellent lateral range of motion! 🌟"

        max_angle = max(self._live_captured_angles) if self._live_captured_angles else live_angle

        return {
            "angle": live_angle,
            "max_angle": max_angle,
            "reps": self._live_rep_count,
            "feedback": feedback_msg,
            "_mid_shoulder": (mid_shoulder_x, mid_shoulder_y),
            "_mid_hip": (mid_hip_x, mid_hip_y),
            "success": True
        }

    def _draw_overlay(self, frame, side, metrics):
        if metrics.get("success", False):
            w = frame.shape[1]
            ms_pixel = (int(metrics["_mid_shoulder"][0]), int(metrics["_mid_shoulder"][1]))
            mh_pixel = (int(metrics["_mid_hip"][0]), int(metrics["_mid_hip"][1]))
            
            cv2.line(frame, mh_pixel, ms_pixel, (0, 255, 255), int(w * 0.003) + 1)
            cv2.circle(frame, ms_pixel, int(w * 0.006) + 1, (0, 0, 255), -1)
            cv2.circle(frame, mh_pixel, int(w * 0.006) + 1, (255, 0, 0), -1)

        self.draw_dashboard(frame, side, metrics["angle"], metrics["max_angle"], metrics["reps"])

    def analyze_video(self, video_path, side="left", estimator=None):
        print(f"[INFO] Running Spine Pipeline for {side.upper()} flexion...")
        if not self._run_frame_loop(video_path, side=side, estimator=estimator,
                                     window_title=f"Spine Assessment Suite - {side.upper()}"):
            return None

        max_flexion = max(self._live_captured_angles)

        return {
            "max_flexion": max_flexion,
            "repetitions": self._live_rep_count,
            "clinical_status": "PASS" if max_flexion >= self.CLINICAL_NORM_FLEXION else "FLAG"
        }

    def print_report(self, left_res, right_res):
        print("\n" + "="*60)
        print("          SPINE LATERAL FLEXION REPORT                 ")
        print("="*60)
        print(f"Left Lateral Flexion:  {left_res['max_flexion']:.1f}° | Reps: {left_res['repetitions']} | [{left_res['clinical_status']}]")
        print(f"Right Lateral Flexion: {right_res['max_flexion']:.1f}° | Reps: {right_res['repetitions']} | [{right_res['clinical_status']}]")
        print("="*60 + "\n")

import os
import cv2

class HipFlexionAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Hip Flexion Assessment (Sagittal Plane).
    Measures the hinge angle between Shoulder, Hip, and Knee using COCO-17 keypoints.
    """
    def __init__(self):
        super().__init__()
        self.CLINICAL_NORM_FLEXION = 120.0 

    def draw_dashboard(self, frame, side, live_angle, max_rom, reps):
        h, w, _ = frame.shape
        box_width = int(w * 0.25) if w > 1000 else 340
        box_height = int(h * 0.15) if h > 1000 else 160
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.8
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"HIP FLEXION: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Angle:   {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Flexion:  {int(max_rom)} Deg", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def _compute_live_metrics_from_kpts(self, keypoints, side="left"):
        """Computes hip flexion angle and reps dynamically from COCO-17 keypoints."""
        # COCO-17 Indices: 5/6 (Shoulder), 11/12 (Hip), 13/14 (Knee)
        is_left = side.lower() == "left"
        shoulder_idx = 5 if is_left else 6
        hip_idx = 11 if is_left else 12
        knee_idx = 13 if is_left else 14

        if not all(idx in keypoints for idx in [shoulder_idx, hip_idx, knee_idx]):
            return {
                "angle": 0.0,
                "max_angle": 0.0,
                "reps": self._live_rep_count,
                "feedback": "Detecting keypoints...",
                "success": False
            }

        shoulder = keypoints[shoulder_idx][:2]
        hip = keypoints[hip_idx][:2]
        knee = keypoints[knee_idx][:2]

        angle_s_h_k = self.calculate_2d_angle(shoulder, hip, knee)
        live_angle = max(0.0, 180.0 - angle_s_h_k)
        self._live_captured_angles.append(live_angle)

        # Repetition tracking
        if self._live_prev_angle is not None:
            if live_angle > self._live_prev_angle + 0.3 and self._live_direction == 0:
                if live_angle > 30: 
                    self._live_direction = 1
            elif live_angle < self._live_prev_angle - 0.3 and self._live_direction == 1:
                if live_angle < 15: 
                    self._live_rep_count += 1
                    self._live_direction = 0
        self._live_prev_angle = live_angle

        # Clinical Feedback Rules
        feedback_msg = "Lift your knee up towards your chest."
        feedback_color = (255, 255, 255)

        if 45.0 < live_angle < self.CLINICAL_NORM_FLEXION:
            feedback_msg = "Good execution, push for maximum height!"
            feedback_color = (0, 165, 255)
        elif live_angle >= self.CLINICAL_NORM_FLEXION:
            feedback_msg = "Target reached! Excellent Hip Mobility. 🌟"
            feedback_color = (0, 255, 0)

        max_angle = max(self._live_captured_angles) if self._live_captured_angles else live_angle

        return {
            "angle": live_angle,
            "max_angle": max_angle,
            "reps": self._live_rep_count,
            "feedback": feedback_msg,
            "feedback_color": feedback_color,
            "_pts": (shoulder, hip, knee),
            "success": True
        }

    def _draw_overlay(self, frame, side, metrics):
        """Renders live joint vectors, feedback banner, and HUD dashboard."""
        h, w, _ = frame.shape

        if metrics.get("success", False):
            s_pt, h_pt, k_pt = metrics["_pts"]
            s_pixel = (int(s_pt[0]), int(s_pt[1]))
            h_pixel = (int(h_pt[0]), int(h_pt[1]))
            k_pixel = (int(k_pt[0]), int(k_pt[1]))

            # Draw vector lines and joint nodes
            cv2.line(frame, s_pixel, h_pixel, (0, 255, 0), int(w * 0.003) + 1)
            cv2.line(frame, h_pixel, k_pixel, (0, 255, 255), int(w * 0.003) + 1)
            for pt in [s_pixel, h_pixel, k_pixel]:
                cv2.circle(frame, pt, int(w * 0.005) + 1, (0, 0, 255), -1)

            # Draw bottom feedback banner
            font_scale = 0.6 if w < 1000 else (w / 1920) * 0.85
            thickness = 2 if w < 1000 else int(w * 0.0015) + 1
            cv2.rectangle(frame, (20, h - int(h * 0.08)), (w - 20, h - int(h * 0.02)), (0, 0, 0), -1)
            cv2.putText(frame, metrics["feedback"], (35, h - int(h * 0.04)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, metrics.get("feedback_color", (255, 255, 255)), thickness, cv2.LINE_AA)

        self.draw_dashboard(frame, side, metrics["angle"], metrics["max_angle"], metrics["reps"])

    def analyze_video(self, video_path, side="left", estimator=None):
        print(f"[INFO] Running Hip Flexion Pipeline for {side.upper()} side...")
        if not self._run_frame_loop(video_path, side=side, estimator=estimator,
                                     window_title=f"Hip Flexion Assessment - {side.upper()}"):
            return None

        max_flexion = max(self._live_captured_angles)
        return {
            "max_flexion": max_flexion,
            "repetitions": self._live_rep_count,
            "clinical_status": "PASS" if max_flexion >= self.CLINICAL_NORM_FLEXION else "FLAG"
        }

    def print_report(self, left_res, right_res):
        print("\n" + "="*60)
        print("              HIP FLEXION CLINICAL REPORT                ")
        print("="*60)
        print(f"Left Hip Flexion:  {left_res['max_flexion']:.1f}° / {self.CLINICAL_NORM_FLEXION}° | Reps: {left_res['repetitions']} | [{left_res['clinical_status']}]")
        print(f"Right Hip Flexion: {right_res['max_flexion']:.1f}° / {self.CLINICAL_NORM_FLEXION}° | Reps: {right_res['repetitions']} | [{right_res['clinical_status']}]")
        print("="*60 + "\n")


import os
import cv2

class FingerFloorAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Finger-to-Floor (FFD) Assessment (Sagittal Plane).
    Measures hip flexion and monitors how close the hands get to the ankles/floor using COCO-17 keypoints.
    """
    def __init__(self):
        super().__init__()
        self.CLINICAL_NORM_FLEXION = 90.0
        self._touch_detected = False

    def draw_dashboard(self, frame, live_angle, max_flexion, touch_detected, reps):
        """Draws dynamic stats box with Live Angle, Floor Touch status, Peak Flexion, and Reps."""
        h, w, _ = frame.shape
        box_width = int(w * 0.28) if w > 1000 else 360
        box_height = int(h * 0.20) if h > 1000 else 200 
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.75
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        touch_str = "YES" if touch_detected else "NO"
        touch_color = (0, 255, 0) if touch_detected else (0, 0, 255)

        cv2.putText(frame, "FINGER-FLOOR COACH", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 100, 0), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Flexion: {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Flexion:  {int(max_flexion)} Deg", (x1 + 15, y1 + int(box_height*0.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Floor Touch:  {touch_str}", (x1 + 15, y1 + int(box_height*0.8)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, touch_color, thickness, cv2.LINE_AA)

    def reset_live_state(self):
        super().reset_live_state()
        self._touch_detected = False

    def _compute_live_metrics_from_kpts(self, keypoints, side="left"):
        """Computes hip flexion and floor touch status using COCO-17 keypoints."""
        # COCO-17 Indices: 5/6 (Shoulder), 9/10 (Wrist), 11/12 (Hip), 13/14 (Knee), 15/16 (Ankle)
        is_left = side.lower() == "left"
        shoulder_idx = 5 if is_left else 6
        wrist_idx = 9 if is_left else 10
        hip_idx = 11 if is_left else 12
        knee_idx = 13 if is_left else 14
        ankle_idx = 15 if is_left else 16

        required_indices = [shoulder_idx, wrist_idx, hip_idx, knee_idx, ankle_idx]
        if not all(idx in keypoints for idx in required_indices):
            return {
                "angle": 0.0,
                "max_angle": 0.0,
                "reps": self._live_rep_count,
                "touch_detected": self._touch_detected,
                "feedback": "Detecting keypoints...",
                "success": False
            }

        shoulder = keypoints[shoulder_idx][:2]
        wrist = keypoints[wrist_idx][:2]
        hip = keypoints[hip_idx][:2]
        knee = keypoints[knee_idx][:2]
        ankle = keypoints[ankle_idx][:2]

        angle_s_h_k = self.calculate_2d_angle(shoulder, hip, knee)
        live_angle = max(0.0, 180.0 - angle_s_h_k)
        self._live_captured_angles.append(live_angle)

        # Touch detection check (Wrist Y >= Ankle Y in screen space)
        if wrist[1] >= ankle[1]:
            self._touch_detected = True

        # Repetition tracking
        if self._live_prev_angle is not None:
            if live_angle > self._live_prev_angle + 0.3 and self._live_direction == 0:
                if live_angle > 30:
                    self._live_direction = 1
            elif live_angle < self._live_prev_angle - 0.3 and self._live_direction == 1:
                if live_angle < 15:
                    self._live_rep_count += 1
                    self._live_direction = 0
        self._live_prev_angle = live_angle

        max_angle = max(self._live_captured_angles) if self._live_captured_angles else live_angle

        return {
            "angle": live_angle,
            "max_angle": max_angle,
            "reps": self._live_rep_count,
            "touch_detected": self._touch_detected,
            "success": True
        }

    def _draw_overlay(self, frame, side, metrics):
        self.draw_dashboard(
            frame, 
            metrics["angle"], 
            metrics["max_angle"], 
            metrics["touch_detected"], 
            metrics["reps"]
        )

    def analyze_video(self, video_path, side="left", estimator=None):
        print(f"[INFO] Running Finger-Floor Pipeline...")
        if not self._run_frame_loop(video_path, side=side, estimator=estimator,
                                     window_title="Finger to Floor Assessment"):
            return None

        max_flexion = max(self._live_captured_angles)
        return {
            "max_flexion": max_flexion,
            "touch_detected": self._touch_detected,
            "repetitions": self._live_rep_count,
            "clinical_status": "PASS" if max_flexion >= self.CLINICAL_NORM_FLEXION else "FLAG"
        }

import os
import math
import cv2

class SpineRotationAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Spine Axial Rotation Assessment (Transverse Plane foreshortening).
    Measures torso rotation relative to a fixed pelvis using shoulder width foreshortening and COCO-17 keypoints.
    """
    def __init__(self):
        super().__init__()
        self.CLINICAL_NORM_ROTATION = 45.0
        self._initial_shoulder_dist = None

    def draw_dashboard(self, frame, side, live_angle, max_rom, reps):
        h, w, _ = frame.shape
        box_width = int(w * 0.25) if w > 1000 else 340
        box_height = int(h * 0.15) if h > 1000 else 160
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.8
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"SPINE ROTATION: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Rotation: {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Rotation:  {int(max_rom)} Deg", (x1 + 15, y1 + int(box_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Completed Reps: {reps}", (x1 + 15, y1 + int(box_height*0.9)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    def reset_live_state(self):
        super().reset_live_state()
        self._initial_shoulder_dist = None

    def _compute_live_metrics_from_kpts(self, keypoints, side="left"):
        """Computes axial spine rotation dynamically using COCO-17 keypoints."""
        # COCO-17 Indices: 5 (Left Shoulder), 6 (Right Shoulder)
        l_shoulder_idx = 5
        r_shoulder_idx = 6

        if not all(idx in keypoints for idx in [l_shoulder_idx, r_shoulder_idx]):
            return {
                "angle": 0.0,
                "max_angle": 0.0,
                "reps": self._live_rep_count,
                "feedback": "Detecting keypoints...",
                "success": False
            }

        l_shoulder = keypoints[l_shoulder_idx][:2]
        r_shoulder = keypoints[r_shoulder_idx][:2]

        curr_dist = math.hypot(r_shoulder[0] - l_shoulder[0], r_shoulder[1] - l_shoulder[1])

        if self._initial_shoulder_dist is None or curr_dist > self._initial_shoulder_dist:
            self._initial_shoulder_dist = curr_dist

        ratio = min(1.0, max(0.0, curr_dist / self._initial_shoulder_dist))
        live_angle = math.degrees(math.acos(ratio))
        self._live_captured_angles.append(live_angle)

        # Repetition tracking
        if self._live_prev_angle is not None:
            if live_angle > self._live_prev_angle + 0.3 and self._live_direction == 0:
                if live_angle > 15:
                    self._live_direction = 1
            elif live_angle < self._live_prev_angle - 0.3 and self._live_direction == 1:
                if live_angle < 5:
                    self._live_rep_count += 1
                    self._live_direction = 0
        self._live_prev_angle = live_angle

        max_angle = max(self._live_captured_angles) if self._live_captured_angles else live_angle

        return {
            "angle": live_angle,
            "max_angle": max_angle,
            "reps": self._live_rep_count,
            "_pts": (l_shoulder, r_shoulder),
            "success": True
        }

    def _draw_overlay(self, frame, side, metrics):
        """Renders live shoulder vector and HUD dashboard."""
        w = frame.shape[1]

        if metrics.get("success", False):
            l_pt, r_pt = metrics["_pts"]
            l_pixel = (int(l_pt[0]), int(l_pt[1]))
            r_pixel = (int(r_pt[0]), int(r_pt[1]))

            # Draw shoulder line and joint nodes
            cv2.line(frame, l_pixel, r_pixel, (0, 165, 255), int(w * 0.003) + 1)
            cv2.circle(frame, l_pixel, int(w * 0.005) + 1, (0, 0, 255), -1)
            cv2.circle(frame, r_pixel, int(w * 0.005) + 1, (0, 0, 255), -1)

        self.draw_dashboard(frame, side, metrics["angle"], metrics["max_angle"], metrics["reps"])

    def analyze_video(self, video_path, side="left", estimator=None):
        print(f"[INFO] Running Spine Rotation Pipeline for {side.upper()} rotation...")
        if not self._run_frame_loop(video_path, side=side, estimator=estimator,
                                     window_title=f"Spine Rotation Suite - {side.upper()}"):
            return None

        max_rotation = max(self._live_captured_angles)
        return {
            "max_rotation": max_rotation,
            "repetitions": self._live_rep_count,
            "clinical_status": "PASS" if max_rotation >= self.CLINICAL_NORM_ROTATION else "FLAG"
        }
####
import os
import cv2

class ShoulderPainfulArcAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Shoulder Painful Arc Assessment (Coronal/Frontal Plane).
    Measures shoulder abduction angle and monitors for painful arc ROM (60° to 120°) using COCO-17 keypoints.
    """
    def __init__(self):
        super().__init__()
        self.PAINFUL_ARC_START = 60.0
        self.PAINFUL_ARC_END = 120.0
        self._painful_arc_triggered = False

    def reset_live_state(self):
        super().reset_live_state()
        self._painful_arc_triggered = False

    def get_session_summary(self):
        summary = super().get_session_summary()
        if summary is None:
            return None
        summary["painful_arc_triggered"] = self._painful_arc_triggered
        return summary

    def draw_dashboard(self, frame, side, live_angle, max_abduction, in_painful_arc, reps):
        """Draws dynamic stats box with Live Angle, Painful Arc status, Peak Abduction, and Reps."""
        h, w, _ = frame.shape
        box_width = int(w * 0.28) if w > 1000 else 360
        box_height = int(h * 0.20) if h > 1000 else 200 
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.75
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        arc_str = "ACTIVE (60-120deg)" if in_painful_arc else "CLEAR"
        arc_color = (0, 0, 255) if in_painful_arc else (0, 255, 0)

        cv2.putText(frame, f"PAINFUL ARC: {side.upper()}", (x1 + 15, y1 + int(box_height*0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Abduction: {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Abduction:  {int(max_abduction)} Deg", (x1 + 15, y1 + int(box_height*0.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Painful Arc Zone: {arc_str}", (x1 + 15, y1 + int(box_height*0.8)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, arc_color, thickness, cv2.LINE_AA)

    def _compute_live_metrics_from_kpts(self, keypoints, side="left"):
        """Computes shoulder abduction angle and painful arc status dynamically using COCO-17 keypoints."""
        # COCO-17 Indices: 5/6 (Shoulder), 7/8 (Elbow), 11/12 (Hip)
        is_left = side.lower() == "left"
        shoulder_idx = 5 if is_left else 6
        elbow_idx = 7 if is_left else 8
        hip_idx = 11 if is_left else 12

        required_indices = [shoulder_idx, elbow_idx, hip_idx]
        if not all(idx in keypoints for idx in required_indices):
            return {
                "angle": 0.0,
                "max_angle": 0.0,
                "reps": self._live_rep_count,
                "in_painful_arc": False,
                "warning": None,
                "success": False
            }

        hip = keypoints[hip_idx][:2]
        shoulder = keypoints[shoulder_idx][:2]
        elbow = keypoints[elbow_idx][:2]

        raw_angle = self.calculate_2d_angle(hip, shoulder, elbow)
        live_angle, is_calibrating = self._get_calibrated_angle(raw_angle, calibration_frames=5)

        if is_calibrating:
            return {
                "angle": 0.0,
                "max_angle": 0.0,
                "reps": self._live_rep_count,
                "feedback": "Calibrating neutral position -- hold still, arm relaxed...",
                "in_painful_arc": False,
                "warning": "Calibrating neutral position -- hold still, arm relaxed...",
                "_pts": (hip, shoulder, elbow),
                "success": True
            }

        self._live_captured_angles.append(live_angle)

        in_painful_arc = self.PAINFUL_ARC_START <= live_angle <= self.PAINFUL_ARC_END
        if in_painful_arc:
            self._painful_arc_triggered = True

        # Repetition tracking
        if self._live_prev_angle is not None:
            if live_angle > self._live_prev_angle + 0.3 and self._live_direction == 0:
                if live_angle > 30:
                    self._live_direction = 1
            elif live_angle < self._live_prev_angle - 0.3 and self._live_direction == 1:
                if live_angle < 20:
                    self._live_rep_count += 1
                    self._live_direction = 0
        self._live_prev_angle = live_angle

        warning = ("⚠️ Entering Painful Arc range (60°-120°) — ask patient about pain."
                   if in_painful_arc else None)

        max_angle = max(self._live_captured_angles) if self._live_captured_angles else live_angle

        return {
            "angle": live_angle,
            "max_angle": max_angle,
            "reps": self._live_rep_count,
            "in_painful_arc": in_painful_arc,
            "warning": warning,
            "_pts": (hip, shoulder, elbow),
            "success": True
        }

    def _draw_overlay(self, frame, side, metrics):
        """Renders live joint vectors and HUD dashboard."""
        w = frame.shape[1]

        if metrics.get("success", False):
            h_pt, s_pt, e_pt = metrics["_pts"]
            h_pixel = (int(h_pt[0]), int(h_pt[1]))
            s_pixel = (int(s_pt[0]), int(s_pt[1]))
            e_pixel = (int(e_pt[0]), int(e_pt[1]))

            # Draw abduction vectors and joint nodes
            cv2.line(frame, h_pixel, s_pixel, (0, 255, 0), int(w * 0.003) + 1)
            cv2.line(frame, s_pixel, e_pixel, (0, 165, 255), int(w * 0.003) + 1)
            for pt in [h_pixel, s_pixel, e_pixel]:
                cv2.circle(frame, pt, int(w * 0.005) + 1, (0, 0, 255), -1)

        self.draw_dashboard(
            frame, 
            side, 
            metrics["angle"], 
            metrics["max_angle"],
            metrics["in_painful_arc"], 
            metrics["reps"]
        )

    def analyze_video(self, video_path, side="left", estimator=None):
        print(f"[INFO] Running Shoulder Painful Arc Pipeline for {side.upper()} side...")
        if not self._run_frame_loop(video_path, side=side, estimator=estimator,
                                     window_title=f"Shoulder Painful Arc Assessment - {side.upper()}"):
            return None

        max_abduction = max(self._live_captured_angles)
        return {
            "max_abduction": max_abduction,
            "painful_arc_range_reached": self._painful_arc_triggered,
            "repetitions": self._live_rep_count,
            "clinical_status": "FLAG (In Painful Arc Range)" if self._painful_arc_triggered else "PASS"
        }

import os
import math
import cv2

class SpineLateralFlexionAssessment(BiomechanicalAssessment):
    """
    Child Class specialized for Spine Lateral Flexion (Coronal/Frontal Plane).
    Measures lateral trunk bending angle relative to vertical using COCO-17 keypoints.
    """
    def __init__(self):
        super().__init__()
        self.CLINICAL_NORM_FLEXION = 35.0

    def draw_dashboard(self, frame, side, live_angle, max_flexion):
        """Draws dynamic status box on the video frame."""
        h, w, _ = frame.shape
        box_width = int(w * 0.28) if w > 1000 else 360
        box_height = int(h * 0.18) if h > 1000 else 180
        padding = int(w * 0.02)
        
        x1, y1 = w - box_width - padding, padding
        x2, y2 = w - padding, padding + box_height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), int(w * 0.002) + 1)
        
        font_scale = 0.55 if w < 1000 else (w / 1920) * 0.75
        thickness = 2 if w < 1000 else int(w * 0.0015) + 1

        cv2.putText(frame, f"LATERAL FLEXION: {side.upper()}", (x1 + 15, y1 + int(box_height*0.25)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Live Bending: {int(live_angle)} Deg", (x1 + 15, y1 + int(box_height*0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Max Bending:  {int(max_flexion)} Deg", (x1 + 15, y1 + int(box_height*0.75)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    def _compute_live_metrics_from_kpts(self, keypoints, side="left"):
        """Computes spine lateral flexion angle dynamically using COCO-17 keypoints."""
        # COCO-17 Indices: 5 (Left Shoulder), 6 (Right Shoulder), 11 (Left Hip), 12 (Right Hip)
        l_shoulder_idx = 5
        r_shoulder_idx = 6
        l_hip_idx = 11
        r_hip_idx = 12

        required_indices = [l_shoulder_idx, r_shoulder_idx, l_hip_idx, r_hip_idx]
        if not all(idx in keypoints for idx in required_indices):
            return {
                "angle": 0.0,
                "max_angle": 0.0,
                "success": False
            }

        l_shoulder = keypoints[l_shoulder_idx][:2]
        r_shoulder = keypoints[r_shoulder_idx][:2]
        l_hip = keypoints[l_hip_idx][:2]
        r_hip = keypoints[r_hip_idx][:2]

        # Midpoints between shoulders and hips
        mid_shoulder_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
        mid_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
        mid_hip_x = (l_hip[0] + r_hip[0]) / 2.0
        mid_hip_y = (l_hip[1] + r_hip[1]) / 2.0

        # Compute lateral deviation from vertical axis
        dx = mid_shoulder_x - mid_hip_x
        dy = mid_hip_y - mid_shoulder_y
        live_angle = math.degrees(math.atan2(abs(dx), abs(dy)))
        self._live_captured_angles.append(live_angle)

        max_angle = max(self._live_captured_angles) if self._live_captured_angles else live_angle

        return {
            "angle": live_angle,
            "max_angle": max_angle,
            "_pts": ((mid_shoulder_x, mid_shoulder_y), (mid_hip_x, mid_hip_y)),
            "success": True
        }

    def _draw_overlay(self, frame, side, metrics):
        """Renders live trunk center vector line and HUD dashboard."""
        w = frame.shape[1]

        if metrics.get("success", False):
            ms_pt, mh_pt = metrics["_pts"]
            ms_pixel = (int(ms_pt[0]), int(ms_pt[1]))
            mh_pixel = (int(mh_pt[0]), int(mh_pt[1]))

            # Draw central spinal line vector
            cv2.line(frame, mh_pixel, ms_pixel, (0, 255, 255), int(w * 0.003) + 1)
            cv2.circle(frame, ms_pixel, int(w * 0.005) + 1, (0, 0, 255), -1)
            cv2.circle(frame, mh_pixel, int(w * 0.005) + 1, (0, 0, 255), -1)

        self.draw_dashboard(frame, side, metrics["angle"], metrics["max_angle"])

    def analyze_video(self, video_path, side="left", estimator=None):
        if not self._run_frame_loop(video_path, side=side, estimator=estimator,
                                     window_title=f"Spine Lateral Flexion - {side.upper()}"):
            return None

        max_flexion = max(self._live_captured_angles)
        return {
            "max_lateral_flexion": max_flexion,
            "clinical_status": "PASS" if max_flexion >= self.CLINICAL_NORM_FLEXION else "FLAG"
        }
####