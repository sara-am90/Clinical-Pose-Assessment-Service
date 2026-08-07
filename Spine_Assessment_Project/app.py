# app.py
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import tempfile
import traceback

from Estimators import get_pose_estimator
from Assessment.registry import ASSESSMENT_REGISTRY, get_assessment
from Assessment.pose_detector import BiomechanicalAssessment

st.set_page_config(page_title="Clinical Spine & ROM Analyzer", layout="wide")

st.title("Clinical Spine Alignment & Range of Motion Dashboard")
st.markdown("A unified framework supporting cross-model pose analysis for kinematic rehabilitation tracking.")

if "recorded_sides" not in st.session_state:
    # Holds at most one completed recording per side: {"left": {...}, "right": {...}}
    # Cleared whenever the assessment changes or the user clicks "Clear recorded sides".
    st.session_state["recorded_sides"] = {}

_status_left = "✅ recorded" if "left" in st.session_state["recorded_sides"] else "not recorded"
_status_right = "✅ recorded" if "right" in st.session_state["recorded_sides"] else "not recorded"
st.caption(f"Recording status — Left: **{_status_left}** | Right: **{_status_right}**")


def process_frame(estimator, assessment, frame, side="left"):
    frame_start = time.time()
    
    # 1. Run selected sidebar model ONCE
    output = estimator.predict(frame)
    inference_ms = (time.time() - frame_start) * 1000

    metrics = {"success": False, "angle": 0.0}

    if output["success"]:
        raw_kpts = output["keypoints"]
        h, w, _ = frame.shape
        
        # Draw overlay dots for selected model
        for idx, pt in raw_kpts.items():
            cx, cy = int(pt[0]), int(pt[1])
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # 2. Compute live metrics using keypoints from the selected model
        metrics = assessment.get_live_metrics_from_keypoints(raw_kpts, frame, side=side)

    current_angle = metrics.get("angle", 0.0)
    detected = metrics.get("success", False)

    return current_angle, inference_ms, detected, metrics


def run_analysis_loop(selected_model, assessment_name, side, cap, tmp_path=None,
                      max_duration_s=None, feed_label="Kinematic Feed"):

    try:
        with st.spinner(f"Initializing {selected_model} Engine & {assessment_name}..."):
            estimator = get_pose_estimator(selected_model)
            estimator.initialize_model()
            # Instantiate assessment class dynamically from registry
            assessment = get_assessment(assessment_name)
            assessment.reset_live_state()  # defensive: guarantees a clean rep/ROM state for this run
    except Exception as e:
        st.error(f"Failed to initialize model '{selected_model}' or assessment. Error: {e}")
        if tmp_path is not None:
            os.remove(tmp_path)
        return None

    if not cap.isOpened():
        st.error("Could not open the selected video source.")
        if tmp_path is not None:
            os.remove(tmp_path)
        return None

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(feed_label)
        video_placeholder = st.empty()
    with col2:
        st.subheader("Clinical Analytics")
        metrics_placeholder = st.empty()
        warning_placeholder = st.empty()  # Added for clinical alerts
        chart_placeholder = st.empty()

    angle_history = []
    time_history = []
    start_time = time.time()

    try:
        while cap.isOpened():
            if max_duration_s is not None and (time.time() - start_time) > max_duration_s:
                st.info(f"Reached the {max_duration_s}s session limit.")
                break

            ret, frame = cap.read()
            if not ret:
                break  # Video finished or webcam disconnected

            try:
                current_angle, inference_ms, detected, metrics = process_frame(
                    estimator, assessment, frame, side=side
                )
            except Exception as e:
                traceback.print_exc()  # Prints the detailed error trace in your terminal
                st.error(f"Model inference failed on this frame: {e}")
                break

            if detected:
                angle_history.append(current_angle)
                time_history.append(time.time() - start_time)

                # Display clinical alerts (e.g. Painful Arc warning)
                if isinstance(metrics, dict) and metrics.get("warning"):
                    warning_placeholder.warning(metrics["warning"])
                else:
                    warning_placeholder.empty()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            max_rom = max(angle_history) if angle_history else 0.0
            min_rom = min(angle_history) if angle_history else 0.0
            metrics_placeholder.markdown(f"""
            **Current Metric Angle:** `{current_angle:.1f}°`  
            **Max Peak Angle:** `{max_rom:.1f}°`  
            **Min Peak Angle:** `{min_rom:.1f}°`  
            **Inference Time:** `{inference_ms:.0f} ms/frame`
            """)

            if angle_history:
                df = pd.DataFrame({
                    "Elapsed Time (s)": time_history[-60:],
                    "Angle (Degrees)": angle_history[-60:]
                }).set_index("Elapsed Time (s)")
                chart_placeholder.line_chart(df)
    finally:
        cap.release()
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    summary = assessment.get_session_summary()
    st.success("Analysis complete.")

    if summary is None:
        st.warning("No frames were successfully analyzed, so this run wasn't saved for comparison.")
        return None

    return {
        "assessment_name": assessment_name,
        "side": side,
        "model": selected_model,
        "summary": summary,
        "angle_history": angle_history,
        "time_history": time_history,
    }


# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration Panel")

selected_model = st.sidebar.selectbox(
    "Select Pose Estimation Engine",
    ["MediaPipe", "MoveNet", "RTMPose", "MeTRAbs"]
)

selected_assessment_name = st.sidebar.selectbox(
    "Select Clinical Assessment", 
    options=list(ASSESSMENT_REGISTRY.keys())
)

selected_side = st.sidebar.radio(
    "Side",
    ["left", "right"],
    horizontal=True,
    help="Which side of the body to evaluate (needed for left/right symmetry comparisons).",
)

input_mode = st.sidebar.radio(
    "Select Input Source",
    ["Live Webcam", "Video File Upload"],
)

video_file = None
max_duration_s = None
if input_mode == "Video File Upload":
    video_file = st.sidebar.file_uploader(
        "Upload Patient Video", type=["mp4", "mov", "avi"]
    )
else:
    max_duration_s = st.sidebar.number_input(
        "Session Length (seconds)", min_value=5, max_value=600, value=15, step=5,
        help="The live session runs for exactly this long, then automatically stops and "
             "saves the result for comparison. Let it finish naturally -- using the app's "
             "own Stop control (square icon, top right) kills the script mid-recording and "
             "skips the save step, which is why a session can seem to 'disappear' with no "
             "comparison showing up afterward.",
    )

st.sidebar.divider()

if input_mode == "Live Webcam":
    st.sidebar.caption(
        f"Runs for {max_duration_s}s once started. Model/assessment changes "
        f"only take effect on the next run."
    )
    if st.sidebar.button("Start Live Session", use_container_width=True):
        cap = cv2.VideoCapture(0)
        result = run_analysis_loop(
            selected_model, selected_assessment_name, selected_side, cap,
            max_duration_s=max_duration_s, feed_label="Real-Time Kinematic Feed",
        )
        if result is not None:
            st.session_state["recorded_sides"][result["side"]] = result
            st.session_state.setdefault("pain_annotations", {})[result["side"]] = []
    else:
        st.info("Press **Start Live Session** in the sidebar to begin.")
else:
    if video_file is None:
        st.info("Upload a video file in the sidebar to begin.")
    else:
        if st.sidebar.button("Run Analysis", use_container_width=True):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_file.write(video_file.read())
            tmp_file.close()
            cap = cv2.VideoCapture(tmp_file.name)
            result = run_analysis_loop(
                selected_model, selected_assessment_name, selected_side, cap,
                tmp_path=tmp_file.name, feed_label="Kinematic Feed",
            )
            if result is not None:
                st.session_state["recorded_sides"][result["side"]] = result
                st.session_state.setdefault("pain_annotations", {})[result["side"]] = []
        else:
            st.info("Press **Run Analysis** in the sidebar to process the uploaded video.")


# --- RECORDING REVIEW: PAIN ANNOTATION & MOTION QUALITY ---
st.divider()
st.subheader("Recording Review")

if "pain_annotations" not in st.session_state:
    # side -> list of [start_seconds, end_seconds] ranges marked painful.
    st.session_state["pain_annotations"] = {}

_recorded = st.session_state["recorded_sides"]
_available_sides = [s for s in ("left", "right") if s in _recorded]

if not _available_sides:
    st.caption("Record a session (above) to review its angle trace and mark which parts were painful.")
else:
    annotate_side = st.radio(
        "Recording to annotate", _available_sides, horizontal=True, key="annotate_side_choice"
    )
    _entry = _recorded[annotate_side]
    _time_hist = _entry.get("time_history", [])
    _angle_hist = _entry.get("angle_history", [])

    if not _time_hist:
        st.caption("This recording has no angle trace to annotate.")
    else:
        ranges = st.session_state["pain_annotations"].setdefault(annotate_side, [])

        trace_df = pd.DataFrame(
            {"Elapsed Time (s)": _time_hist, "Angle (Degrees)": _angle_hist}
        ).set_index("Elapsed Time (s)")
        st.markdown("**Pain Range Annotation**")
        st.line_chart(trace_df)

        max_t = float(_time_hist[-1]) if _time_hist else 0.1
        pain_range = st.slider(
            "Select the time range that was painful (seconds)",
            min_value=0.0, max_value=max(max_t, 0.1), value=(0.0, 0.0), step=0.1,
            key=f"pain_slider_{annotate_side}",
        )

        col_add, col_clear = st.columns(2)
        with col_add:
            if st.button("Mark this range as painful", key=f"add_pain_{annotate_side}"):
                if pain_range[1] > pain_range[0]:
                    ranges.append([pain_range[0], pain_range[1]])
                    st.rerun()
                else:
                    st.warning("Select a range where the end is after the start before marking it.")
        with col_clear:
            if ranges and st.button("Clear painful ranges", key=f"clear_pain_{annotate_side}"):
                st.session_state["pain_annotations"][annotate_side] = []
                st.rerun()

        if ranges:
            st.markdown("**Marked painful ranges:**")
            for i, (s, e) in enumerate(ranges, start=1):
                st.markdown(f"{i}. `{s:.1f}s` – `{e:.1f}s`")

        # --- Motion quality: hesitation / motion-artifact detection ---
        st.markdown("---")
        st.markdown("**Motion Quality (Hesitation / Motion-Artifact Detection)**")
        quality = BiomechanicalAssessment.analyze_motion_quality(_time_hist, _angle_hist)

        if quality["stalls"]:
            st.warning("Possible hesitation / stagnation detected during the movement:")
            for s, e in quality["stalls"]:
                st.markdown(f"- Paused from `{s:.1f}s` to `{e:.1f}s` (`{e - s:.1f}s`)")
        else:
            st.success("No extended pauses detected during the movement.")

        if quality["jitter_events"]:
            shown = quality["jitter_events"][:10]
            more = f" (+{len(quality['jitter_events']) - 10} more)" if len(quality["jitter_events"]) > 10 else ""
            st.warning(
                f"{len(quality['jitter_events'])} possible tracking-artifact spike(s) detected at: "
                + ", ".join(f"{t:.1f}s" for t in shown) + more
            )
        else:
            st.caption("No abrupt tracking-artifact spikes detected.")

        if quality["velocities"]:
            vel_df = pd.DataFrame(
                {"Time (s)": quality["velocity_times"], "Angular Velocity (deg/s)": quality["velocities"]}
            ).set_index("Time (s)")
            st.line_chart(vel_df)

        st.caption(
            "Hesitation and artifact thresholds above are starting defaults, not validated against "
            "real recordings yet -- treat flagged points as worth a manual look, not certainty."
        )


# --- LEFT vs. RIGHT SYMMETRY COMPARISON ---
st.divider()
st.subheader("Left vs. Right Symmetry Comparison")

recorded = st.session_state["recorded_sides"]
left_entry = recorded.get("left")
right_entry = recorded.get("right")

if not left_entry and not right_entry:
    st.caption(
        "Record a **Left** session and a **Right** session (same assessment, "
        "using the side selector in the sidebar) to see a symmetry comparison here."
    )
elif not (left_entry and right_entry):
    have, missing = ("Left", "Right") if left_entry else ("Right", "Left")
    st.info(f"{have} side recorded. Record the **{missing}** side (same assessment) to compare.")
elif left_entry["assessment_name"] != right_entry["assessment_name"]:
    st.warning(
        f"Recorded sides are from different assessments "
        f"(Left: '{left_entry['assessment_name']}', Right: '{right_entry['assessment_name']}'). "
        f"Re-record both sides using the same assessment to compare."
    )
else:
    l_summary, r_summary = left_entry["summary"], right_entry["summary"]
    max_l, max_r = l_summary["max_angle"], r_summary["max_angle"]
    symmetry = BiomechanicalAssessment.compute_symmetry_index(max_l, max_r)

    st.caption(
        f"Comparing **{left_entry['assessment_name']}** — "
        f"Left recorded with {left_entry['model']}, Right recorded with {right_entry['model']}."
        + (" Note: different pose models were used for each side, which can itself "
           "introduce some of the measured difference." if left_entry["model"] != right_entry["model"] else "")
    )

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Left Max Angle", f"{max_l:.1f}°")
    m_col2.metric("Right Max Angle", f"{max_r:.1f}°", delta=f"{max_r - max_l:+.1f}°")
    m_col3.metric("Symmetry Index", f"{symmetry:.1f}%" if symmetry is not None else "N/A")

    chart_df = pd.DataFrame({"Side": ["Left", "Right"], "Max Angle (deg)": [max_l, max_r]}).set_index("Side")
    st.bar_chart(chart_df)

    st.markdown(f"**Reps completed:** Left `{l_summary['reps']}` | Right `{r_summary['reps']}`")

    # Assessment-specific fields (e.g. Painful Arc) only appear if present in both summaries
    if "painful_arc_triggered" in l_summary or "painful_arc_triggered" in r_summary:
        triggered_sides = [name for name, s in (("LEFT", l_summary), ("RIGHT", r_summary))
                            if s.get("painful_arc_triggered")]
        if triggered_sides:
            st.warning(f"⚠️ Painful arc range (60°-120°) was entered on: {', '.join(triggered_sides)}.")
        else:
            st.success("No painful arc range triggered on either side.")

    if symmetry is not None:
        if symmetry < 85.0:
            st.warning(f"Asymmetry detected: {symmetry:.1f}% symmetry (below the 85% "
                       f"threshold used elsewhere in this project).")
        else:
            st.success(f"Good symmetry between both sides ({symmetry:.1f}%).")

    if st.button("Clear recorded sides"):
        st.session_state["recorded_sides"] = {}
        st.session_state["pain_annotations"] = {}
        st.rerun()
