# app.py
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import tempfile
from Estimators import get_pose_estimator
from Assessment.alignment import normalize_and_align_pose, TemporalSmoother
from Assessment.rom_engine import RomeEngine

st.set_page_config(page_title="Clinical Spine & ROM Analyzer", layout="wide")

st.title("Clinical Spine Alignment & Range of Motion Dashboard")
st.markdown("A unified framework supporting cross-model pose analysis for kinematic rehabilitation tracking.")



def process_frame(estimator, smoother, assessment_type, frame):
    
    frame_start = time.time()
    output = estimator.predict(frame)
    inference_ms = (time.time() - frame_start) * 1000

    current_angle = 0.0
    detected = output["success"]

    if detected:
        raw_kpts = output["keypoints"]
        aligned_kpts = normalize_and_align_pose(raw_kpts)
        smoothed_kpts = smoother.smooth(aligned_kpts)

        if assessment_type == "Spine Lateral Bending":
            current_angle = RomeEngine.calculate_spine_lateral_bend(smoothed_kpts)
        elif assessment_type == "Knee Flexion (Left)":
            current_angle = RomeEngine.calculate_knee_flexion(smoothed_kpts, side="left")
        elif assessment_type == "Knee Flexion (Right)":
            current_angle = RomeEngine.calculate_knee_flexion(smoothed_kpts, side="right")

        h, w, _ = frame.shape
        for idx, pt in raw_kpts.items():
            cx, cy = int(pt[0]), int(pt[1])
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

    return current_angle, inference_ms, detected


def run_analysis_loop(selected_model, assessment_type, cap, tmp_path=None,
                       max_duration_s=None, feed_label="Kinematic Feed"):

    try:
        with st.spinner(f"Initializing {selected_model} Engine..."):
            estimator = get_pose_estimator(selected_model)
            estimator.initialize_model()
        smoother = TemporalSmoother(window_size=5)
    except Exception as e:
        st.error(f"Failed to initialize model '{selected_model}'. Ensure weights/dependencies are present. Error: {e}")
        if tmp_path is not None:
            os.remove(tmp_path)
        return

    if not cap.isOpened():
        st.error("Could not open the selected video source.")
        if tmp_path is not None:
            os.remove(tmp_path)
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(feed_label)
        video_placeholder = st.empty()
    with col2:
        st.subheader("Clinical Analytics")
        metrics_placeholder = st.empty()
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
                break  # video finished, or webcam disconnected

            try:
                current_angle, inference_ms, detected = process_frame(
                    estimator, smoother, assessment_type, frame
                )
            except Exception as e:
                st.error(f"Model inference failed on this frame: {e}")
                break

            if detected:
                angle_history.append(current_angle)
                time_history.append(time.time() - start_time)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", width="stretch")

            max_rom = max(angle_history) if angle_history else 0.0
            min_rom = min(angle_history) if angle_history else 0.0
            metrics_placeholder.markdown(f"""
            **Current Metric Angle:** `{current_angle}°`  
            **Max Peak Angle:** `{max_rom}°`  
            **Min Peak Angle:** `{min_rom}°`  
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

    st.success("Analysis complete.")




st.sidebar.header("Configuration Panel")

selected_model = st.sidebar.selectbox(
    "Select Pose Estimation Engine",
    ["MediaPipe", "MoveNet", "RTMPose", "MeTRAbs"]
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
        "Session Length (seconds)", min_value=10, max_value=600, value=60, step=10,
        help="The live session runs for this long, then stops automatically. "
             "To end earlier, refresh the browser tab.",
    )

assessment_type = st.sidebar.selectbox(
    "Select Clinical Assessment",
    ["Spine Lateral Bending", "Knee Flexion (Left)", "Knee Flexion (Right)"]
)

st.sidebar.divider()

if input_mode == "Live Webcam":
    st.sidebar.caption(
        f"Runs for {max_duration_s}s once started. Model/assessment changes "
        f"only take effect on the next run."
    )
    if st.sidebar.button("Start Live Session", width="stretch"):
        cap = cv2.VideoCapture(0)
        run_analysis_loop(
            selected_model, assessment_type, cap,
            max_duration_s=max_duration_s, feed_label="Real-Time Kinematic Feed",
        )
    else:
        st.info("Press **Start Live Session** in the sidebar to begin.")
else:
    if video_file is None:
        st.info("Upload a video file in the sidebar to begin.")
    else:
        if st.sidebar.button("Run Analysis", width="stretch"):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_file.write(video_file.read())
            tmp_file.close()
            cap = cv2.VideoCapture(tmp_file.name)
            run_analysis_loop(
                selected_model, assessment_type, cap,
                tmp_path=tmp_file.name, feed_label="Kinematic Feed",
            )
        else:
            st.info("Press **Run Analysis** in the sidebar to process the uploaded video.")