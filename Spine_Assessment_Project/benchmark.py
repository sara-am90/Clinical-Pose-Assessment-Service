"""
Controlled benchmark for comparing pose estimator adapters.

Runs each selected model over the same fixed video file 
"""
import argparse
import time
import sys
import cv2
import pandas as pd

from Estimators import get_pose_estimator

ALL_MODEL_NAMES = ["MediaPipe", "MoveNet", "RTMPose", "MeTRAbs"]


def benchmark_one_model(model_name: str, video_path: str, max_frames: int, warmup_frames: int):
    """Returns (summary_dict, list_of_raw_ms_timings) or (None, []) on failure."""
    print(f"\n--- Benchmarking {model_name} ---")

    try:
        estimator = get_pose_estimator(model_name)
    except Exception as e:
        print(f"  SKIPPED: could not construct adapter for '{model_name}': {e}")
        return None, []

    load_start = time.perf_counter()
    try:
        estimator.initialize_model()
    except Exception as e:
        print(f"  SKIPPED: model failed to initialize (weights/deps/network?): {e}")
        return None, []
    load_time_s = time.perf_counter() - load_start
    print(f"  Model load time: {load_time_s:.2f} s")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  SKIPPED: could not open video file '{video_path}'")
        return None, []

    raw_timings_ms = []
    frame_idx = 0
    detected_count = 0

    try:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                print(f"  Video ended at frame {frame_idx} (fewer frames than --max-frames)")
                break

            t0 = time.perf_counter()
            try:
                output = estimator.predict(frame)
            except Exception as e:
                print(f"  Frame {frame_idx}: inference error, stopping this model: {e}")
                break
            dt_ms = (time.perf_counter() - t0) * 1000.0

            if output.get("success"):
                detected_count += 1

            if frame_idx >= warmup_frames:
                raw_timings_ms.append(dt_ms)
            frame_idx += 1
    finally:
        cap.release()

    if not raw_timings_ms:
        print(f"  SKIPPED: no timed frames collected (video too short or all frames failed)")
        return None, []

    series = pd.Series(raw_timings_ms)
    summary = {
        "model": model_name,
        "frames_timed": len(series),
        "frames_with_detection": detected_count,
        "load_time_s": round(load_time_s, 2),
        "mean_ms": round(series.mean(), 1),
        "median_ms": round(series.median(), 1),
        "std_ms": round(series.std(), 1),
        "min_ms": round(series.min(), 1),
        "max_ms": round(series.max(), 1),
        "fps": round(1000.0 / series.mean(), 2),
    }
    print(f"  mean={summary['mean_ms']}ms  median={summary['median_ms']}ms  "
          f"std={summary['std_ms']}ms  fps={summary['fps']}")
    return summary, raw_timings_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Path to a fixed test video file (same file used for every model)")
    parser.add_argument("--models", default=",".join(ALL_MODEL_NAMES),
                         help=f"Comma-separated model names to test (default: all -- {', '.join(ALL_MODEL_NAMES)})")
    parser.add_argument("--max-frames", type=int, default=120, help="Max frames to process per model (default: 120)")
    parser.add_argument("--warmup-frames", type=int, default=10,
                         help="Frames to discard at the start of each model's run as warm-up (default: 10)")
    parser.add_argument("--output-csv", default="benchmark_results.csv", help="Summary CSV output path")
    parser.add_argument("--raw-csv", default="benchmark_results_raw.csv", help="Raw per-frame timing CSV output path")
    parser.add_argument("--chart", action="store_true", help="Also save a bar chart (benchmark_chart.png) if matplotlib is installed")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    summaries = []
    raw_rows = []

    for name in model_names:
        summary, raw_timings = benchmark_one_model(name, args.video, args.max_frames, args.warmup_frames)
        if summary is not None:
            summaries.append(summary)
            for i, ms in enumerate(raw_timings):
                raw_rows.append({"model": name, "frame_index": i, "inference_ms": ms})

    if not summaries:
        print("\nNo models completed successfully -- nothing to report.")
        sys.exit(1)

    summary_df = pd.DataFrame(summaries)
    print("\n" + "=" * 70)
    print("SUMMARY (fixed video, warm-up frames excluded)")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved summary to {args.output_csv}")

    if raw_rows:
        pd.DataFrame(raw_rows).to_csv(args.raw_csv, index=False)
        print(f"Saved raw per-frame timings to {args.raw_csv}")

    if args.chart:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(summary_df["model"], summary_df["mean_ms"], yerr=summary_df["std_ms"], capsize=5)
            ax.set_ylabel("Inference time (ms/frame)")
            ax.set_title(f"Pose estimator inference time comparison\n"
                         f"({args.max_frames - args.warmup_frames} frames/model, same video, warm-up excluded)")
            for i, row in summary_df.iterrows():
                ax.text(i, row["mean_ms"] + row["std_ms"] + 2, f"{row['fps']} fps", ha="center", fontsize=9)
            plt.tight_layout()
            plt.savefig("benchmark_chart.png", dpi=150)
            print("Saved chart to benchmark_chart.png")
        except ImportError:
            print("matplotlib not installed -- skipping chart (pip install matplotlib --break-system-packages)")


if __name__ == "__main__":
    main()