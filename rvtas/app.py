"""
app.py — Flask deployment for RVTAS (Real-Time Vehicle Traffic Analytics System)

Usage:
    python app.py
    Open http://localhost:5000 in your browser.
"""

import os
import uuid
import threading
import yaml
from flask import Flask, request, render_template, jsonify, send_from_directory

# Import the existing pipeline
from main import process_video
from utils import write_json_log

# ── Base directory (always the folder where app.py lives) ─────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# ── Folders (absolute paths so OpenCV can always find them) ───────────────────
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "uploads")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Allowed video extensions
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# ── In-memory job store ───────────────────────────────────────────────────────
jobs: dict = {}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Background pipeline worker ────────────────────────────────────────────────
def run_pipeline(job_id: str, input_path: str, output_path: str, cfg: dict) -> None:
    try:
        log = process_video(input_path, output_path, cfg)

        log_path = os.path.join(REPORTS_FOLDER, f"{job_id}_log.json")
        write_json_log(log, log_path)

        # heatmap is saved by the pipeline next to the output video
        heatmap_path = os.path.join(REPORTS_FOLDER, f"{job_id}_heatmap.png")
        # fall back to sample_heatmap.png if pipeline didn't produce one
        if not os.path.exists(heatmap_path):
            heatmap_path = os.path.join(REPORTS_FOLDER, "sample_heatmap.png")

        jobs[job_id] = {
            "status":      "done",
            "log":         log,
            "video_url":   f"/reports/{job_id}_out.mp4",
            "heatmap_url": f"/reports/{os.path.basename(heatmap_path)}",
            "log_url":     f"/reports/{job_id}_log.json",
        }

    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    job_id    = str(uuid.uuid4())[:8]
    safe_name = f"{job_id}_{os.path.basename(f.filename)}"

    # Absolute paths so OpenCV never loses the file
    input_path  = os.path.join(UPLOAD_FOLDER,  safe_name)
    output_path = os.path.join(REPORTS_FOLDER, f"{job_id}_out.mp4")

    f.save(input_path)

    # Verify the file was actually saved and has content
    if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
        return jsonify({"error": "Upload failed — file is empty or missing."}), 400

    # Load config (absolute path)
    config_path = os.path.join(BASE_DIR, "config.yaml")
    with open(config_path) as c:
        cfg = yaml.safe_load(c)

    cfg["input"]  = input_path
    cfg["output"] = output_path

    jobs[job_id] = {"status": "running"}

    t = threading.Thread(
        target=run_pipeline,
        args=(job_id, input_path, output_path, cfg),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


@app.route("/reports/<path:filename>")
def serve_report(filename: str):
    return send_from_directory(REPORTS_FOLDER, filename)


@app.route("/jobs")
def list_jobs():
    summary = {
        jid: {"status": j.get("status"), "error": j.get("error", "")}
        for jid, j in jobs.items()
    }
    return jsonify(summary)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  RVTAS Flask Server")
    print(f"  BASE_DIR     : {BASE_DIR}")
    print(f"  Templates    : {os.path.join(BASE_DIR, 'templates')}")
    print(f"  Uploads      : {UPLOAD_FOLDER}")
    print(f"  Reports      : {REPORTS_FOLDER}")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)