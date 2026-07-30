"""Flask web application for the data profiling and validation pipeline."""
import os
import threading
import uuid
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    redirect,
    url_for,
)

from profiling.core.config import PipelineConfig
from profiling.core.models import PipelineRunRequest
from profiling.pipeline import run as run_pipeline
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB upload limit

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory job store — fine for single-user demo
jobs: dict[str, dict] = {}


def _run_pipeline_job(job_id: str, dataset_paths: list[str], config: PipelineConfig) -> None:
    """Run the pipeline in a background thread, streaming logs to job store."""
    import sys

    job = jobs[job_id]
    job["status"] = "running"

    # Capture print output by redirecting stdout
    class LogCapture:
        def __init__(self):
            self._orig = sys.stdout
        def write(self, text):
            if text.strip():
                job["logs"].append(text.rstrip())
            self._orig.write(text)
        def flush(self):
            self._orig.flush()

    sys.stdout = LogCapture()

    try:
        from profiling.llm.llm_engine import AzureLLMEngine
        llm_client = AzureLLMEngine(
            api_key=os.environ["AZURE_OPENAI_KEY"].strip(),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
        )

        request_obj = PipelineRunRequest(
            dataset_paths=tuple(dataset_paths),
            generate_word=True,
            word_script=Path("generate_word_report.js"),
        )

        result = run_pipeline(
            request=request_obj,
            config=config,
            llm_client=llm_client,
        )

        # Store result summary for display
        job["status"] = "complete"
        job["output_dir"] = str(result.run_directory)
        job["report_path"] = str(result.run_directory / "data_dictionary_report.docx")

        # Build results summary
        summary = []
        for table_name, rules in result.validation_rules.items():
            check = result.validation_check_results.get(table_name, {})
            summary.append({
                "table": table_name,
                "n_rules": len(rules),
                "n_violations": sum(
                    1 for r in check.get("per_rule", [])
                    if (r.get("n_violations") or 0) > 0
                ),
                "n_failing_records": check.get("total_failing_records", 0),
                "violation_records": check.get("violation_records", []),
            })
        job["summary"] = summary

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["logs"].append(f"ERROR: {e}")
    finally:
        sys.stdout = sys.stdout._orig if hasattr(sys.stdout, "_orig") else sys.__stdout__


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def start_run():
    files = request.files.getlist("datasets")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files uploaded"}), 400

    # Save uploaded files
    job_id = str(uuid.uuid4())[:8]
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True)

    dataset_paths = []
    for f in files:
        if f.filename:
            dest = job_upload_dir / f.filename
            f.save(dest)
            dataset_paths.append(str(dest))

    # Parse UI controls
    max_revisions    = int(request.form.get("max_revisions", 2))
    max_regenerations = int(request.form.get("max_regenerations", 1))
    batch_size       = int(request.form.get("batch_size", 200))
    max_retries      = int(request.form.get("max_retries", 3))

    # Build config
    config = PipelineConfig(
        llm_max_retries=max_retries,
        llm_validation_batch_size=batch_size,
    )

    # Patch graph_nodes constants from UI
    import profiling.validation.graph_nodes as gn
    gn.MAX_REVISIONS    = max_revisions
    gn.MAX_REGENERATIONS = max_regenerations

    # Register job
    jobs[job_id] = {
        "status":  "pending",
        "logs":    [],
        "summary": [],
        "error":   None,
        "output_dir":   None,
        "report_path":  None,
    }

    thread = threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, dataset_paths, config),
        daemon=True,
    )
    thread.start()

    return redirect(url_for("progress", job_id=job_id))


@app.route("/progress/<job_id>")
def progress(job_id):
    if job_id not in jobs:
        return "Job not found", 404
    return render_template("progress.html", job_id=job_id)


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    job = jobs[job_id]
    return jsonify({
        "status":  job["status"],
        "logs":    job["logs"],
        "error":   job["error"],
    })


@app.route("/results/<job_id>")
def results(job_id):
    if job_id not in jobs:
        return "Job not found", 404
    job = jobs[job_id]
    if job["status"] != "complete":
        return redirect(url_for("progress", job_id=job_id))
    return render_template("results.html", job_id=job_id, summary=job["summary"])


@app.route("/download/<job_id>")
def download(job_id):
    if job_id not in jobs:
        return "Job not found", 404

    file_type = request.args.get("type", "word")
    output_dir = Path(jobs[job_id].get("output_dir", ""))

    if file_type == "word":
        path = output_dir / "data_dictionary_report.docx"
        if not path.exists():
            return "Report not found", 404
        return send_file(str(path), as_attachment=True,
                         download_name="data_dictionary_report.docx")

    if file_type == "csv":
        import zipfile, io
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for csv_file in output_dir.glob("*_validation_check_results.json"):
                zf.write(csv_file, csv_file.name)
            for csv_file in output_dir.glob("*_data_dictionary.csv"):
                zf.write(csv_file, csv_file.name)
        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name="pipeline_results.zip",
        )

    return "Unknown file type", 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)