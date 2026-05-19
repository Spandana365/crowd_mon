import os
import json
from urllib.request import Request, urlopen
from flask import Flask, render_template, request, redirect, url_for, Response

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")


def _get_json(path):
    req = Request(f"{BACKEND_URL}{path}", headers={"Accept": "application/json"})
    with urlopen(req, timeout=15) as r:
        return json.loads(r.read())


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/", methods=["GET"])
def index():
    return render_template("organizer.html", backend_url=BACKEND_URL)


@app.route("/layout", methods=["GET"])
def layout():
    event_id = request.args.get("event_id", "")
    return render_template("layout.html", event_id=event_id, backend_url=BACKEND_URL)


@app.route("/organizer-monitoring", methods=["GET"])
def organizer_monitoring():
    return render_template("organizer_monitoring.html", backend_url=BACKEND_URL)


@app.route("/monitoring-organizer/<int:event_id>", methods=["GET"])
def monitoring_organizer(event_id: int):
    try:
        event_row = _get_json(f"/api/event/{event_id}")
        layout_data = _get_json(f"/api/layout-data/{event_id}")
    except Exception:
        return redirect(url_for("organizer_monitoring"))
    return render_template(
        "monitoring_organizer_view.html",
        event=event_row,
        boundary=layout_data.get("boundary"),
        zones=layout_data.get("zones"),
        backend_url=BACKEND_URL,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
