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
    return render_template("public.html", backend_url=BACKEND_URL)


@app.route("/monitoring-public/<int:event_id>", methods=["GET"])
def monitoring_public(event_id: int):
    try:
        event_row = _get_json(f"/api/event/{event_id}")
        layout_data = _get_json(f"/api/layout-data/{event_id}")
    except Exception:
        return redirect(url_for("index"))
    return render_template(
        "monitoring_public_view.html",
        event=event_row,
        boundary=layout_data.get("boundary"),
        zones=layout_data.get("zones"),
        backend_url=BACKEND_URL,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=True)
