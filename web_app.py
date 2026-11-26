"""Local dashboard for running the P360 data-pipeline demo."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PAGE_PATH = PROJECT_ROOT / "web" / "index.html"
HOST = "127.0.0.1"
PORT = 8501
DEMO_LOCK = threading.Lock()


def send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/api/status":
            send_json(self, 200, {"running": DEMO_LOCK.locked()})
            return

        if self.path != "/":
            self.send_error(404)
            return

        try:
            body = PAGE_PATH.read_bytes()
        except OSError:
            self.send_error(500, "Dashboard page could not be read")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/api/run-demo":
            self.send_error(404)
            return

        if not DEMO_LOCK.acquire(blocking=False):
            send_json(self, 409, {"error": "The demo is already running."})
            return

        try:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "data_pipeline_demo.py")],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
                check=False,
            )
            send_json(
                self,
                200,
                {
                    "returncode": result.returncode,
                    "output": result.stdout + result.stderr,
                },
            )
        except subprocess.TimeoutExpired as error:
            output = error.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            send_json(
                self,
                504,
                {"error": "The demo exceeded its 15-minute limit.", "output": output},
            )
        finally:
            DEMO_LOCK.release()

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"P360 dashboard running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping P360 dashboard.")
    finally:
        server.server_close()