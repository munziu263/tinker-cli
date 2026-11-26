"""Preview the viewer locally.

    python tools/serve.py                      # http://localhost:8765

viewer/index.html is written as page content only (no <html>/<head>/<body>), the form a
Claude artifact is published in. This wraps it in the same skeleton for a normal browser.
"""
import argparse
import http.server
import mimetypes
import os
import socketserver

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.normpath(os.path.join(HERE, "..", "viewer"))
CDN = "https://cdn.jsdelivr.net/npm/three@0.170.0/"

SKELETON = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
html{box-sizing:border-box}body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>
</head><body>
__PAGE__
</body></html>"""


def make_handler(offline):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=VIEWER, **kw)

        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                html = open(os.path.join(VIEWER, "index.html"), encoding="utf8").read()
                if offline:
                    html = html.replace(CDN, "/vendor/three/")
                return self._send(SKELETON.replace("__PAGE__", html).encode("utf8"), "text/html; charset=utf-8")
            if offline and path.startswith("/vendor/three/"):
                fp = os.path.normpath(os.path.join(offline, path[len("/vendor/three/"):]))
                if fp.startswith(os.path.normpath(offline)) and os.path.isfile(fp):
                    return self._send(open(fp, "rb").read(), mimetypes.guess_type(fp)[0] or "text/javascript")
                return self.send_error(404)
            return super().do_GET()
    return Handler


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--offline", default=None)
    a = ap.parse_args()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", a.port), make_handler(a.offline)) as srv:
        print(f"Garden viewer at http://localhost:{a.port}")
        srv.serve_forever()
