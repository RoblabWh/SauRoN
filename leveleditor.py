"""
Tiny development server for the visual level editor.

Serves LevelEditor/index.html and exposes a minimal JSON API to list, load and
save the level SVG files in svg/. Stdlib only, binds to localhost.

    uv run leveleditor.py [--port 8000]
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Environment.LevelGenerator import generate_level

REPO = Path(__file__).resolve().parent
SVG_DIR = REPO / 'svg'
HTML = REPO / 'LevelEditor' / 'index.html'

MAX_BODY = 2 * 1024 * 1024
NAME_RE = re.compile(r'[A-Za-z0-9_.-]{1,64}')


def safe_path(name):
    """Resolve a level filename to a path inside svg/, or None if unsafe."""
    if not name or name.startswith('.') or not NAME_RE.fullmatch(name):
        return None
    if not name.lower().endswith('.svg'):
        return None
    path = (SVG_DIR / name).resolve()
    if path.parent != SVG_DIR:
        return None
    return path


class EditorHandler(BaseHTTPRequestHandler):

    def _send(self, status, body, content_type='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._send(status, {'error': message})

    def do_GET(self):
        url = urlparse(self.path)

        if url.path == '/':
            try:
                self._send(200, HTML.read_bytes().decode(), 'text/html; charset=utf-8')
            except FileNotFoundError:
                self._error(500, 'LevelEditor/index.html not found')

        elif url.path == '/api/levels':
            levels = sorted(p.name for p in SVG_DIR.iterdir()
                            if p.is_file() and p.suffix.lower() == '.svg')
            self._send(200, {'levels': levels})

        elif url.path == '/api/level':
            name = parse_qs(url.query).get('name', [''])[0]
            path = safe_path(name)
            if path is None:
                return self._error(400, 'invalid level name')
            if not path.is_file():
                return self._error(404, 'level not found')
            self._send(200, path.read_bytes().decode('utf-8', errors='replace'),
                       'image/svg+xml; charset=utf-8')

        else:
            self._error(404, 'not found')

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ('/api/level', '/api/generate'):
            return self._error(404, 'not found')

        length = int(self.headers.get('Content-Length', 0))
        if length <= 0 or length > MAX_BODY:
            return self._error(400, 'bad request size')
        try:
            data = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._error(400, 'invalid JSON body')

        if path == '/api/generate':
            try:
                svg, seed = generate_level(
                    data.get('preset', 'maze'),
                    width=float(data.get('width', 620)),
                    height=float(data.get('height', 280)),
                    robots_min=int(data.get('robots_min', 1)),
                    robots_max=int(data.get('robots_max', 2)),
                    pool=bool(data.get('pool', False)),
                    passage=float(data.get('passage', 1.2)),
                    density=float(data.get('density', 50)),
                    seed=data.get('seed'),
                )
            except (ValueError, TypeError) as e:
                return self._error(400, str(e))
            return self._send(200, {'svg': svg, 'seed': seed})

        name = data.get('name', '')
        svg = data.get('svg', '')
        path = safe_path(name)
        if path is None:
            return self._error(400, 'invalid level name')
        if not isinstance(svg, str) or not svg.strip():
            return self._error(400, 'missing svg content')
        # stdlib ET is not hardened against entity-based attacks (XXE, billion
        # laughs); the editor never emits doctypes, so reject them outright
        if re.search(r'<!\s*(DOCTYPE|ENTITY)', svg, re.IGNORECASE):
            return self._error(400, 'doctype/entity declarations not allowed')
        try:
            ET.fromstring(svg)
        except ET.ParseError as e:
            return self._error(400, f'malformed SVG: {e}')

        if path.exists() and not data.get('overwrite'):
            return self._error(409, 'exists')

        path.write_text(svg, encoding='utf-8')
        self._send(200, {'ok': True, 'path': f'svg/{name}'})

    def log_message(self, format, *args):
        print(f'{self.address_string()} - {format % args}')


def main():
    parser = argparse.ArgumentParser(description='SauRoN level editor server')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer(('127.0.0.1', args.port), EditorHandler)
    print(f'Level editor running at http://127.0.0.1:{args.port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
