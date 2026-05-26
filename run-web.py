from __future__ import annotations

"""Standalone launcher for the SoundStream web UI.

This script sets up the PYTHONPATH so the soundstream package can be imported
from the repository root without installing it.  It then starts the Flask-based
web interface on port 8765.

Usage:
    py run-web.py            # Start on default port 8765
    py run-web.py 8080       # Start on custom port
"""

import os
import sys

# Locate the repository root and add the package source tree to PYTHONPATH.
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "SoundStream", "src")
sys.path.insert(0, SRC)

from soundstream.web import create_app

if __name__ == "__main__":
    # Create the Flask application and start the development server.
    # Accessible on all interfaces at port 8765 by default.
    app = create_app()
    app.run(host="0.0.0.0", port=8765)
