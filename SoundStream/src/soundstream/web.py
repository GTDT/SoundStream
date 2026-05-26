from __future__ import annotations

"""Flask-based web UI for SoundStream.

Provides a browser interface to control streaming and listening sessions.
Runs a background asyncio loop in a separate thread so the synchronous Flask
app can drive the async SoundSession client.
"""

import asyncio
import logging
import threading
from typing import Any

import sounddevice as sd
from flask import Flask, jsonify, render_template, request

from .session import SoundSession
from .signaling.config import DEFAULT_SERVER_URL

logger = logging.getLogger("soundstream.web")


class SessionManager:
    """Bridge between Flask's synchronous world and SoundSession's async API.

    Launches a dedicated asyncio event loop on a background thread.  All async
    SoundSession calls are marshalled into that loop so that Flask request
    handlers can stay synchronous.
    """

    def __init__(self, server_url: str = DEFAULT_SERVER_URL) -> None:
        # Create a private event loop for the background thread.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Build the SoundSession inside the background loop.
        self._session = self._run_coro(self._create_session(server_url))

    def _run_loop(self) -> None:
        """Target function for the background thread — runs the event loop forever."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _create_session(self, server_url: str) -> SoundSession:
        """Coroutine that instantiates a SoundSession on the background loop."""
        return SoundSession(server_url)

    def _run_coro(self, coro: Any) -> Any:
        """Execute a coroutine on the background loop and return its result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def _run_sync(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Wrap a synchronous call so it can be dispatched via the background loop."""

        async def _inner() -> Any:
            return func(*args, **kwargs)

        return self._run_coro(_inner())

    def status(self) -> dict[str, Any]:
        """Return a snapshot of the current session state."""
        return {
            "state": self._session.state,
            "room_id": self._session.room_id,
            "is_streaming": self._session.is_streaming,
            "is_listening": self._session.is_listening,
            "is_connected": self._session.is_connected,
            "server_url": self._session.server_url,
        }

    def list_rooms(self) -> list[dict[str, Any]]:
        """Fetch the list of public rooms from the signaling server."""
        return self._run_coro(self._session.list_rooms())

    def create_room(
        self, room_id: str, is_public: bool = False, title: str | None = None
    ) -> dict[str, Any]:
        """Create a new room on the signaling server."""
        return self._run_coro(
            self._session.create_room(room_id, is_public=is_public, title=title)
        )

    def start_streaming(
        self,
        room_id: str | None = None,
        device: int | None = None,
        is_public: bool = False,
        title: str | None = None,
    ) -> dict[str, Any] | None:
        """Configure input device, create the room if needed, and start streaming."""
        if device is not None:
            self._run_sync(setattr, self._session, "input_device", device)
        if room_id and self._session.state == "idle":
            self.create_room(room_id, is_public=is_public, title=title)
        return self._run_coro(self._session.start_streaming())

    def start_listening(
        self, room_id: str, device: int | None = None
    ) -> dict[str, Any] | None:
        """Configure output device and start listening to the given room."""
        if device is not None:
            self._run_sync(setattr, self._session, "output_device", device)
        return self._run_coro(self._session.start_listening(room_id))

    def stop(self) -> dict[str, Any] | None:
        """Stop the current session (streaming or listening)."""
        return self._run_coro(self._session.stop())

    def send_chat(self, message: str) -> None:
        """Send a chat message through the current session."""
        return self._run_coro(self._session.send_chat(message))


def create_app() -> Flask:
    """Build and configure the Flask application.

    Instantiates SessionManager, wires API routes, and points Flask to the
    templates/ and static/ directories.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    manager = SessionManager()
    # Attach the manager to the app so routes can access it via app.session_manager.
    app.session_manager = manager

    # ------------------------------------------------------------------
    # HTML views
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        """Serve the main web UI page."""
        return render_template("index.html")

    # ------------------------------------------------------------------
    # API: Session status
    # ------------------------------------------------------------------

    @app.route("/api/status", methods=["GET"])
    def api_status():
        """GET /api/status — return the current session state."""
        return jsonify(app.session_manager.status())

    # ------------------------------------------------------------------
    # API: Audio devices
    # ------------------------------------------------------------------

    @app.route("/api/devices", methods=["GET"])
    def api_devices():
        """GET /api/devices — list all audio input/output devices."""
        devices = sd.query_devices()
        sanitized = []
        for idx, dev in enumerate(devices):
            sanitized.append(
                {
                    "index": idx,
                    "name": dev.get("name"),
                    "max_input_channels": dev.get("max_input_channels"),
                    "max_output_channels": dev.get("max_output_channels"),
                    "default_samplerate": dev.get("default_samplerate"),
                }
            )
        return jsonify({"devices": sanitized})

    # ------------------------------------------------------------------
    # API: Rooms
    # ------------------------------------------------------------------

    @app.route("/api/rooms", methods=["GET"])
    def api_rooms():
        """GET /api/rooms — fetch public rooms from the signaling server."""
        try:
            rooms = app.session_manager.list_rooms()
            return jsonify({"rooms": rooms})
        except Exception as exc:
            logger.exception("Failed to list rooms")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/room", methods=["POST"])
    def api_create_room():
        """POST /api/room — create a new room (payload: room_id, is_public, title)."""
        payload = request.get_json(force=True)
        room_id = payload.get("room_id")
        if not room_id:
            return jsonify({"error": "room_id is required"}), 400
        try:
            result = app.session_manager.create_room(
                room_id=room_id,
                is_public=bool(payload.get("is_public", False)),
                title=payload.get("title"),
            )
            return jsonify(result)
        except Exception as exc:
            logger.exception("Failed to create room")
            return jsonify({"error": str(exc)}), 500

    # ------------------------------------------------------------------
    # API: Streaming / Listening / Stop
    # ------------------------------------------------------------------

    @app.route("/api/stream", methods=["POST"])
    def api_start_stream():
        """POST /api/stream — start streaming mic audio (payload: room_id, device, is_public, title)."""
        payload = request.get_json(force=True)
        room_id = payload.get("room_id")
        device = payload.get("device")
        is_public = bool(payload.get("is_public", False))
        title = payload.get("title")

        try:
            app.session_manager.start_streaming(
                room_id=room_id,
                device=device,
                is_public=is_public,
                title=title,
            )
            return jsonify({"success": True})
        except Exception as exc:
            logger.exception("Failed to start streaming")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/listen", methods=["POST"])
    def api_start_listen():
        """POST /api/listen — start listening to a room (payload: room_id, device)."""
        payload = request.get_json(force=True)
        room_id = payload.get("room_id")
        if not room_id:
            return jsonify({"error": "room_id is required"}), 400
        device = payload.get("device")

        try:
            app.session_manager.start_listening(room_id=room_id, device=device)
            return jsonify({"success": True})
        except Exception as exc:
            logger.exception("Failed to start listening")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        """POST /api/stop — stop the current streaming or listening session."""
        try:
            app.session_manager.stop()
            return jsonify({"success": True})
        except Exception as exc:
            logger.exception("Failed to stop session")
            return jsonify({"error": str(exc)}), 500

    # ------------------------------------------------------------------
    # API: Chat
    # ------------------------------------------------------------------

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        """POST /api/chat — send a chat message (payload: message)."""
        payload = request.get_json(force=True)
        message = payload.get("message")
        if not message:
            return jsonify({"error": "message is required"}), 400
        try:
            app.session_manager.send_chat(message)
            return jsonify({"success": True})
        except Exception as exc:
            logger.exception("Failed to send chat")
            return jsonify({"error": str(exc)}), 500

    return app


# ------------------------------------------------------------------------------
# Standalone entry point — useful for quick local testing.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    _port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    _app = create_app()
    _app.run(host="0.0.0.0", port=_port, debug=True)
