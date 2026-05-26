#!/usr/bin/env python3
"""
SoundStream P2P Audio Signaling Server

A lightweight Python REST API for peer discovery and WebRTC connection handshake.

Usage:
    python server.py [--host HOST] [--port PORT]

Environment variables:
    HOST, PORT, DB_PATH
"""

import argparse
import json
import os
import sqlite3
import threading
import time
import uuid
from collections import defaultdict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5642))
DB_PATH = os.environ.get("DB_PATH", "soundstream.db")

ROOM_MAX_LIFETIME = 12 * 3600
ROOM_INACTIVE_TIMEOUT = 120
SIGNALING_TTL = 300
CHAT_TTL = 300
LONG_POLL_TIMEOUT = 10
DEFAULT_LISTENER_CAP = 10
MAX_BODY_SIZE = 1 * 1024 * 1024
CLEANUP_INTERVAL = 60
SSE_KEEPALIVE_INTERVAL = 15


_sse_subscribers: dict[str, list[tuple[str, threading.Event]]] = defaultdict(list)
_sse_lock = threading.Lock()


def sse_publish(room_id: str, signal: dict):
    with _sse_lock:
        subs = list(_sse_subscribers.get(room_id, []))
    for sub_id, event in subs:
        event.signal = signal
        event.set()


class _SSESubscriber:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.sub_id = str(uuid.uuid4())[:8]
        self.event = threading.Event()
        self.signal: dict | None = None
        with _sse_lock:
            _sse_subscribers[room_id].append((self.sub_id, self.event))

    def wait(self, timeout: float) -> dict | None:
        if self.event.wait(timeout):
            sig = self.signal
            self.signal = None
            self.event.clear()
            return sig
        return None

    def remove(self):
        with _sse_lock:
            subs = _sse_subscribers.get(self.room_id, [])
            _sse_subscribers[self.room_id] = [
                (sid, ev) for sid, ev in subs if sid != self.sub_id
            ]


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def log(action, message, color=Colors.BLUE):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(
        f"{Colors.CYAN}[{timestamp}]{Colors.ENDC} {color}{action:20s}{Colors.ENDC} {message}"
    )


def log_request(handler, action, message="", color=Colors.BLUE):
    ip = get_client_ip(handler)
    log(action, f"{ip} - {message}", color)


local = threading.local()
_db_lock = threading.Lock()


def get_db():
    if not hasattr(local, "conn"):
        local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        local.conn.row_factory = sqlite3.Row
        local.conn.execute("PRAGMA journal_mode=WAL")
        init_db(local.conn)
    return local.conn


def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id TEXT UNIQUE NOT NULL,
            streamer_ip TEXT NOT NULL,
            session_key TEXT NOT NULL,
            heartbeat TEXT NOT NULL,
            listener_count INTEGER DEFAULT 0,
            listener_cap INTEGER DEFAULT 10,
            is_public INTEGER DEFAULT 0,
            banned_ips TEXT DEFAULT '[]',
            title TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS signaling (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id TEXT NOT NULL,
            type TEXT NOT NULL,
            sdp TEXT,
            client_ip TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id TEXT NOT NULL,
            sender_ip TEXT NOT NULL,
            sender_session_id TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()


def get_client_ip(handler):
    return handler.client_address[0]


def clean_expired(conn):
    with _db_lock:
        conn.execute(
            """
            DELETE FROM rooms
            WHERE heartbeat < datetime('now', '-' || ? || ' seconds')
            OR created_at < datetime('now', '-' || ? || ' seconds')
        """,
            (ROOM_INACTIVE_TIMEOUT, ROOM_MAX_LIFETIME),
        )
        conn.execute(
            """
            DELETE FROM signaling
            WHERE created_at < datetime('now', '-' || ? || ' seconds')
        """,
            (SIGNALING_TTL,),
        )
        conn.execute(
            """
            DELETE FROM chat_messages
            WHERE created_at < datetime('now', '-' || ? || ' seconds')
        """,
            (CHAT_TTL,),
        )
        conn.commit()


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        log("HTTP", f"{self.address_string()} - {format % args}", Colors.YELLOW)

    def handle(self):
        try:
            super().handle()
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, data, status=200):
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header(
                "Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"
            )
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        except (ConnectionAbortedError, BrokenPipeError):
            pass

    def send_error_json(self, message, status):
        self.send_json({"success": False, "error": message}, status)

    def send_success_json(self, data=None):
        if data is None:
            data = {}
        data["success"] = True
        self.send_json(data)

    def get_json_body(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return {}
        if content_length > MAX_BODY_SIZE:
            raise ValueError(f"Request body too large (max {MAX_BODY_SIZE} bytes)")
        body = self.rfile.read(content_length)
        return json.loads(body.decode())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        action = params.get("action", [""])[0]
        room_id = params.get("room_id", [""])[0]
        client_ip = get_client_ip(self)

        conn = get_db()

        try:
            if action == "rooms":
                clean_expired(conn)
                cursor = conn.execute("""
                    SELECT room_id, title, listener_count, listener_cap, created_at
                    FROM rooms WHERE is_public = 1
                    ORDER BY created_at DESC LIMIT 100
                """)
                rooms = [dict(row) for row in cursor.fetchall()]
                log_request(self, "LIST_ROOMS", f"Found {len(rooms)} rooms")
                self.send_success_json({"rooms": rooms})

            elif action == "room" and room_id:
                cursor = conn.execute(
                    """
                    SELECT room_id, listener_count, listener_cap, is_public, title, created_at
                    FROM rooms WHERE room_id = ?
                """,
                    (room_id,),
                )
                room = cursor.fetchone()
                if room:
                    log_request(self, "GET_ROOM", f"Room: {room_id}")
                    self.send_success_json({"room": dict(room)})
                else:
                    log_request(
                        self, "GET_ROOM", f"Room not found: {room_id}", Colors.RED
                    )
                    self.send_error_json("Room not found", 404)

            elif action == "signaling" and room_id:
                log_request(self, "POLL_SIGNAL", f"Room: {room_id}")
                self.handle_long_poll(room_id, params)

            elif action == "signals.stream" and room_id:
                log_request(self, "SSE_CONNECT", f"Room: {room_id}")
                self.handle_sse_stream(room_id, params)

            elif action == "chat.poll" and room_id:
                log_request(self, "CHAT_POLL", f"Room: {room_id}")
                self.handle_chat_poll(conn, room_id, client_ip, params)

            else:
                self.send_error_json("Not Found", 404)

        except Exception as e:
            if "since_id" in str(e) or isinstance(e, ValueError):
                self.send_error_json(str(e), 400)
            else:
                self.send_error_json(str(e), 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        action = params.get("action", [""])[0]
        room_id = params.get("room_id", [""])[0]
        client_ip = get_client_ip(self)

        try:
            data = self.get_json_body()
        except ValueError as e:
            self.send_error_json(str(e), 400)
            return

        conn = get_db()

        try:
            if action == "room.create":
                self.handle_create_room(conn, data, client_ip)

            elif action == "room.heartbeat" and room_id:
                self.handle_heartbeat(conn, room_id, client_ip)

            elif action == "listener.join" and room_id:
                self.handle_listener_join(conn, room_id, client_ip)

            elif action == "listener.leave" and room_id:
                self.handle_listener_leave(conn, room_id, client_ip)

            elif action == "room.ban" and room_id:
                self.handle_ban(conn, room_id, data, client_ip)

            elif action == "signaling" and room_id:
                self.handle_signaling(conn, room_id, data, client_ip)

            elif action == "chat.send" and room_id:
                self.handle_chat_send(conn, room_id, data, client_ip)

            else:
                self.send_error_json("Not Found", 404)

        except Exception as e:
            self.send_error_json(str(e), 500)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        room_id = params.get("room_id", [""])[0]
        client_ip = get_client_ip(self)

        if room_id:
            conn = get_db()
            try:
                self.handle_delete_room(conn, room_id, client_ip)
            except Exception as e:
                self.send_error_json(str(e), 500)
        else:
            self.send_error_json("Not Found", 404)

    def handle_create_room(self, conn, data, client_ip):
        room_id = data.get("room_id")
        session_key = data.get("session_key")

        if not room_id:
            self.send_error_json("Missing required field: room_id", 400)
            return

        is_public = 1 if data.get("is_public", False) else 0
        listener_cap = data.get("listener_cap", DEFAULT_LISTENER_CAP)
        title = data.get("title")

        with _db_lock:
            conn.execute(
                """
                INSERT INTO rooms (room_id, streamer_ip, session_key, heartbeat, is_public, listener_cap, title)
                VALUES (?, ?, ?, datetime('now'), ?, ?, ?)
                ON CONFLICT(room_id) DO UPDATE SET
                    session_key = excluded.session_key,
                    heartbeat = datetime('now'),
                    is_public = excluded.is_public,
                    listener_cap = excluded.listener_cap,
                    title = excluded.title,
                    listener_count = 0
            """,
                (room_id, client_ip, session_key, is_public, listener_cap, title),
            )
            conn.commit()

        log("CREATE_ROOM", f"Room '{room_id}' created by {client_ip}", Colors.GREEN)
        self.send_success_json(
            {
                "room_id": room_id,
                "is_public": bool(is_public),
                "listener_cap": listener_cap,
            }
        )

    def handle_heartbeat(self, conn, room_id, client_ip):
        cursor = conn.execute(
            "SELECT streamer_ip FROM rooms WHERE room_id = ?", (room_id,)
        )
        room = cursor.fetchone()

        if not room or room["streamer_ip"] != client_ip:
            self.send_error_json("Room not found or unauthorized", 404)
            return

        with _db_lock:
            conn.execute(
                'UPDATE rooms SET heartbeat = datetime("now") WHERE room_id = ?',
                (room_id,),
            )
            conn.commit()
        log("HEARTBEAT", f"Room '{room_id}' heartbeat from {client_ip}", Colors.GREEN)
        self.send_success_json()

    def handle_listener_join(self, conn, room_id, client_ip):
        cursor = conn.execute(
            "SELECT streamer_ip, listener_count, listener_cap, banned_ips FROM rooms WHERE room_id = ?",
            (room_id,),
        )
        room = cursor.fetchone()

        if not room:
            self.send_error_json("Room not found", 404)
            return

        banned_ips = json.loads(room["banned_ips"] or "[]")
        if client_ip in banned_ips:
            self.send_error_json("You are banned from this room", 403)
            return

        if room["listener_count"] >= room["listener_cap"]:
            self.send_error_json("Room is full", 403)
            return

        with _db_lock:
            conn.execute(
                "UPDATE rooms SET listener_count = listener_count + 1 WHERE room_id = ?",
                (room_id,),
            )
            conn.commit()

        log("LISTENER_JOIN", f"Room '{room_id}' - IP: {client_ip}", Colors.GREEN)
        self.send_success_json({"listener_count": room["listener_count"] + 1})

    def handle_listener_leave(self, conn, room_id, client_ip):
        cursor = conn.execute(
            "SELECT streamer_ip, listener_count FROM rooms WHERE room_id = ?",
            (room_id,),
        )
        room = cursor.fetchone()

        if not room:
            self.send_error_json("Room not found", 404)
            return

        if room["listener_count"] <= 0:
            self.send_success_json({"listener_count": 0})
            return

        with _db_lock:
            conn.execute(
                "UPDATE rooms SET listener_count = MAX(listener_count - 1, 0) WHERE room_id = ?",
                (room_id,),
            )
            conn.commit()

        log("LISTENER_LEAVE", f"Room '{room_id}' - IP: {client_ip}", Colors.YELLOW)
        self.send_success_json({"listener_count": room["listener_count"] - 1})

    def handle_ban(self, conn, room_id, data, client_ip):
        cursor = conn.execute(
            "SELECT streamer_ip, banned_ips FROM rooms WHERE room_id = ?", (room_id,)
        )
        room = cursor.fetchone()

        if not room:
            self.send_error_json("Room not found", 404)
            return

        if room["streamer_ip"] != client_ip:
            self.send_error_json("Unauthorized", 403)
            return

        listener_ip = data.get("listener_ip")
        if not listener_ip:
            self.send_error_json("Missing required field: listener_ip", 400)
            return

        banned_ips = json.loads(room["banned_ips"] or "[]")

        if listener_ip not in banned_ips:
            banned_ips.append(listener_ip)
            with _db_lock:
                conn.execute(
                    "UPDATE rooms SET banned_ips = ? WHERE room_id = ?",
                    (json.dumps(banned_ips), room_id),
                )
                conn.commit()

        self.send_success_json(
            {
                "action": "ban",
                "listener_ip": listener_ip,
            }
        )

    def handle_delete_room(self, conn, room_id, client_ip):
        cursor = conn.execute(
            "SELECT streamer_ip FROM rooms WHERE room_id = ?", (room_id,)
        )
        room = cursor.fetchone()

        if not room or room["streamer_ip"] != client_ip:
            self.send_error_json("Room not found or unauthorized", 404)
            return

        with _db_lock:
            conn.execute("DELETE FROM rooms WHERE room_id = ?", (room_id,))
            conn.execute("DELETE FROM signaling WHERE room_id = ?", (room_id,))
            conn.commit()
        log("DELETE_ROOM", f"Room '{room_id}' deleted by {client_ip}", Colors.YELLOW)
        self.send_success_json({"message": "Room deleted"})

    def handle_signaling(self, conn, room_id, data, client_ip):
        cursor = conn.execute(
            "SELECT room_id, banned_ips FROM rooms WHERE room_id = ?", (room_id,)
        )
        room = cursor.fetchone()

        if not room:
            self.send_error_json("Room not found", 404)
            return

        banned_ips = json.loads(room["banned_ips"] or "[]")
        if client_ip in banned_ips:
            self.send_error_json("You are banned from this room", 403)
            return

        signal_type = data.get("type")
        if signal_type not in ("offer", "answer", "chat"):
            self.send_error_json(
                'Invalid signal type. Must be "offer", "answer" or "chat"', 400
            )
            return

        sdp = data.get("sdp")

        if signal_type in ("answer", "chat"):
            cursor = conn.execute(
                "SELECT streamer_ip FROM rooms WHERE room_id = ?", (room_id,)
            )
            room = cursor.fetchone()
            if room["streamer_ip"] != client_ip:
                self.send_error_json(
                    "Only the streamer can submit answers or chat", 403
                )
                return

        with _db_lock:
            cursor = conn.execute(
                """
                INSERT INTO signaling (room_id, type, sdp, client_ip)
                VALUES (?, ?, ?, ?)
            """,
                (room_id, signal_type, sdp, client_ip),
            )
            conn.commit()

        log(
            "SIGNAL",
            f"Room '{room_id}' - {signal_type.upper()} from {client_ip}",
            Colors.GREEN,
        )

        cursor.execute("SELECT last_insert_rowid() as id")
        signal_id = cursor.fetchone()["id"]

        self.send_success_json({"signal_id": signal_id})

        sse_publish(
            room_id,
            {
                "id": signal_id,
                "type": signal_type,
                "sdp": sdp,
            },
        )

    def handle_chat_send(self, conn, room_id, data, client_ip):
        """
        Gintautas: Store a chat message from a listener into the chat_messages queue.
        The streamer will poll and re-broadcast these via the signaling table.
        """
        cursor = conn.execute(
            "SELECT room_id, banned_ips FROM rooms WHERE room_id = ?", (room_id,)
        )
        room = cursor.fetchone()

        if not room:
            self.send_error_json("Room not found", 404)
            return

        banned_ips = json.loads(room["banned_ips"] or "[]")
        if client_ip in banned_ips:
            self.send_error_json("You are banned from this room", 403)
            return

        message = data.get("message", "").strip()
        if not message:
            self.send_error_json("Missing required field: message", 400)
            return

        sender_session_id = data.get("sender_session_id", "").strip()

        with _db_lock:
            conn.execute(
                """
                INSERT INTO chat_messages (room_id, sender_ip, sender_session_id, message)
                VALUES (?, ?, ?, ?)
            """,
                (room_id, client_ip, sender_session_id, message),
            )
            conn.commit()

        log(
            "CHAT_SEND",
            f"Room '{room_id}' - chat from {client_ip}:{sender_session_id}",
            Colors.GREEN,
        )
        self.send_success_json()

    def handle_chat_poll(self, conn, room_id, client_ip, params):
        """
        Gintautas: Poll chat messages for a room. Only the streamer may poll.
        Messages are returned and then deleted so they are not redelivered.
        """
        cursor = conn.execute(
            "SELECT streamer_ip FROM rooms WHERE room_id = ?", (room_id,)
        )
        room = cursor.fetchone()

        if not room:
            self.send_error_json("Room not found", 404)
            return

        if room["streamer_ip"] != client_ip:
            self.send_error_json("Only the streamer can poll chat messages", 403)
            return

        try:
            since_id = int(params.get("since_id", ["0"])[0])
        except (ValueError, TypeError):
            self.send_error_json("Invalid since_id parameter", 400)
            return

        cursor = conn.execute(
            """
            SELECT id, sender_ip, sender_session_id, message
            FROM chat_messages
            WHERE room_id = ? AND id > ?
            ORDER BY id ASC
        """,
            (room_id, since_id),
        )
        rows = cursor.fetchall()

        messages = [
            {
                "id": row["id"],
                "sender_ip": row["sender_ip"],
                "sender_session_id": row["sender_session_id"],
                "message": row["message"],
            }
            for row in rows
        ]

        if rows:
            max_id = rows[-1]["id"]
            with _db_lock:
                conn.execute(
                    "DELETE FROM chat_messages WHERE room_id = ? AND id <= ?",
                    (room_id, max_id),
                )
                conn.commit()

        log(
            "CHAT_POLL",
            f"Room '{room_id}' - {len(messages)} messages polled by {client_ip}",
            Colors.CYAN,
        )
        self.send_success_json({"messages": messages})

    def handle_long_poll(self, room_id, params):
        conn = get_db()
        cursor = conn.execute(
            "SELECT room_id, banned_ips FROM rooms WHERE room_id = ?", (room_id,)
        )
        room = cursor.fetchone()

        if not room:
            self.send_error_json("Room not found", 404)
            return

        client_ip = get_client_ip(self)
        banned_ips = json.loads(room["banned_ips"] or "[]")
        if client_ip in banned_ips:
            self.send_error_json("You are banned from this room", 403)
            return

        try:
            since_id = int(params.get("since_id", [0])[0])
        except (ValueError, TypeError):
            self.send_error_json("Invalid since_id parameter", 400)
            return

        log(
            "POLL",
            f"Room '{room_id}' - Waiting for signal (since_id={since_id})",
            Colors.CYAN,
        )

        start_time = time.time()
        while time.time() - start_time < LONG_POLL_TIMEOUT:
            cursor = conn.execute(
                """
                SELECT id, type, sdp
                FROM signaling
                WHERE room_id = ? AND id > ?
                ORDER BY id ASC LIMIT 1
            """,
                (room_id, since_id),
            )
            signal = cursor.fetchone()

            if signal:
                log(
                    "POLL",
                    f"Room '{room_id}' - Signal found: {signal['type']}",
                    Colors.GREEN,
                )
                self.send_success_json(
                    {
                        "signal": {
                            "id": signal["id"],
                            "type": signal["type"],
                            "sdp": signal["sdp"],
                        }
                    }
                )
                return

            time.sleep(0.5)

        log("POLL", f"Room '{room_id}' - Poll timeout", Colors.YELLOW)
        self.send_success_json({"signal": None, "timeout": True})

    def handle_sse_stream(self, room_id, params):
        conn = get_db()
        cursor = conn.execute(
            "SELECT room_id, banned_ips FROM rooms WHERE room_id = ?", (room_id,)
        )
        room = cursor.fetchone()

        if not room:
            self.send_error_json("Room not found", 404)
            return

        client_ip = get_client_ip(self)
        banned_ips = json.loads(room["banned_ips"] or "[]")
        if client_ip in banned_ips:
            self.send_error_json("You are banned from this room", 403)
            return

        try:
            since_id = int(params.get("since_id", [0])[0])
        except (ValueError, TypeError):
            self.send_error_json("Invalid since_id parameter", 400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

        sub = _SSESubscriber(room_id)
        log(
            "SSE_STREAM",
            f"Room '{room_id}' - Subscriber {sub.sub_id} connected",
            Colors.CYAN,
        )

        try:
            conn_start = time.time()
            last_keepalive = time.time()
            while time.time() - conn_start < 3600:
                if time.time() - last_keepalive >= SSE_KEEPALIVE_INTERVAL:
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except (
                        ConnectionAbortedError,
                        BrokenPipeError,
                        ConnectionResetError,
                    ):
                        break
                    last_keepalive = time.time()

                cursor = conn.execute(
                    """
                    SELECT id, type, sdp
                    FROM signaling
                    WHERE room_id = ? AND id > ?
                    ORDER BY id ASC LIMIT 1
                """,
                    (room_id, since_id),
                )
                signal = cursor.fetchone()

                if signal:
                    since_id = signal["id"]
                    log(
                        "SSE_STREAM",
                        f"Room '{room_id}' - Sending signal: {signal['type']}",
                        Colors.GREEN,
                    )
                    event_data = json.dumps(
                        {
                            "id": signal["id"],
                            "type": signal["type"],
                            "sdp": signal["sdp"],
                        }
                    )
                    self.wfile.write(f"data: {event_data}\n\n".encode())
                    self.wfile.flush()
                    continue

                pushed = sub.wait(0.5)
                if pushed and pushed["id"] > since_id:
                    since_id = pushed["id"]
                    log(
                        "SSE_STREAM",
                        f"Room '{room_id}' - Pushed signal: {pushed['type']}",
                        Colors.GREEN,
                    )
                    event_data = json.dumps(pushed)
                    self.wfile.write(f"data: {event_data}\n\n".encode())
                    self.wfile.flush()
                    continue
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass
        finally:
            sub.remove()
            log(
                "SSE_STREAM",
                f"Room '{room_id}' - Subscriber {sub.sub_id} disconnected",
                Colors.YELLOW,
            )


def _cleanup_loop():
    while True:
        try:
            time.sleep(CLEANUP_INTERVAL)
            conn = get_db()
            clean_expired(conn)
            log("CLEANUP", "Expired rooms and signals cleaned")
        except Exception as e:
            log("CLEANUP", f"Error: {e}", Colors.RED)


def run_server():
    parser = argparse.ArgumentParser(description="SoundStream Signaling Server")
    parser.add_argument("--host", default=HOST, help="Bind address (default: 0.0.0.0)")
    parser.add_argument(
        "--port", type=int, default=PORT, help="Bind port (default: 5642)"
    )
    args = parser.parse_args()

    host = args.host
    port = args.port

    conn = get_db()
    init_db(conn)

    cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
    cleanup_thread.start()

    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"\n{Colors.BOLD}{Colors.GREEN}")
    print("=" * 50)
    print("  SoundStream Signaling Server")
    print("=" * 50)
    print(f"{Colors.ENDC}")
    print(f"{Colors.CYAN}URL:{Colors.ENDC}     http://{host}:{port}")
    print(f"{Colors.CYAN}DB:{Colors.ENDC}      {DB_PATH}")
    print(f"{Colors.CYAN}PID:{Colors.ENDC}     {os.getpid()}")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Shutting down...{Colors.ENDC}")
        server.shutdown()


if __name__ == "__main__":
    run_server()
