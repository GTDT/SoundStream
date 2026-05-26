from __future__ import annotations

import argparse
import asyncio
import sys

import sounddevice as sd
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from ._version import __version__
from .audio import (
    AutoGainControl,
    EffectChain,
    NoiseSuppressor,
    OpusCompressor,
    SpectralNoiseReducer,
)
from .session import SoundSession


# Gintautas: Shared prompt session for async CLI input with concurrent output.
_prompt_session = PromptSession()


def cmd_devices(_args: argparse.Namespace) -> None:
    devices = sd.query_devices()
    print("\nAudio Devices:")
    print(f"  {'Index':<6} {'Name':<40} {'Max In':<8} {'Max Out':<8}")
    print(f"  {'-' * 6} {'-' * 40} {'-' * 8} {'-' * 8}")
    for i, dev in enumerate(devices):
        name = dev["name"][:38]
        print(
            f"  {i:<6} {name:<40} {dev['max_input_channels']:<8} {dev['max_output_channels']:<8}"
        )

    default_in = sd.default.device[0]
    default_out = sd.default.device[1]
    print(f"\n  Default Input:  {default_in} - {devices[default_in]['name']}")
    print(f"  Default Output: {default_out} - {devices[default_out]['name']}")


def cmd_list(args: argparse.Namespace) -> None:
    async def _list() -> None:
        from .session import SoundSession

        session = SoundSession(args.server)
        try:
            rooms = await session.list_rooms()
            if not rooms:
                print("No public rooms found.")
                return
            print(f"\nPublic Rooms ({len(rooms)}):")
            print(f"  {'Room ID':<20} {'Title':<25} {'Listeners':<12} {'Created'}")
            print(f"  {'-' * 20} {'-' * 25} {'-' * 12} {'-' * 19}")
            for room in rooms:
                rid = room["room_id"][:18]
                title = (room.get("title") or "Untitled")[:23]
                lc = f"{room['listener_count']}/{room['listener_cap']}"
                created = room.get("created_at", "")[:19]
                print(f"  {rid:<20} {title:<25} {lc:<12} {created}")
        finally:
            await session.close()

    asyncio.run(_list())


def setup_enhanced_session(server_url: str):
    """Create a SoundSession with an enhanced audio processing chain.

    Configures a session with spectral noise reduction, noise suppression,
    automatic gain control, and opus compression for improved stream quality.
    """
    session = SoundSession(server_url)

    chain = EffectChain(
        [
            SpectralNoiseReducer(noise_floor_len=10),
            NoiseSuppressor(threshold=0.001),
            AutoGainControl(target_rms=0.15),
            OpusCompressor(),
        ]
    )

    session.audio_processor = chain

    return session


async def _chat_input_loop(session) -> None:
    """
    Gintautas: Background loop that reads user input via prompt_toolkit
    and sends it through the active chat manager.
    """
    while session.state in ("streaming", "listening"):
        try:
            with patch_stdout():
                text = await _prompt_session.prompt_async("> ")
            if text.strip():
                await session.send_chat(text.strip())
        except asyncio.CancelledError:
            break
        except Exception:
            pass


def cmd_stream(args: argparse.Namespace) -> None:
    """Start streaming microphone audio to a room with enhanced processing.

    Creates an enhanced session with audio effects and begins streaming
    to the specified room. Enables the chat manager in streamer mode
    so incoming listener messages are re-broadcast.
    """
    from .chat import SignalChatManager

    async def _stream() -> None:
        session = setup_enhanced_session(args.server)

        session.input_device = args.device
        session.on_state_change = lambda old, new: print(f"State: {new}")

        # Gintautas: Wire chat messages to stdout without jumbling the prompt.
        session.on_chat_message = lambda r, m: print(f"[CHAT] {m}")

        try:
            await session.create_room(
                args.room, is_public=args.public, title=args.title
            )
            # Gintautas: Instantiate chat manager in streamer mode.
            session.chat_manager = SignalChatManager(
                session._signaling, mode="streamer"
            )
            await session.start_streaming()

            print(f'Streaming to room "{args.room}"... Press Ctrl+C to stop.')
            # Gintautas: Run chat input loop concurrently with the stream.
            chat_task = asyncio.create_task(_chat_input_loop(session))
            while session.state == "streaming":
                await asyncio.sleep(1)
            chat_task.cancel()
            try:
                await chat_task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            print("\nStopping stream...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await session.close()

    asyncio.run(_stream())


def cmd_listen(args: argparse.Namespace) -> None:
    """Join a room and play received audio.

    Configures the chat manager in listener mode so the user can send
    messages to the streamer and receive broadcast chat messages.
    """
    from .session import SoundSession
    from .chat import SignalChatManager

    async def _listen() -> None:
        session = SoundSession(args.server)
        session.output_device = args.device
        session.volume = args.volume
        session.on_state_change = lambda old, new: print(f"State: {new}")

        # Gintautas: Wire chat messages to stdout without jumbling the prompt.
        session.on_chat_message = lambda r, m: print(f"[CHAT] {m}")

        try:
            # Gintautas: Instantiate chat manager in listener mode.
            session.chat_manager = SignalChatManager(
                session._signaling, mode="listener"
            )
            await session.start_listening(args.room)

            print(f'Listening in room "{args.room}"... Press Ctrl+C to stop.')
            # Gintautas: Run chat input loop concurrently with playback.
            chat_task = asyncio.create_task(_chat_input_loop(session))
            while session.state == "listening":
                await asyncio.sleep(1)
            chat_task.cancel()
            try:
                await chat_task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            print("\nStopping listener...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await session.close()

    asyncio.run(_listen())


def cmd_web(args: argparse.Namespace) -> None:
    """Launch the Flask-based web UI."""
    from .web import create_app

    app = create_app()
    app.run(host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="soundstream",
        description="P2P audio streaming via WebRTC",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (verbose output)",
    )
    parser.add_argument(
        "--server",
        default="http://alpha.cubes.host:25566/",
        help="Signaling server URL (default: http://alpha.cubes.host:25566/)",
    )

    subparsers = parser.add_subparsers(dest="command")

    p_stream = subparsers.add_parser("stream", help="Stream mic audio to a room")
    p_stream.add_argument("--room", required=True, help="Room ID to create")
    p_stream.add_argument("--device", type=int, default=None, help="Input device index")
    p_stream.add_argument("--public", action="store_true", help="List room publicly")
    p_stream.add_argument("--title", default=None, help="Room title")

    p_listen = subparsers.add_parser("listen", help="Listen to audio in a room")
    p_listen.add_argument("--room", required=True, help="Room ID to join")
    p_listen.add_argument(
        "--device", type=int, default=None, help="Output device index"
    )
    p_listen.add_argument(
        "--volume", type=float, default=1.0, help="Playback volume (0.0-10.0)"
    )

    p_web = subparsers.add_parser("web", help="Launch web UI")
    p_web.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    p_web.add_argument(
        "--port", type=int, default=8765, help="Port to bind (default: 8765)"
    )

    subparsers.add_parser("list", help="List public rooms")
    subparsers.add_parser("devices", help="List audio devices")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.debug:
        from . import enable_debug

        enable_debug()

    commands = {
        "devices": cmd_devices,
        "list": cmd_list,
        "stream": cmd_stream,
        "listen": cmd_listen,
        "web": cmd_web,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
