from __future__ import annotations

import logging
import socket

logger = logging.getLogger('soundstream')


def detect_primary_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def setup_ice(stun_timeout: int = 2) -> None:
    import aioice.ice

    primary_ip = detect_primary_ip()

    _orig_get = aioice.ice.get_host_addresses

    def filtered(use_ipv4, use_ipv6):
        if primary_ip and use_ipv4:
            return [primary_ip]
        return _orig_get(use_ipv4, use_ipv6)

    aioice.ice.get_host_addresses = filtered

    _orig_gcc = aioice.ice.Connection.get_component_candidates

    async def patched(self, component, addresses, timeout=5):
        return await _orig_gcc(self, component, addresses, timeout=stun_timeout)

    aioice.ice.Connection.get_component_candidates = patched

    if primary_ip:
        logger.info(f'[ice] Primary interface: {primary_ip} (STUN timeout: {stun_timeout}s)')
    else:
        logger.info(f'[ice] Could not detect primary IP, all interfaces (STUN timeout: {stun_timeout}s)')
