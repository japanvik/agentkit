"""BusParticipant — base class for all networkkit bus clients."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from abc import ABC, abstractmethod
from typing import Any

import zmq
import zmq.asyncio

from networkkit.messages import Message, MessageType

from agentkit.bus.config import BusConfig

log = logging.getLogger("agentkit.bus")


class BusParticipant(ABC):
    """Base class for any process that participates on the networkkit bus.

    Subclass and implement:
        - is_intended_for_me(msg) -> bool
        - handle_message(msg) -> None

    Optionally override:
        - on_start() — called after HELO, before message loop
        - on_stop() — called on shutdown
    """

    name: str = "unnamed"

    SILENT_DEATH_TIMEOUT = 90  # seconds without any ZMQ message triggers reconnect

    def __init__(self, config: BusConfig):
        self.config = config
        if not self.name or self.name == "unnamed":
            self.name = config.name
        self._running = False
        self._http_session = None
        self._zmq_ctx: zmq.asyncio.Context | None = None
        self._zmq_sub: zmq.asyncio.Socket | None = None
        self._last_recv_time: float = 0.0
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    # ── Abstract interface ──────────────────────────────────────────

    @abstractmethod
    def is_intended_for_me(self, message: Message) -> bool:
        """Return True if this message should be handled by this participant."""
        ...

    @abstractmethod
    async def handle_message(self, message: Message) -> None:
        """Process an incoming message."""
        ...

    # ── Lifecycle hooks ─────────────────────────────────────────────

    async def on_start(self) -> None:
        """Called after HELO sent, before entering the message loop."""
        pass

    async def on_stop(self) -> None:
        """Called on graceful shutdown, before cleanup."""
        pass

    # ── Public API ──────────────────────────────────────────────────

    def run(self) -> None:
        """Start the participant (blocking). Sets up bus connections and enters message loop."""
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            pass

    async def send(self, to: str, content: str | dict, message_type: MessageType = MessageType.CHAT, in_reply_to: str = None, id: str = None) -> None:
        """Send a message to the bus."""
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)

        msg = Message(
            source=self.name,
            to=to,
            content=content,
            message_type=message_type,
            in_reply_to=in_reply_to,
            id=id,
        )
        await self._http_send(msg)

    async def send_helo(self) -> None:
        """Broadcast HELO to all peers."""
        payload = json.dumps({
            "type": "HELO",
            "agent": self.name,
            "description": self.config.description or f"{self.name} on the network.",
        })
        msg = Message(
            source=self.name,
            to="ALL",
            content=payload,
            message_type=MessageType.SYSTEM,
        )
        await self._http_send(msg)
        log.info("HELO broadcast sent for '%s'", self.name)

    async def send_ack(self, peer_name: str) -> None:
        """Send ACK to a specific peer."""
        payload = json.dumps({
            "type": "ACK",
            "agent": self.name,
            "description": self.config.description or f"{self.name} on the network.",
        })
        msg = Message(
            source=self.name,
            to=peer_name,
            content=payload,
            message_type=MessageType.SYSTEM,
        )
        await self._http_send(msg)
        log.info("ACK sent to %s", peer_name)

    # ── Internal ────────────────────────────────────────────────────

    async def _run_async(self) -> None:
        loop = asyncio.get_running_loop()

        # Signal handling
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._request_stop)

        # Connect
        await self._connect()
        await self.send_helo()
        await self.on_start()

        log.info("%s running (http=%s zmq=%s)", self.name, self.config.bus_http_url, self.config.bus_zmq_address)
        self._running = True

        # Start HELO beacon
        self._beacon_task = asyncio.create_task(self._helo_beacon())

        # Message loop
        try:
            await self._message_loop()
        finally:
            self._beacon_task.cancel()
            await self._shutdown()

    async def _helo_beacon(self) -> None:
        """Periodically re-send HELO to keep routing table entries fresh."""
        while self._running:
            await asyncio.sleep(self.config.helo_interval)
            if self._running:
                try:
                    await self.send_helo()
                except Exception:
                    log.debug("HELO beacon failed (will retry)")

    async def _connect(self) -> None:
        import aiohttp
        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        self._zmq_ctx = zmq.asyncio.Context()
        self._zmq_sub = self._zmq_ctx.socket(zmq.SUB)
        self._zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._zmq_sub.setsockopt(zmq.RCVTIMEO, 5000)
        self._zmq_sub.connect(self.config.bus_zmq_address)
        self._last_recv_time = time.time()

    async def _message_loop(self) -> None:
        reconnect_delay = 1
        max_reconnect_delay = 30

        while self._running:
            try:
                raw = await self._zmq_sub.recv_json(flags=zmq.NOBLOCK)
                self._last_recv_time = time.time()
                message = Message.model_validate(raw)

                # Handle HELO/ACK at base level
                if message.message_type == MessageType.SYSTEM.value:
                    self._handle_system(message)
                    continue

                if self.is_intended_for_me(message):
                    try:
                        await self.handle_message(message)
                    except Exception:
                        log.exception("Error handling message from %s", message.source)

                reconnect_delay = 1  # reset on success

            except zmq.Again:
                # Check for silent death: no messages received for too long
                silence = time.time() - self._last_recv_time
                if silence > self.SILENT_DEATH_TIMEOUT:
                    log.warning(
                        "ZMQ silent death detected (no messages for %.0fs) — reconnecting",
                        silence,
                    )
                    await self._reconnect_zmq()
                    self._last_recv_time = time.time()
                    await self.send_helo()
                else:
                    await asyncio.sleep(0.1)
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break
                log.error("ZMQ error: %s — reconnecting in %ds", e, reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                await self._reconnect_zmq()
            except asyncio.CancelledError:
                break

    def _handle_system(self, message: Message) -> None:
        """Handle HELO/ACK system messages."""
        try:
            payload = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            return

        msg_type = payload.get("type")
        peer_name = payload.get("agent", "")

        if not peer_name or peer_name == self.name:
            return

        if msg_type == "HELO":
            self.on_peer_discovered(peer_name, payload.get("description", ""))
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.send_ack(peer_name))
            except RuntimeError:
                pass  # no loop running (test context)
        elif msg_type == "ACK":
            self.on_peer_discovered(peer_name, payload.get("description", ""))

    def on_peer_discovered(self, peer_name: str, description: str) -> None:
        """Called when a peer is discovered via HELO/ACK. Override for custom behavior."""
        log.info("Peer discovered: %s — %s", peer_name, description[:60])

    async def _reconnect_zmq(self) -> None:
        """Reconnect ZMQ subscriber socket."""
        try:
            if self._zmq_sub:
                self._zmq_sub.close(linger=0)
        except Exception:
            pass

        self._zmq_sub = self._zmq_ctx.socket(zmq.SUB)
        self._zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._zmq_sub.setsockopt(zmq.RCVTIMEO, 5000)
        self._zmq_sub.connect(self.config.bus_zmq_address)
        log.info("ZMQ reconnected to %s", self.config.bus_zmq_address)

    async def _http_send(self, message: Message) -> None:
        """Send a message via HTTP POST to the bus."""
        if not self._http_session:
            import aiohttp
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        try:
            async with self._http_session.post(
                f"{self.config.bus_http_url}/data",
                json=message.model_dump(),
            ) as resp:
                if resp.status != 200:
                    log.warning("HTTP send failed: %d", resp.status)
        except Exception as e:
            log.warning("HTTP send error: %s", e)

    def _request_stop(self) -> None:
        """Signal handler to initiate graceful shutdown."""
        log.info("Shutdown requested for %s", self.name)
        self._running = False

    async def _shutdown(self) -> None:
        """Clean up resources."""
        log.info("Shutting down %s...", self.name)
        await self.on_stop()

        if self._zmq_sub:
            try:
                self._zmq_sub.close(linger=0)
            except Exception:
                pass
        if self._zmq_ctx:
            try:
                self._zmq_ctx.term()
            except Exception:
                pass
        if self._http_session:
            await self._http_session.close()

        log.info("%s stopped.", self.name)
