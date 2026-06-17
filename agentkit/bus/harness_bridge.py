"""HarnessBridge — BusParticipant that bridges NetworkKit bus to a tmux-based agent.

Supports two harness types:
  - CC (Claude Code): detects idle via ❯ prompt, stale via pane comparison
  - Kiro CLI: detects idle via 'ask a question' marker, done via ▸ Credits marker

Handles:
  - Priority queue (agent > user > heartbeat)
  - HELO/ACK peer discovery
  - Telegram busy-ACK when processing
  - Stale session detection + auto-restart (CC only)
  - Auto-forward responses to bus (skip for telegram sources)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from networkkit.messages import Message, MessageType

from agentkit.bus.config import BusConfig
from agentkit.bus.participant import BusParticipant

log = logging.getLogger("agentkit.harness_bridge")

PRIORITY_AGENT = 0
PRIORITY_USER = 1
PRIORITY_HEARTBEAT = 2


# ── Enforcement Layer ──────────────────────────────────────────────────────


class EnforcementLayer:
    """Post-response inspection: checks if agent updated state files after task-relevant interactions."""

    def __init__(self, workdir: str, tracked_files: list[str] | None = None):
        self._workdir = Path(workdir).expanduser()
        self._tracked = tracked_files or ["TASKS.md", "WORKING.md", "MEMORY.md"]
        self._nudge_count = 0
        self._interaction_count = 0
        self._update_count = 0
        self._log_path = self._workdir / ".bridge-enforcement.jsonl"

    def snapshot_mtimes(self) -> dict[str, float]:
        result = {}
        for f in self._tracked:
            p = self._workdir / f
            try:
                result[f] = p.stat().st_mtime
            except FileNotFoundError:
                result[f] = 0.0
        return result

    def check_state_update(self, before: dict[str, float], after: dict[str, float]) -> bool:
        return any(after.get(f, 0) > before.get(f, 0) for f in self._tracked)

    def is_task_relevant(self, source: str) -> bool:
        return source.startswith("telegram:") or source.startswith("scheduler:")

    def should_nudge(self, source: str, state_updated: bool) -> bool:
        if not self.is_task_relevant(source):
            return False
        self._interaction_count += 1
        if state_updated:
            self._update_count += 1
            return False
        return True

    def record(self, source: str, state_updated: bool, nudged: bool):
        import json as _json
        entry = {
            "ts": time.time(),
            "source": source,
            "state_updated": state_updated,
            "nudged": nudged,
            "total": self._interaction_count,
            "updates": self._update_count,
            "nudges": self._nudge_count,
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(_json.dumps(entry) + "\n")
        except Exception:
            pass

    @property
    def compliance_rate(self) -> float:
        if self._interaction_count == 0:
            return 1.0
        return self._update_count / self._interaction_count

    NUDGE_MESSAGE = "📋 Task checkpoint — did this advance a task? Update TASKS.md or WORKING.md if so."


class HarnessConfig(BusConfig):
    """Config for the harness bridge."""

    def __init__(self, **kwargs):
        self.harness_type: str = kwargs.pop("harness_type", "cc")
        self.tmux_session: str = kwargs.pop("tmux_session", "")
        self.workdir: str = kwargs.pop("workdir", ".")
        self.send_timeout: int = kwargs.pop("send_timeout", 120)
        self.stable_idle_seconds: float = kwargs.pop("stable_idle_seconds", 3.0)
        self.peer_dir: str = kwargs.pop("peer_dir", "")
        self.auto_restart: bool = kwargs.pop("auto_restart", True)
        self.alert_destination: str = kwargs.pop("alert_destination", "")
        self.enforce: bool = kwargs.pop("enforce", False)
        self.enforce_nudge: bool = kwargs.pop("enforce_nudge", False)
        super().__init__(**kwargs)

    @classmethod
    def from_env(cls, name: str | None = None, dotenv_path: str | None = None) -> HarnessConfig:
        from dotenv import load_dotenv
        if dotenv_path:
            load_dotenv(dotenv_path)
        else:
            load_dotenv()

        agent_name = name or os.environ.get("AGENT_NAME", "kiro")
        return cls(
            name=agent_name,
            bus_http_url=os.environ.get("BUS_HTTP_URL", "http://127.0.0.1:8000"),
            bus_zmq_address=os.environ.get("BUS_ZMQ_ADDRESS", "tcp://127.0.0.1:5555"),
            description=os.environ.get("AGENT_DESCRIPTION", f"Agent '{agent_name}' on the network."),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            harness_type=os.environ.get("HARNESS_TYPE", "cc"),
            tmux_session=os.environ.get("TMUX_SESSION", agent_name),
            workdir=os.environ.get("AGENT_WORKDIR", "."),
            send_timeout=int(os.environ.get("SEND_TIMEOUT", "120")),
            stable_idle_seconds=float(os.environ.get("STABLE_IDLE_SECONDS", "3.0")),
            peer_dir=os.environ.get("PEER_DIR", ""),
            auto_restart="1" in os.environ.get("AUTO_RESTART", "1"),
            alert_destination=os.environ.get("ALERT_DESTINATION", ""),
            enforce="1" in os.environ.get("BRIDGE_ENFORCE", "0"),
            enforce_nudge="1" in os.environ.get("BRIDGE_ENFORCE_NUDGE", "0"),
        )


# ── Harness implementations ───────────────────────────────────────────────


class BaseHarness:
    """Common tmux interactions."""

    def __init__(self, session: str):
        self.session = session
        self._last_response: str | None = None
        self._stale_count: int = 0

    def is_running(self) -> bool:
        r = subprocess.run(["tmux", "has-session", "-t", self.session], capture_output=True)
        return r.returncode == 0

    def _capture_pane(self) -> str:
        r = subprocess.run(
            ["tmux", "capture-pane", "-t", self.session, "-p"],
            capture_output=True, text=True,
        )
        return r.stdout

    def _capture_pane_long(self) -> str:
        r = subprocess.run(
            ["tmux", "capture-pane", "-t", self.session, "-p", "-S", "-200"],
            capture_output=True, text=True,
        )
        return r.stdout

    def _send_keys(self, text: str):
        # Use tmux buffers to avoid blocking on pty input
        # Load text into a tmux buffer, then paste it
        subprocess.run(
            ["tmux", "set-buffer", "-t", self.session, text],
            capture_output=True, timeout=5,
        )
        subprocess.run(
            ["tmux", "paste-buffer", "-t", self.session, "-d"],
            capture_output=True, timeout=5,
        )
        # Send Enter keystroke to submit (paste-buffer \n doesn't trigger submit in TUIs)
        subprocess.run(
            ["tmux", "send-keys", "-t", self.session, "Enter"],
            capture_output=True, timeout=5,
        )

    def is_idle(self) -> bool:
        raise NotImplementedError

    def send(self, message: str, timeout: int = 120) -> str | None:
        raise NotImplementedError

    def restart_session(self) -> bool:
        raise NotImplementedError


class CCHarness(BaseHarness):
    """Claude Code harness — idle via ❯ prompt on its own line."""

    IDLE_MARKER = "❯"

    def is_idle(self) -> bool:
        pane = self._capture_pane()
        # CC is idle when ❯ appears on one of the LAST few lines (actual input prompt)
        # Not just anywhere in scrollback
        lines = [l.strip() for l in pane.rstrip().split("\n") if l.strip()]
        # Check last 5 non-empty lines for the idle marker
        tail = lines[-5:] if len(lines) >= 5 else lines
        for line in tail:
            if line == self.IDLE_MARKER or line == self.IDLE_MARKER + " ":
                return True
        return False

    def is_busy(self) -> bool:
        """CC is actively generating (pane changing, no idle prompt)."""
        return not self.is_idle()

    def send(self, message: str, timeout: int = 120) -> str | None:
        # Wait for CC to be idle (prompt empty) before injecting
        if not self._wait_for_stable_idle(timeout=60, stable_seconds=2.0):
            return None
        self._send_keys(message)
        self._stale_count = 0
        return "__DELIVERED__"

    def restart_session(self) -> bool:
        subprocess.run(["tmux", "send-keys", "-t", self.session, "C-c"], capture_output=True)
        time.sleep(2)
        subprocess.run(["tmux", "send-keys", "-t", self.session, "/exit", "Enter"], capture_output=True)
        time.sleep(3)
        subprocess.run(["tmux", "send-keys", "-t", self.session, "C-c"], capture_output=True)
        time.sleep(1)
        subprocess.run(["tmux", "send-keys", "-t", self.session, "C-c"], capture_output=True)
        time.sleep(2)

        subprocess.run(
            ["tmux", "send-keys", "-t", self.session,
             "claude --dangerously-skip-permissions", "Enter"],
            capture_output=True,
        )

        self._last_response = None
        self._stale_count = 0
        return self._wait_for_stable_idle(timeout=90, stable_seconds=5.0)

    def _wait_for_stable_idle(self, timeout: int = 60, stable_seconds: float = 3.0) -> bool:
        deadline = time.time() + timeout
        stable_since = 0.0
        was_idle = False

        while time.time() < deadline:
            if not self.is_idle():
                stable_since = 0.0
                was_idle = False
                time.sleep(1)
                continue

            if not was_idle:
                was_idle = True
                stable_since = time.time()

            if time.time() - stable_since >= stable_seconds:
                return True

            time.sleep(0.5)
        return False

    def _wait_for_done(self, timeout: int = 300) -> bool:
        """Wait for CC to finish processing and return to idle."""
        deadline = time.time() + timeout
        # First wait for CC to become non-idle (it's processing)
        time.sleep(2)
        # Then wait for it to come back to idle
        while time.time() < deadline:
            if self.is_idle():
                time.sleep(1)
                if self.is_idle():
                    return True
            time.sleep(1)
        return False


class KiroCliHarness(BaseHarness):
    """Kiro CLI harness — idle via 'ask a question' marker, done via ▸ Credits."""

    IDLE_MARKER = "ask a question or describe a task"
    DONE_MARKER = re.compile(r"▸ Credits: [\d.]+ • Time: \d+")

    def is_idle(self) -> bool:
        pane = self._capture_pane_long()
        return self.IDLE_MARKER in pane

    def send(self, message: str, timeout: int = 120) -> str | None:
        if not self._wait_for_stable_idle(timeout=60):
            return None

        subprocess.run(["tmux", "clear-history", "-t", self.session], capture_output=True)
        pane_before = self._capture_pane_long()

        subprocess.run(["tmux", "send-keys", "-t", self.session, "-X", "cancel"], capture_output=True)
        time.sleep(0.1)
        self._send_keys(message)

        deadline_accept = time.time() + 15
        while time.time() < deadline_accept:
            if self._capture_pane_long() != pane_before:
                break
            time.sleep(0.3)

        if not self._wait_for_done(timeout=timeout):
            return None

        return self._extract_response()

    def restart_session(self) -> bool:
        return False

    def _wait_for_stable_idle(self, timeout: int = 60, stable_seconds: float = 3.0) -> bool:
        deadline = time.time() + timeout
        last_pane = ""
        stable_since = 0.0

        while time.time() < deadline:
            pane = self._capture_pane_long()
            if self.IDLE_MARKER not in pane:
                last_pane = ""
                stable_since = 0.0
                time.sleep(1)
                continue

            if pane == last_pane:
                if time.time() - stable_since >= stable_seconds:
                    return True
            else:
                last_pane = pane
                stable_since = time.time()

            time.sleep(0.5)
        return False

    def _wait_for_done(self, timeout: int = 300) -> bool:
        deadline = time.time() + timeout
        start = time.time()
        while time.time() < start + 10:
            if not self.is_idle():
                break
            time.sleep(0.5)

        while time.time() < deadline:
            pane = self._capture_pane_long()
            if self.DONE_MARKER.search(pane) and self.IDLE_MARKER in pane:
                return True
            time.sleep(1)
        return False

    def _extract_response(self) -> str:
        pane = self._capture_pane_long()
        lines = pane.split("\n")

        # Find last Credits line
        credits_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if self.DONE_MARKER.search(lines[i]):
                credits_idx = i
                break
        if credits_idx is None:
            return ""

        # Find separator before this response
        sep_idx = None
        for i in range(credits_idx - 1, -1, -1):
            if "────" in lines[i]:
                sep_idx = i
                break
        if sep_idx is None:
            return ""

        # Between separator and Credits: input echo block, empty line, response block, empty line
        block = lines[sep_idx + 1:credits_idx]

        # Split into segments by empty lines
        segments = []
        current = []
        for line in block:
            if line.strip() == "":
                if current:
                    segments.append(current)
                    current = []
            else:
                current.append(line.strip())
        if current:
            segments.append(current)

        # Last non-empty segment is the response
        if segments:
            return "\n".join(segments[-1])
        return ""


# ── Bridge BusParticipant ──────────────────────────────────────────────────


def _classify_priority(source: str, content: str) -> int:
    if "scheduler:" in source or content.strip().lower() in ("hb", "heartbeat", "check heartbeat", "continue"):
        return PRIORITY_HEARTBEAT
    if any(x in source for x in ("router:", "megu", "agent")):
        return PRIORITY_AGENT
    return PRIORITY_USER


class HarnessBridge(BusParticipant):
    """Bridges the NetworkKit bus to a tmux-based agent (CC or Kiro CLI)."""

    def __init__(self, config: HarnessConfig):
        super().__init__(config)
        self.hc: HarnessConfig = config
        self.name = config.name

        if config.harness_type == "cc":
            self._harness: BaseHarness = CCHarness(config.tmux_session)
        else:
            self._harness: BaseHarness = KiroCliHarness(config.tmux_session)

        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._processing = asyncio.Event()
        self._consecutive_failures = 0
        self._seen_hashes: set[str] = set()
        self._seen_max = 100
        self._known_agents: set[str] = {"sophia", "kiro", "megu", "rin"}
        self._peer_dir = Path(config.peer_dir or config.workdir).expanduser() / "peers"

        # Enforcement layer (opt-in via BRIDGE_ENFORCE=1)
        self._enforce: EnforcementLayer | None = None
        if config.enforce:
            self._enforce = EnforcementLayer(config.workdir)
            log.info("Enforcement layer enabled (workdir=%s, nudge=%s)", config.workdir, config.enforce_nudge)

    def is_intended_for_me(self, message: Message) -> bool:
        to = message.to.lower()
        if to in (self.name.lower(), "all"):
            if message.source == self.name:
                return False
            if message.source.startswith(f"harness:{self.name}"):
                return False
            return True
        return False

    async def handle_message(self, message: Message) -> None:
        source = message.source
        content = message.content

        if message.message_type in (MessageType.SYSTEM.value, "SYSTEM"):
            return

        if message.message_type not in (MessageType.CHAT.value, "CHAT"):
            return

        # Dedup — use message id if available, else source+content hash (no timestamp)
        msg_id = getattr(message, 'id', None)
        if msg_id:
            msg_hash = f"id:{msg_id}"
        else:
            msg_hash = hashlib.md5(f"{source}:{content}".encode()).hexdigest()
        if msg_hash in self._seen_hashes:
            return
        self._seen_hashes.add(msg_hash)
        if len(self._seen_hashes) > self._seen_max:
            to_remove = list(self._seen_hashes)[:self._seen_max // 2]
            for h in to_remove:
                self._seen_hashes.discard(h)

        # Extract text from JSON content
        if content.strip().startswith("{"):
            try:
                parsed = json.loads(content)
                text = parsed.get("text", "")
                file_path = parsed.get("file_path", "")
                message_id = parsed.get("message_id", "")
                if file_path and "[Attached file:" not in text:
                    text = f"{text}\n[Attached file: {file_path}]" if text else f"[Attached file: {file_path}]"
                if message_id:
                    text = f"{text}\n[message_id: {message_id}]" if text else f"[message_id: {message_id}]"
                content = text
            except json.JSONDecodeError:
                pass

        if not content:
            return

        clean_source = source.replace("router:", "") if source.startswith("router:") else source
        priority = _classify_priority(source, content)

        await self._queue.put((priority, time.time(), {
            "source": source,
            "clean_source": clean_source,
            "content": content,
            "in_reply_to": getattr(message, 'in_reply_to', None),
            "message_id": getattr(message, 'id', None),
        }))

        label = ["AGENT", "USER", "HB"][priority]
        log.info("Queued [%s] from %s: %s", label, clean_source, content[:60])

        # Busy ACK for telegram
        if clean_source.startswith("telegram:") and (self._processing.is_set() or not self._harness.is_idle()):
            await self.send(
                clean_source,
                json.dumps({"text": "📨 受け取ったよ！今別の作業中。少し待ってね。"}, ensure_ascii=False),
            )

    async def on_start(self) -> None:
        if not self._harness.is_running():
            log.error("tmux session '%s' not running. Start it first.", self.hc.tmux_session)
            raise SystemExit(1)
        self._peer_dir.mkdir(parents=True, exist_ok=True)
        asyncio.get_running_loop().create_task(self._process_loop())

    def on_peer_discovered(self, peer_name: str, description: str) -> None:
        """Write peer file on discovery."""
        super().on_peer_discovered(peer_name, description)
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        (self._peer_dir / f"{peer_name}.md").write_text(
            f"# {peer_name}\n\n*Last seen: {now}*\n*Status: online*\n\n## Description\n{description}\n"
        )

    async def _process_loop(self) -> None:
        """Main processing loop — pulls from priority queue and sends to harness."""
        while self._running:
            try:
                priority, ts, msg = await asyncio.wait_for(self._queue.get(), timeout=5)
            except asyncio.TimeoutError:
                continue

            source = msg["source"]
            clean_source = msg["clean_source"]
            content = msg["content"]

            # Skip stale heartbeats
            if priority == PRIORITY_HEARTBEAT:
                if time.time() - ts > 60:
                    log.info("Skipping stale heartbeat (%.0fs old)", time.time() - ts)
                    continue
                if not self._queue.empty():
                    await self._queue.put((priority, ts, msg))
                    await asyncio.sleep(1)
                    continue

            label = ["AGENT", "USER", "HB"][priority]
            log.info("Processing [%s] from %s: %s", label, clean_source, content[:60])

            # Snapshot file mtimes before interaction (for enforcement)
            mtimes_before = self._enforce.snapshot_mtimes() if self._enforce else None

            self._processing.set()
            # Replace newlines with spaces to prevent paste-buffer splitting
            # into multiple CC inputs (each \n acts as Enter in terminal)
            clean_content = content.replace("\n", " ")
            prefixed = f"[from: {clean_source}] {clean_content}"

            # Long messages crash kiro-cli TUI — write to file and send path instead
            if len(prefixed) > 800:
                msg_file = Path(self.hc.workdir).expanduser() / ".bridge-incoming.md"
                msg_file.write_text(prefixed, encoding="utf-8")
                prefixed = f"[from: {clean_source}]\n(Long message saved to {msg_file} — read it with your file tools)"

            # Run harness.send in a thread (it's blocking)
            response = await asyncio.get_running_loop().run_in_executor(
                None, self._harness.send, prefixed, self.hc.send_timeout
            )
            self._processing.clear()

            # Post-response enforcement check
            if self._enforce and mtimes_before and response and response not in ("__BUSY__", "__DELIVERED__"):
                mtimes_after = self._enforce.snapshot_mtimes()
                state_updated = self._enforce.check_state_update(mtimes_before, mtimes_after)
                should_nudge = self._enforce.should_nudge(clean_source, state_updated)
                nudged = False

                if should_nudge and self.hc.enforce_nudge:
                    self._enforce._nudge_count += 1
                    nudged = True
                    log.info("Enforcement: nudging (no state update after task-relevant interaction)")
                    await asyncio.get_running_loop().run_in_executor(
                        None, self._harness.send, self._enforce.NUDGE_MESSAGE, 60
                    )
                elif should_nudge:
                    log.info("Enforcement: would nudge (logging only, BRIDGE_ENFORCE_NUDGE=0)")

                self._enforce.record(clean_source, state_updated, nudged)

            if response == "__BUSY__":
                log.info("CC busy — re-queuing message from %s (will retry when idle)", clean_source)
                await self._queue.put((priority, ts, msg))
                await asyncio.sleep(10)
                continue

            if response == "__DELIVERED__":
                self._consecutive_failures = 0
                log.info("Delivered to CC (agent replies directly)")
                continue

            if response:
                self._consecutive_failures = 0
                log.info("Response (%d chars): %s", len(response), response[:80])

                if clean_source.startswith("telegram:"):
                    log.info("Telegram source — agent replies via netkit send")
                elif clean_source in self._known_agents:
                    log.info("Agent source (%s) — agent replies via netkit send", clean_source)
                elif msg.get("in_reply_to"):
                    log.info("Response (in_reply_to=%s) — skipping auto-forward", msg["in_reply_to"])
                elif priority != PRIORITY_HEARTBEAT:
                    await self.send(clean_source, response)
                    log.info("Published reply to %s (%d chars)", clean_source, len(response))
            else:
                self._consecutive_failures += 1
                stale = self._harness._stale_count
                log.warning("No response — consecutive failures: %d, stale: %d", self._consecutive_failures, stale)

                if stale >= 2 or self._consecutive_failures >= 3:
                    log.error("SESSION HUNG — stale responses detected. Auto-restarting.")
                    if self.hc.alert_destination:
                        await self.send(
                            self.hc.alert_destination,
                            json.dumps({"text": f"⚠️ {self.name} session hung (stale={stale}, failures={self._consecutive_failures}). Auto-restarting CC..."}, ensure_ascii=False),
                        )

                    if self.hc.auto_restart and self._harness.restart_session():
                        log.info("Session restarted successfully")
                        if self.hc.alert_destination:
                            await self.send(
                                self.hc.alert_destination,
                                json.dumps({"text": f"✅ {self.name} session restarted and idle. Resuming."}, ensure_ascii=False),
                            )
                    else:
                        log.error("Restart failed or disabled — backing off 120s")
                        if self.hc.alert_destination:
                            await self.send(
                                self.hc.alert_destination,
                                json.dumps({"text": f"❌ {self.name} auto-restart failed. Manual intervention needed."}, ensure_ascii=False),
                            )
                        await asyncio.sleep(120)

                    self._consecutive_failures = 0


def main():
    """Entry point for the harness bridge."""
    config = HarnessConfig.from_env()
    if not config.tmux_session:
        config.tmux_session = config.name

    bridge = HarnessBridge(config)
    bridge.run()


if __name__ == "__main__":
    main()
