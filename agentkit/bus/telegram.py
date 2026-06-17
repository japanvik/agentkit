"""TelegramAdapter — BusParticipant subclass for Telegram <-> NetworkKit bridging.

Handles both directions:
  Inbound:  Telegram messages → NetworkKit bus
  Outbound: NetworkKit bus messages → Telegram
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from networkkit.messages import Message, MessageType

from agentkit.bus.config import BusConfig
from agentkit.bus.participant import BusParticipant

log = logging.getLogger("agentkit.telegram")


class TelegramConfig(BusConfig):
    """Extended config for Telegram adapters."""

    def __init__(self, **kwargs):
        self.bot_token: str = kwargs.pop("bot_token", "")
        self.allowed_chat_ids: set[int] = kwargs.pop("allowed_chat_ids", set())
        self.agent_name: str = kwargs.pop("agent_name", "")
        self.adapter_prefix: str = kwargs.pop("adapter_prefix", "telegram")
        self.stt_api_url: str = kwargs.pop("stt_api_url", "")
        self.tts_api_url: str = kwargs.pop("tts_api_url", "")
        super().__init__(**kwargs)

    @classmethod
    def from_env(cls, name: str | None = None, dotenv_path: str | None = None) -> TelegramConfig:
        from dotenv import load_dotenv
        if dotenv_path:
            load_dotenv(dotenv_path)
        else:
            load_dotenv()

        chat_ids_raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "")
        chat_ids = {int(x.strip()) for x in chat_ids_raw.split(",") if x.strip()}

        return cls(
            name=name or os.environ.get("AGENT_NAME", os.environ.get("MEGU_AGENT_NAME", "unnamed")),
            bus_http_url=os.environ.get("BUS_HTTP_URL", "http://127.0.0.1:8000"),
            bus_zmq_address=os.environ.get("BUS_ZMQ_ADDRESS", "tcp://127.0.0.1:5555"),
            description=os.environ.get("AGENT_DESCRIPTION", "Telegram adapter"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            allowed_chat_ids=chat_ids,
            agent_name=os.environ.get("MEGU_AGENT_NAME", "megu"),
            adapter_prefix=os.environ.get("ADAPTER_NAME_PREFIX", "telegram"),
            stt_api_url=os.environ.get("STT_API_URL", ""),
            tts_api_url=os.environ.get("TTS_API_URL", ""),
        )


class TelegramAdapter(BusParticipant):
    """Bridges Telegram <-> NetworkKit bus for a single agent."""

    def __init__(self, config: TelegramConfig):
        super().__init__(config)
        self.tc: TelegramConfig = config
        self.name = f"{config.adapter_prefix}-{config.agent_name}"
        self._app = None
        self._voice_chats: set[int] = set()
        self._seen: set[str] = set()
        self._seen_max = 50

    def _source_for_chat(self, chat_id: int) -> str:
        return f"{self.tc.adapter_prefix}:{chat_id}"

    def _chat_id_from_to(self, to: str) -> int | None:
        prefix = f"{self.tc.adapter_prefix}:"
        if to.startswith(prefix):
            try:
                return int(to[len(prefix):])
            except ValueError:
                return None
        return None

    # ── BusParticipant interface ────────────────────────────────────

    def is_intended_for_me(self, message: Message) -> bool:
        # Match if addressed directly to us (e.g. to="telegram-megu")
        if message.to == self.name:
            log.debug("ACCEPT (direct name match): source=%s to=%s", message.source, message.to)
            return True
        # Match if from our agent and addressed to a telegram chat
        src = message.source.lower().removeprefix("router:")
        agent = self.tc.agent_name.lower()
        if not (src == agent or src == f"harness:{agent}" or src.startswith(f"{agent}:")):
            return False
        has_chat = self._chat_id_from_to(message.to) is not None
        if has_chat:
            log.info("ACCEPT outbound: source=%s to=%s (agent=%s)", message.source, message.to, agent)
        return has_chat

    async def handle_message(self, message: Message) -> None:
        """Outbound: agent response → Telegram."""
        import hashlib
        msg_hash = hashlib.md5(f"{message.source}:{message.to}:{message.content}".encode()).hexdigest()
        if msg_hash in self._seen:
            return
        self._seen.add(msg_hash)
        if len(self._seen) > self._seen_max:
            self._seen.clear()

        chat_id = self._chat_id_from_to(message.to)
        if chat_id is None and message.to == self.name:
            # Agent replied to us by name — send to default chat (first allowed)
            chat_id = next(iter(self.tc.allowed_chat_ids), None)
        if chat_id is None:
            return
        if self.tc.allowed_chat_ids and chat_id not in self.tc.allowed_chat_ids:
            log.warning("Outbound blocked chat_id=%s", chat_id)
            return

        try:
            content = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            log.warning("Outbound content not JSON-parseable: %r", message.content[:200])
            content = {"text": str(message.content)}

        # Double-stringified check: if content is still a string after first parse, try again
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                content = {"text": content}

        text = str(content.get("text", "")).strip()
        file_path = str(content.get("file_path", "")).strip()
        if file_path:
            from pathlib import Path as _P
            log.info("Outbound file_path=%s exists=%s", file_path, _P(file_path).is_file())
        caption = str(content.get("caption", "")).strip() or None

        # Handle reactions
        react_emoji = content.get("react")
        react_message_id = content.get("message_id")
        if react_emoji and react_message_id:
            try:
                from telegram import ReactionTypeEmoji
                await self._app.bot.set_message_reaction(
                    chat_id=chat_id,
                    message_id=int(react_message_id),
                    reaction=[ReactionTypeEmoji(emoji=react_emoji)],
                )
                log.info("Reaction chat_id=%s msg=%s emoji=%s", chat_id, react_message_id, react_emoji)
            except Exception:
                log.warning("Failed to set reaction", exc_info=True)
            return

        bot = self._app.bot

        if file_path and Path(file_path).is_file():
            doc_caption = caption or text or None
            if doc_caption and len(doc_caption) > 1024:
                doc_caption = doc_caption[:1024]
            with open(file_path, "rb") as f:
                await bot.send_document(chat_id=chat_id, document=f, caption=doc_caption)
            log.info("Outbound file chat_id=%s path=%s caption=%s", chat_id, file_path, bool(doc_caption))
        elif text:
            for i in range(0, len(text), 4096):
                await bot.send_message(chat_id=chat_id, text=text[i:i + 4096])
            log.info("Outbound text chat_id=%s chars=%d", chat_id, len(text))

            # TTS voice reply
            should_voice = content.get("voice") or chat_id in self._voice_chats
            if self.tc.tts_api_url and should_voice and len(text) < 2000:
                await self._send_voice(chat_id, text)
        else:
            log.warning("Outbound empty message chat_id=%s", chat_id)

    async def on_start(self) -> None:
        """Start Telegram bot polling."""
        from telegram import Bot
        from telegram.ext import ApplicationBuilder, MessageHandler, MessageReactionHandler, CommandHandler, filters

        self._app = ApplicationBuilder().token(self.tc.bot_token).build()

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("model", self._cmd_forward))
        self._app.add_handler(CommandHandler("newsession", self._cmd_forward))
        self._app.add_handler(CommandHandler("help", self._cmd_forward))
        self._app.add_handler(MessageHandler(
            (filters.TEXT | filters.PHOTO | filters.Document.ALL | filters.VIDEO) & ~filters.COMMAND,
            self._handle_inbound,
        ))
        self._app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._handle_voice))
        self._app.add_handler(MessageReactionHandler(self._handle_reaction))

        # Start polling (non-blocking)
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        log.info("Telegram polling started for agent=%s", self.tc.agent_name)

    async def on_stop(self) -> None:
        """Stop Telegram bot."""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    # ── Inbound handlers ────────────────────────────────────────────

    async def _handle_inbound(self, update, context) -> None:
        """Telegram message → NetworkKit bus."""
        chat = update.effective_chat
        if chat is None or (self.tc.allowed_chat_ids and chat.id not in self.tc.allowed_chat_ids):
            if update.message:
                await update.message.reply_text("Unauthorized chat.")
            return

        text = (update.message.text or update.message.caption or "").strip() if update.message else ""
        file_path = ""

        if update.message:
            tg_file = None
            if update.message.photo:
                tg_file = await update.message.photo[-1].get_file()
            elif update.message.document:
                tg_file = await update.message.document.get_file()
            elif update.message.video:
                tg_file = await update.message.video.get_file()

            if tg_file:
                import tempfile
                ext = Path(tg_file.file_path or "file").suffix or ".bin"
                dl_dir = Path(tempfile.gettempdir()) / "telegram-uploads"
                dl_dir.mkdir(parents=True, exist_ok=True)
                local_path = dl_dir / f"{tg_file.file_unique_id}{ext}"
                await tg_file.download_to_drive(local_path)
                file_path = str(local_path)

        if not text and not file_path:
            return

        content_data: dict[str, Any] = {"text": text, "message_id": update.message.message_id}
        if file_path:
            content_data["file_path"] = file_path
            if not text:
                content_data["text"] = f"User sent a file: {file_path}"

        await self._send_as_chat(chat.id, self.tc.agent_name, content_data)
        await context.bot.send_chat_action(chat_id=chat.id, action="typing")
        log.info("Inbound chat_id=%s text=%s", chat.id, text[:80])

    async def _handle_voice(self, update, context) -> None:
        """Voice message → STT → bus."""
        chat = update.effective_chat
        if chat is None or (self.tc.allowed_chat_ids and chat.id not in self.tc.allowed_chat_ids):
            return
        if not self.tc.stt_api_url:
            if update.message:
                await update.message.reply_text("Voice not configured.")
            return

        import httpx
        import tempfile

        try:
            voice = update.message.voice or update.message.audio
            if not voice:
                return

            tg_file = await voice.get_file()
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                await tg_file.download_to_drive(tmp.name)
                tmp_path = tmp.name

            async with httpx.AsyncClient(timeout=30, verify=False) as client:
                with open(tmp_path, "rb") as f:
                    resp = await client.post(
                        f"{self.tc.stt_api_url}/stt",
                        files={"file": ("voice.ogg", f, "audio/ogg")},
                    )
                resp.raise_for_status()
                result = resp.json()

            os.unlink(tmp_path)
            text = result.get("text", "").strip()

            if not text:
                await update.message.reply_text("(couldn't transcribe voice)")
                return

            self._voice_chats.add(chat.id)
            await self._send_as_chat(chat.id, self.tc.agent_name, {"text": text, "voice": True})
            await context.bot.send_chat_action(chat_id=chat.id, action="typing")
            log.info("Voice chat_id=%s text=%s", chat.id, text[:80])

        except Exception:
            log.exception("Voice handling failed chat_id=%s", chat.id)
            if update.message:
                await update.message.reply_text("Voice processing failed.")

    async def _handle_reaction(self, update, context) -> None:
        """Forward reactions to bus."""
        reaction = update.message_reaction
        if not reaction:
            return
        chat_id = reaction.chat.id
        new_emojis = [r.emoji for r in (reaction.new_reaction or []) if hasattr(r, "emoji")]
        if not new_emojis:
            return
        await self._send_as_chat(reaction.chat.id, self.tc.agent_name, {"text": f"[reaction: {' '.join(new_emojis)}]", "reaction": new_emojis})

    async def _cmd_start(self, update, context) -> None:
        if update.message:
            await update.message.reply_text(f"{self.tc.agent_name} adapter online.")

    async def _cmd_status(self, update, context) -> None:
        if update.message:
            await update.message.reply_text(
                f"Adapter: running\nAgent: {self.tc.agent_name}\nBus: {self.tc.bus_http_url}"
            )

    async def _cmd_forward(self, update, context) -> None:
        """Forward /command to agent."""
        chat = update.effective_chat
        if chat is None or (self.tc.allowed_chat_ids and chat.id not in self.tc.allowed_chat_ids):
            return
        full_text = (update.message.text or "").strip() if update.message else ""
        await self._send_as_chat(chat.id, self.tc.agent_name, {"text": full_text})

    # ── Helpers ─────────────────────────────────────────────────────

    async def _send_as_chat(self, chat_id: int, to: str, content: str | dict) -> None:
        """Send to bus with source=telegram:{chat_id} so agent knows where to reply."""
        import json as _json
        if isinstance(content, dict):
            content = _json.dumps(content, ensure_ascii=False)
        from networkkit.messages import Message as _Msg, MessageType as _MT
        msg = _Msg(
            source=self._source_for_chat(chat_id),
            to=to,
            content=content,
            message_type=_MT.CHAT,
        )
        await self._http_send(msg)

    async def _send_voice(self, chat_id: int, text: str) -> None:
        """TTS and send voice message."""
        try:
            import httpx, io
            lang = "ja" if any("　" <= c <= "鿿" or "゠" <= c <= "ヿ" for c in text) else "en"
            async with httpx.AsyncClient(timeout=30, verify=False) as client:
                resp = await client.post(
                    f"{self.tc.tts_api_url}/tts",
                    json={"text": text, "lang": lang},
                )
                resp.raise_for_status()
                audio_bytes = resp.content

            if len(audio_bytes) > 100:
                await self._app.bot.send_voice(chat_id=chat_id, voice=io.BytesIO(audio_bytes))
                log.info("Voice reply chat_id=%s lang=%s", chat_id, lang)
                self._voice_chats.discard(chat_id)
        except Exception:
            log.warning("TTS failed", exc_info=True)


def main():
    """Entry point for the adapter."""
    config = TelegramConfig.from_env()
    if not config.bot_token:
        print("TELEGRAM_BOT_TOKEN is required", file=__import__("sys").stderr)
        raise SystemExit(1)
    if not config.allowed_chat_ids:
        print("TELEGRAM_ALLOWED_CHAT_IDS is required", file=__import__("sys").stderr)
        raise SystemExit(1)

    adapter = TelegramAdapter(config)
    adapter.run()


if __name__ == "__main__":
    main()
