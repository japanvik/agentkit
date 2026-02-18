from typing import Optional

from networkkit.messages import Message, MessageType

from agentkit.agents.human_agent import HumanAgent, ParsedInput


class MockMessageSender:
    def __init__(self) -> None:
        self._attention: Optional[str] = None
        self.sent_messages = []

    @property
    def attention(self) -> Optional[str]:
        return self._attention

    @attention.setter
    def attention(self, value: str) -> None:
        self._attention = value

    async def send_message(self, message: Message) -> None:
        self.sent_messages.append(message)


def _build_agent() -> HumanAgent:
    return HumanAgent(
        name="Human",
        config={"description": "A human", "default_channel": "general"},
        message_sender=MockMessageSender(),
    )


def test_parse_command_input():
    agent = _build_agent()
    parsed = agent.parse_user_input("/dm Sophia hello there")

    assert parsed.kind == "command"
    assert parsed.command == "dm"
    assert parsed.args == "Sophia hello there"


def test_parse_legacy_target_input():
    agent = _build_agent()
    parsed = agent.parse_user_input("@Sophia,Julia deploy status")

    assert parsed.kind == "message"
    assert parsed.targets == ["Sophia", "Julia"]
    assert parsed.content == "deploy status"


def test_plain_message_broadcasts_to_channel():
    agent = _build_agent()
    parsed = ParsedInput(kind="message", content="status update")

    outgoing = agent.build_outbound_messages(parsed)

    assert len(outgoing) == 1
    assert outgoing[0].to == "ALL"
    assert outgoing[0].message_type == MessageType.CHAT
    assert outgoing[0].content == "[#general] status update"


def test_dm_does_not_add_channel_prefix():
    agent = _build_agent()
    parsed = ParsedInput(kind="message", content="private", targets=["Sophia"])

    outgoing = agent.build_outbound_messages(parsed)

    assert len(outgoing) == 1
    assert outgoing[0].to == "Sophia"
    assert outgoing[0].content == "private"


def test_japanese_content_is_preserved():
    agent = _build_agent()
    parsed = ParsedInput(kind="message", content="こんにちは、進捗を教えてください")

    outgoing = agent.build_outbound_messages(parsed)

    assert outgoing[0].content == "[#general] こんにちは、進捗を教えてください"


def test_extract_channel_prefix():
    channel, content = HumanAgent._extract_channel("[#ops] incident started")

    assert channel == "ops"
    assert content == "incident started"
