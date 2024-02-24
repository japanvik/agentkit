from typing import Any, Protocol
import requests
import zmq
import asyncio
from agentkit.messages import Message

class Subscriber(Protocol):
    name: str
    
    async def handle_message(self, message: Message) -> Any:
        ...
    
    def is_intended_for_me(self, message: Message) -> bool:
        ...

class MessageSender(Protocol):
    def send_message(self, message: Message) -> Any:
        """Send message out via the network"""
        ...


class ZMQMessageReceiver:
    def __init__(self, subscribe_address: str="tcp://127.0.0.1:5555"):
        self.subscribe_address = subscribe_address
        self.subscribers: list[Subscriber] = []
        # ZMQ stuff
        self.context = zmq.Context()
        self.pubsub_subscriber = self.context.socket(zmq.SUB)
        self.pubsub_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self.pubsub_subscriber.connect(subscribe_address)
        self.running = False

    async def start(self):
        self.running = True
        while self.running:
            try:
                raw_message = await asyncio.to_thread(self.pubsub_subscriber.recv_json)
                message = Message.model_validate(raw_message)
                await self.handle_message(message)
            except zmq.ZMQError as e:
                if e.errno != zmq.EAGAIN:
                    raise
            except asyncio.CancelledError:
                # Handle task cancellation gracefully if needed
                break

    def stop(self):
        self.running = False
        self.pubsub_subscriber.close()
        self.context.term()

    def register_subscriber(self, subscriber: Subscriber):
        self.subscribers.append(subscriber)

    async def handle_message(self, message: Message):
        for subscriber in self.subscribers:
            if subscriber.is_intended_for_me(message):
                await subscriber.handle_message(message)


class HTTPMessageSender:
    def __init__(self, publish_address: str="http://127.0.0.1:8000") -> None:
       self.publish_address = publish_address
    
    def send_message(self, message: Message) -> requests.Response:
        response = requests.post(f"{self.publish_address}/data", json=message.dict())
        return response