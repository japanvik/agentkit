"""
This module defines the message data structures used for communication within the AgentKit framework. 

It provides two primary classes:

* `MessageType`: An enumeration class representing the different message types used in AgentKit.
* `Message`: A Pydantic model class representing a message object exchanged through the data bus.
"""
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

from enum import Enum

class MessageType(str, Enum):
    """Enumeration of message types used within AgentKit communication.

    HELO: Indicates a login request or checking if the agent "to" is available.
    ACK: Response to a HELO request, indicating the agent is available.
    CHAT: Text message intended for conversation.
    SYSTEM: System message coming from the data hub.
    SENSOR: Messages for data coming from sensors.
    ERROR: Error messages.
    MEMORY: A memory cluster.
    INFO: A response from a previous information request.
    """
    HELO = "HELO"
    ACK = "ACK"
    CHAT = "CHAT"
    SYSTEM = "SYSTEM"
    SENSOR = "SENSOR"
    ERROR = "ERROR"
    MEMORY = "MEMORY"
    INFO = "INFO"

class Message(BaseModel):
    """
    A message object representing data exchanged through the AgentKit data bus. 

    This class is built using Pydantic for data validation and type hints.

    Attributes:
        source (str): The source of the message (e.g., agent name, sensor name).
        to (str): The intended recipient of the message (e.g., agent name, or 'ALL' for broadcast).
        content (str): The actual message content in string format.
        topic (MessageType): The type of message as defined by the `MessageType` enumeration.
        created_at (str, optional): The timestamp of when the message was created. If not provided, it will be automatically set to the current time.
    """

    source: str
    to: str
    content: str
    created_at: str = None  # Optional, set to current time by default
    message_type: MessageType

    class Config:
        use_enum_values = True

    def __init__(self, **data):
        """
        Initializer for the Message class.

        If 'created_at' is not provided in the data dictionary, it will be automatically set to the current time.
        """
        if 'created_at' not in data or data['created_at'] is None:
            data['created_at'] = datetime.now().strftime("%a %Y-%m-%d %H:%M:%S")
        super().__init__(**data)

    def prompt(self):
        """
        Provides a human-readable representation of the message.
        """
        return f"source: {self.source}: to: {self.to} ({self.message_type}): {self.content} on {self.created_at})"

        
class MessageResponse(BaseModel):
    """Pydantic model for response
    """
    status: str
