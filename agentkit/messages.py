# Pydantic model for request validation
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

class MessageType(str, Enum):
    HELO = "HELO"# indicating a login or checking if the agent "to" is available 
    ACK = "ACK" # response to a HELO request. Indicates that the agent is available
    CHAT = "CHAT" # text message intended for conversation
    SYSTEM = "SYSTEM" # system message coming from the datahub
    SENSOR = "SENSOR" # messages for data coming from sensors
    ERROR = "ERROR" # error messages
    MEMORY = "MEMORY" # A memory cluster
    INFO = "INFO" # A response from a previous information request

class Message(BaseModel):
    """A Message Object that is sent out on the PUB channel of the PUBHUB mechanism.
    The description of the fields is as follows:
    source: the source of the message (e.g., the name of the sender, usually another agent or a sensor)
    to: the intended recipient of the message (e.g., the name of the receiver, usually another agent, or 'ALL' for broadcast)
    content: the actual message content in string
    topic: Enum of the topic.
    created_at: Optional timestamp for when the message was created.
    """
    source: str
    to: str
    content: str
    created_at: str
    message_type: MessageType

    class Config:  
        use_enum_values = True  
        
    def __init__(self, **data):
        if 'created_at' not in data or data['created_at'] is None:
            data['created_at'] = datetime.now().strftime("%a %Y-%m-%d %H:%M:%S")
        super().__init__(**data)
    
    def prompt(self):
        """Print the message in a human-readable format"""
        return f"source: {self.source}: to: {self.to} ({self.message_type}): {self.content} on {self.created_at})"
        
# Pydantic model for response
class MessageResponse(BaseModel):
    status: str
