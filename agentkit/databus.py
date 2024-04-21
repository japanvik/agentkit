"""
This module provides a data bus service for publishing messages using ZeroMQ and a FastAPI interface for receiving messages.

The core functionality is provided by the `send_message` function, which sends a message object to a ZeroMQ publisher channel. It also offers a FastAPI endpoint (`/data`) to receive messages via an HTTP POST request.

Additionally, this module includes a background task (`time_publisher_task`) that publishes system messages containing the current time every 5 minutes.

**To run the databus:**

1. (Optional) Activate your virtual environment (if applicable).
2. Execute the script from the console: `python -m "agentkit.databus"`
"""
import asyncio
import datetime

import zmq
from fastapi import FastAPI

from agentkit.messages import Message, MessageType

app = FastAPI()

# ZeroMQ Publisher setup
context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://*:5555")  # Allow connections from any IP address

async def send_message(message: Message):
    try:
        # Make sure we have a created_at
        if not message.created_at:
            message.created_at = datetime.datetime.now().strftime("%a %Y-%m-%d %H:%M:%S")
        # Send the message to the ZeroMQ publisher channel in JSON format
        publisher.send_json(message.dict())

        # Return success status
        return {"status": "success"}

    except Exception as e:
        # Log the exception
        print(f"Error sending message: {e}")
        # Return error status
        return {"status": "error"}

async def time_publisher_task():
    while True:
        current_time = datetime.datetime.now().strftime("%a %Y-%m-%d %H:%M:%S")
        message = Message(
            source="TimePublisher",
            to="ALL",
            content=f"Current time: {current_time}",
            created_at=current_time,
            message_type=MessageType.SYSTEM
        )
        await send_message(message)
        await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(time_publisher_task())

@app.post("/data")
async def post_message(message: Message):
    # This endpoint calls the same send_message method
    return await send_message(message)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
