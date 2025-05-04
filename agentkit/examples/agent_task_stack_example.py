"""
Example demonstrating the Agent Task Stack system.

This example shows how to create and use task-aware agents that can maintain
multiple conversations and manage tasks across different agents.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from networkkit.messages import Message, MessageType

from agentkit.agents.task_aware_agent import TaskAwareAgent
from agentkit.brains.simple_brain import SimpleBrain
from agentkit.memory.threaded_memory import ThreadedMemory


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Run the example."""
    # Create agents
    agent1 = TaskAwareAgent(
        name="Agent1",
        config={
            "name": "Agent1",
            "description": "A task-aware agent that can handle multiple conversations"
        },
        brain=SimpleBrain(name="Agent1Brain")
    )
    
    agent2 = TaskAwareAgent(
        name="Agent2",
        config={
            "name": "Agent2",
            "description": "Another task-aware agent that can handle multiple conversations"
        },
        brain=SimpleBrain(name="Agent2Brain")
    )
    
    agent3 = TaskAwareAgent(
        name="Agent3",
        config={
            "name": "Agent3",
            "description": "A third task-aware agent that can handle multiple conversations"
        },
        brain=SimpleBrain(name="Agent3Brain")
    )
    
    # Start agents
    await agent1.start()
    await agent2.start()
    await agent3.start()
    
    try:
        # Simulate direct conversation between Agent1 and Agent2
        logger.info("Starting direct conversation between Agent1 and Agent2")
        
        # Agent1 sends a message to Agent2
        message1 = Message(
            source="Agent1",
            to="Agent2",
            content="Hello Agent2, how are you?",
            message_type=MessageType.CHAT
        )
        await agent1.send_message(message1)
        
        # Wait for Agent2 to process the message
        await asyncio.sleep(1)
        
        # Agent2 sends a message to Agent1
        message2 = Message(
            source="Agent2",
            to="Agent1",
            content="I'm doing well, Agent1. How can I help you today?",
            message_type=MessageType.CHAT
        )
        await agent2.send_message(message2)
        
        # Wait for Agent1 to process the message
        await asyncio.sleep(1)
        
        # Simulate broadcast message from Agent3
        logger.info("Agent3 sending a broadcast message")
        broadcast_message = Message(
            source="Agent3",
            to="ALL",
            content="Attention all agents: This is a broadcast message!",
            message_type=MessageType.CHAT
        )
        await agent3.send_message(broadcast_message)
        
        # Wait for agents to process the broadcast message
        await asyncio.sleep(1)
        
        # Agent1 continues the direct conversation with Agent2
        message3 = Message(
            source="Agent1",
            to="Agent2",
            content="I need help with a task. Can you process some data for me?",
            message_type=MessageType.CHAT
        )
        await agent1.send_message(message3)
        
        # Wait for Agent2 to process the message
        await asyncio.sleep(1)
        
        # Agent1 starts a new conversation with Agent3
        logger.info("Starting direct conversation between Agent1 and Agent3")
        message4 = Message(
            source="Agent1",
            to="Agent3",
            content="Hello Agent3, I have a different question for you.",
            message_type=MessageType.CHAT
        )
        await agent1.send_message(message4)
        
        # Wait for Agent3 to process the message
        await asyncio.sleep(1)
        
        # Add a custom task for Agent2
        logger.info("Adding a custom task for Agent2")
        task = await agent2.add_task(
            description="Process data from Agent1",
            priority=8,
            due_time=datetime.now() + timedelta(minutes=5)
        )
        
        # Wait for a moment
        await asyncio.sleep(1)
        
        # Display conversation and task information
        logger.info("Displaying conversation and task information")
        
        # Get active conversations for each agent
        agent1_convs = await agent1.get_active_conversations()
        agent2_convs = await agent2.get_active_conversations()
        agent3_convs = await agent3.get_active_conversations()
        
        logger.info(f"Agent1 active conversations: {agent1_convs}")
        logger.info(f"Agent2 active conversations: {agent2_convs}")
        logger.info(f"Agent3 active conversations: {agent3_convs}")
        
        # Get pending tasks for each agent
        agent1_tasks = await agent1.get_pending_tasks()
        agent2_tasks = await agent2.get_pending_tasks()
        agent3_tasks = await agent3.get_pending_tasks()
        
        logger.info(f"Agent1 pending tasks: {len(agent1_tasks)}")
        logger.info(f"Agent2 pending tasks: {len(agent2_tasks)}")
        logger.info(f"Agent3 pending tasks: {len(agent3_tasks)}")
        
        # Display Agent2's task details
        if agent2_tasks:
            logger.info(f"Agent2 task details: {agent2_tasks[0]}")
        
    finally:
        # Stop agents
        await agent1.stop()
        await agent2.stop()
        await agent3.stop()


if __name__ == "__main__":
    asyncio.run(main())
