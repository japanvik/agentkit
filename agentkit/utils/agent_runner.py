"""
Agent runner utility for simplified agent lifecycle management.

This module provides the AgentRunner class, which simplifies the process of
creating, running, and gracefully shutting down agents.
"""

import asyncio
import logging
import signal
from typing import Dict, List, Optional, Any

from networkkit.network import ZMQMessageReceiver
from agentkit.agents.simple_agent_factory import simple_agent_factory

class AgentRunner:
    """
    Runner for agents created with simple_agent_factory that handles
    complete lifecycle management including graceful shutdown.
    
    This class simplifies the process of creating, running, and gracefully
    shutting down agents. It handles signal registration, message receiver
    setup, and proper cleanup of resources.
    
    Example usage:
    ```python
    import asyncio
    from agentkit.utils import AgentRunner
    
    async def main():
        runner = AgentRunner(
            name="Julia",
            description="A friendly AI assistant",
            model="ollama/qwen3",
            system_prompt="You are a friendly and helpful AI agent called Julia."
        )
        
        await runner.run()
    
    if __name__ == "__main__":
        asyncio.run(main())
    ```
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        model: str,
        system_prompt: str = None,
        user_prompt: str = None,
        agent_type: str = "SimpleAgent",
        brain_type: str = "SimpleBrain",
        memory_type: str = "SimpleMemory",
        plugins_dir: str = "plugins",
        bus_ip: str = "127.0.0.1",
        api_config: Dict[str, Any] = None
    ):
        """
        Initialize the AgentRunner with parameters for agent creation.
        
        Args:
            name: Name of the agent
            description: Description of the agent
            model: Model to use for the agent's brain
            system_prompt: System prompt for the agent's brain
            user_prompt: User prompt for the agent's brain
            agent_type: Type of agent to create
            brain_type: Type of brain to create
            memory_type: Type of memory to create
            plugins_dir: Directory containing plugins
            bus_ip: IP address of the message bus
            api_config: API configuration for the agent
        """
        # Store parameters for agent creation
        self.agent_params = {
            "name": name,
            "description": description,
            "model": model,
            "system_prompt": system_prompt or f"You are {name}, {description}",
            "user_prompt": user_prompt,
            "agent_type": agent_type,
            "brain_type": brain_type,
            "memory_type": memory_type,
            "plugins_dir": plugins_dir,
            "bus_ip": bus_ip,
            "api_config": api_config
        }
        
        self.bus_ip = bus_ip
        self.agent = None
        self.message_receiver = None
        self.tasks = []
        self.shutdown_event = asyncio.Event()
    
    def register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig, 
                    lambda s=sig: asyncio.create_task(self.handle_signal(s))
                )
                logging.info(f"Registered signal handler for {signal.Signals(sig).name}")
            except NotImplementedError:
                logging.warning(f"Signal handler for {signal.Signals(sig).name} not implemented on this platform.")
    
    async def handle_signal(self, sig):
        """Handle shutdown signals."""
        logging.info(f"Received exit signal {signal.Signals(sig).name}...")
        self.shutdown_event.set()
    
    async def setup(self):
        """Create and set up the agent and message receiver."""
        # Create agent using the factory
        self.agent = simple_agent_factory(**self.agent_params)
        
        # Create message receiver
        self.message_receiver = ZMQMessageReceiver(
            subscribe_address=f"tcp://{self.bus_ip}:5555"
        )
        
        # Register agent with message receiver
        self.message_receiver.register_subscriber(self.agent)
    
    async def run(self):
        """Run the agent with proper lifecycle management."""
        # Register signal handlers
        self.register_signal_handlers()
        
        # Set up agent and message receiver
        await self.setup()
        
        # Start message receiver
        receiver_task = asyncio.create_task(
            self.message_receiver.start(),
            name="ReceiverTask"
        )
        self.tasks.append(receiver_task)
        
        # Start agent
        agent_task = asyncio.create_task(
            self.agent.start(),
            name=f"AgentTask-{self.agent.name}"
        )
        self.tasks.append(agent_task)
        
        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()
        finally:
            # Always ensure shutdown happens
            await self.shutdown()
    
    async def shutdown(self):
        """Perform graceful shutdown sequence."""
        logging.info("Initiating shutdown sequence...")
        
        # Stop the message receiver
        if self.message_receiver:
            await self.message_receiver.stop()
            logging.info("Message receiver stopped.")
        
        # Stop the agent
        if self.agent:
            await self.agent.stop()
            logging.info(f"Agent {self.agent.name} stopped.")
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logging.info(f"Task '{task.get_name()}' cancelled successfully.")
        
        # Close any client sessions in the message sender
        # This is a workaround for the "Unclosed client session" error from networkkit
        if self.agent and hasattr(self.agent, '_message_sender'):
            message_sender = self.agent._message_sender
            
            # Try different possible locations/names for the client session
            session_closed = False
            for attr_name in ['client_session', '_client_session', 'session', '_session']:
                if hasattr(message_sender, attr_name):
                    session = getattr(message_sender, attr_name)
                    if session and hasattr(session, 'close'):
                        try:
                            await session.close()
                            logging.info(f"Closed HTTP client session ({attr_name}).")
                            session_closed = True
                            break
                        except Exception as e:
                            logging.error(f"Error closing client session: {e}")
            
            # If we couldn't find a session attribute, try calling close() directly on the message sender
            if not session_closed and hasattr(message_sender, 'close') and callable(getattr(message_sender, 'close')):
                try:
                    await message_sender.close()
                    logging.info("Closed message sender.")
                except Exception as e:
                    logging.error(f"Error closing message sender: {e}")
        
        logging.info("Shutdown complete.")
