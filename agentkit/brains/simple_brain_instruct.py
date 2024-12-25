from agentkit.memory.memory_protocol import Memory
from agentkit.processor import llm_chat_processor, remove_emojis
from networkkit.messages import Message, MessageType
from agentkit.brains.simple_brain import SimpleBrain
import ollama

class SimpleBrainInstruct(SimpleBrain):
    """
    SimpleBrain version that will create instruct type context TODO: it might be called 'chat' 
    Specially created for Qwen2.5 models but probably Chat interface.
    """
    async def create_completion_message(self, agent) -> Message:
        """ Override
        Generate a chat message response using the LLM based on the current context and prompts.
        
        Requirements for this method:
        Create a list of dictionaries containing the system promt, the user and agent histories in a list.
        The list should be in the following format:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]

        Args:
            agent: The agent object for which the chat message is being generated.

        Returns:
            Message: The generated message object (type: agentkit.messages.Message) containing the response content.
        """
        context=""
        system_prompt = self.system_prompt.format(name=self.name, description=self.description, context=context, target=agent.attention)
        prompt = self.create_chat_messages_prompt(agent, system_prompt)
        #print(f"prompt: {prompt}")
        model = self.model.split('/')[-1]
        response = ollama.chat(model=model, messages=prompt)
        #print(response)

        msg = self.format_response(agent, response['message']['content'])
        return msg


