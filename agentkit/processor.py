import json
import re
from typing import Dict, List

from litellm import acompletion

class JSONParseError(ValueError):
    """
    Custom exception class raised when encountering errors during JSON parsing in the `extract_json` function.
    """

    pass


async def llm_processor(llm_model: str,
                       system_prompt: str = "",
                       user_prompt: str = "",
                       api_base: str = "http://localhost:11434",
                       stop: list[str] = [],
                       ) -> str:
    """
    Asynchronous function that utilizes a Large Language Model (LLM) to generate text in response to prompts.

    This function leverages the `acompletion` function from the `litellm` library to interact with an LLM model.
    It constructs messages with system and user prompts (if provided) and sends them to the LLM for response generation.
    The generated response is then returned after stripping any leading/trailing whitespace.

    Args:
        llm_model (str): The name of the LLM model to be used.
        system_prompt (str, optional): The system prompt to provide context to the LLM. Defaults to "".
        user_prompt (str, optional): The user prompt to be used for response generation. Defaults to "".
        api_base (str, optional): The base URL of the LLM API endpoint. Defaults to "http://localhost:11434".
        stop (list[str], optional): A list of stop tokens to signal the LLM to terminate response generation. Defaults to ["\\n"].

    Returns:
        str: The generated text response from the LLM.
    """

    # Create message list with system and user prompts (if provided)
    messages = [{"content": system_prompt, "role": "system"}]
    if user_prompt:
        messages.append({"content": user_prompt, "role": "user"})

    # Generate response using LLM
    response = await acompletion(
        model=llm_model,
        messages=messages,
        api_base=api_base,
        stop=stop
    )
    return response.choices[0].message.content.strip()

async def llm_chat_processor(llm_model: str, prompt: List[Dict], api_base: str = "http://localhost:11434") -> str:
    """
    Asynchronous function that utilizes a Large Language Model (LLM) to generate text in response to prompts.
    This is the 'chat' variant of the prompt handling.

    Args:
        llm_model (str): The name of the LLM model to be used.
        prompt: (List[Dict]) : The system prompt to provide context to the LLM.
        api_base (str, optional): The base URL of the LLM API endpoint. Defaults to "http://localhost:11434".

    Returns:
        str: The generated text response from the LLM.
    """

    #set_verbose(True)

    # Generate response using LLM
    response = await acompletion(model=llm_model, prompt=prompt, api_base=api_base)
    print(f"response: {response}")
    return response.choices[0].message.content.strip()


def extract_json(text: str) -> dict:
    """
    Function to extract a JSON object from a given text string.

    This function attempts to locate a JSON structure within the provided text using regular expressions.
    If a match is found, it converts the matched string into a Python dictionary and returns it.
    If no JSON structure is found or there are errors during parsing, a `JSONParseError` exception is raised.

    Args:
        text (str): The text string to extract the JSON object from.

    Returns:
        dict: The extracted JSON object as a Python dictionary.

    Raises:
        JSONParseError: If no JSON structure is found or there are errors during parsing.
    """

    # Extract JSON part using regular expression
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        raise JSONParseError("No JSON structure was found in your input while parsing. Make sure the JSON formatting is correct.")

    try:
        match = json_match.group(0)
        # Convert the matched JSON string to a dictionary
        data = json.loads(match)
        return data
    except json.JSONDecodeError:
        raise JSONParseError(f"Invalid JSON structure: {match}. Make sure the JSON formatting is correct. Always use double quotes and don't use single quotes in your formatting.")


def remove_emojis(data):
    """
    Function to remove emojis from a given text string.

    This function utilizes a compiled regular expression to identify and remove emojis from the provided text data.
    It returns a new string with all emojis stripped out.

    Args:
        data (str): The text string from which to remove emojis.

    Returns:
        str: The text string without emojis.
    """


    emoj = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            #u"\U00002702-\U000027B0"
            #u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)

    return emoj.sub(r'', data) 
