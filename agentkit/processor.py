import json
import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from litellm import acompletion

# Load environment variables from .env file
load_dotenv()

class JSONParseError(ValueError):
    """
    Custom exception class raised when encountering errors during JSON parsing in the `extract_json` function.
    """

    pass

async def llm_chat(
    llm_model: str,
    messages: List[Dict],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Generate a chat response using the specified LLM model.

    Args:
        llm_model (str): The name of the LLM model to use
        messages (List[Dict]): List of message dictionaries in the format expected by the model
        api_base (Optional[str]): Base URL for the API. If not provided, will use OPENAI_API_BASE 
                                 from environment variables, or default to "http://localhost:11434"
        api_key (Optional[str]): API key for authentication. If not provided, will use OPENAI_API_KEY
                                from environment variables

    Returns:
        str: The generated response text

    Note:
        This function will attempt to load API configuration from environment variables if not provided:
        - OPENAI_API_BASE: The base URL for the API
        - OPENAI_API_KEY: The API key for authentication
        
        You can set these in a .env file in your project root:
        OPENAI_API_BASE=your-api-base-url
        OPENAI_API_KEY=your-api-key
    """
    # Get API base URL from parameter, environment, or default
    final_api_base = api_base or os.getenv('OPENAI_API_BASE', "http://localhost:11434")
    
    # Get API key from parameter or environment
    final_api_key = api_key or os.getenv('OPENAI_API_KEY')

    # Prepare completion parameters
    completion_params = {
        "model": llm_model,
        "messages": messages,
        "api_base": final_api_base
    }

    # Add API key if available
    if final_api_key:
        completion_params["api_key"] = final_api_key

    # Generate response using LLM
    response = await acompletion(**completion_params)
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
