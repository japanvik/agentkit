import json
import re

from litellm import acompletion, set_verbose

class JSONParseError(ValueError):
    pass
    

async def llm_processor(llm_model: str, 
                    system_prompt: str = "", 
                    user_prompt: str = "", 
                    api_base: str = "http://localhost:11434",
                    stop: list[str] = ["\n"],
                    ) -> str:

    """
    A text processor that uses a LLM to generate a response.
    """
    set_verbose=True
    # Use the LLM to generate a response
    messages=[ {"content": system_prompt, "role": "system"} ]
    if user_prompt:
        messages.append({"content": user_prompt, "role": "user"})
    
    response = await acompletion(
        model = llm_model,
        messages = messages,
        api_base= api_base,
        stop=stop
    )
    return response.choices[0].message.content.strip()


def extract_json(text: str) -> dict:
    """ Extract the JSON object from the text.
    :param text: The text to extract the JSON from.
    :return: The JSON object as a dictionary.
    """
    # Extract JSON part from the message
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        raise JSONParseError("No JSON structure was found in your input while parsing. Make sure the JSON formatting is correct.")
    try:
        match = json_match.group(0)
        # convert it to dict
        #data = ast.literal_eval(match)
        data = json.loads(match)
        return data
    except json.JSONDecodeError:
        raise JSONParseError(f"Invalid JSON structure: {match}. Make sure the JSON formatting is correct. Always use double quotes and don't use single quotes in your formatting.")


def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data) 