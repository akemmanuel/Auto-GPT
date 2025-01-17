from ast import List
import time
from typing import Dict, Optional

import requests

from colorama import Fore

from autogpt.config import Config
from transformers import AutoTokenizer, AutoModel
import torch

CFG = Config()


def call_ai_function(
    function: str, args: List, description: str, model: Optional[str] = None
) -> str:
    """Call an AI function

    This is a magic function that can do anything with no-code. See
    https://github.com/Torantulino/AI-Functions for more info.

    Args:
        function (str): The function to call
        args (list): The arguments to pass to the function
        description (str): The description of the function
        model (str, optional): The model to use. Defaults to None.

    Returns:
        str: The response from the function
    """
    if model is None:
        model = CFG.smart_llm_model
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    args = ", ".join(args)
    messages = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}"
            f"\n{function}```\n\nOnly respond with your `return` value.",
        },
        {"role": "user", "content": args},
    ]

    return create_chat_completion(model=model, messages=messages, temperature=0)


# Overly simple abstraction until we create something better
# simple retry mechanism when getting a rate error or a bad gateway
def create_chat_completion(
    messages: List,  # type: ignore
    model: Optional[str] = None,
    temperature: float = CFG.temperature,
    max_tokens: Optional[int] = None,
) -> str:
    """Create a chat completion using the LLM API

    Args:
        messages (List[Dict[str, str]]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.

    Returns:
        str: The response from the chat completion
    """
    response = None
    num_retries = 10
    if CFG.debug_mode:
        print(
            Fore.GREEN
            + f"Creating chat completion with model {model}, temperature {temperature},"
            f" max_tokens {max_tokens}" + Fore.RESET
        )
    for attempt in range(num_retries):
        backoff = 2 ** (attempt + 2)
        try:
            model = "Meta-Llama-3.3-70B-Instruct"
            url = "https://api.sambanova.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {CFG.sambanova_api_key}",
                "Content-Type": "application/json"
            }
            data = {"stream": False, "model": model, "messages": messages}
            response = requests.post(url, headers=headers, json=data)
            break
        except Exception as e:
            if CFG.debug_mode:
                print(
                    Fore.RED + "Error: ",
                    str(e) + Fore.RESET,
                )

        time.sleep(backoff)
    if response is None:
        raise RuntimeError(f"Failed to get response after {num_retries} retries")

    return response.json()["choices"][0]["message"]["content"]


    
def create_embedding_with_hf(text, num_retries=10, debug_mode=True):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    for attempt in range(num_retries):
        backoff = 2 ** (attempt + 2)
        try:
            # Tokenisierung und Verarbeitung
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Mittelwert über die Hidden States für das Satzembedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embedding
        
        except Exception as e:
            # Fehler behandeln und ggf. erneut versuchen
            if debug_mode:
                print(f"Error on attempt {attempt + 1}: {e}")
            if attempt == num_retries - 1:
                raise
            time.sleep(backoff)
