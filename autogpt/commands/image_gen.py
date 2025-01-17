""" Image Generation Module for AutoGPT."""
import io
import os.path
import uuid
from base64 import b64decode

import requests
from PIL import Image
from pathlib import Path
from autogpt.config import Config

CFG = Config()

WORKING_DIRECTORY = Path(__file__).parent.parent / "auto_gpt_workspace"


def generate_image(prompt: str) -> str:
    """Generate an image from a prompt.

    Args:
        prompt (str): The prompt to use

    Returns:
        str: The filename of the image
    """
    filename = f"{str(uuid.uuid4())}.jpg"

    # DALL-E
    if CFG.image_provider == "pollinations":
        return generate_image_with_pollinations(prompt, filename)
    elif CFG.image_provider == "sd":
        return generate_image_with_hf(prompt, filename)
    else:
        return "No Image Provider Set"


def generate_image_with_hf(prompt: str, filename: str) -> str:
    """Generate an image with HuggingFace's API.

    Args:
        prompt (str): The prompt to use
        filename (str): The filename to save the image to

    Returns:
        str: The filename of the image
    """
    API_URL = (
        "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
    )
    if CFG.huggingface_api_token is None:
        raise ValueError(
            "You need to set your Hugging Face API token in the config file."
        )
    headers = {"Authorization": f"Bearer {CFG.huggingface_api_token}"}

    response = requests.post(
        API_URL,
        headers=headers,
        json={
            "inputs": prompt,
        },
    )

    image = Image.open(io.BytesIO(response.content))
    print(f"Image Generated for prompt:{prompt}")

    image.save(os.path.join(WORKING_DIRECTORY, filename))

    return f"Saved to disk:{filename}"



def generate_image_with_pollinations(prompt: str, filename: str) -> str:
    url = f"https://image.pollinations.ai/prompt/{prompt}?width={1000}&height={1000}&model=flux&seed=234&nologo=true&private=true"
    response = requests.get(url)
    print(response.text)
    with open(f"{WORKING_DIRECTORY}/{filename}", 'wb') as png:
        png.write(response.content)

    return f"Saved to disk:{filename}"