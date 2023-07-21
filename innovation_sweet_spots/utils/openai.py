"""
Utils for using the OpenAI API
"""
from innovation_sweet_spots import PROJECT_DIR
import os
import dotenv
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
from typing import List, Dict


def get_api_key() -> str:
    # If .env file exists
    if os.path.isfile(PROJECT_DIR / ".env"):
        # Load the .env file
        dotenv.load_dotenv(PROJECT_DIR / ".env")
        try:
            # Try loading API key
            return open(os.environ["OPENAI_API_KEY"], "r").read()
        except:
            # If the key is not in the .env file
            raise ValueError("No path to an OpenAI API key found")


def num_tokens(text: str) -> int:
    return len(encoding.encode(text))


def print_prompt(prompt: List[Dict[str, str]]):
    for line in prompt:
        print(f"{line['role']}:")
        print(line["content"])
        print("-------")
