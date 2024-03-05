"""
Initialization for the OpenAI configuration module.

This module ensures the OpenAI key is loaded at the start of the application.
"""

import os
import openai
from dotenv import load_dotenv
from .utils import get_project_root
from . import data_processing
from . import term_extraction

# Get the absolute path to the .env file
dotenv_path = os.path.join(get_project_root(), ".env")

# Load environment variables from .env
load_dotenv(dotenv_path)

# Check and set the OpenAI API key
open_api_key = os.environ.get("OPENAI_API_KEY")
if not open_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")
openai.api_key = open_api_key

# to validate key is loaded, print the first two letters
print(openai.api_key[:2])