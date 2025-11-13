"""Settings for the DSI Throwaway Scripts project."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

DATA_DIR = Path(os.environ["DATA_DIR"])
