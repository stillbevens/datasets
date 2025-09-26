import json
import logging
import os
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

CONFIG_ENV_VARIABLE = "HF_ENDPOINT_CONFIG"
CONFIG_DIRNAME = ".datasets"
CONFIG_FILENAME = "config.json"
CONFIG_KEY = "hf_endpoint"


def get_config_path() -> Path:
    """Return the absolute path to the endpoint configuration file."""
    env_value = os.getenv(CONFIG_ENV_VARIABLE)
    if env_value:
        return Path(env_value).expanduser()
    return Path.home() / CONFIG_DIRNAME / CONFIG_FILENAME


def load_endpoint_from_config() -> Optional[str]:
    """Load the persisted endpoint configuration if it exists."""

    config_path = get_config_path()
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            payload = json.load(config_file)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as error:
        logger.warning("Unable to parse endpoint configuration at %s: %s", config_path, error)
        return None

    raw_endpoint = payload.get(CONFIG_KEY)
    if not raw_endpoint:
        return None

    endpoint = raw_endpoint.strip().rstrip("/")
    if not endpoint:
        return None

    return endpoint


def save_endpoint_to_config(endpoint: str) -> Path:
    """Persist the provided endpoint value into the configuration file."""

    normalized = endpoint.strip().rstrip("/")
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump({CONFIG_KEY: normalized}, config_file, indent=2)
        config_file.write("\n")
    logger.info("Saved datasets endpoint configuration to %s", config_path)
    return config_path
