import os
import sys
from argparse import ArgumentParser
from typing import Optional

from datasets.commands import BaseDatasetsCLICommand
from datasets._endpoint_config import load_endpoint_from_config, save_endpoint_to_config


def _command_factory(args, **_kwargs):
    return ConfigureEndpointCommand(endpoint=args.endpoint, non_interactive=args.non_interactive)


class ConfigureEndpointCommand(BaseDatasetsCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        description = "Persist a default Hugging Face endpoint used by the datasets library."
        configure_parser = parser.add_parser("configure-endpoint", help=description)
        configure_parser.add_argument(
            "--endpoint",
            type=str,
            help="The base URL of your private datasets hub. If omitted an interactive prompt is shown.",
        )
        configure_parser.add_argument(
            "--non-interactive",
            action="store_true",
            help="Fail instead of prompting when no endpoint value is provided.",
        )
        configure_parser.set_defaults(func=_command_factory)

    def __init__(self, endpoint: Optional[str], non_interactive: bool = False):
        self.endpoint = endpoint
        self.non_interactive = non_interactive

    def run(self):
        endpoint = self.endpoint or load_endpoint_from_config()
        if endpoint:
            endpoint = self._normalize(endpoint)
        elif self.non_interactive:
            raise SystemExit("No endpoint provided. Re-run with --endpoint or in interactive mode.")
        else:
            endpoint = self._prompt_for_endpoint()
            if not endpoint:
                print("No changes written.")
                return

        config_path = save_endpoint_to_config(endpoint)
        print(f"Saved endpoint '{endpoint}' to {config_path}.")

    def _prompt_for_endpoint(self) -> Optional[str]:
        if not sys.stdin.isatty():
            raise SystemExit("Cannot prompt for endpoint in a non-interactive environment.")

        current = load_endpoint_from_config()
        default_hint = current or os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
        print("Configure the default Hugging Face endpoint used by the datasets library.")
        print("Press <enter> to keep the current value (shown in brackets).")
        value = input(f"Endpoint [{default_hint}]: ").strip()
        if not value:
            return current or default_hint
        return self._normalize(value)

    @staticmethod
    def _normalize(value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise SystemExit("Endpoint value cannot be empty.")
        if not stripped.startswith("http://") and not stripped.startswith("https://"):
            stripped = "https://" + stripped
        return stripped.rstrip("/")


def main() -> None:
    parser = ArgumentParser(description="Configure the default datasets endpoint.")
    parser.add_argument("--endpoint", type=str, help="Endpoint URL to persist.")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail instead of prompting when no endpoint is provided.",
    )
    args = parser.parse_args()
    ConfigureEndpointCommand(endpoint=args.endpoint, non_interactive=args.non_interactive).run()
