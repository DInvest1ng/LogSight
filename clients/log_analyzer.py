from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from .yandex_gpt_client import CompletionOptions, LLMClient, Message


DEFAULT_CONFIG_PATH = Path("configs/log_analysis_prompt.yaml")


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return {
        "temperature": config["generation_params"]["temperature"],
        "max_tokens": config["generation_params"]["max_tokens"],
        "top_p": config["generation_params"].get("top_p", 0.95),
        "prompt": config["system_prompt"],
    }


def _load_log(path: Path) -> str:
    with path.open("r", encoding="utf-8") as file:
        return file.read()


async def log_response(log_file: str, config_file: Optional[str] = None) -> str | dict:
    load_dotenv()

    config_path = Path(config_file) if config_file else DEFAULT_CONFIG_PATH
    config = _load_config(config_path)
    log_text = _load_log(Path(log_file))

    async with LLMClient() as client:
        messages = [
            Message(role="system", text=config["prompt"]),
            Message(role="user", text=log_text),
        ]
        options = CompletionOptions(
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
        )
        try:
            return await client.generate_text(messages=messages, options=options)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


# Backward-compatible alias for old typoed API name.
async def log_responce(log_file: str, config_file: Optional[str] = None) -> str | dict:
    return await log_response(log_file=log_file, config_file=config_file)


async def main() -> None:
    print(await log_response("YOUR_FILE"))


if __name__ == "__main__":
    asyncio.run(main())
