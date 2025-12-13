# scripts/log_analyzer.py

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import yaml
import json
import asyncio
from clients import *
from dotenv import load_dotenv
from typing import Optional


async def _load_config(f: Optional[str] = "config/log_analysis_prompt.yaml") -> dict:
    """
    Асинхронно загружает конфигурационный файл YAML и возвращает словарь с параметрами генерации и промптом.

    :param f: Путь к конфигурационному файлу. По умолчанию 'config/log_analysis_prompt.yaml'.
    :return: Словарь с ключами 'temperature', 'max_tokens' и 'prompt'.
    """
    with open(f, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return {
        "temperature": config["generation_params"]["temperature"],
        "max_tokens": config["generation_params"]["max_tokens"],
        "prompt": config["system_prompt"],
    }


async def _load_log(f: str) -> str:
    """
    Асинхронно загружает содержимое текстового файла с логами.

    :param f: Путь к файлу с логами.
    :return: Содержимое файла в виде строки.
    """

    with open(f, "r", encoding="utf-8") as file:
        return file.read()


async def log_responce(log_file: str, config_file: Optional[str] = None) -> json:
    """
    Асинхронная функция для анализа лога с помощью LLM.

    Загружает переменные окружения, конфигурацию и лог-файл,
    создает сообщения для модели, отправляет запрос и возвращает JSON-ответ.
    В случае ошибки возвращает словарь с описанием ошибки.

    :param log_file: Путь к файлу с логами для анализа.
    :param config_file: Путь к файлу конфигурации (опционально).
    :return: JSON-строка с ответом модели или словарь с ошибкой.
    """
    load_dotenv()

    config = await _load_config(config_file) if config_file else await _load_config()
    log = await _load_log(log_file)

    async with LLMClient() as client:
        messages = [
            Message(role="system", text=f"{config['prompt']}"),
            Message(role="user", text=f"{log}"),
        ]

        options = CompletionOptions(
            max_tokens=config["max_tokens"], temperature=config["temperature"]
        )

        try:
            response = await client.generate_text(messages=messages, options=options)
            return json.dumps(response)

        except Exception as e:
            return {"error": e}


async def main():

    print(await log_responce("YOUR_FILE"))


if __name__ == "__main__":
    asyncio.run(main())
