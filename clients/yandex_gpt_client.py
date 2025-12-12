# clients/yandex_gpt_client.py


import os
import aiohttp
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio


@dataclass
class Message:
    """Структура сообщения для запроса"""

    role: str  # 'system', 'user', или 'assistant'
    text: str


@dataclass
class CompletionOptions:
    """Опции для генерации текста"""

    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.95


class YandexGPTClient:
    """
    Клиент для взаимодействия с YandexGPT API
    """

    def __init__(self, api_key: Optional[str] = None, folder_id: Optional[str] = None):
        """
        Инициализация клиента

        Args:
            api_key: API ключ (если не передан, берется из окружения по переменной YANDEX_CLOUD_API_KEY)
            folder_id: ID каталога (берется из окружения по переменной YANDEX_CLOUD_FOLDER, если не передан)
        """
        self.api_key = api_key or os.getenv("YANDEX_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API ключ не найден. Установите переменную окружения YANDEX_CLOUD_API_KEY"
            )

        self.folder_id = folder_id or os.getenv("YANDEX_CLOUD_FOLDER")
        if not self.folder_id:
            raise ValueError(
                "ID каталога не найден. Установите переменную окружения YANDEX_CLOUD_FOLDER"
            )

        self.base_url = (
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        )
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Контекстный менеджер для сессии"""
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Закрытие сессии"""
        if self.session:
            await self.session.close()

    async def generate_text(
        self,
        messages: list[Message],
        model_uri: str = "gpt://<your-folder-id>/yandexgpt-lite",
        options: CompletionOptions = CompletionOptions(),
    ) -> str:
        """
        Генерация текста с помощью YandexGPT

        Args:
            messages: Список сообщений для контекста
            model_uri: URI модели (замените <your-folder-id> на реальный ID)
            options: Опции генерации

        Returns:
            Сгенерированный текст
        """
        if not self.session:
            raise RuntimeError("Клиент должен быть открыт через async with")

        # Заменяем placeholder в URI модели на реальный ID каталога
        actual_model_uri = model_uri.replace("<your-folder-id>", self.folder_id)

        payload = {
            "modelUri": actual_model_uri,
            "completionOptions": {
                "maxTokens": options.max_tokens,
                "temperature": options.temperature,
                "topP": options.top_p,
            },
            "messages": [{"role": msg.role, "text": msg.text} for msg in messages],
        }

        try:
            async with self.session.post(self.base_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ошибка API: {response.status}, {error_text}")

                result: Dict[str, Any] = await response.json()

                # Возвращаем текст первого варианта ответа
                if result.get("result") and result["result"].get("alternatives"):
                    return result["result"]["alternatives"][0]["message"]["text"]
                else:
                    raise Exception("Неверный формат ответа от API")

        except aiohttp.ClientError as e:
            raise Exception(f"Ошибка HTTP запроса: {str(e)}")


async def main():

    load_dotenv()

    async with YandexGPTClient() as client:
        messages = [
            Message(role="system", text="Ты - полезный ассистент"),
            Message(role="user", text="Привет! Расскажи коротко о себе"),
        ]

        options = CompletionOptions(max_tokens=300, temperature=0.6)

        try:
            response = await client.generate_text(messages=messages, options=options)
            print(response)
        except Exception as e:
            print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
