# main.py


import asyncio
from clients import *
from dotenv import load_dotenv


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