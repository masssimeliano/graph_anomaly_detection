import os
import sys

import telebot
from dotenv import load_dotenv

load_dotenv(dotenv_path="./tg.env")

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = telebot.TeleBot(BOT_TOKEN)


class TelegramLogger(object):
    def __init__(self):
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        if message.strip():
            try:
                bot.send_message(CHAT_ID, message)
            except Exception as e:
                self.terminal.write(f"Telegram error: {e}\n")

    def flush(self):
        pass
