import logging

import telebot

BOT_TOKEN = '7603189034:AAGCeQDK3Oqzn0o7qeQzJLgrZiDGm-KHkqA'
CHAT_ID = '340101376'

bot = telebot.TeleBot(BOT_TOKEN)


class TelegramLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        try:
            if log_entry.strip():
                bot.send_message(CHAT_ID, log_entry)
        except Exception as e:
            print(f"Telegram error: {e}")
