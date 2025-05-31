import sys
import telebot

BOT_TOKEN = '7603189034:AAGCeQDK3Oqzn0o7qeQzJLgrZiDGm-KHkqA'
CHAT_ID = '340101376'

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