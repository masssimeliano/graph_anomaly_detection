import os
import requests
from pathlib import Path

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_file_to_telegram(file_path: Path):
    print(f"[INFO] Sending: {file_path}")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    with open(file_path, "rb") as f:
        files = {"document": f}
        data = {"chat_id": CHAT_ID}
        response = requests.post(url, data=data, files=files)
        if response.status_code != 200:
            print(f"[ERROR] Error sending {file_path.name}: {response.text}")
        else:
            print(f"[OK] Sent: {file_path.name}")

def main():
    base_dir = Path(__file__).resolve().parents[2]  # корень проекта
    txt_files = list(base_dir.rglob("*.txt"))

    if not txt_files:
        print("[INFO] No txt files found.")
        return

    for file in txt_files:
        send_file_to_telegram(file)

if __name__ == "__main__":
    main()
