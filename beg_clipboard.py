"""
Press T / t to copy the next unused quote this session.
Close this window to quit. Order is reshuffled on exit.
"""

from __future__ import annotations

import atexit
import random
import sys
import time
from pathlib import Path

try:
    import pyperclip
    from pynput import keyboard
except ImportError:
    print("Missing packages. Run:")
    print("  pip install pynput pyperclip")
    input("Press Enter to exit...")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
QUOTES_FILE = BASE_DIR / "quotes.txt"
COOLDOWN_SEC = 1.2


def load_quotes() -> list[str]:
    if not QUOTES_FILE.exists():
        print(f"Missing {QUOTES_FILE}")
        input("Press Enter to exit...")
        sys.exit(1)

    quotes: list[str] = []
    seen: set[str] = set()
    for line in QUOTES_FILE.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        quotes.append(text)

    if not quotes:
        print("quotes.txt is empty.")
        input("Press Enter to exit...")
        sys.exit(1)
    return quotes


def save_quotes(quotes: list[str]) -> None:
    QUOTES_FILE.write_text("\n".join(quotes) + "\n", encoding="utf-8")


def main() -> None:
    quotes = load_quotes()
    random.shuffle(quotes)
    queue = list(quotes)
    last_copy_at = 0.0

    def reshuffle_on_exit() -> None:
        shuffled = list(quotes)
        random.shuffle(shuffled)
        save_quotes(shuffled)
        print("\nOrder reshuffled for next run.")

    atexit.register(reshuffle_on_exit)

    print("=" * 56)
    print(" MVP++ Quote Clipboard")
    print("=" * 56)
    print(f" Quotes loaded : {len(quotes)}")
    print(f" Remaining     : {len(queue)}")
    print("-" * 56)
    print(" [T] / [t]   copy next quote (no repeat this run)")
    print(" Close window to quit (order will reshuffle)")
    print("=" * 56)
    print(" Tip: Run as Admin if Minecraft is Admin.")
    print(" Tip: Press T -> Ctrl+V paste in chat.")
    print("\nListening...\n")

    def copy_next() -> None:
        nonlocal queue, last_copy_at
        now = time.monotonic()
        if now - last_copy_at < COOLDOWN_SEC:
            return
        if not queue:
            print("[!] All quotes used this run. Close window and reopen for a new shuffled round.")
            return

        quote = queue.pop(0)
        pyperclip.copy(quote)
        last_copy_at = now
        used = len(quotes) - len(queue)
        print(f"[{used}/{len(quotes)}] Copied:")
        print(f"  {quote}")
        print(f"  Remaining: {len(queue)}\n")

    def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
        vk = getattr(key, "vk", None)
        char = getattr(key, "char", None)
        char_l = char.lower() if isinstance(char, str) else ""
        if vk == 0x54 or char_l == "t":
            copy_next()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}")
        input("Press Enter to exit...")
