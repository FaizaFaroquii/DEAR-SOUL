# chatbot.py
import faiss
import pickle
import pandas as pd
import numpy as np
import re
import random
from datetime import datetime
from sentence_transformers import SentenceTransformer
import ollama
import os

# Session info
SESSION_ID = f"DS-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
TODAY_DATE = datetime.now().strftime("%Y-%m-%d")
os.makedirs("logs", exist_ok=True)
os.makedirs("transcripts", exist_ok=True)

# Load FAISS
index = faiss.read_index("dearsoul_index.faiss")
with open("dearsoul_meta.pkl", "rb") as f:
    metadata = pickle.load(f)
df = pd.DataFrame(metadata)

# Load embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load tagged quotes
with open("Labeled_Quotes.txt", "r", encoding="utf-8") as f:
    quotes = [line.strip() for line in f if line.strip()]

# Extracted topic-quote map
def get_quote_by_topic(topic=None):
    if topic:
        filtered = [q for q in quotes if q.lower().startswith(f"[{topic.lower()}]")]
        if filtered:
            return random.choice(filtered).split("]", 1)[-1].strip()
    return random.choice(quotes).split("]", 1)[-1].strip() if "]" in quotes[0] else random.choice(quotes)

# System persona
system_prompt = {
    "role": "system",
    "content": (
        "You are DearSoul — a warm, emotionally intelligent motivational coach who sounds like a smart best friend. "
        "Start by validating the user's feeling, then offer an encouraging mindset shift, followed by one small practical step. "
        "Structure your reply naturally — don't use bullets or numbered points. "
        "Keep your tone real, clear, and supportive (not robotic or poetic). Limit your reply to 2–3 short sentences max. Use 1 emoji if it feels right."
    )
}

def get_inspiration(user_input):
    vec = embed_model.encode([user_input])
    D, I = index.search(np.array(vec), k=1)
    return df.iloc[I[0][0]]['text']

def log_blocked_input(user_input):
    log = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "BlockedInput": user_input
    }])
    log.to_csv("blocked_log.csv", mode="a", header=not pd.io.common.file_exists("blocked_log.csv"), index=False)

def log_chat(user_input, reply):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_only = datetime.now().strftime("%I:%M %p")

    # CSV
    chat_row = pd.DataFrame([
        {"SessionID": SESSION_ID, "Timestamp": timestamp, "ChatTurn": f"🧍‍♀️ You said: {user_input}"},
        {"SessionID": SESSION_ID, "Timestamp": timestamp, "ChatTurn": f"🌸 DearSoul said: {reply}"}
    ])
    csv_path = f"logs/dearsoul_log_{TODAY_DATE}.csv"
    chat_row.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    # WhatsApp-style .txt
    with open(f"transcripts/dearsoul_{TODAY_DATE}.txt", "a", encoding="utf-8") as f:
        f.write(f"You [{time_only}]: {user_input}\n")
        f.write(f"DearSoul [{time_only}]: {reply}\n\n")

def run_chat():
    print("💬 DearSoul is ready to motivate you!\n(Type 'exit' to quit or try 'quote study', 'quote healing', etc 🌟)\n")

    while True:
        user_input = input("🧍‍♀️ You: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("🌸 DearSoul: Stay strong, bestie. You’ve got this! 💖")
            break

        # Handle empty or emoji-only
        if not user_input or re.fullmatch(r'["\']{1,2}', user_input):
            print("⚠️  Type something real, bestie 💬\n")
            log_blocked_input(user_input)
            continue

        if user_input.strip() == ":(":
            print("🌸 DearSoul: Aww, you typed a sad face 😢 — want to talk about it?\n")
            continue

        if re.fullmatch(r'^[\W\d_]+$', user_input):
            print("⚠️  Emojis are cute, but I need words too 🥹\n")
            log_blocked_input(user_input)
            continue

        # Check for quote + topic
        match = re.match(r"quote\s*(\w+)?", user_input.lower())
        if match:
            topic = match.group(1) if match.group(1) else None
            quote = get_quote_by_topic(topic)
            print(f"\n💡 Here's a DearSoul quote for you:\n\"{quote}\"\n")
            log_chat(user_input, quote)
            continue

        # Full RAG response
        inspiration = get_inspiration(user_input)
        messages = [
            system_prompt,
            {
                "role": "user",
                "content": (
                    f"I’m struggling: {user_input}\n\n"
                    f"Here’s a motivational thought to guide your tone:\n\"{inspiration}\""
                )
            }
        ]
        response = ollama.chat(
            model='mistral',
            messages=messages,
            options={"num_predict": 80}
        )
        reply = response['message']['content']
        print(f"\n🌸 DearSoul: {reply}\n")
        log_chat(user_input, reply)

# Run it
if __name__ == "__main__":
    run_chat()
