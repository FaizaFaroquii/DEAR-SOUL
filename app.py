# © 2025 Faiza Farooqui - All Rights Reserved
# This code is not licensed for reuse, distribution, or modification.import streamlit as st
import faiss
import pickle
import numpy as np
import pandas as pd
import re
import random
import ollama
from datetime import datetime
from sentence_transformers import SentenceTransformer
import os

index = faiss.read_index("dearsoul_index.faiss")
with open("dearsoul_meta.pkl", "rb") as f:
    meta = pickle.load(f)
df = pd.DataFrame(meta)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

with open("Labeled_Quotes.txt", "r", encoding="utf-8") as f:
    quotes = [line.strip() for line in f if line.strip()]


def get_quote(tag=None):
    if tag:
        matches = [q.split("]", 1)[-1].strip() for q in quotes if q.lower().startswith(f"[{tag.lower()}]")]
        if matches:
            return random.choice(matches)
    return random.choice([q.split("]", 1)[-1].strip() for q in quotes])


def get_inspiration(user_input):
    vec = embed_model.encode([user_input])
    D, I = index.search(np.array(vec), k=1)
    return df.iloc[I[0][0]]['text']


system_prompt = {
    "role": "system",
    "content": (
        "You are DearSoul — a warm, emotionally intelligent motivational coach who speaks like a caring, smart best friend. "
        "Your replies must be emotionally validating, empowering, and feel like they're coming from someone who *truly cares*.\n\n"
        
        "Here’s how you respond:\n"
        "User: I'm feeling overwhelmed, everything is too much lately.\n"
        "You: Aww I get it, love 💛 Life can really pile on sometimes. Just breathe — you don't have to fix everything today. Start with one small thing, just for you.\n\n"
        
        "User: I feel so behind in life, everyone’s moving forward except me.\n"
        "You: You're not behind, you're just on a different path. 🌿 Life isn’t a race, bestie — trust your pace and take the next step that feels kind to you.\n\n"

        "User: I failed again and feel like I’m not good enough.\n"
        "You: Oof, I hear you 💔 But failing doesn’t make you a failure. It just means you're learning. Be proud that you're still showing up — that’s strength.\n\n"

        "User: I’m too lazy, I keep procrastinating everything.\n"
        "You: Bestie, it’s okay to feel unmotivated sometimes. 💫 Start tiny — do *one* small thing today, just to build momentum. You’ve got this.\n\n"

        "Your tone should always be:\n"
        "- Supportive and friendly (no robotic or poetic)\n"
        "- Use 2–3 natural sentences max\n"
        "- Add 1 emoji if it fits naturally\n"
        "- NEVER use numbered or bullet points\n"
        "- Always end with a gentle, doable action the user can take today"
    )
}


if "chat" not in st.session_state:
    st.session_state.chat = []

st.set_page_config(page_title="DearSoul 💖", page_icon="🌷")
st.title("🌷 DearSoul — Your Motivational Bestie")
st.markdown("Hi bestie! I'm always here to support you. Type how you're feeling 💬")

 
user_input = st.chat_input("Type here...")


with st.expander("Need some instant motivation?"):
    selected_topic = st.selectbox("Choose a topic for quote:", ["random", "study", "healing", "life", "love", "confidence", "procrastination", "growth", "sad"])
    if st.button("Give me a quote 🌟"):
        picked = get_quote(selected_topic if selected_topic != "random" else None)
        st.success(f"💡 {picked}")
        st.session_state.chat.append(("🧍‍♀️ You (quote)", f"quote {selected_topic}"))
        st.session_state.chat.append(("🌷 DearSoul", picked))


if user_input:
    if user_input.strip() == ":(":
        reply = "Aww, you typed a sad face 😢 — want to talk about it?"
    elif re.fullmatch(r'^[\W\d_]+$', user_input):
        reply = "⚠️ Emojis are cute, but I need some words too 🥹"
    else:
        inspiration = get_inspiration(user_input)
        messages = [
            system_prompt,
            {
                "role": "user",
                "content": (
                    f"My friend said: \"{user_input}\"\n\n"
                    f"Here’s a motivational quote to guide your tone:\n\"{inspiration}\"\n\n"
                    "Now reply like a caring best friend — warm, clear, and emotionally supportive. End with one helpful step the user can take today."
                )
            }
        ]
        res = ollama.chat(
            model="mistral",
            messages=messages,
            options={"num_predict": 300}
        )
        reply = res['message']['content'].strip()
        if reply.count('.') > 3:
            reply = '.'.join(reply.split('.')[:3]) + '.'

    st.session_state.chat.append(("🧍‍♀️ You", user_input))
    st.session_state.chat.append(("🌷 DearSoul", reply))

   
    os.makedirs("logs", exist_ok=True)
    log_row = pd.DataFrame([{
        "SessionID": f"DS-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "You": user_input,
        "DearSoul": reply
    }])
    log_file = f"logs/dearsoul_log_{datetime.now().strftime('%Y-%m-%d')}.csv"
    log_row.to_csv(log_file, mode='a', index=False, header=not os.path.exists(log_file))


st.markdown("### 💬 Chat History")
for i, (sender, msg) in enumerate(st.session_state.chat):
    st.markdown(f"**{sender}**: {msg}")
    if sender == "🌷 DearSoul":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful", key=f"up_{i}"):
                st.success("Thanks for the feedback 💖")
        with col2:
            if st.button("👎 Needs work", key=f"down_{i}"):
                os.makedirs("feedback", exist_ok=True)
                bad_row = pd.DataFrame([{
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "UserMessage": st.session_state.chat[i-1][1] if i >= 1 else "N/A",
                    "BotReply": msg
                }])
                bad_row.to_csv("feedback/retrain_me.csv", mode='a', index=False, header=not os.path.exists("feedback/retrain_me.csv"))
                st.warning("Got it! We'll learn from this 💡")
