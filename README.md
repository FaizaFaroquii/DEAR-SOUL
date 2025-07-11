## 🌸 DearSoul – Motivational Chatbot

🎥 **Demo Video**  
[Watch DearSoul in Action (Google Drive)](https://drive.google.com/file/d/1MO0JQTrW8sj-WGJ2OB5hJZfKvZ_9PM_Q/view?usp=sharing)

---

### 🌷 About the Project  
**DearSoul** is an emotionally intelligent AI chatbot built for one purpose — to motivate, uplift, and encourage users like a warm best friend 💖  
It uses **LLMs (Mistral)** with **Retrieval-Augmented Generation (RAG)**, and a custom **quote engine**, offering hyper-personalized motivational replies.

---

### ✨ Key Features
- 💬 Retrieval-Augmented Generation (RAG) via FAISS
- 🤖 Mistral LLM for emotionally intelligent, short-form motivational replies
- 🌟 Quote classification by topic (e.g., study, sadness, healing)
- ❌ Emoji-only / empty input handling
- 💡 Streamlit UI (optional)
- 🔐 Session logging, chat history, and TXT/JSON output
- 🛡️ Designed with privacy and emotional support in mind

---

### 📦 Full Project Download  
> **Note:** Due to GitHub size limits, the full project (360MB) including FAISS index, dataset, and environment setup is not uploaded here.  
🗂️ If you'd like access for review purposes, feel free to request it.

---

### 🛠️ How to Run (Locally)

```bash
# Optional: Set up virtual environment
python -m venv dearsoul_env
dearsoul_env\Scripts\activate     # For Windows

# Install required packages
pip install -r requirements.txt

# Run chatbot (console mode)
python chatbot.py
