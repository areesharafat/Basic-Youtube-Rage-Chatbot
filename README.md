# 🎥 YouTube RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about YouTube videos.

The application extracts the video's transcript, splits it into chunks, converts the chunks into embeddings, stores them in a FAISS vector database, retrieves relevant context for a user query, and uses Qwen3-8B to generate an answer.

## 🚀 Features

* 🔗 Accepts a YouTube video URL
* 📝 Automatically loads the video transcript
* ✂️ Splits transcripts into overlapping chunks
* 🧠 Generates semantic embeddings using `all-MiniLM-L6-v2`
* 🔎 Performs similarity-based retrieval using FAISS
* 🤖 Uses Qwen3-8B for answer generation
* 💬 Streamlit-based user interface
* 📚 Answers questions using the retrieved transcript context

## 🏗️ Architecture

```text
YouTube URL
     │
     ▼
YouTube Transcript
     │
     ▼
Text Splitting
     │
     ▼
Hugging Face Embeddings
     │
     ▼
FAISS Vector Store
     │
     ▼
Similarity Retrieval
     │
     ▼
Relevant Transcript Chunks
     │
     ▼
Prompt + Context + Question
     │
     ▼
Qwen3-8B
     │
     ▼
Generated Answer
```

## 🛠️ Tech Stack

* **Python**
* **Streamlit** — Web interface
* **LangChain** — RAG pipeline and document processing
* **FAISS** — Vector similarity search
* **Hugging Face** — Embeddings and LLM inference
* **Qwen3-8B** — Generative language model
* **YouTube Transcript** — Source knowledge

## 📁 Project Structure

```text
ytchatbot/
│
├── assets/
│   └── youtube.webp
│
├── venv/
├── .env
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ytchatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

#### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

#### Windows CMD

```cmd
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

If Streamlit is not included in `requirements.txt`:

```bash
pip install streamlit
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_ACCESS_TOKEN=your_huggingface_token
```

You can generate a Hugging Face access token from your Hugging Face account.

**Never commit your `.env` file or API token to GitHub.**

Add this to `.gitignore`:

```text
.env
venv/
__pycache__/
```

## ▶️ Running the Application

Activate your virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Then start Streamlit:

```bash
streamlit run app.py
```

The application will open locally in your browser.

## 💡 How It Works

### 1. Video Ingestion

The user provides a YouTube URL. The application loads the available transcript using LangChain's YouTube loader.

### 2. Text Chunking

The transcript is divided into smaller overlapping chunks using `RecursiveCharacterTextSplitter`.

Current configuration:

```python
chunk_size = 1000
chunk_overlap = 200
```

### 3. Embedding Generation

Each chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These embeddings represent the semantic meaning of the transcript chunks.
