# AI-Powered GitHub Repository Question Answering Chatbot

## 📌 Project Overview

The AI-Powered GitHub Repository Question Answering Chatbot is an intelligent application that allows users to ask questions about GitHub repositories and receive accurate answers using Retrieval-Augmented Generation (RAG).

Instead of manually reading large README files and source code, users can simply provide repository information and ask questions in natural language.

## 🎯 Objectives

- Automatically understand GitHub repository documentation.
- Retrieve relevant information based on user questions.
- Generate meaningful answers using Large Language Models (LLMs).
- Reduce the time required to understand complex codebases.
- Improve answer accuracy using hybrid search and reranking.

## 🚀 Features

- GitHub repository URL input.
- Automatic README and repository content processing.
- Document splitting using text chunking.
- Embedding generation.
- Vector storage using ChromaDB.
- Keyword-based retrieval using BM25.
- Hybrid retrieval combining keyword and vector search.
- Reciprocal Rank Fusion (RRF).
- Reranking using CrossEncoder.
- Context-aware answer generation using an LLM.
- Interactive Streamlit user interface.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | User interface |
| LangChain | RAG pipeline development |
| ChromaDB | Vector database |
| BM25 | Keyword-based retrieval |
| Hugging Face | Embeddings and reranking |
| Groq API | LLM inference |
| CrossEncoder | Document reranking |
| GitHub API | Repository content retrieval |

## ⚙️ How It Works

1. **Repository Input:** The user provides a GitHub repository URL.
2. **Data Extraction:** The application retrieves repository documentation and supported source files.
3. **Text Splitting:** Extracted content is divided into smaller chunks.
4. **Embedding Generation:** Text chunks are converted into numerical vectors.
5. **Vector Storage:** Embeddings are stored in ChromaDB.
6. **Hybrid Retrieval:** BM25 and vector search retrieve relevant chunks.
7. **RRF:** Reciprocal Rank Fusion combines the retrieval results.
8. **Reranking:** CrossEncoder reranks the retrieved documents based on query relevance.
9. **Answer Generation:** The LLM generates an answer using the retrieved context.
10. **Output:** The answer is displayed through the Streamlit interface.

## 📂 Project Structure

```text
github-repository-qa-chatbot/
│
├── app.py
├── requirements.txt
├── .env
├── README.md
│
├── modules/
│   ├── github_loader.py
│   ├── retriever.py
│   ├── embeddings.py
│   └── rag_pipeline.py
│
└── evaluation/
    ├── evaluate_rag.py
    └── eval_dataset.json
```

*Note: Update this structure according to your actual project files.*

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd github-repository-qa-chatbot
```

### 3. Create a Virtual Environment

```bash
python -m venv myenv
```

### 4. Activate the Environment

Windows:

```bash
myenv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Never upload your actual API keys to GitHub.

### 7. Run the Application

```bash
streamlit run app.py
```

## 📊 Retrieval and Answer Generation

This project uses a hybrid retrieval approach:

- **Dense Retrieval:** Finds semantically similar documents using embeddings.
- **Sparse Retrieval:** Finds relevant documents using keyword matching with BM25.
- **RRF:** Combines rankings from multiple retrieval methods.
- **Reranking:** Uses a CrossEncoder to improve the relevance of retrieved documents.

## 🔮 Future Improvements

- Support for larger repositories.
- Recursive source-code retrieval.
- Multi-language programming support.
- Improved RAG evaluation using RAGAS.
- Conversation memory.
- React-based frontend.
- Advanced retrieval optimization.

## 👨‍💻 Author

**Shivakumar Burugupelli**

- GitHub: https://github.com/shivakumarburugupelli
- LinkedIn: https://www.linkedin.com/in/shiva-kumar-burugupelli-5b51bb354/
