# Movie GraphRAG Assistant 🎬

A hybrid GraphRAG movie assistant combining **Neo4j** (knowledge graph for factual relationships) and **Pinecone** (vector search for movie plots) powered by **Google Gemini** and **LangChain**. Includes a **Streamlit** chat interface with Cypher query guardrails and self-healing error correction.

---

## 📊 Dataset

Download `wiki_movie_plots_deduped.csv` from Kaggle and place it in the project root:
- **Kaggle Dataset**: [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)

*(Note: The CSV file is ignored by `.gitignore` due to size).*

---

## 🚀 Quick Setup

### 1. Clone & Install
```bash
git clone <repo-url>
cd project_graph_rag
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```
Required keys:
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
- `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`
- `GOOGLE_API_KEY`, `MISTRAL_API_KEY`

### 3. Run the App

**Streamlit Web UI:**
```bash
streamlit run main_ui.py
```

**Terminal CLI:**
```bash
python main_new.py
```
*(On first run, the app will automatically ingest data if databases are empty).*
