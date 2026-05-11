# SHL Assessment Recommender

Conversational agent that recommends SHL Individual Test Solutions via a FastAPI service.
Built with **LangGraph + LangChain + Groq (llama-3.3-70b-versatile) + FAISS**.

## Architecture

```
POST /chat
    │
    ▼
[LangGraph]
    │
  route ──► clarify  → ask one follow-up question
    │
    ├──► retrieve → FAISS search (top-20) → rank (LLM picks 1-10) → shortlist
    │
    ├──► compare  → FAISS search → grounded comparison from catalog data
    │
    └──► refuse   → politely decline off-topic / injection attempts
```

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (free, local, no API key)
- **LLM**: Groq `llama-3.3-70b-versatile` (free tier)
- **Vector store**: FAISS IndexFlatIP (cosine similarity after L2 normalisation)
- **Catalog**: scraped from `https://www.shl.com/solutions/products/product-catalog/?type=1`

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key
```bash
# .env
GROQ_API_KEY=your_key_here
```
Get a free key at https://console.groq.com

### 3. Scrape catalog (one-time)
```bash
python scraper.py
```
Produces `catalog.json` (~377 products).

### 4. Build FAISS index (one-time)
```bash
python vector_store.py
```
Produces `catalog_index.pkl`.

### 5. Start server
```bash
flask --app main run --host 0.0.0.0 --port 8000
```

Or run everything in one shot:
```bash
python setup_and_run.py
```

## API

### GET /health
```json
{"status": "ok"}
```

### POST /chat
Request:
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```
Response:
```json
{
  "end_of_conversation": true,
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "test_type": "K",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/"
    },
    {
      "name": "Global Skills Assessment",
      "test_type": "C",
      "url": "https://www.shl.com/products/product-catalog/view/global-skills-assessment/"
    },
    {
      "name": "Entry Level Technical Support Solution",
      "test_type": "P",
      "url": "https://www.shl.com/products/product-catalog/view/entry-level-technical-support-solution/"
    }
  ],
  "reply": "Got it. Here are 3 assessments that fit your requirements."
}
```

## Evaluation
```bash
# Place trace JSON files in traces/ directory
python eval.py --traces traces/ --url http://localhost:8000
```
