# Data Study Tool

An LLM-powered data profiling and validation pipeline developed as an internship project. Automatically profiles datasets, generates data dictionaries, infers validation rules, and flags data quality violations using an agentic LangGraph refinement loop.

---

## Tech Stack

| Component | Technology |
|---|---|
| Profiling | YData Profiling |
| Relationship detection | MinHash / LSH (datasketch) |
| LLM | Kimi K2.5 via Azure AI Foundry |
| Agentic orchestration | LangGraph |
| Report generation | Node.js |
| Web interface | Flask |
| Language | Python 3.11 |

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Azure AI Foundry access (Kimi K2.5 deployment)

### Installation

```bash
pip install -r requirements.txt
npm install
```

### Environment Variables

Create a `.env` file at the project root:

```env
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_API_VERSION=2024-05-01-preview
ENDPOINT_KIMI=https://your-endpoint.openai.azure.com/
DEPLOYMENT_KIMI=your-kimi-deployment-name
```

---

## Usage

### Web Interface

```bash
python app.py
```

Open `http://localhost:5000`, upload CSV files, configure settings, and click **Run Pipeline**.

### CLI

```bash
# Interactive dataset selection
python main.py

# Explicit dataset paths
python main.py --datasets data/raw/table_a.csv data/raw/table_b.csv

# Skip cached results
python main.py --no-resume
```
