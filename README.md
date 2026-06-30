# Core RAG Assistant: University Student Performance Monitor

An enterprise-grade Retrieval-Augmented Generation (RAG) backend engineered with **FastAPI**, **LangChain (LCEL)**, **MongoDB**, and **Qdrant**. The system enables academic institutions to securely ingest course curriculum documents, systematically generate deterministic, multi-level assessments, and automatically execute asynchronous, criteria-mapped evaluation of subjective and objective student responses.

---

## 🗂️ Project Structure

```text
mini-rag-omar/
├── pyproject.toml             # Project metadata and dependencies (uv engine)
├── uv.lock                    # Locked cross-platform dependency graph
├── README.md                  # System documentation
├── docker/
│   └── docker-compose.yml     # Infrastructure layer (MongoDB, Qdrant)
└── src/
    ├── main.py                # ASGI application entry point (FastAPI)
    ├── .env.example           # Infrastructure configuration blueprint
    ├── assets/                # Local secure payload store & transient filesystem
    ├── chains/                # Custom LangChain Expression Language (LCEL) chains
    ├── controllers/           # Modular business logic controllers 
    ├── models/                # Document Schemas & Pydantic validation boundaries
    ├── routes/                # REST API routing layer & controllers mappings
    ├── stores/                # Multi-provider Vector & LLM structural bindings
    └── helper/                # System utilities, configurations & prompt wrappers

```

---

## 🛠️ Technology Stack & Engine Architecture

* **Runtime & Optimization:** `Python` optimized via `uv` dependency engine.
* **Application Layer:** `FastAPI` powered by asynchronous execution loops (`AsyncIO`).
* **Orchestration Engine:** `LangChain` & `LCEL` (LangChain Expression Language) for linear execution piping.
* **Persistence Stores:** `MongoDB (via Motor Async Driver)` for structural schema analytics and metadata management.
* **Vector Compute Engine:** `Qdrant DB` running high-performance cosine similarity indexing.
* **Embedding Pipeline:** Local transformer-based dense embeddings using `BAAI/bge-base-en-v1.5`.

---

## 🔌 Core Service API Endpoints & Contract Payloads

### 1. Document Upload & Ingestion Pipeline

* **Endpoint:** `POST /api/v1/professor/upload_docs/{project_id}`
* **Content-Type:** `multipart/form-data`
* **Lifecycle Flow:**

```text
       [Document Input Stream]
                  │
                  ▼
       [DataController Validation] ──(Fails Size/MIME)──► [HTTP 400 Exception]
                  │ (Passes Max 1GB / PDF or Word)
                  ▼
     [ProcessController File Parser] ──(LangChain TextLoader)
                  │
                  ▼
    [RecursiveCharacterTextSplitter] ──(Size: 400 Tokens / Overlap: 50 Chars)
                  │
                  ├──► [MongoDB Base Persistence Engine]
                  ▼
       [nlpController Encoder] ──(BAAI/bge-base-en-v1.5 Transformers)
                  │
                  ▼
         [Qdrant Collection] ──(Vector Matrix Indexing)

```

---

### 2. Intelligent Assessment Generation Engine (`QA_enhance`)

* **Endpoint:** `POST /api/v1/data/QA_enhance`
* **Description:** Assembles context from filtered top-k semantic segments across selected projects, automatically mapping out balanced difficulty patterns across targeted nodes.

#### Request Schema

```json
{
  "project_ids": ["247", "246"],
  "questions_number": 9,
  "questions_types": [
    {"MCQ": 6},
    {"TrueFalse": 2},
    {"Written": 1}
  ],
  "difficulty_levels": {
    "easy": 20,
    "medium": 60,
    "hard": 20
  },
  "topics": ["Replication"],
  "human_query": "make Define in written questions"
}

```

#### Core Logic Lifecycle:

1. **Context Boundary Assembly:** Checks if targeted explicit topics are assigned. Normalizes strings to eliminate token discrepancies.
2. **Dense Vector Search:** Employs cosine metrics over Qdrant collections to return the closest matching matrices. Filters structural matches beneath strict threshold metrics.
3. **LCEL Task-Batch Compilation:** Compiles a standard runnable block matching specific criteria distributions via parallel `RunnableLambda` instances executing `prompt | llm | output_parser`.
4. **Structured Mapping:** Enforces output patterns cleanly through a programmatic `Pydantic` mapping validator layer.

---

### 3. Distributed Student Grading Engine (`submit`)

* **Endpoint:** `POST /api/grading/submit`
* **Description:** Parses bulk exam answers against contextual course criteria schemas through vectorized batch loops.

#### Request Schema

```json
{
  "project_Ids": [],
  "exam_id": "22",
  "studentAnswers": [
    {
      "id": "402",
      "questionText": "What is inheritance in OOP?",
      "type": "Written",
      "mark": 10,
      "studentAnswer": "Inheritance allows a child class to acquire properties and methods from a parent class.",
      "questionAnswer": "Inheritance allows a child class to acquire properties and methods from a parent class.",
      "instructorCriteria": [
        { "criteria": "Define inheritance correctly", "weight": 5 },
        { "criteria": "Mention parent-child relationship", "weight": 3 },
        { "criteria": "Provide valid example", "weight": 2 }
      ]
    }
  ]
}

```

#### Core Logic Lifecycle:

1. **Dynamic Routing & Prompt Mapping:** Automatically selects specialized engine tasks (`GRADING_SYSTEM_PROMPT`, Batch MCQ templates, or custom Instructor-weighted multi-criteria rubrics).
2. **AsyncIO Micro-Batching Engine:** Splices array batches into uniform segments ($\text{chunk\_size} = 15$) executing in parallel over non-blocking asynchronous routines.
3. **Deterministic Fallback Loops:**
* *Objective Prompts:* Drops immediately back to strict Case-Insensitive programmatic string matching.
* *Subjective Prompts:* Re-queues failed schema Extractions back through single-item atomic evaluation layers.


4. **Weak Topic & Concept Gap Extraction:** Isolates penalized points across submissions to build tailored concept gap arrays for student dashboards.

---


## 🚀 Setup & Running (Step by Step)

### 1 — Clone the repo

```bash
git clone <repo-url>
cd mini-rag-omar

```

### 2 — Create uv environment

```bash
cd src
uv venv

```

### 3 — Install Python dependencies

```bash
uv sync

```

### 4 — Create `.env` file

```bash
# run from src/
cp .env.example .env   # if example exists, otherwise create manually

```

### 5 — Start MongoDB & Qdrant Infra (Docker)

```bash
# from project root
docker compose -f docker/docker-compose.yml up -d

```

Verify the containers are running healthy:

```bash
docker ps   # should show active containers for mongodb and qdrant

```

### 6 — Configure Environment Credentials

Open your newly created `.env` file and insert your respective LLM provider authentication tokens and configurations:

```env
LLM_API_KEY=your_actual_api_key_here
# Ensure database and vector URLs point to your running Docker ports

```

### 7 — Run the FastAPI server

```bash
# make sure you're inside the src/ directory
cd src
uv run uvicorn main:app --reload --host 0.0.0.0 --port 5000

```

Once the worker initializes, you can access the interactive API docs at `http://localhost:5000/docs`.