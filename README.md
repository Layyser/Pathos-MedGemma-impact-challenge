# Pathos

Privacy-first clinical retrieval and evidence summarization with MedGemma.

Pathos is a local-first medical RAG application that combines:
- Patient document retrieval (ephemeral/in-memory vectors).
- External clinical knowledge retrieval (persistent vectors built from PubMed Central full text).
- Grounded MedGemma summaries over retrieved evidence.

It is designed to help clinicians or clinical researchers review patient context and literature faster, while keeping sensitive data local.

## First-Run Data State
Pathos does **not** require a preloaded default database to run.

- Patient vectors are built only from files you upload.
- Dataset vectors are built only from datasets you create/load.
- `db/`, `data/`, `raw_data/`, and `datasets.json` are user-generated runtime artifacts.

If you start from a clean clone, dataset search returns no results until you ingest datasets.

## What It Does
- Ingests patient files (`.txt`, `.md`, `.pdf`, `.docx`).
- Chunks and embeds text with medical-aware chunking rules.
- Stores patient vectors in RAM (ephemeral Chroma collection).
- Builds specialty datasets from PMC:
  - Generates PubMed/MeSH query (cloud LLM step).
  - Fetches/cleans PMC XML.
  - Indexes cleaned text in persistent Chroma (`db/`).
- Supports search scopes in Streamlit:
  - `selected_docs`
  - `docs`
  - `dataset`
  - `all`
- Uses MedGemma to stream evidence summaries with phrase-level grounding highlights.
- Preloads the MedGemma provider when the Streamlit app starts.

## Architecture (High Level)
- `app/services/ingestion.py`: Parse -> Chunk -> Embed -> Route to storage.
- `app/services/retrieval.py`: Query embedding + patient/dataset vector search.
- `app/application/search_context.py`: Scope-aware orchestration for UI.
- `app/application/chunk_grounding.py`: MedGemma prompting, summary streaming, grounding/highlighting.
- `app/providers/llm/local.py`: Local MedGemma runtime + optional LoRA adapter loading.
- `app/providers/embeddings/local.py`: Sentence-transformers embeddings.
- `app/persistence/patient_repo.py`: Ephemeral patient vector store.
- `app/persistence/datasets_repo.py`: Persistent dataset vector store.
- `streamlit.py`: Main UI.

## Project Structure
```text
newwww/
  streamlit.py
  pyproject.toml
  datasets.json
  app/
    application/
    domain/
    persistence/
    providers/
    services/
    shared/
  data/
  raw_data/
  db/
```

## Requirements
- Python `>=3.10`
- CUDA-capable GPU recommended for local MedGemma (`google/medgemma-4b-it`)
- Internet access for:
  - First-time model downloads (Hugging Face / sentence-transformers)
  - Dataset creation (Google API + Entrez PMC fetch)

## Installation
```bash
pip install -e .
```

Optional dev extras:
```bash
pip install -e .[dev]
```

## Configuration
Config defaults are in `app/shared/config.py`.

Environment variables used:
- `GOOGLE_API_KEY`: required for cloud query generation in dataset creation.
- `EMAIL`: used by Entrez for PMC acquisition.

Create a `.env` file in project root:
```env
GOOGLE_API_KEY=your_key_here
EMAIL=your_email_here
```
For more info on the API key, go here: https://aistudio.google.com/app/api-keys


Important defaults:
- MedGemma model: `google/medgemma-4b-it`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Chunk size/overlap: `512 / 50`
- Top-k retrieval: `5`

## Run the App
```bash
streamlit run streamlit.py
```

On startup, Streamlit initializes the service container and preloads the MedGemma ask service.

## Typical Workflow
1. Start Streamlit.
2. Create a new dataset in the sidebar (or load your own folder via service code).
3. Wait for ingestion/indexing to finish (this populates `db/`).
4. Upload patient files and set patient name.
5. Choose search scope (`all`, `selected_docs`, `docs`, `dataset`).
6. Enter clinical query.
7. Inspect retrieved chunks and MedGemma grounded summary.

## Loading Your Own Data
- Datasets: create them in UI (`New dataset`) to fetch/clean/index PMC documents.
- Patient files: upload `.txt`, `.md`, `.pdf`, `.docx` in sidebar.
- Optional local folder ingestion is available via `DatasetService.create_dataset_from_folder(...)`.

Without these steps, there is no default persistent knowledge base.

## Current State and Limitations
- Remote MedGemma provider (`hospital_api`) is scaffolded but not implemented.
- Remote embedding provider is scaffolded but not implemented.
- LoRA adapter loading is implemented, but no adapter checkpoints are bundled.
- Test sources are not fully present in this repository snapshot.

## Submission Context
For the full judging-focused narrative, see:
- `WRITEUP.md`

It includes:
- rubric alignment,
- impact estimates,
- feasibility notes,
- next milestones.

## License
MIT (as declared in `pyproject.toml`).
