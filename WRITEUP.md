# Pathos: A Locally Grounded Clinical Synthesis Agent using MedGemma
## 1. Project Summary
Pathos is a privacy-first agentic workflow application built to solve a specific, highly complex clinical process: reviewing patient context alongside massive volumes of current medical literature. The goal is simple: deploy AI as intelligent agents to cut down research time while keeping the clinician in total control of their data, traceability, and privacy.

## 2. Detailed Description
### Architecture and Model Strategy
I deployed MedGemma (google/medgemma-4b-it) strictly as a Synthesis Agent rather than an open-ended chatbot. Working with a 4B parameter model means you have to design around its limits. During early testing, I noticed that while MedGemma-4b-it understands clinical terminology perfectly, asking a small model to do math or complex logic leads to hallucinations. For example, if it retrieved a cardiology formula, it might miscalculate the math or make flawed logical leaps.

To fix this, I locked down the prompt. MedGemma is barred from computing or deducing answers; it is only allowed to synthesize explicit facts found in the retrieved chunks.

To make this actually trustworthy and adaptable in a clinical setting, I built the system around two key technical features:

- **Phrase-Level Grounding:** When getting an LLM answer, it's usually hard to tell what was actually in the source text and what the model just generated. To solve this, the UI streams MedGemma's output and uses n-gram matching to map generated phrases back to the retrieved chunks. When a doctor sees underscored text in the UI, it means it is a verbatim, 1-to-1 quote from the source literature. This visual proof heavily reduces hallucination risks.

- **Agentic Dataset Builder:** Doctors on shared machines need to switch specialties fast. I built an autonomous dataset-creator agent powered by the Google AI API. Instead of manually searching, a doctor can just type "Create a dataset about COVID-19 long-term consequences." The agent then acts autonomously using callable tools: it translates the prompt into a precise PubMed/MeSH query, fetches the XML data from Entrez, cleans it, and builds a persistent vector corpus for that exact topic. Crucially, because this agent is already hooked into the Google AI ecosystem, the architecture is perfectly positioned to link up directly with Google's massive healthcare knowledge graphs and advanced retrieval APIs (like Vertex AI Search or Google Scholar data) in the future. This would exponentially enhance the depth and quality of the specialized datasets the agent can generate on the fly.

The backend also supports loading PEFT LoRA adapters at runtime, meaning the system can specialize (e.g., swapping to an oncology adapter) without rewriting the core code.

### Solving the Hospital IT Bottleneck
I built Pathos around the reality of hospital IT. Doctors work under strict time limits and heavy privacy laws like HIPAA. They often share underpowered workstations and face a bad trade-off: either do a shallow manual search, or risk violating data privacy by pasting patient notes into public cloud AI tools.

To bypass these data compliance hurdles, Pathos splits how it handles memory into two distinct RAG pipelines:

- **Ephemeral RAG (RAM-only) for Patient Data:** Patient files are vectorized into an in-memory Chroma database. You can load raw, unanonymized patient data safely because it never touches the hard drive and gets completely wiped from memory when the session ends.

- **Persistent RAG (Disk) for Literature:** The curated clinical datasets (like the PubMed corpus) are saved persistently on a local disk so they don't need to be rebuilt and can be loaded instantly depending on the case.

#### Current workflow for a clinician:

1. Read scattered patient files manually.

2. Open separate search engines for guidelines/literature.

3. Manually compare information and create a narrative (while actively avoiding cloud AI tools due to privacy laws).

#### Improved workflow with Pathos:

1. **Agentic Generation:** Trigger the autonomous agent to generate a new specialty dataset via natural language

2. **Secure Upload:** Load unanonymized patient files directly into the secure, RAM-only ephemeral store.

3. **Query & Synthesize:** Submit one query and instantly receive patient similarity matches, targeted literature matches, and a MedGemma grounded summary with phrase-level traceability.

### Impact and Feasibility
This isn't just a theoretical wrapper; the repo ships with real, ready-to-use data. It includes 946 cleaned PubMed Central articles across oncology, neurology, pediatrics, and cardiology. That's about 29MB of cleaned text broken into roughly 63,000 dense retrieval chunks. The persistent vector store is highly optimized, taking up only ~366 MB of disk space, making it incredibly easy to deploy.

The MVP runs locally, but the architecture is abstracted so an IT team could easily route the LLM inference to a centralized hospital GPU API if the local machines are too weak. I wanted this to be usable right out of the box. The whole system is packaged standardly in ``pyproject.toml``. You just run:
```bash
pip install -e . 
streamlit run streamlit.py.
```

## 3. Known Gaps and Next Milestones
During the hackathon, I prioritized shipping a working, safe end-to-end user experience over writing extensive test suites. Because of that, there are some practical gaps I plan to address:

- **Lack of Formal Benchmarks:** I relied heavily on "gut-checking" the RAG retrieval to ensure it felt clinically accurate. The next major milestone is building a formal quantitative evaluation harness (like precision@k and strict grounding metrics) to mathematically prove the dataset quality.

- **Missing LoRA Checkpoints:** The codebase is fully ready to load and unload PEFT LoRA adapters, but I lacked the GPU hardware and time to actually train them.

- **Date Parsing and Complex PDFs:** The ingestion pipeline can be a bit brittle when it encounters messy date formats in patient files. Also, complex tables in scanned PDFs might lose their structural context during embedding.

- **Deployment:** While the hooks to offload inference to a Hospital GPU API are in the code, the whole setup needs proper Dockerization for true plug-and-play deployment.

### 4. One-Paragraph Submission Narrative
Pathos solves a practical clinical bottleneck: safely cross-referencing patient history with medical literature without violating data privacy laws. By splitting memory—using ephemeral RAM for unanonymized patient data and persistent disk storage for specialized clinical datasets—it runs securely on standard hospital hardware. To work around the limitations of smaller models, MedGemma-4B is strictly constrained to act as an evidence synthesizer rather than an open-ended chatbot, utilizing UI-level text grounding to actively prevent clinical hallucinations. Complete with an autonomous dataset-building agent and shipping with nearly 1,000 processed clinical articles, Pathos reimagines a highly fragmented, legally risky research task into a single, secure, and fully traceable agentic workflow.