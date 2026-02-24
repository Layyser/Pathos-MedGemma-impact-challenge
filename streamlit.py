import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List

from streamlit.runtime.scriptrunner import add_script_run_ctx

import streamlit as st
from app.application.chunk_grounding import (
    build_summary_highlighter,
    stream_chunk_evidence_summary,
)
from app.application.container import ServiceContainer, build_default_container
from app.application.datasets import create_dataset
from app.application.ingest_patient_files import run_patient_files_ingestion_pipeline
from app.application.search_context import run_search
from app.services.ask import AskService

st.set_page_config(
    page_title="Pathos",
    page_icon=str(Path(__file__).with_name("logo.svg")),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {
        padding-top: 2rem !important;
        max-width: 1400px;
    }
    footer {visibility: hidden;}

    .streamlit-expanderHeader p {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #4DA6FF !important;
    }

    .stMain .stExpander summary {
        background-color: rgb(30, 31, 32) !important;
        padding: 1rem 0.75rem !important;

    }

    .stMain .stExpander details {
        border: 0px;
        background-color: rgb(30, 31, 32);
    }

    .stMain .stExpander details[open] {
        background-color: rgb(30, 31, 32);
    }

    .stMain .stExpander [data-testid="stExpanderDetails"] {
        border-top: 0px;
        padding-top: 0px;
        padding-right: 2rem;
    }

    .stMain .stExpander [data-testid="stExpanderDetails"] > p {
        text-align: justify;
        text-justify: inter-word;
    }


    /* Sidebar */
    .stSidebar .stExpander summary {
        background-color: rgb(30, 31, 32) !important;
        padding-left: 0px;
        padding-right: 0px;
        color: rgba(255, 255, 255, 0.6);
    }

    .stSidebar .stExpander details {
        border: 0px;
        background-color: rgb(30, 31, 32);
        padding-left: 0px;
        padding-right: 0px;
    }

    .stSidebar .stExpander details[open] {
        background-color: rgb(30, 31, 32);
        padding-left: 0px;
        padding-right: 0px;
        padding-top: 0px;
    }

    .stSidebar .stExpander [data-testid="stExpanderDetails"] {
        border-top: 0px;
        padding-left: 0px;
        padding-right: 0px;
        padding-top: 0.5rem;
        padding-bottom: 0px;
    }

    .stSidebar .stExpander [data-testid="stExpanderDetails"] > p {
        text-align: justify;
        text-justify: inter-word;
        padding-left: 1rem;
    }

    .stFileUploader > div {
        display: none;
    }
</style>
""",
    unsafe_allow_html=True,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
state = st.session_state


@st.cache_resource
def _get_shared_ask_service() -> AskService:
    return AskService()


@st.cache_resource
def _get_shared_container() -> ServiceContainer:
    return build_default_container()


if "container" not in state:
    state["container"] = _get_shared_container()
if "ask_service" not in state:
    state["ask_service"] = _get_shared_ask_service()
if "selected_datasets" not in state:
    state["selected_datasets"] = []
if "selected_dataset" in state and state["selected_dataset"] and not state["selected_datasets"]:
    state["selected_datasets"] = [state["selected_dataset"]]
if "bg_task_status" not in state:
    state["bg_task_status"] = "idle"  # idle, running, complete, error
if "bg_task_message" not in state:
    state["bg_task_message"] = ""
if "bg_task_progress" not in state:
    state["bg_task_progress"] = 0.0
if "patient_ingestion_report" not in state:
    state["patient_ingestion_report"] = None
if "selected_patient_document_ids" not in state:
    state["selected_patient_document_ids"] = []
if "last_auto_patient_ingest_key" not in state:
    state["last_auto_patient_ingest_key"] = None

RESULT_SCORE_THRESHOLD = 0.2


def _list_dataset_names(container: ServiceContainer) -> List[str]:
    datasets = container.datasets.list_datasets()
    names = sorted({dataset.name for dataset in datasets if dataset.name})
    return names


def _render_dataset_checklist(container: ServiceContainer, disabled: bool) -> None:
    names = _list_dataset_names(container)
    selected = set(state.get("selected_datasets", []))
    selected.intersection_update(names)

    if not names:
        st.info("No datasets available yet. Create one below.")
        state["selected_datasets"] = []
        return

    updated: List[str] = []
    for name in names:
        checked = st.checkbox(
            label=name,
            value=name in selected,
            key=f"dataset_check_{name}",
            disabled=disabled,
        )
        if checked:
            updated.append(name)

    state["selected_datasets"] = updated


def _list_patient_files(container: ServiceContainer) -> List[Dict[str, str]]:
    doc_repo = container.ingestion.doc_meta_repo
    rows: List[Dict[str, str]] = []
    for patient_id in sorted(doc_repo._patient_docs.keys()):
        docs = doc_repo.fetch_by_patient_id(patient_id)
        for doc in docs:
            rows.append(
                {
                    "document_id": doc.id,
                    "patient_id": patient_id,
                    "file_name": doc.file_name,
                    "date": doc.effective_date.isoformat(),
                }
            )

    rows.sort(
        key=lambda row: (
            row["date"],
            row["file_name"].lower(),
            row["document_id"],
        ),
        reverse=True,
    )
    return rows


def _render_patient_file_checklist(container: ServiceContainer, disabled: bool) -> None:
    files = _list_patient_files(container)
    selected = set(state.get("selected_patient_document_ids", []))
    valid_ids = {row["document_id"] for row in files}
    selected.intersection_update(valid_ids)

    if not files:
        state["selected_patient_document_ids"] = []
        return

    updated: List[str] = []
    for row in files:
        doc_id = row["document_id"]
        label = f"{row['file_name']} - {row['date']} ({row['patient_id']})"
        checked = st.checkbox(
            label=label,
            value=doc_id in selected,
            key=f"patient_file_check_{doc_id}",
            disabled=disabled,
        )
        if checked:
            updated.append(doc_id)

    state["selected_patient_document_ids"] = updated


def _on_dataset_progress(event: Dict[str, Any]) -> None:
    message = str(event.get("message", "")).strip()
    progress_raw = event.get("progress", 0.0)

    try:
        progress = float(progress_raw)
    except (TypeError, ValueError):
        progress = 0.0

    state["bg_task_progress"] = max(0.0, min(1.0, progress))
    if message:
        state["bg_task_message"] = message


def run_dataset_creation_thread(container: ServiceContainer, ds_name: str, ds_topic: str) -> None:
    try:
        created = create_dataset(
            container,
            dataset_name=ds_name,
            topic=ds_topic,
            progress_callback=_on_dataset_progress,
        )
        if created:
            state["bg_task_status"] = "complete"
            state["bg_task_progress"] = 1.0
            state["bg_task_message"] = f"Created {created.name} ({created.document_count} docs)."
        else:
            state["bg_task_status"] = "error"
            state["bg_task_progress"] = 1.0
            state["bg_task_message"] = "Ingestion failed. No documents found."
    except Exception as exc:
        state["bg_task_status"] = "error"
        state["bg_task_progress"] = 1.0
        state["bg_task_message"] = f"Error: {exc}"


def _write_uploaded_patient_files(uploaded_files: List[Any]) -> tuple[List[Path], Path]:
    temp_dir = Path(tempfile.mkdtemp(prefix="patient_upload_"))
    saved_paths: List[Path] = []

    for index, uploaded in enumerate(uploaded_files):
        original_name = Path(str(getattr(uploaded, "name", f"upload_{index}.txt"))).name
        unique_dir = temp_dir / str(index)
        unique_dir.mkdir(parents=True, exist_ok=True)
        target_path = unique_dir / original_name
        target_path.write_bytes(bytes(uploaded.getbuffer()))
        saved_paths.append(target_path)

    return saved_paths, temp_dir


def _build_patient_upload_key(patient_id: str, uploaded_files: List[Any]) -> str:
    parts: List[str] = [patient_id.strip()]
    for uploaded in uploaded_files:
        name = Path(str(getattr(uploaded, "name", "unknown"))).name
        size = int(getattr(uploaded, "size", 0))
        file_id = str(getattr(uploaded, "file_id", ""))
        parts.append(f"{name}:{size}:{file_id}")
    return "|".join(parts)


def _render_streamed_summary(
    *,
    query: str,
    search_payload: Dict[str, Any],
    ask_service: AskService,
    state_key: str,
) -> None:
    state[state_key] = []
    highlighter = build_summary_highlighter(search_payload=search_payload)
    stream_box = st.empty()
    assembled = ""

    try:
        for token in stream_chunk_evidence_summary(
            query=query,
            search_payload=search_payload,
            ask_service=ask_service,
        ):
            assembled += token
            if highlighter is not None:
                annotation = highlighter.annotate(assembled)
                state[state_key] = annotation.get("matches", [])
                rendered_html = str(annotation.get("html", "")).strip()
                if rendered_html:
                    stream_box.markdown(rendered_html, unsafe_allow_html=True)
                else:
                    stream_box.markdown(str(annotation.get("text", "")))
            else:
                stream_box.markdown(assembled)
    except Exception as exc:
        st.error(f"Could not generate LLM evidence summary: {exc}")
        return

    if not assembled.strip():
        st.info("The LLM returned an empty summary.")
    elif state.get(state_key):
        st.caption(
            "Underlined text indicates n-gram matches against retrieved chunks. "
            "Hover to inspect source."
        )


@st.fragment(run_every="1s")
def render_progress_tracker() -> None:
    if state["bg_task_status"] == "running":
        progress_val = float(state.get("bg_task_progress", 0.0))
        percent_str = int(progress_val * 100)
        status_text = state["bg_task_message"] or "Creating dataset..."
        st.progress(progress_val, text=f"{status_text} ({percent_str}%)")

    elif state["bg_task_status"] == "complete":
        st.success(state["bg_task_message"])
        if st.button("Dismiss and reload datasets", use_container_width=True):
            state["bg_task_status"] = "idle"
            state["bg_task_progress"] = 0.0
            st.rerun()

    elif state["bg_task_status"] == "error":
        st.error(state["bg_task_message"])
        if st.button("Dismiss", use_container_width=True):
            state["bg_task_status"] = "idle"
            state["bg_task_progress"] = 0.0
            st.rerun()


with st.sidebar:
    is_running = state["bg_task_status"] == "running"

    with st.expander("Selected Datasets", expanded=True):
        _render_dataset_checklist(container=state["container"], disabled=is_running)

    with st.expander("Selected Patient Files", expanded=True):
        _render_patient_file_checklist(container=state["container"], disabled=is_running)

        patient_ingest_id = st.text_input(
            "Patient Name",
            placeholder="e.g., Jorge Vico",
            disabled=is_running,
            key="patient_ingest_id",
        )
        uploaded_patient_files = st.file_uploader(
            "Upload patient files",
            type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True,
            disabled=is_running,
            key="patient_file_uploader",
            help="Supported formats: TXT, MD, PDF, DOCX. Files are auto-ingested when Patient ID is set.",
        )

        normalized_patient_id = patient_ingest_id.strip()
        if uploaded_patient_files and not normalized_patient_id:
            st.info("Set Patient ID to auto-ingest uploaded files.")

        if uploaded_patient_files and normalized_patient_id and not is_running:
            auto_ingest_key = _build_patient_upload_key(
                patient_id=normalized_patient_id,
                uploaded_files=uploaded_patient_files,
            )
            if state.get("last_auto_patient_ingest_key") != auto_ingest_key:
                saved_paths: List[Path] = []
                temp_dir: Path | None = None
                try:
                    with st.spinner("Ingesting patient files..."):
                        saved_paths, temp_dir = _write_uploaded_patient_files(uploaded_patient_files)
                        report = run_patient_files_ingestion_pipeline(
                            file_paths=saved_paths,
                            patient_id=normalized_patient_id,
                            ingestion_svc=state["container"].ingestion,
                        )
                    state["patient_ingestion_report"] = report
                    state["last_auto_patient_ingest_key"] = auto_ingest_key
                    st.rerun()
                except Exception as exc:
                    state["last_auto_patient_ingest_key"] = auto_ingest_key
                    st.error(f"Could not ingest patient files: {exc}")
                finally:
                    if temp_dir is not None:
                        shutil.rmtree(temp_dir, ignore_errors=True)

        patient_report = state.get("patient_ingestion_report")
        if patient_report:
            summary = patient_report.get("summary", {})
            total = int(summary.get("total", 0))
            success = int(summary.get("success", 0))
            failed = int(summary.get("failed", 0))
            if failed == 0 and total > 0:
                st.success(f"Ingested {success}/{total} files successfully.")
            elif total > 0:
                st.warning(f"Ingested {success}/{total} files. {failed} failed.")

    with st.expander("New dataset", expanded=True):
        new_ds_name = st.text_input(
            "Dataset Name",
            placeholder="e.g., oncology_v1",
            disabled=is_running,
        )
        new_ds_topic = st.text_input(
            "Topic",
            placeholder="e.g., lung cancer protocols",
            disabled=is_running,
        )

        if st.button("Create Dataset", use_container_width=True, disabled=is_running):
            if not new_ds_name:
                st.error("Name is required.")
            else:
                state["bg_task_status"] = "running"
                state["bg_task_message"] = f"Building '{new_ds_name}'..."
                state["bg_task_progress"] = 0.0

                thread = threading.Thread(
                    target=run_dataset_creation_thread,
                    args=(state["container"], new_ds_name, new_ds_topic or new_ds_name),
                )
                add_script_run_ctx(thread)
                thread.start()

                st.rerun()

    render_progress_tracker()

st.title("Clinical Search")

scope = st.radio(
    "Scope",
    options=["all", "selected_docs", "docs", "dataset"],
    format_func=lambda value: {
        "all": "Search in Docs and Datasets",
        "selected_docs": "Selected Patient Docs",
        "docs": "Search in Patient Docs",
        "dataset": "Search in Datasets",
    }[value],
    horizontal=True,
    label_visibility="collapsed",
)

with st.container():
    query = st.chat_input("Enter clinical terms, patient IDs, or topics...")

if query:
    st.markdown(f"**Showing results for:** `{query}`")
    if scope in {"all", "dataset"}:
        if state["selected_datasets"]:
            st.caption(f"Dataset filter: {', '.join(state['selected_datasets'])}")
        else:
            st.caption("Dataset filter: all datasets")

    if scope == "selected_docs":
        selected_patient_doc_count = len(state.get("selected_patient_document_ids", []))
        if selected_patient_doc_count:
            st.caption(f"Selected patient files: {selected_patient_doc_count}")
        else:
            st.caption("Selected patient files: none")
    elif scope in {"docs", "all"}:
        st.caption("Patient docs retrieval: similarity top-k (5)")

    if scope != "selected_docs":
        st.caption(f"Similarity threshold: >= {RESULT_SCORE_THRESHOLD:.2f}")
    st.write("")

    with st.spinner("Searching knowledge base..."):
        payload: Dict[str, Any] = run_search(
            container=state["container"],
            scope=scope,
            query=query,
            selected_datasets=state["selected_datasets"],
            selected_document_ids=state["selected_patient_document_ids"],
            score_threshold=RESULT_SCORE_THRESHOLD,
        )

    docs = payload.get("docs", [])
    if payload["scope"] in {"selected_docs", "docs", "all"} and docs:
        if payload["scope"] == "selected_docs":
            st.markdown("### Selected Patient Docs")
        else:
            st.markdown("### Patient Similarity Matches")

        for item in docs:
            score = item.get("score")
            score_label = (
                f" - Score: {float(score):.3f}"
                if isinstance(score, (int, float))
                else ""
            )
            with st.expander(f"{item['source']} - Patient: {item['patient_id']}{score_label}"):
                effective_date = str(item.get("effective_date", "")).strip()
                if effective_date:
                    st.caption(f"Date: {effective_date}")
                st.caption(f"Chunk ID: `{item['id']}`")
                st.write(item["text"])

    dataset_results = payload.get("dataset", [])
    if payload["scope"] in {"dataset", "all"} and dataset_results:
        if payload["scope"] == "all" and docs:
            st.write("")

        st.markdown("### Dataset Context Matches")
        for item in dataset_results:
            metadata = item.get("metadata", {}) or {}
            title = str(metadata.get("title", "")).strip()
            header = f"{item['source']} - Score: {item['score']:.3f}"
            if title:
                header = f"{title} - Score: {item['score']:.3f}"

            with st.expander(header):
                if title:
                    st.caption(f"Source: {item['source']}")
                st.caption(f"Chunk ID: `{item['id']}`")
                st.write(item["text"])

    if payload["scope"] == "selected_docs" and not state.get("selected_patient_document_ids"):
        st.info("No patient files selected. Use the patient-file checklist in the sidebar.")

    has_results = bool(payload.get("docs") or payload.get("dataset"))
    if not has_results:
        st.info("No results found for your query in the current scope.")
    else:
        st.write("")
        if payload["scope"] == "all" and docs and dataset_results:
            st.markdown("### MedGemma overviews")
        else:
            st.markdown("### MedGemma summary")
        st.caption(
            "Generated from your query and the retrieved chunks. "
            "It should explain the evidence in natural language and avoid directly answering the query."
        )

        if payload["scope"] == "all" and docs and dataset_results:
            st.markdown("#### Patient overview")
            patient_payload = {"docs": docs, "dataset": []}
            _render_streamed_summary(
                query=query,
                search_payload=patient_payload,
                ask_service=state["ask_service"],
                state_key="summary_phrase_matches_docs",
            )

            st.write("")
            st.markdown("#### Dataset overview")
            dataset_payload = {"docs": [], "dataset": dataset_results}
            _render_streamed_summary(
                query=query,
                search_payload=dataset_payload,
                ask_service=state["ask_service"],
                state_key="summary_phrase_matches_dataset",
            )
        else:
            _render_streamed_summary(
                query=query,
                search_payload=payload,
                ask_service=state["ask_service"],
                state_key="summary_phrase_matches",
            )
