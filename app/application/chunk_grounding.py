import html
import json
import re
from typing import Any, Dict, Iterator, List, Optional

from app.services.ask import AskService
from app.services.ports import AskServicePort

_TOKEN_RE = re.compile(
    r"\d+(?:[./:-]\d+)+(?:[A-Za-z%°]+)?"
    r"|"
    r"\d+(?:\.\d+)?(?:[A-Za-z%°]+)?"
    r"|"
    r"[A-Za-z]+(?:'[A-Za-z]+)?"
)
_NUMERIC_SIGNAL_RE = re.compile(
    r"^\d+(?:[./:-]\d+)+(?:[A-Za-z%°]+)?$"
    r"|"
    r"^\d+(?:\.\d+)?(?:%|mmhg|bpm|mg/dl|mmol/l|kg|g|cm|mm|°f|°c|f|c)$",
    re.IGNORECASE,
)
_DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "here", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "may",
    "me", "might", "more", "most", "my", "no", "nor", "not", "of", "on", "or", "our", "ours",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "then", "there", "these", "they", "this", "those", "to", "too", "under", "up", "us", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "would", "you", "your", "yours",
}
_AGE_TIME_UNITS = {
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "yr",
    "yrs",
    "yo",
    "y/o",
    "mo",
    "mos",
    "hour",
    "hours",
    "hr",
    "hrs",
}


def _is_per_file_query(query: str) -> bool:
    normalized = query.lower()
    patterns = (
        r"\bper file\b",
        r"\bfile by file\b",
        r"\beach file\b",
        r"\bby file\b",
        r"\bper document\b",
        r"\bdocument by document\b",
        r"\beach document\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def _is_dataset_only_chunks(chunks: List[Dict[str, Any]]) -> bool:
    if not chunks:
        return False

    has_dataset = any(str(chunk.get("bucket", "")) == "dataset" for chunk in chunks)
    has_docs = any(str(chunk.get("bucket", "")) == "docs" for chunk in chunks)
    return has_dataset and not has_docs


def _sentence_dedupe_key(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return re.sub(r"[^a-z0-9/.: -]", "", lowered)


def _clean_json_response(llm_output: str) -> str:
    clean_output = re.sub(r"```(?:json)?", "", llm_output, flags=re.IGNORECASE).strip()
    return clean_output.strip("`").strip()


def _normalize_chunks(search_payload: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    result_rank = 0

    for item in search_payload.get("docs", []):
        chunk_id = str(item.get("id", "")).strip()
        text = str(item.get("text", "")).strip()
        if not chunk_id or not text:
            continue
        result_rank += 1
        source = str(item.get("source", "Unknown source"))
        title = source

        chunks.append(
            {
                "id": chunk_id,
                "bucket": "docs",
                "source": source,
                "patient_id": item.get("patient_id"),
                "score": None,
                "effective_date": str(item.get("effective_date", "")).strip(),
                "text": text,
                "result_rank": result_rank,
                "title": title,
            }
        )

    for item in search_payload.get("dataset", []):
        chunk_id = str(item.get("id", "")).strip()
        text = str(item.get("text", "")).strip()
        if not chunk_id or not text:
            continue
        result_rank += 1
        source = str(item.get("source", "Unknown source"))
        metadata = item.get("metadata", {}) or {}
        raw_title = str(metadata.get("title", "")).strip()
        title = " ".join(raw_title.split()) if raw_title else source

        chunks.append(
            {
                "id": chunk_id,
                "bucket": "dataset",
                "source": source,
                "patient_id": None,
                "score": item.get("score"),
                "effective_date": "",
                "text": text,
                "result_rank": result_rank,
                "title": title,
            }
        )

    return chunks[:limit]


def _prompt_chunks(chunks: List[Dict[str, Any]], max_text_len: int = 800) -> str:
    rendered: List[str] = []
    for chunk in chunks:
        text = chunk["text"].replace("\n", " ").strip()
        if len(text) > max_text_len:
            text = text[:max_text_len]

        rendered.append(
            (
                f"id={chunk['id']}\n"
                f"bucket={chunk['bucket']}\n"
                f"source={chunk['source']}\n"
                # f"title={chunk.get('title', chunk['source'])}\n"
                f"patient_id={chunk['patient_id'] if chunk['patient_id'] else 'n/a'}\n"
                f"effective_date={chunk.get('effective_date') or 'n/a'}\n"
                f"score={chunk['score'] if chunk['score'] is not None else 'n/a'}\n"
                f"result_rank={chunk.get('result_rank', 'n/a')}\n"
                f"text={text}"
            )
        )

    return "\n\n---\n\n".join(rendered)


def _build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    chunk_block = _prompt_chunks(chunks)
    return f"""
You are a retrieval-grounding auditor.
You must NOT answer the user query and must NOT provide medical advice.
You must only identify which retrieved chunks contain relevant evidence.

Return ONLY strict JSON with this schema:
{{
  "query": "string",
  "matches": [
    {{
      "chunk_id": "string",
      "reason": "short reason tied to query terms",
      "quotes": ["exact verbatim quote from chunk text"]
    }}
  ]
}}

Rules:
1) Do not answer the query.
2) Use only chunk IDs from the provided list.
3) Each quote must be copied exactly from the chunk text.
4) Prefer at most 5 chunks, sorted by relevance.
5) If nothing is relevant, return "matches": [].

User query:
{query}

Retrieved chunks:
{chunk_block}
""".strip()


def _build_summary_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    chunk_block = _prompt_chunks(chunks)
    per_file_mode = _is_per_file_query(query)
    dataset_only_mode = _is_dataset_only_chunks(chunks)
    granularity_rules = ""
    length_rules = "Target length: 120-220 words."
    output_style = """
Output style:
- 2 short paragraphs.
- Paragraph 1: main evidence.
- Paragraph 2: limits or uncertainty.
""".strip()

    if per_file_mode:
        granularity_rules = """
11) The user asked for file-level detail. Cover each relevant retrieved file explicitly.
12) Use chronological order (oldest to newest) when effective_date is present.
13) Start each paragraph with source and date in plain text:
   source (effective_date): evidence summary for that file.
14) Do not collapse file-level evidence into only a global min/max range.
15) Avoid repeating the same sentence or restating identical ranges.
""".strip()
        length_rules = "Target length: 140-260 words, depending on number of files."
        output_style = """
Output style:
- Multiple short paragraphs, one per relevant file.
- No bullets or numbering.
- Keep each paragraph factual and grounded in that file's text.
""".strip()
    elif dataset_only_mode:
        granularity_rules = """
11) This is a dataset search summary. Write exactly two paragraphs.
12) Paragraph 1 should synthesize the strongest evidence from retrieved chunks.
13) Paragraph 2 should cover nuance, caveats, or uncertainty for the SAME evidence themes as paragraph 1.
14) Paragraph 2 must NOT introduce a new topic, disease subtype, or population that was not already part of paragraph 1.
15) Avoid meta-research filler (e.g., "future studies", "basis of knowledge", "improve completeness of the field")
    unless the user explicitly asks for research-gap commentary.
16) Do not use bullets, numbering, headers, or template labels.
17) Keep both paragraphs grounded to retrieved text and avoid repeating paragraph 1.
""".strip()
        length_rules = "Target length: 150-260 words total."
        output_style = """
Output style:
- Exactly 2 paragraphs separated by one blank line.
- Paragraph 1: main evidence synthesis.
- Paragraph 2: limitations, uncertainty, or boundaries of the evidence.
""".strip()

    return f"""
You are an evidence summarizer for retrieved chunks.
Your task is to write ONE grounded explanation based only on the retrieved chunks below.

Important constraints:
1) Do NOT answer the user query directly.
2) Do NOT provide diagnosis, recommendations, or medical advice.
3) Only point to what the retrieved chunks explicitly contain.
4) Write in plain natural language paragraphs, not as bullets or numbered lists.
5) Do NOT use labels like "Finding", "Quote", "Why it matters", or section headers.
6) If evidence is weak, say so clearly.
7) If there is no relevant evidence, output exactly:
No relevant evidence found in retrieved chunks.
8) Never append that sentence if you already produced evidence text.
9) Keep the narrative cohesive: do not stitch disconnected sentences from different chunks.
10) Reuse key chunk wording in short spans so phrase matching works, but synthesize naturally.
11) Keep dates, measurements, percentages, and medical entities exactly as written in the chunks.
12) Prefer concrete evidence details over generic concluding language.
{granularity_rules}

{output_style}
{length_rules}

Keep it concise and factual.

User query:
{query}

Retrieved chunks:
{chunk_block}
""".strip()


def normalize_summary_text(
    summary_text: str,
) -> str:
    """
    Normalizes common unwanted model formats into plain narrative text.
    """
    text = summary_text.strip()
    if not text:
        return ""

    no_evidence_sentence = "No relevant evidence found in retrieved chunks."
    if no_evidence_sentence in text and text.strip() != no_evidence_sentence:
        text = text.replace(no_evidence_sentence, "")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

    triplet_pattern = re.compile(
        r"(?is)Finding:\s*(.*?)\s*Quote:\s*\"?(.*?)\"?\s*Why it matters:\s*(.*?)(?=(?:\n\s*Finding:)|\Z)"
    )
    triplets = triplet_pattern.findall(text)
    if triplets:
        blocks: List[str] = []
        for finding, quote, why in triplets:
            parts = [finding.strip(), quote.strip(), why.strip()]
            cleaned = [part.rstrip(" .") + "." for part in parts if part]
            if cleaned:
                blocks.append(" ".join(cleaned))
        if blocks:
            return "\n\n".join(blocks).strip()

    text = re.sub(r"(?im)^\s*\d+\)\s*", "", text)
    text = re.sub(r"(?im)^\s*(Finding|Quote|Why it matters)\s*:\s*", "", text)
    text = re.sub(r"(?im)^\s*(EVIDENCE SUMMARY|LIMITATIONS)\s*$", "", text)
    text = text.replace("•", " ").replace("- ", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text]

    deduped_paragraphs: List[str] = []
    seen_paragraph_keys: set[str] = set()
    for paragraph in paragraphs:
        sentence_chunks = re.split(r"(?<=[.!?])\s+", paragraph)
        deduped_sentences: List[str] = []
        previous_sentence_key = ""
        for sentence in sentence_chunks:
            clean_sentence = sentence.strip()
            if not clean_sentence:
                continue
            current_key = _sentence_dedupe_key(clean_sentence)
            if current_key and current_key == previous_sentence_key:
                continue
            deduped_sentences.append(clean_sentence)
            previous_sentence_key = current_key

        if not deduped_sentences:
            continue

        paragraph_text = " ".join(deduped_sentences).strip()
        paragraph_key = _sentence_dedupe_key(paragraph_text)
        if paragraph_key and paragraph_key in seen_paragraph_keys:
            continue
        if paragraph_key:
            seen_paragraph_keys.add(paragraph_key)
        deduped_paragraphs.append(paragraph_text)

    normalized = "\n\n".join(deduped_paragraphs).strip()
    return normalized


def _tokenize_with_spans(text: str) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []
    for match in _TOKEN_RE.finditer(text):
        raw = match.group(0)
        tokens.append(
            {
                "raw": raw,
                "norm": raw.lower(),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return tokens


def _is_content_token(token: str) -> bool:
    normalized = token.lower()

    if normalized in _DEFAULT_STOPWORDS:
        return False
    if _NUMERIC_SIGNAL_RE.match(normalized):
        return True
    if len(token) <= 2:
        return False
    if token.isdigit():
        return False
    return True


def _is_plain_number_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", token))


def _extract_content_tokens(window_tokens: List[str]) -> List[str]:
    content_tokens: List[str] = []
    for idx, token in enumerate(window_tokens):
        if _is_content_token(token):
            content_tokens.append(token)
            continue

        if not _is_plain_number_token(token):
            continue

        prev_token = window_tokens[idx - 1] if idx > 0 else ""
        next_token = window_tokens[idx + 1] if idx + 1 < len(window_tokens) else ""
        if prev_token in _AGE_TIME_UNITS or next_token in _AGE_TIME_UNITS:
            content_tokens.append(token)

    return content_tokens


def _extract_content_token_entries(
    window_tokens: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    norms = [token["norm"] for token in window_tokens]
    content_entries: List[Dict[str, Any]] = []
    for idx, token_entry in enumerate(window_tokens):
        norm = norms[idx]
        if _is_content_token(norm):
            content_entries.append(token_entry)
            continue

        if not _is_plain_number_token(norm):
            continue

        prev_norm = norms[idx - 1] if idx > 0 else ""
        next_norm = norms[idx + 1] if idx + 1 < len(norms) else ""
        if prev_norm in _AGE_TIME_UNITS or next_norm in _AGE_TIME_UNITS:
            content_entries.append(token_entry)

    return content_entries


def _build_chunk_ngram_index(
    chunks: List[Dict[str, Any]],
    min_n: int = 2,
    max_n: int = 8,
) -> Dict[str, str]:
    # Pattern index: chunk_id -> normalized content-token stream.
    # Matching then uses direct pattern checks against these streams.
    index: Dict[str, str] = {}
    for chunk in chunks:
        chunk_id = str(chunk["id"])
        token_entries = _tokenize_with_spans(chunk["text"])
        norms = [token["norm"] for token in token_entries]
        content_tokens = _extract_content_tokens(norms)
        if not content_tokens:
            continue
        index[chunk_id] = " ".join(content_tokens)

    return index


def _match_chunk_ids_for_pattern(
    content_tokens: List[str],
    chunk_pattern_index: Dict[str, str],
    cache: Dict[str, List[str]],
) -> List[str]:
    if not content_tokens:
        return []

    pattern_key = " ".join(content_tokens).strip()
    if not pattern_key:
        return []
    cached = cache.get(pattern_key)
    if cached is not None:
        return cached

    padded_pattern = f" {pattern_key} "
    matched_chunk_ids: List[str] = []
    for chunk_id, token_stream in chunk_pattern_index.items():
        if padded_pattern in f" {token_stream} ":
            matched_chunk_ids.append(chunk_id)

    cache[pattern_key] = matched_chunk_ids
    return matched_chunk_ids


def _find_summary_matches(
    summary_text: str,
    chunk_ngram_index: Dict[str, str],
    min_n: int = 2,
    max_n: int = 8,
    max_matches: int = 120,
) -> List[Dict[str, Any]]:
    tokens = _tokenize_with_spans(summary_text)
    matches: List[Dict[str, Any]] = []
    pattern_cache: Dict[str, List[str]] = {}

    i = 0
    while i < len(tokens):
        best_match: Optional[Dict[str, Any]] = None
        max_window = min(max_n, len(tokens) - i)

        for width in range(max_window, 0, -1):
            window_tokens = tokens[i : i + width]
            content_token_entries = _extract_content_token_entries(window_tokens)
            content_tokens = [token["norm"] for token in content_token_entries]
            if not content_tokens:
                continue
            numeric_signal_tokens = [
                token
                for token in content_tokens
                if _NUMERIC_SIGNAL_RE.match(token)
            ]
            if width < min_n and not numeric_signal_tokens:
                continue
            if len(content_tokens) < 2 and not numeric_signal_tokens:
                continue

            chunk_ids = _match_chunk_ids_for_pattern(
                content_tokens=content_tokens,
                chunk_pattern_index=chunk_ngram_index,
                cache=pattern_cache,
            )
            if not chunk_ids:
                continue

            # Use only the content-token span for cleaner highlighting
            # (avoids underlining leading/trailing stopwords such as "it was").
            char_start = content_token_entries[0]["start"]
            char_end = content_token_entries[-1]["end"]
            phrase = summary_text[char_start:char_end].strip()
            if len(phrase) < 8 and not numeric_signal_tokens:
                continue

            best_match = {
                "start": char_start,
                "end": char_end,
                "phrase": phrase,
                "chunk_ids": chunk_ids,
                "token_end": i + width - 1,
            }
            break

        if best_match is None:
            i += 1
            continue

        if matches and best_match["start"] < matches[-1]["end"]:
            i += 1
            continue

        matches.append(best_match)
        if len(matches) >= max_matches:
            break

        i = int(best_match["token_end"]) + 1

    return matches


def _render_annotated_summary_html(summary_text: str, matches: List[Dict[str, Any]]) -> str:
    styles = """
<style>
.evidence-summary {
    line-height: 1.65;
    font-size: 1.12rem;
    text-align: justify;
    text-justify: inter-word;
}
.evidence-match {
    text-decoration-line: underline;
    text-decoration-thickness: 2px;
    text-decoration-style: solid;
    text-decoration-color: #4DA6FF;
    text-underline-offset: 3px;
    background: rgba(77, 166, 255, 0.14);
    border-radius: 4px;
    cursor: help;
}
</style>
""".strip()

    if not matches:
        plain = html.escape(summary_text).replace("\n", "<br>")
        return f"{styles}<div class=\"evidence-summary\">{plain}</div>"

    parts: List[str] = []
    cursor = 0

    for match in matches:
        start = int(match["start"])
        end = int(match["end"])
        result_ranks = [int(rank) for rank in match.get("result_ranks", []) if isinstance(rank, int)]
        result_labels = [str(label) for label in match.get("result_labels", []) if isinstance(label, str)]
        chunk_ids = [str(chunk_id) for chunk_id in match.get("chunk_ids", [])]

        parts.append(html.escape(summary_text[cursor:start]))

        if result_labels:
            tooltip = ", ".join(result_labels[:4])
        elif result_ranks:
            tooltip = ", ".join(f"Result {rank}" for rank in result_ranks[:4])
        else:
            tooltip = ", ".join(f"chunk[{chunk_id}]" for chunk_id in chunk_ids[:4])
        highlighted_text = html.escape(summary_text[start:end])
        parts.append(
            "<span class=\"evidence-match\" title=\""
            + html.escape(tooltip, quote=True)
            + "\">"
            + highlighted_text
            + "</span>"
        )
        cursor = end

    parts.append(html.escape(summary_text[cursor:]))
    body = "".join(parts).replace("\n", "<br>")
    return f"{styles}<div class=\"evidence-summary\">{body}</div>"


class SummaryNgramHighlighter:
    """
    Precomputes chunk n-grams once and supports repeated live annotation while streaming.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        min_n: int = 2,
        max_n: int = 8,
    ):
        self._chunks = chunks
        self._min_n = min_n
        self._max_n = max_n
        self._index = _build_chunk_ngram_index(
            chunks=chunks,
            min_n=min_n,
            max_n=max_n,
        )
        self._rank_by_chunk_id = {
            str(chunk["id"]): int(chunk.get("result_rank", idx + 1))
            for idx, chunk in enumerate(chunks)
        }
        self._title_by_chunk_id = {
            str(chunk["id"]): str(chunk.get("title", chunk.get("source", "Untitled"))).strip()
            for chunk in chunks
        }

    def annotate(self, summary_text: str) -> Dict[str, Any]:
        normalized_text = normalize_summary_text(summary_text)
        if not normalized_text:
            return {"text": "", "html": "", "matches": []}

        matches = _find_summary_matches(
            summary_text=normalized_text,
            chunk_ngram_index=self._index,
            min_n=self._min_n,
            max_n=self._max_n,
        )
        enriched_matches: List[Dict[str, Any]] = []
        for match in matches:
            chunk_ids = [str(chunk_id) for chunk_id in match.get("chunk_ids", [])]
            result_ranks = sorted(
                {
                    self._rank_by_chunk_id[chunk_id]
                    for chunk_id in chunk_ids
                    if chunk_id in self._rank_by_chunk_id
                }
            )
            result_labels: List[str] = []
            for rank in result_ranks:
                title_for_rank = ""
                for chunk_id in chunk_ids:
                    chunk_rank = self._rank_by_chunk_id.get(chunk_id)
                    if chunk_rank != rank:
                        continue
                    title_for_rank = self._title_by_chunk_id.get(chunk_id, "")
                    break

                safe_title = " ".join(title_for_rank.split()) if title_for_rank else "Untitled"
                if len(safe_title) > 90:
                    safe_title = safe_title[:87].rstrip() + "..."
                result_labels.append(f"Result {rank}: {safe_title}")

            enriched = dict(match)
            enriched["chunk_ids"] = chunk_ids
            enriched["result_ranks"] = result_ranks
            enriched["result_labels"] = result_labels
            enriched_matches.append(enriched)

        html_output = _render_annotated_summary_html(
            summary_text=normalized_text,
            matches=enriched_matches,
        )

        return {
            "text": normalized_text,
            "html": html_output,
            "matches": [
                {
                    "phrase": str(match.get("phrase", "")),
                    "start": int(match.get("start", 0)),
                    "end": int(match.get("end", 0)),
                    "chunk_ids": [str(chunk_id) for chunk_id in match.get("chunk_ids", [])],
                    "result_ranks": [int(rank) for rank in match.get("result_ranks", [])],
                    "result_labels": [str(label) for label in match.get("result_labels", [])],
                }
                for match in enriched_matches
            ],
        }


def _parse_llm_matches(llm_output: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunk_by_id = {chunk["id"]: chunk for chunk in chunks}

    try:
        payload = json.loads(_clean_json_response(llm_output))
    except json.JSONDecodeError:
        return []

    raw_matches = payload.get("matches", [])
    if not isinstance(raw_matches, list):
        return []

    parsed: List[Dict[str, Any]] = []
    for item in raw_matches:
        if not isinstance(item, dict):
            continue

        chunk_id = str(item.get("chunk_id", "")).strip()
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            continue

        raw_quotes = item.get("quotes", [])
        if not isinstance(raw_quotes, list):
            raw_quotes = []

        quotes: List[str] = []
        for quote in raw_quotes:
            if not isinstance(quote, str):
                continue

            quote_text = quote.strip()
            if not quote_text:
                continue

            if quote_text in chunk["text"] and quote_text not in quotes:
                quotes.append(quote_text)

        if not quotes:
            continue

        reason = str(item.get("reason", "")).strip()
        if not reason:
            reason = "This chunk contains terms directly related to the query."

        parsed.append(
            {
                "chunk_id": chunk["id"],
                "bucket": chunk["bucket"],
                "source": chunk["source"],
                "patient_id": chunk["patient_id"],
                "score": chunk["score"],
                "reason": reason,
                "quotes": quotes[:2],
            }
        )

        if len(parsed) >= 5:
            break

    return parsed


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]{3,}", text.lower())


def _extract_fallback_quote(text: str, tokens: List[str], max_len: int = 220) -> str:
    lowered = text.lower()
    positions = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
    if not positions:
        return text[:max_len]

    start = min(positions)
    slice_start = max(0, start - 60)
    slice_end = min(len(text), slice_start + max_len)
    return text[slice_start:slice_end]


def _fallback_matches(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    query_tokens = sorted(set(_tokenize(query)))
    if not query_tokens:
        return []

    scored: List[Dict[str, Any]] = []
    for chunk in chunks:
        chunk_text_lower = chunk["text"].lower()
        matching_tokens = [token for token in query_tokens if token in chunk_text_lower]
        if not matching_tokens:
            continue

        quote = _extract_fallback_quote(chunk["text"], matching_tokens)
        if not quote:
            continue

        scored.append(
            {
                "token_hits": len(matching_tokens),
                "chunk_id": chunk["id"],
                "bucket": chunk["bucket"],
                "source": chunk["source"],
                "patient_id": chunk["patient_id"],
                "score": chunk["score"],
                "reason": (
                    "Fallback relevance from lexical overlap with query terms: "
                    + ", ".join(matching_tokens[:6])
                ),
                "quotes": [quote],
            }
        )

    scored.sort(key=lambda item: item["token_hits"], reverse=True)
    return [
        {
            "chunk_id": item["chunk_id"],
            "bucket": item["bucket"],
            "source": item["source"],
            "patient_id": item["patient_id"],
            "score": item["score"],
            "reason": item["reason"],
            "quotes": item["quotes"],
        }
        for item in scored[:5]
    ]


def build_chunk_grounding(
    query: str,
    search_payload: Dict[str, Any],
    ask_service: Optional[AskServicePort] = None,
    max_chunks_for_model: int = 12,
    max_new_tokens: int = 500,
) -> Dict[str, Any]:
    chunks = _normalize_chunks(search_payload=search_payload, limit=max_chunks_for_model)
    if not chunks:
        return {"query": query, "matches": [], "mode": "empty"}

    svc = ask_service or AskService()
    prompt = _build_prompt(query=query, chunks=chunks)

    llm_output = ""
    matches: List[Dict[str, Any]] = []
    mode = "llm"

    try:
        llm_output = svc.ask(prompt=prompt, max_new_tokens=max_new_tokens)
        matches = _parse_llm_matches(llm_output=llm_output, chunks=chunks)
    except Exception:
        matches = []

    if not matches:
        matches = _fallback_matches(query=query, chunks=chunks)
        mode = "fallback"

    return {
        "query": query,
        "matches": matches,
        "mode": mode,
    }


def stream_chunk_evidence_summary(
    query: str,
    search_payload: Dict[str, Any],
    ask_service: Optional[AskServicePort] = None,
    max_chunks_for_model: int = 12,
    max_new_tokens: int = 650,
) -> Iterator[str]:
    chunks = _normalize_chunks(search_payload=search_payload, limit=max_chunks_for_model)
    if not chunks:
        return

    svc = ask_service or AskService()
    prompt = _build_summary_prompt(query=query, chunks=chunks)
    token_budget = max_new_tokens
    if _is_dataset_only_chunks(chunks):
        token_budget = max(token_budget, 700)

    yield from svc.ask_stream(prompt=prompt, max_new_tokens=token_budget)


def annotate_summary_with_chunk_matches(
    summary_text: str,
    search_payload: Dict[str, Any],
    max_chunks_for_matching: int = 20,
    min_n: int = 2,
    max_n: int = 8,
) -> Dict[str, Any]:
    chunks = _normalize_chunks(
        search_payload=search_payload,
        limit=max_chunks_for_matching,
    )
    if not chunks:
        return {"text": normalize_summary_text(summary_text), "html": "", "matches": []}

    highlighter = SummaryNgramHighlighter(chunks=chunks, min_n=min_n, max_n=max_n)
    return highlighter.annotate(summary_text)


def build_summary_highlighter(
    search_payload: Dict[str, Any],
    max_chunks_for_matching: int = 20,
    min_n: int = 2,
    max_n: int = 8,
) -> Optional[SummaryNgramHighlighter]:
    chunks = _normalize_chunks(
        search_payload=search_payload,
        limit=max_chunks_for_matching,
    )
    if not chunks:
        return None

    return SummaryNgramHighlighter(chunks=chunks, min_n=min_n, max_n=max_n)
