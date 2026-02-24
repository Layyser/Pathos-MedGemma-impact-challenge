import logging
import re
from typing import List

from app.domain.chunk import Chunk
from app.domain.document import Document
from app.shared.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


class MedicalChunkingService:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        
        # Medical-aware abbreviations to prevent false sentence splits
        abbreviations = [
            'dr', 'prof', 'mr', 'ms', 'mrs', 'st', 'ave', 
            'mg', 'dl', 'ml', 'kg', 'oz', 'tab', 'cap',
            'bid', 'tid', 'qid', 'po', 'iv', 'im', 'stat',
            'vol', 'no', 'vs', 'approx', 'min', 'max', 'mcg'
        ]
        
        # Regex: Look for . ! or ? followed by whitespace,
        # but NOT if preceded by a known abbreviation + period (e.g., "Dr.", "mg.")
        lookbehind = "".join(
            rf"(?<!\b{re.escape(abbr)}\.)"
            for abbr in abbreviations
        )
        self.split_pattern = re.compile(
            rf"{lookbehind}(?<=[.!?])\s+",
            flags=re.IGNORECASE,
        )


    def _get_sentences_with_offsets(self, text: str) -> List[dict]:
        sentences = []
        last_pos = 0
        for match in self.split_pattern.finditer(text):
            sentence_text = text[last_pos:match.start()].strip()
            if sentence_text:
                sentences.append({"text": sentence_text, "start": last_pos, "end": match.start()})
            last_pos = match.end()
        
        if last_pos < len(text):
            final_text = text[last_pos:].strip()
            if final_text:
                sentences.append({"text": final_text, "start": last_pos, "end": len(text)})
        return sentences


    def create_chunks(self, doc: Document) -> List[Chunk]:
        if not doc.text:
            logger.warning("Document %s has no extractable text.", doc.id)
            return []

        sentences = self._get_sentences_with_offsets(doc.text)
        
        chunk_objs = []
        current_sentences = []
        current_len = 0
        chunk_index = 0

        for i, s in enumerate(sentences):
            s_len = len(s["text"])
            
            # If adding this sentence exceeds chunk_size, flush
            if current_len + s_len > self.chunk_size and current_sentences:
                chunk_objs.append(self._build_chunk_object(doc, current_sentences, chunk_index))
                chunk_index += 1
                
                # --- OVERLAP LOGIC ---
                new_sentences = []
                overlap_len = 0
                # Move backwards from current point to collect overlap
                for j in range(i - 1, -1, -1):
                    prev_s = sentences[j]
                    if overlap_len + len(prev_s["text"]) <= self.chunk_overlap:
                        new_sentences.insert(0, prev_s)
                        overlap_len += len(prev_s["text"]) + 1
                    else:
                        break
                
                current_sentences = new_sentences
                current_len = overlap_len
            
            current_sentences.append(s)
            current_len += s_len + 1

        if current_sentences:
            chunk_objs.append(self._build_chunk_object(doc, current_sentences, chunk_index))

        logger.info("Chunking complete", extra={"doc_id": doc.id, "count": len(chunk_objs)})
        return chunk_objs


    def _build_chunk_object(self, doc: Document, sentences: List[dict], index: int) -> Chunk:
        """ Helper to construct the Chunk domain object from a list of sentence dicts """
        content = " ".join([s["text"] for s in sentences])
        
        # The start of the first sentence in the group
        start_offset = sentences[0]["start"]
        # The end of the last sentence in the group
        end_offset = sentences[-1]["end"]

        return Chunk(
            id=f"{doc.id}_ch_{index}",
            document_id=doc.id,
            patient_id=doc.patient_id,
            text=content,            # Mirroring content for your specific dataclass
            start_offset=start_offset,
            end_offset=end_offset
        )
