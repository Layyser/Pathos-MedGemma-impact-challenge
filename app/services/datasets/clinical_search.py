import json
import logging
import re
from typing import Dict, Optional

from app.providers.cloud_llm.factory import CloudLLMFactory
from app.services.ports import CloudLLMPort

logger = logging.getLogger(__name__)


class ClinicalQueryBuilder:
    """
    Service responsible for constructing high-precision clinical and biomedical 
    search queries (PubMed/MeSH) by leveraging a Cloud LLM dynamically.
    """

    def __init__(self, llm_provider: Optional[CloudLLMPort] = None):
        self.llm_provider = llm_provider or CloudLLMFactory.get_provider()

    def build_query(self, topic: str, user_context: Optional[str] = None) -> Dict[str, str]:
        """
        Generates a query string and explanation for ANY biomedical/scientific topic.
        
        Args:
            topic: The core subject (e.g., "Animal Neurology", "Gene Editing").
            user_context: Optional specific instructions (e.g., "Mice models only").
            
        Returns:
            Dict containing the 'query', 'explanation', and 'source'.
        """
        
        # Default context if the user doesn't specify anything
        context_instruction = user_context if user_context else "Use standard filters appropriate for the topic. Do not restrict to humans unless the topic explicitly requires it."

        # 1. Construct the prompt requesting strict JSON
        prompt = f"""
You are an elite Medical Librarian and PubMed Boolean Search Architect. 
Your ONLY job is to translate natural language user queries into flawlessly formatted, highly optimized PubMed search strings.
You are interacting with an automated API pipeline. If you output anything other than strict, parseable JSON, the system will crash.

### THE 5 UNBREAKABLE RULES OF PUBMED QUERY CONSTRUCTION:

1. THE ME-TIAB GOLDEN RULE (NO ORPHANED CONCEPTS):
To prevent returning zero results, EVERY medical concept MUST be paired with its free-text Title/Abstract equivalent using OR. 
- WRONG: "Myocardial Infarction"[MeSH]
- CORRECT: ("Myocardial Infarction"[MeSH] OR "heart attack"[tiab] OR "myocardial infarction"[tiab])

2. THE HUMAN FILTER MANDATE:
Clinical search is for humans. You MUST append `AND "Humans"[MeSH]` to the very end of EVERY query, UNLESS the user explicitly mentions:
- Animals (dogs, cats, mice, rats, veterinary)
- In-vitro studies (cell lines, petri dish, assays)
If they mention animals/in-vitro, DO NOT use the Humans filter. Use the specific animal MeSH (e.g., "Dogs"[MeSH]).

3. THE BILINGUAL NET (LANGUAGE FALLBACK):
95% of high-impact medical research is published in English. If the user writes in a language other than English (e.g., Spanish, Catalan), you MUST include both English AND their detected language to avoid zero-result queries.
- FORMAT: `AND ("English"[lang] OR "Spanish"[lang])`
- If the user writes in English, just use `AND "English"[lang]`.

4. BOOLEAN SYNTAX & NESTING:
- Operators MUST be uppercase: AND, OR, NOT.
- Parentheses MUST be perfectly balanced. Group synonyms with OR inside parentheses, then combine concepts with AND.
- Example structure: `((Concept1) AND (Concept2)) AND Filter1 AND Filter2`

5. ABSOLUTE JSON ENFORCEMENT:
- Output ONLY valid JSON.
- NO markdown code blocks (do not use ```json).
- NO conversational text before or after the JSON.
- The explanation must be written in the same language the user used.

---
### EXHAUSTIVE EXAMPLES:

Input: "Breast cancer treatments"
{{
    "query": "((\\"Breast Neoplasms\\"[MeSH] OR \\"breast cancer\\"[tiab] OR \\"breast tumor\\"[tiab]) AND (\\"Therapeutics\\"[MeSH] OR \\"treatment\\"[tiab] OR \\"therapy\\"[tiab])) AND \\"Humans\\"[MeSH] AND \\"English\\"[lang]",
    "explanation": "Mapped 'breast cancer' to Breast Neoplasms MeSH and tiab terms. Mapped 'treatments' to Therapeutics. Applied default Humans filter and English language."
}}

Input: "Quiero estudios sobre asma severo en niños pequeños"
{{
    "query": "((\\"Asthma\\"[MeSH] OR \\"asma\\"[tiab] OR \\"asthma\\"[tiab]) AND (\\"Severity of Illness Index\\"[MeSH] OR \\"severe\\"[tiab] OR \\"severo\\"[tiab]) AND (\\"Child, Preschool\\"[MeSH] OR \\"Infant\\"[MeSH] OR \\"niños pequeños\\"[tiab] OR \\"toddler\\"[tiab])) AND \\"Humans\\"[MeSH] AND (\\"English\\"[lang] OR \\"Spanish\\"[lang])",
    "explanation": "Se combinaron términos MeSH y de texto libre para asma, severidad y niños en edad preescolar/infantil. Se aplicó el filtro predeterminado de humanos y se incluyó tanto inglés como español."
}}

Input: "Eficàcia de la quimioteràpia en gossos amb limfoma"
{{
    "query": "((\\"Lymphoma\\"[MeSH] OR \\"lymphoma\\"[tiab] OR \\"limfoma\\"[tiab]) AND (\\"Drug Therapy\\"[MeSH] OR \\"Antineoplastic Agents\\"[MeSH] OR \\"quimioteràpia\\"[tiab] OR \\"chemotherapy\\"[tiab]) AND (\\"Dogs\\"[MeSH] OR \\"dogs\\"[tiab] OR \\"gossos\\"[tiab])) AND (\\"English\\"[lang] OR \\"Catalan\\"[lang])",
    "explanation": "S'han utilitzat termes MeSH i text lliure per limfoma, quimioteràpia i gossos. Com que l'usuari especifica gossos, NO s'ha aplicat el filtre d'humans. S'ha ampliat la cerca a anglès i català per garantir resultats."
}}

Input: "CRISPR gene editing in HeLa cell lines"
{{
    "query": "((\\"CRISPR-Cas Systems\\"[MeSH] OR \\"CRISPR\\"[tiab]) AND (\\"Gene Editing\\"[MeSH] OR \\"gene editing\\"[tiab]) AND (\\"HeLa Cells\\"[MeSH] OR \\"HeLa\\"[tiab] OR \\"cell line\\"[tiab])) AND \\"English\\"[lang]",
    "explanation": "Mapped CRISPR, gene editing, and HeLa cells. Because this is an in-vitro cell line study, the Humans filter was explicitly excluded."
}}

Input: "Side effects of aspirin NOT including bleeding"
{{
    "query": "((\\"Aspirin\\"[MeSH] OR \\"aspirin\\"[tiab]) AND (\\"Drug-Related Side Effects and Adverse Reactions\\"[MeSH] OR \\"side effects\\"[tiab] OR \\"adverse effects\\"[tiab])) NOT (\\"Hemorrhage\\"[MeSH] OR \\"bleeding\\"[tiab]) AND \\"Humans\\"[MeSH] AND \\"English\\"[lang]",
    "explanation": "Combined aspirin with side effects, and used the NOT operator to exclude hemorrhage/bleeding. Applied Humans and English filters."
}}

---
### ACTUAL TASK:
Input: "{topic} {context_instruction}"
{{
    "query":
}}
        """

        # 2. Call the Remote LLM
        logger.info(f"Generating dynamic query for topic: {topic} | Context: {user_context}")
        # A slightly higher temperature helps with MeSH term creativity for obscure topics
        generated_response = self.llm_provider.generate(prompt, temperature=0.2)

        # 3. Parse the JSON response
        parsed_data = self._parse_json_response(generated_response)

        return {
            "query": self._normalize_query(parsed_data.get("query", "")),
            "explanation": parsed_data.get("explanation", "No explanation provided."),
            "source": "MeSH"
        }

    def _parse_json_response(self, llm_output: str) -> Dict[str, str]:
        """
        Safely extracts and parses JSON from the LLM output, handling cases 
        where the LLM wraps the response in markdown code blocks.
        """
        try:
            # Strip markdown formatting if the LLM includes it (e.g., ```json ... ```)
            clean_output = re.sub(r'```(?:json)?', '', llm_output, flags=re.IGNORECASE).strip()
            clean_output = clean_output.strip('`')
            
            return json.loads(clean_output)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM: {e}. Raw output: {llm_output}")
            return {
                "query": "",
                "explanation": "Error: Could not parse LLM response into JSON."
            }

    def _normalize_query(self, query: str) -> str:
        """
        Ensures consistent spacing around boolean operators.
        """
        if not query:
            return ""
        # Clean up stray quotes around AND/OR/NOT that sometimes get generated
        query = re.sub(r"[\'\"]\s*(AND|OR|NOT)\s*[\'\"]", r" \1 ", query, flags=re.IGNORECASE)
        return " ".join(query.split())