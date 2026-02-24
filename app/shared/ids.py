import hashlib


def generate_document_id(content: str, patient_id: str) -> str:
    """
    Generates a deterministic ID based on content and patient_id.
    Including patient_id prevents collisions if two patients have 
    identical documents.
    """
    payload = f"{patient_id}:{content}"
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()