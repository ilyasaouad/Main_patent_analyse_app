import os
import json
from typing import Optional, Tuple

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")


def extract_sections_with_llm(
    text: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Use LLM to extract description, claims, and abstract from patent text.
    Returns: (description, claims, abstract) or (None, None, None) if failed.
    """
    try:
        from openai import OpenAI

        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

        prompt = f"""Extract the following sections from this patent document text. Return as JSON with keys: "description", "claims", "abstract". If a section is not found, set it to null.

Rules:
- "description": The detailed description/background/summary sections (everything before claims)
- "claims": The numbered claims section (starts with "Claims" or "What is claimed")
- "abstract": The abstract section (usually starts with "Abstract")

Return ONLY valid JSON, no other text.

Document text:
{text[:8000]}"""

        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000,
        )

        content = response.choices[0].message.content.strip()

        # Try to parse JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content)
            description = result.get("description")
            claims = result.get("claims")
            abstract = result.get("abstract")

            print(
                f"[LLM Fallback] Extracted - description: {bool(description)}, claims: {bool(claims)}, abstract: {bool(abstract)}"
            )
            return description, claims, abstract

        except json.JSONDecodeError as e:
            print(f"[LLM Fallback] JSON parse error: {e}")
            return None, None, None

    except Exception as e:
        print(f"[LLM Fallback] Error: {e}")
        return None, None, None


def is_ollama_available() -> bool:
    """Check if Ollama is running and model is available."""
    try:
        import requests

        resp = requests.get(OLLAMA_BASE_URL.replace("/v1", "/api/tags"), timeout=2)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            if OLLAMA_MODEL.split(":")[0] in model_names or any(
                OLLAMA_MODEL.split(":")[0] in m for m in model_names
            ):
                return True
            print(
                f"[LLM Fallback] Model '{OLLAMA_MODEL}' not found. Available: {model_names}"
            )
        return False
    except Exception as e:
        print(f"[LLM Fallback] Ollama not available: {e}")
        return False
