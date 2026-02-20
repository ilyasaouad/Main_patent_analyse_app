import os
import sys
import re
import tempfile
from pathlib import Path
import torch

try:
    import doclayout_yolo.nn.tasks

    torch.serialization.add_safe_globals(
        [doclayout_yolo.nn.tasks.YOLOv10DetectionModel]
    )
except ImportError:
    pass

# Setup internal paths so the nested mineru/demo can find each other
os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
MODULE_ROOT = Path(__file__).parent.absolute()
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

try:
    from mineru.cli.common import read_fn, prepare_env
    from demo.demo import parse_doc
    from mineru.utils.enum_class import MakeMode

    MINERU_AVAILABLE = True
except ImportError as e:
    MINERU_AVAILABLE = False
    print(f"[DescriptionReaderSubAgent] Initialization Error: {e}")


class DescriptionReaderSubAgent:
    """
    A LangGraph-ready agent that handles patent description extraction.
    Input: File path (PDF, DOCX, or image)
    Output: (text_content, path_to_txt_file)

    If claims_uploaded=False, the agent will:
      1. Detect the claims section inside the description text.
      2. Save everything BEFORE the claims header as describtion_text.txt (clean description + abstract).
      3. Save everything FROM the claims header onward as claims_text.txt,
         stripping any abstract that appears after the claims.

    If claims_uploaded=True, the full text is saved as describtion_text.txt as-is,
    and no claims extraction is attempted.
    """

    # Signals the START of the claims section
    CLAIMS_HEADER_PATTERN = re.compile(
        r"(?im)"
        r"("
        r"^\s*claims?\s*$"
        r"|^\s*what\s+is\s+claimed\s*(is)?\s*[:\.]?\s*$"
        r"|^\s*what\s+we\s+claim\s*(is)?\s*[:\.]?\s*$"
        r"|^\s*i\s+claim\s*[:\.]?\s*$"
        r"|^\s*we\s+claim\s*[:\.]?\s*$"
        r")"
    )

    # Signals the START of an abstract section (to strip from end of claims)
    ABSTRACT_HEADER_PATTERN = re.compile(
        r"(?im)"
        r"("
        r"^\s*abstract\s*$"
        r"|^\s*abstract\s+of\s+the\s+disclosure\s*[:\.]?\s*$"
        r"|^\s*abstract\s+of\s+the\s+invention\s*[:\.]?\s*$"
        r"|^\s*abstract\s*[:\.]\s*$"
        r"|^\s*abstract\s+"
        r")"
    )

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or tempfile.gettempdir()

    def _clear_output_files(self):
        for filename in [
            "describtion_text.txt",
            "claims_text.txt",
            "abstract_text.txt",
            "drawings_text.txt",
        ]:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("")

    def run(self, file_path, method="auto", lang="en", claims_uploaded=False):
        """
        Equivalent to a LangGraph 'node'.
        Takes a file, returns the description text and its file path.

        Args:
            file_path      : Path to the description document (PDF, DOCX, or image).
            method         : MinerU parsing method ("auto", "ocr", etc.).
            lang           : Language hint for MinerU.
            claims_uploaded: Set to True if the user uploaded a separate claims file.
                             If False, this agent will split the text into clean
                             description and claims parts automatically.
        """
        self._clear_output_files()
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            text, _ = self._process_pdf(path, method, lang)
        elif ext == ".docx":
            text = self._process_docx(path)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            text, _ = self._process_image(path, method, lang)
        else:
            raise ValueError(f"Unsupported format: {ext}. Use PDF, DOCX, or Image.")

        text = self._clean_text(text)

        abstract_text = ""
        if claims_uploaded:
            # No splitting needed — save full text as description
            description_text = text
            self._save_text("describtion_text.txt", description_text)
            print(
                "[DescriptionReaderSubAgent] Claims file uploaded separately. "
                "Full description saved to describtion_text.txt"
            )
        else:
            # Split into clean description, claims, and abstract
            description_text, abstract_text = self._split_and_save(text)

        return description_text, "describtion_text.txt", abstract_text

    # ------------------------------------------------------------------
    # Core splitting logic
    # ------------------------------------------------------------------

    def _split_and_save(self, full_text):
        """
        Finds the claims header in the full text and splits it into:

          - Description (everything before abstract and claims headers)
            → describtion_text.txt

          - Abstract (if found, extracted from description section)
            → abstract_text.txt

          - Claims (everything from the claims header onward,
            with any trailing abstract stripped out)
            → claims_text.txt

        Returns the clean description text.
        """
        claims_match = self.CLAIMS_HEADER_PATTERN.search(full_text)

        if not claims_match:
            # No claims section found — try to extract abstract anyway
            description_text, abstract_text = self._extract_abstract(full_text.strip())
            self._save_text("describtion_text.txt", description_text)
            if abstract_text:
                self._save_text("abstract_text.txt", abstract_text)
                print(
                    "[DescriptionReaderSubAgent] Warning: No claims section detected. "
                    "Full text saved as description. abstract_text.txt created."
                )
            else:
                print(
                    "[DescriptionReaderSubAgent] Warning: No claims section detected. "
                    "Full text saved as description."
                )
            return description_text, abstract_text

        # --- Split at the claims header ---
        claims_start = claims_match.start()

        # Description = everything before the claims header
        description_text = full_text[:claims_start].strip()

        # Claims = everything from the claims header onward
        claims_text = full_text[claims_start:].strip()

        # --- Strip abstract from end of claims text (abstract often appears after claims) ---
        claims_text, abstract_from_claims = self._strip_abstract_from_claims(
            claims_text
        )

        # --- Extract abstract from description (if abstract appears before claims) ---
        description_text, abstract_from_desc = self._extract_abstract(description_text)

        # Use abstract from claims if found, otherwise from description
        abstract_text = abstract_from_claims or abstract_from_desc

        # --- Save all files ---
        self._save_text("describtion_text.txt", description_text)
        self._save_text("claims_text.txt", claims_text)
        if abstract_text:
            self._save_text("abstract_text.txt", abstract_text)

        print("[DescriptionReaderSubAgent] Successfully split document:")
        print(
            f"  → describtion_text.txt  ({len(description_text)} chars) — description"
        )
        print(f"  → claims_text.txt       ({len(claims_text)} chars) — claims only")
        if abstract_text:
            print(f"  → abstract_text.txt     ({len(abstract_text)} chars) — abstract")

        return description_text, abstract_text

    def _extract_abstract(self, text):
        """
        Extracts abstract section from text and returns (text_without_abstract, abstract).
        Abstract is usually at the beginning of the document or before claims.
        """
        abstract_match = self.ABSTRACT_HEADER_PATTERN.search(text)

        if not abstract_match:
            print("[DescriptionReaderSubAgent] No abstract header found in text")
            return text, None

        print(
            f"[DescriptionReaderSubAgent] Abstract header found at position {abstract_match.start()}: '{abstract_match.group()}'"
        )
        abstract_start = abstract_match.end()

        # Find where abstract ends - either at claims header or next major section
        # Look for the next section header or end of text
        next_section = self.CLAIMS_HEADER_PATTERN.search(text, abstract_start)
        if next_section:
            abstract_end = next_section.start()
            print(
                f"[DescriptionReaderSubAgent] Abstract ends at claims section (position {abstract_end})"
            )
        else:
            # Look for common section headers after abstract
            next_header = re.search(
                r"(?im)^(?:description|background|summary|detailed\s+description|brief\s+description)\s*$",
                text[abstract_start:],
            )
            if next_header:
                abstract_end = abstract_start + next_header.start()
                print(
                    f"[DescriptionReaderSubAgent] Abstract ends at section header (position {abstract_end})"
                )
            else:
                # Try to find paragraph break - abstract is usually 1-2 paragraphs
                # Look for double newline or end of text
                para_break = re.search(r"\n\s*\n", text[abstract_start:])
                if para_break:
                    # Take first paragraph as abstract (max ~500 chars for typical abstract)
                    potential_end = abstract_start + para_break.start()
                    if potential_end - abstract_start < 500:
                        abstract_end = potential_end
                    else:
                        abstract_end = len(text)
                else:
                    abstract_end = len(text)
                print(
                    f"[DescriptionReaderSubAgent] Abstract ends at position {abstract_end}"
                )

        abstract_text = text[abstract_start:abstract_end].strip()

        if not abstract_text:
            print("[DescriptionReaderSubAgent] Abstract text is empty after extraction")
            return text, None

        # Remove abstract section from description
        description_without_abstract = (
            text[: abstract_match.start()] + text[abstract_end:]
        )
        description_without_abstract = description_without_abstract.strip()

        print(
            f"[DescriptionReaderSubAgent] Abstract extracted ({len(abstract_text)} chars)"
        )
        return description_without_abstract, abstract_text

    def _strip_abstract_from_claims(self, claims_text):
        """
        Removes any abstract section that appears inside the claims text.
        Patents sometimes print the abstract after the claims block,
        so we find the abstract header and cut everything from there onward.

        Returns: (claims_without_abstract, abstract_text or None)
        """
        abstract_match = self.ABSTRACT_HEADER_PATTERN.search(claims_text)

        if abstract_match:
            stripped = claims_text[: abstract_match.start()].strip()
            abstract_text = claims_text[abstract_match.end() :].strip()
            print(
                "[DescriptionReaderSubAgent] Abstract detected at end of claims — extracted."
            )
            return stripped, abstract_text

        return claims_text, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_text(self, filename, text):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)

    def _clean_text(self, text):
        # Remove markdown image links: ![](images/...)
        text = re.sub(r"!\[.*?\]\(images/.*?\)", "", text)
        # Remove '#' characters (common in Markdown output from MinerU)
        text = text.replace("#", "")
        # Convert HTML tables to markdown
        text = self._html_tables_to_markdown(text)

        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            # Remove patent line numbers that are multiples of 5 (5, 10, 15 ...)
            match = re.match(r"^(\s*)(\d+)(\s+)", line)
            if match:
                num = int(match.group(2))
                if num > 0 and num % 5 == 0:
                    line = line[match.end() :]
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _html_tables_to_markdown(self, text):
        def convert_table(match):
            html = match.group(0)
            rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
            if not rows:
                return ""
            md_rows = []
            for i, row in enumerate(rows):
                cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
                cells = [c.strip() for c in cells]
                md_rows.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
            return "\n".join(md_rows)

        return re.sub(r"<table>.*?</table>", convert_table, text, flags=re.DOTALL)

    # ------------------------------------------------------------------
    # Document processors
    # ------------------------------------------------------------------

    def _process_image(self, path, method, lang):
        return self._process_pdf(path, method, lang)

    def _process_pdf(self, path, method, lang):
        if not MINERU_AVAILABLE:
            return (
                "Error: MinerU/Magic-PDF components not correctly imported. "
                "Check your installation.",
                None,
            )

        run_output = tempfile.mkdtemp(prefix="desc_agent_out_", dir=self.output_dir)

        try:
            os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
            parse_doc(
                path_list=[path],
                output_dir=run_output,
                lang=lang,
                backend="pipeline",
                method=method,
                formula_enable=False,
                table_enable=True,
            )

            for root, _, files in os.walk(run_output):
                for f in files:
                    if f.endswith(".md") and not f.endswith("_layout.md"):
                        res_path = os.path.join(root, f)
                        with open(res_path, "r", encoding="utf-8") as fh:
                            return fh.read(), res_path

            return (
                f"Error: Extraction completed but no Markdown output was found in "
                f"{run_output}. This often happens if the PDF is encrypted, corrupted, "
                f"or if the model weights (like doclayout_yolo) are missing.",
                None,
            )
        except Exception as e:
            return f"Description Extraction Exception: {str(e)}", None

    def _process_docx(self, path):
        try:
            from docx import Document

            doc = Document(path)
            return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except ImportError:
            return "Error: python-docx not installed."
