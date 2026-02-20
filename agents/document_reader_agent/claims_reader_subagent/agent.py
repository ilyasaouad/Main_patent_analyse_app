import os
import sys
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
    # Log error internally but allow the class to exist
    print(f"[ClaimsReaderSubAgent] Initialization Error: {e}")


class ClaimsReaderSubAgent:
    """
    A LangGraph-ready agent that handles patent claims extraction.
    Input: File path (PDF, DOCX, or image)
    Output: (text_content, path_to_txt_file)
    """

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or tempfile.gettempdir()

    def run(self, file_path, method="auto", lang="en"):
        """
        Equivalent to a LangGraph 'node'.
        Takes a file, returns text and the resulting text file path.
        """
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

        # Save to txt file as requested
        text = self._clean_text(text)
        txt_path = "claims_text.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        return text, txt_path

    def _clean_text(self, text):
        import re

        # Remove '#' characters
        text = text.replace("#", "")

        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            # Check for numbers at start of line that are multiples of 5
            # This handles typical patent line numbering (5, 10, 15...)
            match = re.match(r"^(\s*)(\d+)(\s+)", line)
            if match:
                num = int(match.group(2))
                if num > 0 and num % 5 == 0:
                    # Remove the number and immediate following whitespace
                    line = line[match.end() :]
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _process_image(self, path, method, lang):
        return self._process_pdf(path, method, lang)

    def _process_pdf(self, path, method, lang):
        if not MINERU_AVAILABLE:
            return "Error: MinerU/Magic-PDF components not correctly imported.", None

        run_output = tempfile.mkdtemp(prefix="claims_agent_out_", dir=self.output_dir)

        try:
            os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
            parse_doc(
                path_list=[path],
                output_dir=run_output,
                lang=lang,
                backend="pipeline",
                method=method,
                formula_enable=False,
                table_enable=False,
            )

            for root, _, files in os.walk(run_output):
                for f in files:
                    if f.endswith(".md") and not f.endswith("_layout.md"):
                        res_path = os.path.join(root, f)
                        with open(res_path, "r", encoding="utf-8") as fh:
                            return fh.read(), res_path

            return (
                f"Error: No output found in {run_output}. Check model weights or file format.",
                None,
            )
        except Exception as e:
            return f"Claims Extraction Exception: {str(e)}", None

    def _process_docx(self, path):
        try:
            from docx import Document

            doc = Document(path)
            return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except ImportError:
            return "Error: python-docx not installed."
