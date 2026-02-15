
import os
import sys
import tempfile
from pathlib import Path
import torch
try:
    import doclayout_yolo.nn.tasks
    torch.serialization.add_safe_globals([doclayout_yolo.nn.tasks.YOLOv10DetectionModel])
except ImportError:
    pass

# Setup internal paths so the nested mineru/demo can find each other
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
    print(f"[DocumentReaderAgent] Initialization Error: {e}")

class DocumentReaderAgent:
    """
    A LangGraph-ready agent that handles document extraction.
    Input: File path (PDF or DOCX)
    Output: (text_content, path_to_generated_file)
    """
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or tempfile.gettempdir()

    def run(self, file_path, method="auto", lang="en"):
        """
        Equivalent to a LangGraph 'node'. 
        Takes a file, returns text and the resulting file path.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() == '.pdf':
            return self._process_pdf(path, method, lang)
        elif path.suffix.lower() == '.docx':
            return self._process_docx(path), str(path)
        else:
            raise ValueError("Unsupported format. Use PDF or DOCX.")

    def _process_pdf(self, path, method, lang):
        if not MINERU_AVAILABLE:
            return "Error: Magic-PDF not available", None

        # Create localized output for this specific run
        run_output = tempfile.mkdtemp(prefix="agent_out_", dir=self.output_dir)
        
        try:
            parse_doc(
                path_list=[path],
                output_dir=run_output,
                lang=lang,
                backend="pipeline",
                method=method,
                formula_enable=False,
                table_enable=False
            )
            
            # Find the resulting Markdown file
            for root, _, files in os.walk(run_output):
                for f in files:
                    if f.endswith('.md') and not f.endswith('_layout.md'):
                        res_path = os.path.join(root, f)
                        with open(res_path, 'r', encoding='utf-8') as fh:
                            return fh.read(), res_path
            
            return "", None
        except Exception as e:
            return f"Extraction Agent Error: {str(e)}", None

    def _process_docx(self, path):
        try:
            from docx import Document
            doc = Document(path)
            return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except ImportError:
            return "Error: python-docx not installed."
