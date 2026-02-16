
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
os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
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
    print(f"[DrawingReaderSubAgent] Initialization Error: {e}")

class DrawingReaderSubAgent:
    """
    A LangGraph-ready agent that handles patent drawings extraction.
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
        if ext == '.pdf':
            text, _ = self._process_pdf(path, method, lang)
        elif ext == '.docx':
            text = self._process_docx(path)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            text, _ = self._process_image(path, method, lang)
        else:
            raise ValueError(f"Unsupported format: {ext}. Use PDF, DOCX, or Image.")

        # Save to txt file as requested
        txt_path = "drawings_text.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        return text, txt_path

    def _process_image(self, path, method, lang):
        # For pure images, we use tesseract directly or via MinerU ocr
        return self._process_pdf(path, method="ocr", lang=lang)

    def _process_pdf(self, path, method, lang):
        """
        Uses a combination of PyMuPDF for structural text/element counts 
        and MinerU OCR for layout extraction.
        """
        extracted_parts = []
        stats = {
            'images_found': 0,
            'shapes_found': 0,
            'pages_with_drawings': [],
            'total_pages': 0
        }

        # 1. Structural Analysis using PyMuPDF (Optimized from drawings_extractor.py)
        try:
            import fitz
            doc = fitz.open(str(path))
            stats['total_pages'] = len(doc)
            
            extracted_parts.append("="*80)
            extracted_parts.append("DRAWINGS AND FIGURES STRUCTURAL ANALYSIS")
            extracted_parts.append("="*80)
            
            text_blocks = []
            for page_num, page in enumerate(doc):
                page_text = page.get_text().strip()
                image_list = page.get_images()
                drawings = page.get_drawings()
                
                stats['images_found'] += len(image_list)
                stats['shapes_found'] += len(drawings)
                
                if page_text:
                    text_blocks.append(f"--- Page {page_num + 1} ---\n{page_text}")
                
                if drawings or image_list:
                    stats['pages_with_drawings'].append(page_num + 1)

            extracted_parts.append(f"Total Pages: {stats['total_pages']}")
            extracted_parts.append(f"Images Found: {stats['images_found']}")
            extracted_parts.append(f"Vector Shapes Found: {stats['shapes_found']}")
            extracted_parts.append(f"Pages with Visuals: {stats['pages_with_drawings']}")
            
            if text_blocks:
                extracted_parts.append("\nEXTRACTED PAGE TEXT:")
                extracted_parts.append("\n".join(text_blocks))
            
            doc.close()
        except ImportError:
            extracted_parts.append("[System Info: PyMuPDF not installed, skipping structural pass]")
        except Exception as e:
            extracted_parts.append(f"[System Info: Structural analysis failed: {str(e)}]")

        # 2. Layout OCR using MinerU (Magic-PDF)
        if MINERU_AVAILABLE:
            run_output = tempfile.mkdtemp(prefix="draw_agent_out_", dir=self.output_dir)
            try:
                os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
                parse_doc(
                    path_list=[path],
                    output_dir=run_output,
                    lang=lang,
                    backend="pipeline",
                    method="ocr",
                    formula_enable=False,
                    table_enable=False
                )
                
                extracted_parts.append("\n" + "="*80)
                extracted_parts.append("LAYOUT-AWARE EXTRACTION (MinerU)")
                extracted_parts.append("="*80)
                
                # Get the standard Markdown text
                import re
                md_text = ""
                for root, _, files in os.walk(run_output):
                    for f in files:
                        if f.endswith('.md') and not f.endswith('_layout.md'):
                            with open(os.path.join(root, f), 'r', encoding='utf-8') as fh:
                                content = fh.read()
                                # 1. Remove markdown image links: ![](images/...)
                                content = re.sub(r'!\[.*?\]\(images/.*?\)', '', content)
                                # 2. Remove redundant "Figure X" captions that often follow image tags
                                content = re.sub(r'(?i)^\s*figure\s+\d+[a-z]?\s*$', '', content, flags=re.MULTILINE)
                                md_text += content
                
                if md_text.strip():
                    # Clean up multiple newlines
                    md_text = re.sub(r'\n{3,}', '\n\n', md_text).strip()
                    extracted_parts.append(md_text)

                # 3. Deep OCR on internal images (Tesseract)
                try:
                    import pytesseract
                    from PIL import Image
                    
                    # Better Tesseract detection
                    import shutil
                    tesseract_cmd = shutil.which('tesseract')
                    if tesseract_cmd:
                        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                    elif os.name == 'nt':
                        # Windows fallback
                        for tp in [r"C:\Program Files\Tesseract-OCR\tesseract.exe", 
                                   r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
                            if os.path.exists(tp):
                                pytesseract.pytesseract.tesseract_cmd = tp
                                break
                    
                    # Search for images recursively
                    img_files = []
                    for root, _, files in os.walk(run_output):
                        for f in files:
                            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                                img_files.append(Path(root) / f)
                    
                    print(f"DEBUG: Found {len(img_files)} images to OCR")
                    
                    if img_files:
                        temp_ocr_results = []
                        # Check if tesseract binary is actually found before looping
                        has_tesseract = False
                        try:
                             # Test running tesseract version to see if engine exists
                             import subprocess
                             cmd = pytesseract.pytesseract.tesseract_cmd or "tesseract"
                             subprocess.run([cmd, "--version"], capture_output=True, check=True)
                             has_tesseract = True
                        except:
                             print("DEBUG: Tesseract engine not found. Skipping image OCR.")
                        
                        if has_tesseract:
                            for img_file in img_files:
                                try:
                                    img = Image.open(img_file)
                                    print(f"DEBUG: OCR on {img_file.name} ({img.size})")
                                    img_text = pytesseract.image_to_string(img, config='--psm 11')
                                    if img_text.strip():
                                        temp_ocr_results.append(f"\n[Text from {img_file.name}]:\n{img_text.strip()}")
                                    else:
                                        print(f"DEBUG: No text found in {img_file.name}")
                                except Exception as e:
                                    print(f"DEBUG: Error on {img_file.name}: {e}")
                            
                            if temp_ocr_results:
                                extracted_parts.append(f"\n\n--- DETAILED FIGURE OCR ({len(img_files)} images analyzed) ---")
                                extracted_parts.extend(temp_ocr_results)
                        
                except ImportError as e:
                    print(f"DEBUG: Tesseract library not available: {e}")
                except Exception as e:
                    print(f"DEBUG: General OCR error: {e}")

            except Exception as e:
                extracted_parts.append(f"\n[MinerU Error]: {str(e)}")
        
        final_text = "\n\n".join(extracted_parts)
        return final_text, None

    def _process_docx(self, path):
        try:
            from docx import Document
            doc = Document(str(path))
            text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
            inline_shapes = len(doc.inline_shapes)
            
            output = [
                "="*80, "DOCX DRAWING EXTRACTION", "="*80,
                f"Inline Shapes: {inline_shapes}",
                "\n".join(text_parts)
            ]
            return "\n".join(output)
        except Exception as e:
            return f"DOCX Error: {str(e)}"
