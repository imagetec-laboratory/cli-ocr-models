from typing import List, Dict, Any
from pathlib import Path
import time
import json
import asyncio
import concurrent.futures
import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
import uuid
from datetime import datetime
from . import config

console = Console()

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class OCREngine:
    def __init__(self, name: str):
        self.name = name
        self.color = config.COLORS.get(name.lower(), (255, 0, 0))

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        raise NotImplementedError
    
    async def extract_text_async(self, image_path: str) -> Dict[str, Any]:
        """Async wrapper for extract_text using thread pool"""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.extract_text, image_path)


class PaddleOCREngine(OCREngine):
    def __init__(self):
        super().__init__("PaddleOCR")
        self._ocr = None
        self.color = config.COLORS.get("paddle", (255, 0, 0))

    def _load_model(self):
        if self._ocr is None:
            try:
                import paddleocr
                console.print(f"[yellow]PaddleOCR version: {paddleocr.__version__ if hasattr(paddleocr, '__version__') else 'unknown'}[/yellow]")
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(lang=config.PADDLE_LANG, use_gpu=config.USE_GPU, use_mp=config.PADDLE_USE_MP, show_log=False)
            except ImportError as e:
                console.print(f"[red]PaddleOCR import error: {str(e)}[/red]")
                console.print("[red]Install with: uv add paddleocr paddlepaddle[/red]")
                return False
            except Exception as e:
                console.print(f"[red]PaddleOCR initialization error: {str(e)}[/red]")
                return False
        return True

    def _preprocess_image(self, image_path: str) -> str:
        """Preprocess image for better OCR compatibility"""
        try:
            import cv2
            import numpy as np
            
            # Read image with different modes for compatibility
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                # Try different reading modes
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if img is None:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        console.print(f"[yellow]Could not load image with OpenCV, trying with PIL...[/yellow]")
                        # Try with PIL as fallback
                        from PIL import Image as PILImage
                        pil_img = PILImage.open(image_path)
                        img = np.array(pil_img)
                        if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
                            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                        elif len(img.shape) == 3:  # RGB
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            if img is None:
                return image_path  # Return original path if can't load
            
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Ensure we have a valid grayscale image
            if gray is None or gray.size == 0:
                return image_path
            
            # Normalize intensity for better OCR
            if gray.dtype != np.uint8:
                # Normalize to 0-255 range
                gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                gray = gray_norm.astype(np.uint8)
            
            # Apply adaptive histogram equalization for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Save preprocessed image temporarily
            import os
            temp_path = os.path.splitext(image_path)[0] + '_paddle_processed.jpg'
            cv2.imwrite(temp_path, enhanced)
            
            # Verify the temporary image was created successfully
            if not os.path.exists(temp_path):
                return image_path
                
            return temp_path
            
        except Exception as e:
            console.print(f"[yellow]Image preprocessing failed: {str(e)}, using original[/yellow]")
            return image_path

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        if not self._load_model():
            return {"error": "PaddleOCR not available", "text": "", "time": 0, "success": False, "boxes": []}
        
        start_time = time.time()
        processed_image_path = image_path  # Default to original
        
        try:
            # Try preprocessing first, fallback to original on failure
            try:
                processed_image_path = self._preprocess_image(image_path)
            except Exception as preprocess_error:
                console.print(f"[yellow]Image preprocessing failed: {str(preprocess_error)}, using original[/yellow]")
                processed_image_path = image_path
            
            result = self._ocr.ocr(img=processed_image_path, cls=False)
                
            text = ""
            boxes = []
            
            if result and len(result) > 0:
                for line in result:
                    if line:
                        for word_info in line:
                            if word_info and len(word_info) >= 2:
                                box, text_conf = word_info[0], word_info[1]
                                if text_conf and len(text_conf) >= 2:
                                    word_text, confidence = text_conf[0], text_conf[1]
                                    if confidence >= config.MIN_TEXT_CONFIDENCE and confidence >= config.MIN_BOX_CONFIDENCE:
                                        text += word_text + " "
                                        boxes.append(box)
                text = text.strip().replace(" ", "\n")
            
            processing_time = time.time() - start_time
            
            # Clean up temporary processed image if it was created
            if processed_image_path != image_path:
                try:
                    import os
                    if os.path.exists(processed_image_path):
                        os.remove(processed_image_path)
                except:
                    pass  # Ignore cleanup errors
            
            return {
                "engine": self.name,
                "text": text,
                "time": processing_time,
                "success": True,
                "boxes": boxes
            }
        except Exception as e:
            console.print(f"[red]PaddleOCR Error: {str(e)}[/red]")
            # Try to load and preprocess image for better error diagnosis
            try:
                import cv2
                img = cv2.imread(image_path)
                if img is None:
                    console.print(f"[red]Could not load image: {image_path}[/red]")
                else:
                    console.print(f"[yellow]Image shape: {img.shape}, dtype: {img.dtype}[/yellow]")
            except Exception as img_error:
                console.print(f"[red]Image loading error: {str(img_error)}[/red]")
            
            return {
                "engine": self.name,
                "error": str(e),
                "text": "",
                "time": time.time() - start_time,
                "success": False,
                "boxes": []
            }


class EasyOCREngine(OCREngine):
    def __init__(self):
        super().__init__("EasyOCR")
        self._reader = None
        self.color = config.COLORS.get("easyocr", (0, 255, 0))

    def _load_model(self):
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(config.EASYOCR_LANG, gpu=config.USE_GPU)
            except ImportError:
                console.print("[red]EasyOCR not available. Install with: uv add easyocr[/red]")
                return False
        return True

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        if not self._load_model():
            return {"error": "EasyOCR not available", "text": "", "time": 0}
        
        start_time = time.time()
        try:
            results = self._reader.readtext(image_path)
            # Filter results based on confidence thresholds
            filtered_results = [result for result in results if result[2] >= config.MIN_TEXT_CONFIDENCE and result[2] >= config.MIN_BOX_CONFIDENCE]
            text = "\n".join([result[1] for result in filtered_results])
            # Extract bounding boxes from filtered EasyOCR results
            boxes = [result[0] for result in filtered_results]
            processing_time = time.time() - start_time
            
            return {
                "engine": self.name,
                "text": text,
                "time": processing_time,
                "success": True,
                "boxes": boxes
            }
        except Exception as e:
            return {
                "engine": self.name,
                "error": str(e),
                "text": "",
                "time": time.time() - start_time,
                "success": False,
                "boxes": []
            }


class TesseractEngine(OCREngine):
    def __init__(self):
        super().__init__("Tesseract")
        self.color = config.COLORS.get("tesseract", (0, 0, 255))

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            
            # Get bounding box data from Tesseract
            boxes = []
            filtered_text_parts = []
            try:
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                for i in range(len(data['text'])):
                    conf = int(data['conf'][i])
                    text_part = data['text'][i].strip()
                    if conf >= (config.MIN_TEXT_CONFIDENCE * 100) and conf >= (config.MIN_BOX_CONFIDENCE * 100) and text_part:  # Tesseract confidence is 0-100
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        boxes.append([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
                        filtered_text_parts.append(text_part)
                
                # Use filtered text instead of original
                if filtered_text_parts:
                    text = '\n'.join(filtered_text_parts)
            except Exception:
                pass  # If bounding box extraction fails, continue without boxes
            
            processing_time = time.time() - start_time
            
            return {
                "engine": self.name,
                "text": text.strip(),
                "time": processing_time,
                "success": True,
                "boxes": boxes
            }
        except ImportError:
            return {
                "engine": self.name,
                "error": "Pytesseract not available. Install with: uv add pytesseract",
                "text": "",
                "time": time.time() - start_time,
                "success": False,
                "boxes": []
            }
        except Exception as e:
            return {
                "engine": self.name,
                "error": str(e),
                "text": "",
                "time": time.time() - start_time,
                "success": False,
                "boxes": []
            }


class OCRManager:
    def __init__(self):
        self.engines = {
            "paddle": PaddleOCREngine(),
            "easyocr": EasyOCREngine(),
            "tesseract": TesseractEngine()
        }
    
    def _clean_boxes_for_json(self, boxes: List) -> List:
        """Helper method to clean numpy arrays from boxes for JSON serialization"""
        clean_boxes = []
        for box in boxes:
            if hasattr(box, 'tolist'):  # numpy array
                clean_boxes.append(box.tolist())
            elif isinstance(box, list):
                # Handle nested numpy arrays
                clean_box = []
                for item in box:
                    if hasattr(item, 'tolist'):
                        clean_box.append(item.tolist())
                    elif hasattr(item, 'item'):  # numpy scalar
                        clean_box.append(item.item())
                    else:
                        clean_box.append(item)
                clean_boxes.append(clean_box)
            else:
                clean_boxes.append(box)
        return clean_boxes
        
    def create_output_folder_structure(self) -> str:
        """Create organized folder structure: project_root/yyyy/mm/dd/uuid/"""
        now = datetime.now()
        year = now.strftime("%Y")
        month = now.strftime("%m")
        day = now.strftime("%d")
        uuid_str = str(uuid.uuid4())
        
        # Create the folder path relative to current working directory
        folder_path = Path.cwd() / config.OUTPUT_DIR / year / month / day / uuid_str
        folder_path.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[blue]Created output folder: {folder_path}[/blue]")
        return str(folder_path)

    def create_organized_output(self, image_path: str, results: List[Dict[str, Any]], output_folder: str):
        """Create organized output with individual model images and comparison image"""
        try:
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                console.print(f"[red]Could not load image: {image_path}[/red]")
                return False
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_filename = Path(image_path).stem
            
            # Create individual images for each model
            individual_results = []
            for result in results:
                if not result.get("success", False):
                    continue
                    
                engine_name = result["engine"]
                boxes = result.get("boxes", [])
                
                # Map engine names to keys for color
                engine_key_map = {
                    "PaddleOCR": "paddle",
                    "EasyOCR": "easyocr", 
                    "Tesseract": "tesseract"
                }
                engine_key = engine_key_map.get(engine_name, engine_name.lower().replace(" ", ""))
                engine = self.engines.get(engine_key, None)
                color = engine.color if engine else (128, 128, 128)
                
                # Create individual image for this engine
                individual_image = image_rgb.copy()
                
                # Draw bounding boxes for this engine only
                for box in boxes:
                    if len(box) >= 4:
                        points = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(individual_image, [points], isClosed=True, color=color, thickness=config.BOX_LINE_THICKNESS)
                        
                        # Add engine label near the first point
                        if len(points) > 0:
                            label_pos = (int(points[0][0][0]), max(int(points[0][0][1]) - 10, 15))
                            # cv2.putText(individual_image, engine_name, label_pos, 
                            #           cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, color, config.FONT_THICKNESS, cv2.LINE_AA)
                
                # Save individual model image
                individual_path = Path(output_folder) / f"{original_filename}_{engine_name.lower()}.{config.IMAGE_FORMAT}"
                cv2.imwrite(str(individual_path), cv2.cvtColor(individual_image, cv2.COLOR_RGB2BGR))
                individual_results.append(str(individual_path))
                console.print(f"[green]✓ {engine_name} image saved to: {individual_path}[/green]")
            
            # Create comparison image with all engines
            comparison_image = image_rgb.copy()
            for result in results:
                if not result.get("success", False) or not result.get("boxes"):
                    continue
                
                engine_name = result["engine"]
                boxes = result["boxes"]
                engine_key_map = {
                    "PaddleOCR": "paddle",
                    "EasyOCR": "easyocr", 
                    "Tesseract": "tesseract"
                }
                engine_key = engine_key_map.get(engine_name, engine_name.lower().replace(" ", ""))
                engine = self.engines.get(engine_key, None)
                color = engine.color if engine else (128, 128, 128)
                
                # Draw bounding boxes
                for box in boxes:
                    if len(box) >= 4:
                        points = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(comparison_image, [points], isClosed=True, color=color, thickness=config.BOX_LINE_THICKNESS)
                        
                        # Add engine label
                        if len(points) > 0:
                            label_pos = (int(points[0][0][0]), max(int(points[0][0][1]) - 10, 15))
                            # cv2.putText(comparison_image, engine_name, label_pos, 
                            #           cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, color, config.FONT_THICKNESS, cv2.LINE_AA)
            
            # Save comparison image
            comparison_path = Path(output_folder) / f"{original_filename}_comparison.{config.IMAGE_FORMAT}"
            cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison_image, cv2.COLOR_RGB2BGR))
            console.print(f"[green]✓ Comparison image saved to: {comparison_path}[/green]")
            
            return {
                "individual_images": individual_results,
                "comparison_image": str(comparison_path),
                "success": True
            }
            
        except Exception as e:
            console.print(f"[red]Error creating organized output: {str(e)}[/red]")
            return {"success": False, "error": str(e)}

    def run_single_engine(self, engine_name: str, image_path: str) -> Dict[str, Any]:
        if engine_name not in self.engines:
            return {"error": f"Unknown engine: {engine_name}"}
        
        return self.engines[engine_name].extract_text(image_path)

    def run_all_engines(self, image_path: str) -> List[Dict[str, Any]]:
        results = []
        for engine in self.engines.values():
            console.print(f"[blue]Running {engine.name}...[/blue]")
            result = engine.extract_text(image_path)
            results.append(result)
        
        return results

    async def run_all_engines_async(self, image_path: str) -> List[Dict[str, Any]]:
        """Run all OCR engines concurrently using asyncio"""
        console.print("[blue]Running all OCR engines concurrently...[/blue]")
        
        # Create tasks for all engines
        tasks = []
        for engine in self.engines.values():
            task = engine.extract_text_async(image_path)
            tasks.append(task)
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        return list(results)

    def display_results(self, results: List[Dict[str, Any]], show_text: bool = False):
        table = Table(title="OCR Results", title_justify="left")
        table.add_column("Engine", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Time (s)", style="yellow")
        table.add_column("Boxes", style="magenta")
        
        if show_text:
            table.add_column("Text Preview", style="white")

        for result in results:
            status = "✓ Success" if result.get("success", False) else "✗ Error"
            time_str = f"{result.get('time', 0):.2f}s"
            boxes_count = len(result.get('boxes', []))
            boxes_str = f"{boxes_count} boxes" if boxes_count != 1 else "1 box"
            
            row_data = [
                result.get('engine', 'Unknown'),
                status,
                time_str,
                boxes_str
            ]
            
            if show_text:
                text_preview = result.get('text', result.get('error', ''))[:50] + "..." if len(result.get('text', result.get('error', ''))) > 50 else result.get('text', result.get('error', ''))
                row_data.append(text_preview)
            
            table.add_row(*row_data)

        console.print(table)

    def save_organized_results(self, results: List[Dict[str, Any]], image_path: str, output_folder: str) -> dict:
        """Save OCR results in organized format with NDJSON and images"""
        try:
            # Save NDJSON results
            original_filename = Path(image_path).stem
            ndjson_path = Path(output_folder) / f"{original_filename}_{config.NDJSON_PREFIX}.ndjson"
            
            with open(ndjson_path, 'w') as f:
                # Add metadata as first line
                metadata = {
                    "type": "metadata",
                    "image_path": str(Path(image_path).resolve()),
                    "timestamp": time.time(),
                    "output_folder": output_folder,
                    "total_engines": len(results),
                    "successful_engines": len([r for r in results if r.get("success", False)])
                }
                f.write(json.dumps(metadata, cls=NumpyEncoder) + '\n')
                
                # Add each engine result as separate JSON lines
                for result in results:
                    # Create a clean result dict for JSON serialization
                    json_result = {
                        "type": "ocr_result",
                        "file_path": result.get("file_path", str(Path(image_path).resolve())),
                        "engine": result.get("engine", "unknown"),
                        "success": result.get("success", False),
                        "text": result.get("text", ""),
                        "time": float(result.get("time", 0)),
                        "boxes": self._clean_boxes_for_json(result.get("boxes", [])),
                        "error": result.get("error", None)
                    }
                    f.write(json.dumps(json_result, cls=NumpyEncoder) + '\n')
            
            # Create organized images
            image_results = self.create_organized_output(image_path, results, output_folder)
            
            console.print(f"[green]✓ NDJSON results saved to: {ndjson_path}[/green]")
            
            return {
                "ndjson_path": str(ndjson_path),
                "images": image_results,
                "success": True
            }
                
        except Exception as e:
            console.print(f"[red]Error saving organized results: {str(e)}[/red]")
            import traceback
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
            return {"success": False, "error": str(e)}