from pathlib import Path
from typing import Optional
import asyncio
import json
import time
import os
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .ocr_engines import OCRManager

app = typer.Typer(help="OCR Models CLI - Run multiple OCR engines on images")
console = Console()

@app.command()
def paddle(
    image_path: str = typer.Argument(..., help="Path to the image file"),
    show_text: bool = typer.Option(False, "--show-text", help="Display extracted text")
):
    """Run PaddleOCR on an image"""
    if not Path(image_path).exists():
        console.print(f"[red]Error: Image file '{image_path}' not found[/red]")
        raise typer.Exit(1)
    
    console.print(Panel(f"[blue]Running PaddleOCR on: {image_path}[/blue]"))
    
    manager = OCRManager()
    result = manager.run_single_engine("paddle", image_path)
    
    if result.get("success"):
        console.print(f"[green]✓ Completed in {result['time']:.2f}s[/green]")
        boxes_count = len(result.get('boxes', []))
        console.print(f"[magenta]Found {boxes_count} bounding boxes[/magenta]")
        
        if show_text:
            console.print(Panel(result["text"], title="Extracted Text"))
        
        # Add file path to result
        result['file_path'] = str(Path(image_path).resolve())
        
        # Create organized folder and save results
        output_folder = manager.create_output_folder_structure()
        manager.save_organized_results([result], image_path, output_folder)
    else:
        console.print(f"[red]✗ Error: {result.get('error', 'Unknown error')}[/red]")
        
        # Add file path to result even on error
        result['file_path'] = str(Path(image_path).resolve())
        
        # Create organized folder and save results even for errors
        output_folder = manager.create_output_folder_structure()
        manager.save_organized_results([result], image_path, output_folder)


@app.command()
def easyocr(
    image_path: str = typer.Argument(..., help="Path to the image file"),
    show_text: bool = typer.Option(False, "--show-text", help="Display extracted text")
):
    """Run EasyOCR on an image"""
    if not Path(image_path).exists():
        console.print(f"[red]Error: Image file '{image_path}' not found[/red]")
        raise typer.Exit(1)
    
    console.print(Panel(f"[blue]Running EasyOCR on: {image_path}[/blue]"))
    
    manager = OCRManager()
    result = manager.run_single_engine("easyocr", image_path)
    
    if result.get("success"):
        console.print(f"[green]✓ Completed in {result['time']:.2f}s[/green]")
        boxes_count = len(result.get('boxes', []))
        console.print(f"[magenta]Found {boxes_count} bounding boxes[/magenta]")
        
        if show_text:
            console.print(Panel(result["text"], title="Extracted Text"))
        
        # Add file path to result
        result['file_path'] = str(Path(image_path).resolve())
        
        # Create organized folder and save results
        output_folder = manager.create_output_folder_structure()
        manager.save_organized_results([result], image_path, output_folder)
    else:
        console.print(f"[red]✗ Error: {result.get('error', 'Unknown error')}[/red]")
        
        # Add file path to result even on error
        result['file_path'] = str(Path(image_path).resolve())
        
        # Create organized folder and save results even for errors
        output_folder = manager.create_output_folder_structure()
        manager.save_organized_results([result], image_path, output_folder)


@app.command()
def tesseract(
    image_path: str = typer.Argument(..., help="Path to the image file"),
    show_text: bool = typer.Option(False, "--show-text", help="Display extracted text")
):
    """Run Tesseract OCR on an image"""
    if not Path(image_path).exists():
        console.print(f"[red]Error: Image file '{image_path}' not found[/red]")
        raise typer.Exit(1)
    
    console.print(Panel(f"[blue]Running Tesseract OCR on: {image_path}[/blue]"))
    
    manager = OCRManager()
    result = manager.run_single_engine("tesseract", image_path)
    
    if result.get("success"):
        console.print(f"[green]✓ Completed in {result['time']:.2f}s[/green]")
        boxes_count = len(result.get('boxes', []))
        console.print(f"[magenta]Found {boxes_count} bounding boxes[/magenta]")
        
        if show_text:
            console.print(Panel(result["text"], title="Extracted Text"))
        
        # Add file path to result
        result['file_path'] = str(Path(image_path).resolve())
        
        # Create organized folder and save results
        output_folder = manager.create_output_folder_structure()
        manager.save_organized_results([result], image_path, output_folder)
    else:
        console.print(f"[red]✗ Error: {result.get('error', 'Unknown error')}[/red]")
        
        # Add file path to result even on error
        result['file_path'] = str(Path(image_path).resolve())
        
        # Create organized folder and save results even for errors
        output_folder = manager.create_output_folder_structure()
        manager.save_organized_results([result], image_path, output_folder)


@app.command()
def all(
    input_path: str = typer.Argument(..., help="Path to image file or directory containing images"),
    async_mode: bool = typer.Option(True, "--async/--sync", help="Run OCR engines concurrently (faster)"),
    show_text: bool = typer.Option(False, "--show-text", help="Display extracted text in results table"),
    image_extensions: str = typer.Option("jpg,jpeg,png,bmp,tiff,tif,webp", help="Comma-separated image extensions for folder processing")
):
    """Run all OCR engines on an image file or all images in a directory"""
    input_path_obj = Path(input_path)
    
    if not input_path_obj.exists():
        console.print(f"[red]Error: Path '{input_path}' not found[/red]")
        raise typer.Exit(1)
    
    manager = OCRManager()
    
    if input_path_obj.is_file():
        # Single file processing
        console.print(Panel(f"[bold blue]Running ALL OCR engines on: {input_path}[/bold blue]"))
        
        # Run engines either async or sync based on user preference
        if async_mode:
            results = asyncio.run(manager.run_all_engines_async(input_path))
        else:
            results = manager.run_all_engines(input_path)
        
        # Add file path to each result
        for result in results:
            result['file_path'] = str(input_path_obj.resolve())
        
        # Display comparison table
        manager.display_results(results, show_text)
        
        # Create organized folder structure
        output_folder = manager.create_output_folder_structure()
        
        # Save organized results (NDJSON + individual images + comparison image)
        manager.save_organized_results(results, input_path, output_folder)
        
    elif input_path_obj.is_dir():
        # Directory processing
        console.print(Panel(f"[bold blue]Processing ALL images in directory: {input_path}[/bold blue]"))
        
        # Find all image files in directory and subdirectories using os.walk
        extensions = {f".{ext.strip().lower()}" for ext in image_extensions.split(",")}
        image_files = []
        
        for root, _, files in os.walk(input_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in extensions:
                    image_files.append(file_path)
        
        if not image_files:
            console.print(f"[yellow]No images found in '{input_path}' with extensions: {image_extensions}[/yellow]")
            return
        
        console.print(f"[cyan]Found {len(image_files)} images to process[/cyan]")
        
        # Create organized folder structure for batch processing
        output_folder = manager.create_output_folder_structure()
        all_results = []
        
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            console.print(f"[cyan]Processing {i}/{len(image_files)}: {image_file.name}[/cyan]")
            
            try:
                if async_mode:
                    results = asyncio.run(manager.run_all_engines_async(str(image_file)))
                else:
                    results = manager.run_all_engines(str(image_file))
                
                # Add file path and processing info to each result
                for result in results:
                    result['file_path'] = str(image_file.resolve())
                    result['batch_index'] = i
                    result['total_images'] = len(image_files)
                
                all_results.extend(results)
                
                # Save organized results for each individual file in the batch
                manager.save_organized_results(results, str(image_file), output_folder)
                
            except Exception as e:
                console.print(f"[red]Error processing {image_file.name}: {str(e)}[/red]")
        
        # Display summary table for all results
        console.print(f"\n[bold green]Summary: Processed {len(image_files)} images[/bold green]")
        console.print(f"[green]Total OCR results: {len(all_results)}[/green]")
        
        # Save comprehensive batch results to organized folder
        folder_name = input_path_obj.name
        batch_ndjson_path = Path(output_folder) / f"{folder_name}_batch_results.ndjson"
        
        # Create a comprehensive NDJSON with metadata
        try:
            with open(batch_ndjson_path, 'w') as f:
                # Write metadata as first line
                metadata = {
                    "type": "batch_metadata",
                    "input_directory": str(input_path_obj.resolve()),
                    "total_images": len(image_files),
                    "total_results": len(all_results),
                    "timestamp": time.time(),
                    "image_extensions": image_extensions
                }
                from .ocr_engines import NumpyEncoder
                f.write(json.dumps(metadata, cls=NumpyEncoder) + '\n')
                
                # Write all results
                for result in all_results:
                    # Clean result for JSON serialization
                    boxes = result.get("boxes", [])
                    clean_boxes = []
                    for box in boxes:
                        if hasattr(box, 'tolist'):
                            clean_boxes.append(box.tolist())
                        elif isinstance(box, list):
                            clean_box = []
                            for item in box:
                                if hasattr(item, 'tolist'):
                                    clean_box.append(item.tolist())
                                elif hasattr(item, 'item'):
                                    clean_box.append(item.item())
                                else:
                                    clean_box.append(item)
                            clean_boxes.append(clean_box)
                        else:
                            clean_boxes.append(box)
                    
                    json_result = {
                        "type": "ocr_result",
                        "file_path": result.get("file_path"),
                        "batch_index": result.get("batch_index"),
                        "total_images": result.get("total_images"),
                        "engine": result.get("engine", "unknown"),
                        "success": result.get("success", False),
                        "text": result.get("text", ""),
                        "time": float(result.get("time", 0)),
                        "boxes": clean_boxes,
                        "error": result.get("error", None)
                    }
                    f.write(json.dumps(json_result, cls=NumpyEncoder) + '\n')
            
            console.print(f"[green]✓ Batch results saved to: {batch_ndjson_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error saving batch results: {str(e)}[/red]")
    
    else:
        console.print(f"[red]Error: '{input_path}' is neither a file nor a directory[/red]")
        raise typer.Exit(1)


@app.command()
def batch_async(
    image_dir: str = typer.Argument(..., help="Directory containing images to process"),
    output_file: str = typer.Argument(..., help="Output NDJSON file path"),
    image_extensions: str = typer.Option("jpg,jpeg,png,bmp,tiff,tif,webp", help="Comma-separated image extensions")
):
    """Process multiple images asynchronously and save all results to one NDJSON file"""
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists() or not image_dir_path.is_dir():
        console.print(f"[red]Error: Directory '{image_dir}' not found[/red]")
        raise typer.Exit(1)
    
    # Find all images in directory
    extensions = {f".{ext.strip().lower()}" for ext in image_extensions.split(",")}
    image_files = [
        img for img in image_dir_path.iterdir()
        if img.is_file() and img.suffix.lower() in extensions
    ]
    
    if not image_files:
        console.print(f"[yellow]No images found in '{image_dir}' with extensions: {image_extensions}[/yellow]")
        return
    
    console.print(f"[blue]Found {len(image_files)} images to process[/blue]")
    
    async def process_images():
        manager = OCRManager()
        all_results = []
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            console.print(f"[cyan]Processing {i}/{len(image_files)}: {image_path.name}[/cyan]")
            
            try:
                results = await manager.run_all_engines_async(str(image_path))
                
                # Add image metadata to each result
                for result in results:
                    result['image_file'] = str(image_path)
                    result['batch_index'] = i
                
                all_results.extend(results)
                
            except Exception as e:
                console.print(f"[red]Error processing {image_path.name}: {str(e)}[/red]")
        
        return all_results
    
    # Run async processing
    console.print("[blue]Starting async batch processing...[/blue]")
    all_results = asyncio.run(process_images())
    
    # Save all results to single NDJSON file
    try:
        with open(output_file, 'w') as f:
            # Write metadata
            metadata = {
                "type": "batch_metadata",
                "total_images": len(image_files),
                "total_results": len(all_results),
                "image_directory": str(image_dir_path),
                "timestamp": time.time()
            }
            # Import the encoder class
            from .ocr_engines import NumpyEncoder
            f.write(json.dumps(metadata, cls=NumpyEncoder) + '\n')
            
            # Write all results
            for result in all_results:
                # Convert numpy arrays to Python lists for JSON serialization
                boxes = result.get("boxes", [])
                if boxes:
                    boxes = [box.tolist() if hasattr(box, 'tolist') else box for box in boxes]
                
                # Clean result for JSON serialization
                json_result = {
                    "type": "batch_ocr_result",
                    "image_file": result.get("image_file"),
                    "batch_index": result.get("batch_index"),
                    "engine": result.get("engine", "unknown"),
                    "success": result.get("success", False),
                    "text": result.get("text", ""),
                    "time": float(result.get("time", 0)),
                    "boxes": boxes,
                    "error": result.get("error", None)
                }
                f.write(json.dumps(json_result, cls=NumpyEncoder) + '\n')
        
        console.print(f"[green]✓ Batch processing complete! Results saved to: {output_file}[/green]")
        console.print(f"[green]✓ Processed {len(image_files)} images with {len(all_results)} total OCR results[/green]")
        
    except Exception as e:
        console.print(f"[red]Error saving batch results: {str(e)}[/red]")


def main():
    app()


if __name__ == "__main__":
    main()