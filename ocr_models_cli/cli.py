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

from ocr_models_cli.ocr_engines import OCRManager, NumpyEncoder
from ocr_models_cli import config

app = typer.Typer(
    help="OCR Models CLI - Run multiple OCR engines on images",
    add_completion=True,
)
console = Console()

@app.command()
def all(
    input_path: str = typer.Argument(..., help="Path to image file or directory containing images"),
    async_mode: bool = typer.Option(True, "--async/--sync", help="Run OCR engines concurrently (faster)"),
    show_text: bool = typer.Option(False, "--show-text", help="Display extracted text in results table"),
    image_extensions: str = typer.Option(",".join(config.IMAGE_EXTENSIONS), help="Comma-separated image extensions for folder processing")
):
    """
    Run all OCR engines on an image file or all images in a directory
    """
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

        # Prepare batch NDJSON file for incremental writing
        folder_name = input_path_obj.name
        batch_ndjson_path = Path(output_folder) / f"{folder_name}_batch_results.ndjson"

        # Write metadata as first line
        with open(batch_ndjson_path, 'w') as f:
            metadata = {
                "type": "batch_metadata",
                "input_directory": str(input_path_obj.resolve()),
                "total_images": len(image_files),
                "timestamp": time.time(),
                "image_extensions": image_extensions
            }
            f.write(json.dumps(metadata, cls=NumpyEncoder) + '\n')

        console.print(f"[green]✓ Batch results will be saved incrementally to: {batch_ndjson_path}[/green]")

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
                
                # Get image dimensions
                width, height = manager.get_image_dimensions(str(image_file))
                
                # Append results to batch file immediately after processing each image
                with open(batch_ndjson_path, 'a') as f:
                    for result in results:
                        # Clean result for JSON serialization
                        boxes = result.get("boxes", [])
                        clean_boxes = manager._clean_boxes_for_json(boxes)
                        
                        json_result = {
                            "type": "ocr_result",
                            "file_path": result.get("file_path"),
                            "image_width": width,
                            "image_height": height,
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
                
                console.print(f"[green]✓ Results for {image_file.name} saved to batch file[/green]")

            except Exception as e:
                console.print(f"[red]Error processing {image_file.name}: {str(e)}[/red]")

        # Display summary table for all results
        console.print(f"\n[bold green]Summary: Processed {len(image_files)} images[/bold green]")
        console.print(f"[green]Total OCR results: {len(all_results)}[/green]")
        console.print(f"[green]✓ All results saved incrementally to: {batch_ndjson_path}[/green]")

    else:
        console.print(f"[red]Error: '{input_path}' is neither a file nor a directory[/red]")
        raise typer.Exit(1)


@app.command()
def preload(
    engines: str = typer.Option("all", "--engines", help="Comma-separated list of engines to preload (paddle,easyocr,tesseract) or 'all'")
):
    """Preload OCR model(s) to reduce initialization time on first use"""
    manager = OCRManager()
    
    if engines.lower() == "all":
        engine_list = list(manager.engines.keys())
    else:
        engine_list = [e.strip().lower() for e in engines.split(",")]
    
    console.print(Panel(f"[blue]Preloading OCR engines: {', '.join(engine_list)}[/blue]"))
    
    success_count = 0
    total_start_time = time.time()
    
    for engine_name in engine_list:
        if engine_name not in manager.engines:
            console.print(f"[red]✗ Unknown engine: {engine_name}[/red]")
            continue
            
        console.print(f"[cyan]Preloading {engine_name}...[/cyan]")
        start_time = time.time()
        
        if manager.preload_engine(engine_name):
            load_time = time.time() - start_time
            console.print(f"[green]✓ {engine_name} loaded in {load_time:.2f}s[/green]")
            success_count += 1
        else:
            load_time = time.time() - start_time
            console.print(f"[red]✗ Failed to load {engine_name} ({load_time:.2f}s)[/red]")
    
    total_time = time.time() - total_start_time
    console.print(f"[green]✓ Preloaded {success_count}/{len(engine_list)} engines in {total_time:.2f}s[/green]")

def main():
    app()


if __name__ == "__main__":
    main()