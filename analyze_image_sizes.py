#!/usr/bin/env python3
"""
Analyze image size frequency distribution from OCR results.
"""
import json
from pathlib import Path
from collections import Counter
from rich.console import Console
from rich.table import Table
import typer

app = typer.Typer()
console = Console()

def analyze_image_sizes(ndjson_file: Path) -> None:
    """Analyze image size frequency distribution from NDJSON file."""
    console.print(f"[bold blue]Analyzing image sizes from: {ndjson_file}[/bold blue]")
    
    # Collect image dimensions
    dimensions = []
    unique_images = set()
    
    with open(ndjson_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get('type') == 'ocr_result' and 'image_width' in data and 'image_height' in data:
                    file_path = data['file_path']
                    # Only count each unique image once (OCR engines create multiple entries per image)
                    if file_path not in unique_images:
                        unique_images.add(file_path)
                        width = data['image_width']
                        height = data['image_height']
                        dimensions.append((width, height))
            except json.JSONDecodeError:
                continue
    
    console.print(f"[green]Found {len(dimensions)} unique images[/green]")
    
    # Create frequency distribution
    size_counter = Counter(dimensions)
    
    # Sort by frequency (descending)
    sorted_sizes = sorted(size_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Create table for display
    table = Table(title="Image Size Frequency Distribution")
    table.add_column("Width", style="cyan", justify="right")
    table.add_column("Height", style="cyan", justify="right")
    table.add_column("Count", style="magenta", justify="right")
    table.add_column("Percentage", style="green", justify="right")
    
    total_images = len(dimensions)
    
    for (width, height), count in sorted_sizes:
        percentage = (count / total_images) * 100
        table.add_row(
            str(width),
            str(height), 
            str(count),
            f"{percentage:.1f}%"
        )
    
    console.print(table)
    
    # Summary statistics
    console.print("\n[bold yellow]Summary Statistics:[/bold yellow]")
    console.print(f"Total unique images: {total_images}")
    console.print(f"Number of unique sizes: {len(size_counter)}")
    
    # Most common size
    most_common_size, most_common_count = sorted_sizes[0]
    console.print(f"Most common size: {most_common_size[0]}x{most_common_size[1]} ({most_common_count} images, {(most_common_count/total_images)*100:.1f}%)")
    
    # Calculate aspect ratios
    aspect_ratios = []
    for width, height in dimensions:
        aspect_ratios.append(width / height)
    
    aspect_ratio_counter = Counter([round(ar, 2) for ar in aspect_ratios])
    console.print(f"\n[bold yellow]Most common aspect ratios:[/bold yellow]")
    for ratio, count in sorted(aspect_ratio_counter.items(), key=lambda x: x[1], reverse=True)[:5]:
        percentage = (count / total_images) * 100
        console.print(f"  {ratio:.2f}: {count} images ({percentage:.1f}%)")

@app.command()
def main(
    results_dir: str = typer.Argument(default="results", help="Results directory to analyze"),
):
    """Analyze image size frequency distribution from OCR results."""
    
    # Find NDJSON files in results directory
    results_path = Path(results_dir)
    if not results_path.exists():
        console.print(f"[red]Results directory not found: {results_dir}[/red]")
        return
    
    ndjson_files = list(results_path.rglob("*.ndjson"))
    
    if not ndjson_files:
        console.print(f"[red]No NDJSON files found in {results_dir}[/red]")
        return
    
    console.print(f"[blue]Found {len(ndjson_files)} NDJSON files:[/blue]")
    for i, file in enumerate(ndjson_files, 1):
        console.print(f"  {i}. {file}")
    
    # Analyze each file
    for ndjson_file in ndjson_files:
        analyze_image_sizes(ndjson_file)
        console.print()

if __name__ == "__main__":
    app()