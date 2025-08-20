#!/usr/bin/env python3
"""
Unstructured Image to Markdown Converter for OmniDocBench

This script processes images from the OmniDocBench dataset and converts them
to markdown format using the Unstructured open source library.

Requirements:
    pip install unstructured[all-docs]
    pip install pillow
    
Usage:
    python unstructured_img2md.py --input_dir /path/to/images --output_dir /path/to/markdowns
    python unstructured_img2md.py --pdf_dir /path/to/pdfs --output_dir /path/to/markdowns
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import time

try:
    from unstructured.partition.image import partition_image
    from unstructured.partition.pdf import partition_pdf
    from unstructured.staging.base import convert_to_dict
    from unstructured.chunking.title import chunk_by_title
except ImportError as e:
    print(f"Error importing unstructured: {e}")
    print("Please install unstructured with: pip install unstructured[all-docs]")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnstructuredProcessor:
    """
    Processor class for converting images/PDFs to markdown using Unstructured
    """
    
    def __init__(self, output_dir: str, strategy: str = "auto", languages: List[str] = None):
        """
        Initialize the processor
        
        Args:
            output_dir: Directory to save markdown files
            strategy: Unstructured strategy ('auto', 'fast', 'hi_res', 'ocr_only')
            languages: List of languages for OCR (e.g., ['eng', 'chi_sim'] for English and Chinese)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.strategy = strategy
        # Optimize language order for OmniDocBench (Chinese first for better recognition)
        self.languages = languages or ['chi_sim', 'eng']  # Chinese first, then English
        
        # Statistics
        self.processed_count = 0
        self.failed_count = 0
        self.failed_files = []
        
    def process_image(self, image_path: str) -> Optional[str]:
        """
        Process a single image and convert to markdown
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Markdown content as string, or None if failed
        """
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Partition the image using Unstructured
            elements = partition_image(
                filename=image_path,
                strategy=self.strategy,
                infer_table_structure=True,
                extract_images_in_pdf=False,
                include_page_breaks=True,
                languages=self.languages
            )
            
            # Convert elements to markdown
            markdown_content = self._elements_to_markdown(elements)
            
            self.processed_count += 1
            return markdown_content
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            self.failed_count += 1
            self.failed_files.append(image_path)
            return None
    
    def process_pdf(self, pdf_path: str, page_number: Optional[int] = None) -> Optional[str]:
        """
        Process a PDF file and convert to markdown
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Specific page to process (0-indexed), None for all pages
            
        Returns:
            Markdown content as string, or None if failed
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Partition the PDF using Unstructured
            elements = partition_pdf(
                filename=pdf_path,
                strategy=self.strategy,
                infer_table_structure=True,
                extract_images_in_pdf=False,
                include_page_breaks=True,
                languages=self.languages
            )
            
            # Filter by page number if specified
            if page_number is not None:
                elements = [elem for elem in elements if getattr(elem.metadata, 'page_number', 1) == page_number + 1]
            
            # Convert elements to markdown
            markdown_content = self._elements_to_markdown(elements)
            
            self.processed_count += 1
            return markdown_content
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            self.failed_count += 1
            self.failed_files.append(pdf_path)
            return None
    
    def _elements_to_markdown(self, elements) -> str:
        """
        Convert Unstructured elements to markdown format
        
        Args:
            elements: List of Unstructured elements
            
        Returns:
            Markdown formatted string
        """
        markdown_parts = []
        
        for element in elements:
            element_type = type(element).__name__
            text = element.text.strip() if element.text else ""
            
            if not text:
                continue
                
            # Handle different element types
            if element_type == "Title":
                # Use ## for titles
                markdown_parts.append(f"## {text}")
                
            elif element_type == "NarrativeText":
                # Regular paragraph text
                markdown_parts.append(text)
                
            elif element_type == "ListItem":
                # List items
                markdown_parts.append(f"- {text}")
                
            elif element_type == "Table":
                # Tables - preserve HTML format as expected by OmniDocBench
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'text_as_html'):
                    # Use HTML table directly as OmniDocBench expects this format
                    html_table = element.metadata.text_as_html
                    markdown_parts.append(html_table)
                else:
                    # Try to convert to HTML table format
                    table_html = self._text_table_to_html(text)
                    markdown_parts.append(table_html)
                    
            elif element_type == "Formula":
                # Mathematical formulas
                markdown_parts.append(f"$${text}$$")
                
            elif element_type == "FigureCaption":
                # Figure captions
                markdown_parts.append(f"*{text}*")
                
            elif element_type == "Header":
                # Headers
                markdown_parts.append(f"# {text}")
                
            elif element_type == "Footer":
                # Footers - add as small text
                markdown_parts.append(f"<small>{text}</small>")
                
            else:
                # Default handling for other element types
                markdown_parts.append(text)
        
        # Join all parts with double newlines for proper markdown spacing
        return "\n\n".join(markdown_parts)
    
    def _text_table_to_html(self, text_table: str) -> str:
        """
        Convert text table to HTML table format
        
        Args:
            text_table: Text representation of table
            
        Returns:
            HTML table string
        """
        try:
            lines = text_table.strip().split('\n')
            if len(lines) < 2:
                return f"<pre>{text_table}</pre>"
            
            html_parts = ["<table>"]
            
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                    
                # Split by common delimiters (tab, multiple spaces, |)
                if '\t' in line:
                    cells = line.split('\t')
                elif '|' in line:
                    cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                else:
                    # Split by multiple spaces
                    cells = [cell.strip() for cell in line.split('  ') if cell.strip()]
                
                if cells:
                    html_parts.append("    <tr>")
                    for cell in cells:
                        html_parts.append(f"        <td>{cell.strip()}</td>")
                    html_parts.append("    </tr>")
            
            html_parts.append("</table>")
            return "\n".join(html_parts)
            
        except Exception as e:
            logger.warning(f"Failed to convert text table to HTML: {e}")
            return f"<pre>{text_table}</pre>"
    
    def process_batch_images(self, input_dir: str, image_extensions: List[str] = None):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing images
            image_extensions: List of image file extensions to process
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files to process")
        
        for image_file in image_files:
            # Generate output filename - keep exact base name, just change extension
            output_filename = image_file.stem + ".md"
            output_path = self.output_dir / output_filename
            
            # Skip if already processed
            if output_path.exists():
                logger.info(f"Skipping {image_file.name} - output already exists")
                continue
            
            # Process the image
            markdown_content = self.process_image(str(image_file))
            
            if markdown_content:
                # Save the markdown
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                logger.info(f"Saved: {output_path}")
            else:
                logger.error(f"Failed to process: {image_file}")
    
    def process_batch_pdfs(self, pdf_dir: str):
        """
        Process all PDFs in a directory
        
        Args:
            pdf_dir: Directory containing PDF files
        """
        pdf_path = Path(pdf_dir)
        
        if not pdf_path.exists():
            logger.error(f"PDF directory does not exist: {pdf_dir}")
            return
        
        # Find all PDF files
        pdf_files = list(pdf_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            # Generate output filename
            output_filename = pdf_file.stem + ".md"
            output_path = self.output_dir / output_filename
            
            # Skip if already processed
            if output_path.exists():
                logger.info(f"Skipping {pdf_file.name} - output already exists")
                continue
            
            # Process the PDF
            markdown_content = self.process_pdf(str(pdf_file))
            
            if markdown_content:
                # Save the markdown
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                logger.info(f"Saved: {output_path}")
            else:
                logger.error(f"Failed to process: {pdf_file}")
    
    def print_statistics(self):
        """Print processing statistics"""
        logger.info(f"\n=== Processing Statistics ===")
        logger.info(f"Successfully processed: {self.processed_count}")
        logger.info(f"Failed: {self.failed_count}")
        
        if self.failed_files:
            logger.info(f"Failed files:")
            for failed_file in self.failed_files:
                logger.info(f"  - {failed_file}")


def process_omnidocbench_dataset(omnidocbench_json: str, images_dir: str, output_dir: str, strategy: str = "auto", languages: List[str] = None):
    """
    Process OmniDocBench dataset using the JSON metadata
    
    Args:
        omnidocbench_json: Path to OmniDocBench.json file
        images_dir: Directory containing the images
        output_dir: Directory to save markdown files
        strategy: Unstructured processing strategy
        languages: List of languages for OCR
    """
    logger.info("Processing OmniDocBench dataset with metadata...")
    
    # Load the dataset JSON
    with open(omnidocbench_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    processor = UnstructuredProcessor(output_dir, strategy, languages)
    
    for sample in dataset:
        try:
            # Get image information
            image_name = os.path.basename(sample['page_info']['image_path'])
            image_path = os.path.join(images_dir, image_name)
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Generate output filename (same as image name but with .md extension)
            output_filename = os.path.splitext(image_name)[0] + ".md"
            output_path = os.path.join(output_dir, output_filename)
            
            # Skip if already processed
            if os.path.exists(output_path):
                logger.info(f"Skipping {image_name} - output already exists")
                continue
            
            # Process the image
            markdown_content = processor.process_image(image_path)
            
            if markdown_content:
                # Save the markdown
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                logger.info(f"Processed: {image_name} -> {output_filename}")
            else:
                # Save empty file to track the attempt
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("# No content extracted\n\nUnstructured was unable to extract meaningful content from this image.")
                logger.warning(f"No content extracted from {image_name}, saved placeholder file")
            
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
    
    processor.print_statistics()


def main():
    parser = argparse.ArgumentParser(description="Convert images/PDFs to markdown using Unstructured")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", type=str, help="Directory containing images to process")
    group.add_argument("--pdf_dir", type=str, help="Directory containing PDFs to process")
    group.add_argument("--omnidocbench_json", type=str, help="Path to OmniDocBench.json file")
    
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save markdown files")
    parser.add_argument("--images_dir", type=str, help="Directory containing images (required with --omnidocbench_json)")
    parser.add_argument("--strategy", type=str, default="auto", 
                        choices=["auto", "fast", "hi_res", "ocr_only"],
                        help="Unstructured processing strategy (default: auto)")
    parser.add_argument("--languages", type=str, nargs='+', 
                        default=['eng', 'chi_sim', 'chi_tra', 'jpn', 'kor'],
                        help="Languages for OCR (default: chi_sim eng - optimized for OmniDocBench). Common options: eng, chi_sim, chi_tra, jpn, kor, fra, deu, spa, rus")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.omnidocbench_json and not args.images_dir:
        parser.error("--images_dir is required when using --omnidocbench_json")
    
    # Create processor
    start_time = time.time()
    
    if args.omnidocbench_json:
        # Process OmniDocBench dataset
        process_omnidocbench_dataset(
            args.omnidocbench_json,
            args.images_dir,
            args.output_dir,
            args.strategy,
            args.languages
        )
    elif args.input_dir:
        # Process directory of images
        processor = UnstructuredProcessor(args.output_dir, args.strategy, args.languages)
        processor.process_batch_images(args.input_dir)
        processor.print_statistics()
    elif args.pdf_dir:
        # Process directory of PDFs
        processor = UnstructuredProcessor(args.output_dir, args.strategy, args.languages)
        processor.process_batch_pdfs(args.pdf_dir)
        processor.print_statistics()
    
    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main() 