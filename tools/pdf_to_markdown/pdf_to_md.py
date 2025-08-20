#!/usr/bin/env python3
"""
PDF to Markdown Converter for OmniDocBench

This script processes PDF files from the OmniDocBench dataset and converts them
to markdown format using the Unstructured open source library.

Requirements:
    pip install unstructured[all-docs]
    pip install pillow
    
Usage:
    python tools/pdf_to_markdown/pdf_to_md.py \
        --omnidocbench_json ./demo_data/omnidocbench_demo/OmniDocBench_demo.json \
        --pdf_dir ./demo_data/omnidocbench_demo/pdfs/ \
        --output_dir ./custom_results/unstructured/hi_res \
        --strategy hi_res
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import time

try:
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
    Processor class for converting PDF files to markdown using Unstructured
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
    
    def process_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Process a single PDF file and convert to markdown
        
        Args:
            pdf_path: Path to the PDF file
            
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
    
    def print_statistics(self):
        """Print processing statistics"""
        logger.info(f"\n=== Processing Statistics ===")
        logger.info(f"Successfully processed: {self.processed_count}")
        logger.info(f"Failed: {self.failed_count}")
        
        if self.failed_files:
            logger.info(f"Failed files:")
            for failed_file in self.failed_files:
                logger.info(f"  - {failed_file}")


def process_omnidocbench_dataset(omnidocbench_json: str, pdf_dir: str, output_dir: str, 
                                strategy: str = "auto", languages: List[str] = None):
    """
    Process OmniDocBench dataset using the JSON metadata for PDFs
    
    Args:
        omnidocbench_json: Path to OmniDocBench.json file
        pdf_dir: Directory containing the PDF files
        output_dir: Directory to save markdown files
        strategy: Unstructured processing strategy
        languages: List of languages for OCR
    """
    logger.info("Processing OmniDocBench dataset with PDF files...")
    
    # Load the dataset JSON
    with open(omnidocbench_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    processor = UnstructuredProcessor(output_dir, strategy, languages)
    
    for sample in dataset:
        try:
            # Get PDF filename from the sample - following OmniDocBench format
            # The JSON contains 'page_info' with 'image_path' that we map to PDF
            if 'page_info' in sample and 'image_path' in sample['page_info']:
                image_name = os.path.basename(sample['page_info']['image_path'])
            elif 'image_path' in sample:
                image_name = sample['image_path']
            else:
                logger.warning(f"No image_path found in sample: {sample}")
                continue
            
            # Convert image filename to PDF filename (replace image extension with .pdf)
            pdf_name = os.path.splitext(image_name)[0] + '.pdf'
            pdf_path = os.path.join(pdf_dir, pdf_name)
            
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF not found: {pdf_path}")
                continue
            
            # Generate output filename (same as image name but with .md extension)
            # This ensures output filenames match image names as required by OmniDocBench
            output_filename = os.path.splitext(image_name)[0] + ".md"
            output_path = os.path.join(output_dir, output_filename)
            
            # Skip if already processed
            if os.path.exists(output_path):
                logger.info(f"Skipping {pdf_name} - output already exists")
                continue
            
            # Process the PDF
            markdown_content = processor.process_pdf(pdf_path)
            
            if markdown_content:
                # Save the markdown
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                logger.info(f"Processed: {pdf_name} -> {output_filename}")
            else:
                # Save empty file to track the attempt
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("# No content extracted\n\nUnstructured was unable to extract meaningful content from this PDF.")
                logger.warning(f"No content extracted from {pdf_name}, saved placeholder file")
            
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
    
    processor.print_statistics()


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to markdown using Unstructured")
    
    # Main options matching unstructured_img2md.py pattern exactly
    parser.add_argument("--omnidocbench_json", type=str, required=True,
                        help="Path to OmniDocBench.json file")
    parser.add_argument("--pdf_dir", type=str, required=True,
                        help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save markdown files")
    parser.add_argument("--strategy", type=str, default="auto", 
                        choices=["auto", "fast", "hi_res", "ocr_only"],
                        help="Unstructured processing strategy (default: auto)")
    parser.add_argument("--languages", type=str, nargs='+', 
                        default=['eng', 'chi_sim', 'chi_tra', 'jpn', 'kor'],
                        help="Languages for OCR (default: eng chi_sim chi_tra jpn kor - optimized for OmniDocBench). Common options: eng, chi_sim, chi_tra, jpn, kor, fra, deu, spa, rus")
    
    args = parser.parse_args()
    
    # Create processor and run
    start_time = time.time()
    
    # Process OmniDocBench dataset
    process_omnidocbench_dataset(
        args.omnidocbench_json,
        args.pdf_dir,
        args.output_dir,
        args.strategy,
        args.languages
    )
    
    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main() 