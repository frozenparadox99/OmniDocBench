#!/usr/bin/env python3
"""
Marker Image to Markdown Converter for OmniDocBench

This script processes images from the OmniDocBench dataset and converts them
to markdown format using the Marker library - a high-accuracy PDF/image to markdown converter.

Marker Features:
- Supports PDF, image, PPTX, DOCX, XLSX, HTML, EPUB files
- High accuracy for tables, equations, code blocks
- Extracts and saves images
- Works on GPU, CPU, or MPS
- Optional LLM enhancement for highest accuracy

Requirements:
    pip install marker-pdf
    # For full document support:
    pip install marker-pdf[full]
    
Usage:
    python marker_img2md.py --omnidocbench_json ./data.json --images_dir ./images --output_dir ./output
    python marker_img2md.py --omnidocbench_json ./data.json --images_dir ./images --output_dir ./output --use_llm
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
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    from marker.config.parser import ConfigParser
except ImportError as e:
    print(f"Error importing marker: {e}")
    print("Please install marker with: pip install marker-pdf")
    print("For full document support: pip install marker-pdf[full]")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarkerProcessor:
    """
    Processor class for converting images/PDFs to markdown using Marker
    """
    
    def __init__(self, output_dir: str, use_llm: bool = False, force_ocr: bool = False, 
                 output_format: str = "markdown", torch_device: str = None):
        """
        Initialize the Marker processor
        
        Args:
            output_dir: Directory to save markdown files
            use_llm: Whether to use LLM for enhanced accuracy
            force_ocr: Force OCR on all content for better math/table handling
            output_format: Output format ('markdown', 'json', 'html', 'chunks')
            torch_device: Torch device to use ('cuda', 'cpu', 'mps', 'auto')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_llm = use_llm
        self.force_ocr = force_ocr
        self.output_format = output_format
        
        # Set torch device if specified
        if torch_device:
            os.environ['TORCH_DEVICE'] = torch_device
        
        # Configure Marker
        config = {
            "output_format": output_format,
            "force_ocr": force_ocr,
            "extract_images": True,  # Extract images as expected by OmniDocBench
            "paginate_output": False,  # Don't paginate for single images
        }
        
        if use_llm:
            logger.info("🚀 Initializing Marker with LLM enhancement for highest accuracy")
            config["use_llm"] = True
        
        # Initialize Marker components
        logger.info("Loading Marker models...")
        self.config_parser = ConfigParser(config)
        self.artifact_dict = create_model_dict()
        
        # Create converter
        self.converter = PdfConverter(
            config=self.config_parser.generate_config_dict(),
            artifact_dict=self.artifact_dict,
            processor_list=self.config_parser.get_processors(),
            renderer=self.config_parser.get_renderer(),
            llm_service=self.config_parser.get_llm_service() if use_llm else None
        )
        
        logger.info(f"✅ Marker initialized with output format: {output_format}")
        if use_llm:
            logger.info("✅ LLM enhancement enabled")
        if force_ocr:
            logger.info("✅ Force OCR enabled for better math/table handling")
        
        # Statistics
        self.processed_count = 0
        self.failed_count = 0
        self.failed_files = []
        
    def process_image(self, image_path: str) -> Optional[str]:
        """
        Process a single image and convert to markdown using Marker
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Markdown content as string, or None if failed
        """
        try:
            logger.info(f"Processing image with Marker: {image_path}")
            
            # Convert image using Marker
            rendered = self.converter(image_path)
            
            # Extract text content based on output format
            if self.output_format == "markdown":
                # For markdown, get the text content
                text_content, _, images = text_from_rendered(rendered)
                markdown_content = text_content
                
                # Handle extracted images if any
                if images:
                    logger.info(f"Extracted {len(images)} images from {image_path}")
            
            elif self.output_format == "json":
                # For JSON format, convert to markdown-like text for OmniDocBench compatibility
                markdown_content = self._json_to_markdown(rendered)
                
            else:
                # For other formats, get text representation
                text_content, _, images = text_from_rendered(rendered)
                markdown_content = text_content
            
            self.processed_count += 1
            return markdown_content
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            self.failed_count += 1
            self.failed_files.append(image_path)
            return None
    
    def _json_to_markdown(self, rendered) -> str:
        """
        Convert Marker JSON output to markdown format for OmniDocBench compatibility
        
        Args:
            rendered: Marker rendered output in JSON format
            
        Returns:
            Markdown formatted string
        """
        try:
            markdown_parts = []
            
            # Handle the JSON structure from Marker
            if hasattr(rendered, 'children'):
                # Process each page/block
                for page in rendered.children:
                    page_content = self._process_json_block(page)
                    if page_content:
                        markdown_parts.append(page_content)
            
            return "\n\n".join(markdown_parts)
            
        except Exception as e:
            logger.warning(f"Failed to convert JSON to markdown: {e}")
            # Fallback to string representation
            return str(rendered)
    
    def _process_json_block(self, block) -> str:
        """
        Process a single JSON block and convert to markdown
        
        Args:
            block: A single block from Marker JSON output
            
        Returns:
            Markdown formatted string for the block
        """
        try:
            block_type = getattr(block, 'block_type', 'Unknown')
            
            # Handle different block types
            if block_type == "SectionHeader":
                # Extract text content and format as header
                text = self._extract_text_from_block(block)
                return f"## {text}" if text else ""
                
            elif block_type == "Table":
                # For tables, preserve HTML format as expected by OmniDocBench
                if hasattr(block, 'html'):
                    return block.html
                else:
                    text = self._extract_text_from_block(block)
                    return f"<table><tr><td>{text}</td></tr></table>" if text else ""
                    
            elif block_type == "Equation":
                # Format equations with LaTeX delimiters
                text = self._extract_text_from_block(block)
                return f"$${text}$$" if text else ""
                
            elif block_type == "Code":
                # Format code blocks
                text = self._extract_text_from_block(block)
                return f"```\n{text}\n```" if text else ""
                
            elif block_type == "ListItem":
                # Format list items
                text = self._extract_text_from_block(block)
                return f"- {text}" if text else ""
                
            else:
                # Default handling for text blocks
                text = self._extract_text_from_block(block)
                return text if text else ""
                
        except Exception as e:
            logger.warning(f"Failed to process block type {block_type}: {e}")
            return ""
    
    def _extract_text_from_block(self, block) -> str:
        """
        Extract text content from a Marker JSON block
        
        Args:
            block: Marker JSON block
            
        Returns:
            Text content as string
        """
        try:
            # Try different ways to extract text
            if hasattr(block, 'text'):
                return block.text.strip()
            elif hasattr(block, 'html'):
                # Strip HTML tags for text content
                import re
                return re.sub(r'<[^>]+>', '', block.html).strip()
            else:
                return str(block).strip()
        except:
            return ""
    
    def process_batch_images(self, input_dir: str, image_extensions: List[str] = None):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing images
            image_extensions: List of image file extensions to process
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf']
        
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} files to process")
        
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
    
    def print_statistics(self):
        """Print processing statistics"""
        logger.info(f"\n=== Marker Processing Statistics ===")
        logger.info(f"Successfully processed: {self.processed_count}")
        logger.info(f"Failed: {self.failed_count}")
        
        if self.failed_files:
            logger.info(f"Failed files:")
            for failed_file in self.failed_files:
                logger.info(f"  - {failed_file}")


def process_omnidocbench_dataset(omnidocbench_json: str, images_dir: str, output_dir: str, 
                                use_llm: bool = False, force_ocr: bool = False, 
                                output_format: str = "markdown", torch_device: str = None):
    """
    Process OmniDocBench dataset using Marker
    
    Args:
        omnidocbench_json: Path to OmniDocBench.json file
        images_dir: Directory containing the images
        output_dir: Directory to save markdown files
        use_llm: Whether to use LLM for enhanced accuracy
        force_ocr: Force OCR on all content
        output_format: Output format ('markdown', 'json', 'html', 'chunks')
        torch_device: Torch device to use
    """
    logger.info("Processing OmniDocBench dataset with Marker...")
    
    # Load the dataset JSON
    with open(omnidocbench_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    processor = MarkerProcessor(output_dir, use_llm, force_ocr, output_format, torch_device)
    
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
                logger.info(f"Processed with Marker: {image_name} -> {output_filename}")
            else:
                # Save placeholder file to track the attempt
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("# No content extracted\n\nMarker was unable to extract meaningful content from this image.")
                logger.warning(f"No content extracted from {image_name}, saved placeholder file")
            
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
    
    processor.print_statistics()


def main():
    parser = argparse.ArgumentParser(description="Convert images to markdown using Marker")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", type=str, help="Directory containing images to process")
    group.add_argument("--omnidocbench_json", type=str, help="Path to OmniDocBench.json file")
    
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save markdown files")
    parser.add_argument("--images_dir", type=str, help="Directory containing images (required with --omnidocbench_json)")
    
    # Marker-specific options
    parser.add_argument("--use_llm", action="store_true", 
                        help="Use LLM for enhanced accuracy (slower but higher quality)")
    parser.add_argument("--force_ocr", action="store_true",
                        help="Force OCR on all content for better math/table handling")
    parser.add_argument("--output_format", type=str, default="markdown",
                        choices=["markdown", "json", "html", "chunks"],
                        help="Output format (default: markdown)")
    parser.add_argument("--torch_device", type=str, 
                        choices=["cuda", "cpu", "mps", "auto"],
                        help="Torch device to use (auto-detected by default)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.omnidocbench_json and not args.images_dir:
        parser.error("--images_dir is required when using --omnidocbench_json")
    
    start_time = time.time()
    
    if args.omnidocbench_json:
        # Process OmniDocBench dataset
        process_omnidocbench_dataset(
            args.omnidocbench_json,
            args.images_dir,
            args.output_dir,
            args.use_llm,
            args.force_ocr,
            args.output_format,
            args.torch_device
        )
    else:
        # Process directory of images
        processor = MarkerProcessor(
            args.output_dir, 
            args.use_llm, 
            args.force_ocr, 
            args.output_format, 
            args.torch_device
        )
        processor.process_batch_images(args.input_dir)
        processor.print_statistics()
    
    end_time = time.time()
    logger.info(f"Total Marker processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main() 