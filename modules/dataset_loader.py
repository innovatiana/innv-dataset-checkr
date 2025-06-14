"""
Dataset Loader Module
Handles loading and parsing of various dataset formats including COCO, YOLO, and custom formats
"""

import os
import json
import zipfile
import csv
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from PIL import Image
import yaml
import logging
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import mimetypes
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Container for dataset information"""
    name: str
    total_files: int
    file_types: Dict[str, int] = field(default_factory=dict)
    annotation_format: str = "unknown"
    classes: List[str] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    annotators: List[str] = field(default_factory=list)
    media_stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetLoader:
    """Main class for loading and parsing datasets"""
    
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    SUPPORTED_TEXT_FORMATS = {'.txt', '.json', '.jsonl', '.csv'}
    
    def __init__(self, temp_dir: str = "./temp_datasets"):
        """
        Initialize dataset loader
        
        Args:
            temp_dir: Directory for temporary file extraction
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.current_dataset_path = None
        self.dataset_info = None
        
    def load_dataset(self, file_path: Union[str, Path], 
                    extract_dir: Optional[str] = None) -> DatasetInfo:
        """
        Load dataset from file or directory
        
        Args:
            file_path: Path to dataset file (zip) or directory
            extract_dir: Optional directory to extract files to
            
        Returns:
            DatasetInfo object with dataset details
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        # Handle ZIP files
    if file_path.suffix.lower() == '.zip':
        extract_path = Path(extract_dir) if extract_dir else self.temp_dir / file_path.stem
        self._extract_zip(file_path, extract_path)
        self.current_dataset_path = extract_path
        self.dataset_dir = extract_path 
    else:
        self.current_dataset_path = file_path
        self.dataset_dir = file_path
        
        # Initialize dataset info
        self.dataset_info = DatasetInfo(name=file_path.stem, total_files=0)
        
        # Scan dataset structure
        self._scan_dataset_structure()
        
        # Detect and parse annotation format
        self._detect_annotation_format()
        
        # Parse annotations based on detected format
        self._parse_annotations()
        
        # Calculate statistics
        self._calculate_statistics()
        
        return self.dataset_info
    
    def _extract_zip(self, zip_path: Path, extract_path: Path):
        """Extract ZIP file to specified directory"""
        logger.info(f"Extracting {zip_path} to {extract_path}")
        
        # Clean up existing directory if exists
        if extract_path.exists():
            shutil.rmtree(extract_path)
        
        extract_path.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    
    def _scan_dataset_structure(self):
        """Scan dataset directory structure and count file types"""
        file_counts = defaultdict(int)
        total_files = 0
        
        for file_path in self.current_dataset_path.rglob('*'):
            if file_path.is_file():
                total_files += 1
                ext = file_path.suffix.lower()
                
                # Categorize file types
                if ext in self.SUPPORTED_IMAGE_FORMATS:
                    file_counts['images'] += 1
                elif ext in self.SUPPORTED_AUDIO_FORMATS:
                    file_counts['audio'] += 1
                elif ext in self.SUPPORTED_VIDEO_FORMATS:
                    file_counts['video'] += 1
                elif ext in self.SUPPORTED_TEXT_FORMATS:
                    file_counts['text'] += 1
                elif ext in {'.json', '.xml', '.txt'}:
                    file_counts['annotations'] += 1
                else:
                    file_counts['other'] += 1
        
        self.dataset_info.total_files = total_files
        self.dataset_info.file_types = dict(file_counts)
        
        logger.info(f"Found {total_files} files: {dict(file_counts)}")
    
    def _detect_annotation_format(self):
        """Detect annotation format (COCO, YOLO, etc.)"""
        # Check for COCO format
        coco_files = list(self.current_dataset_path.glob('**/annotations*.json'))
        if not coco_files:
            coco_files = list(self.current_dataset_path.glob('**/*coco*.json'))
        
        if coco_files:
            # Verify COCO format
            try:
                with open(coco_files[0], 'r') as f:
                    data = json.load(f)
                    if 'images' in data and 'annotations' in data:
                        self.dataset_info.annotation_format = 'coco'
                        logger.info("Detected COCO format")
                        return
            except:
                pass
        
        # Check for YOLO format
        yolo_files = list(self.current_dataset_path.glob('**/*.txt'))
        if yolo_files:
            # Check if there's a classes file
            classes_files = list(self.current_dataset_path.glob('**/classes.txt')) + \
                           list(self.current_dataset_path.glob('**/obj.names'))
            if classes_files or self._is_yolo_format(yolo_files[0]):
                self.dataset_info.annotation_format = 'yolo'
                logger.info("Detected YOLO format")
                return
        
        # Check for JSONL format
        jsonl_files = list(self.current_dataset_path.glob('**/*.jsonl'))
        if jsonl_files:
            self.dataset_info.annotation_format = 'jsonl'
            logger.info("Detected JSONL format")
            return
        
        # Check for CSV format
        csv_files = list(self.current_dataset_path.glob('**/*.csv'))
        if csv_files:
            self.dataset_info.annotation_format = 'csv'
            logger.info("Detected CSV format")
            return
        
        logger.warning("Could not detect annotation format")
    
    def _is_yolo_format(self, file_path: Path) -> bool:
        """Check if file follows YOLO format"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    # YOLO format: class_id center_x center_y width height
                    parts = lines[0].strip().split()
                    if len(parts) >= 5:
                        # Check if values are numeric and normalized
                        values = [float(p) for p in parts[1:5]]
                        if all(0 <= v <= 1 for v in values):
                            return True
        except:
            pass
        return False
    
    def _parse_annotations(self):
        """Parse annotations based on detected format"""
        if self.dataset_info.annotation_format == 'coco':
            self._parse_coco_annotations()
        elif self.dataset_info.annotation_format == 'yolo':
            self._parse_yolo_annotations()
        elif self.dataset_info.annotation_format == 'jsonl':
            self._parse_jsonl_annotations()
        elif self.dataset_info.annotation_format == 'csv':
            self._parse_csv_annotations()
        else:
            logger.warning("No specific parser for annotation format")
    
    def _parse_coco_annotations(self):
        """Parse COCO format annotations"""
        coco_files = list(self.current_dataset_path.glob('**/annotations*.json'))
        if not coco_files:
            coco_files = list(self.current_dataset_path.glob('**/*coco*.json'))
        
        if not coco_files:
            self.dataset_info.errors.append("No COCO annotation files found")
            return
        
        try:
            with open(coco_files[0], 'r') as f:
                coco_data = json.load(f)
            
            # Extract categories
            if 'categories' in coco_data:
                self.dataset_info.classes = [cat['name'] for cat in coco_data['categories']]
            
            # Count annotations per category
            if 'annotations' in coco_data:
                category_counts = Counter()
                for ann in coco_data['annotations']:
                    category_counts[ann.get('category_id', 'unknown')] += 1
                
                # Map category IDs to names
                if 'categories' in coco_data:
                    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
                    self.dataset_info.class_distribution = {
                        cat_id_to_name.get(cat_id, f'category_{cat_id}'): count
                        for cat_id, count in category_counts.items()
                    }
            
            # Extract image information
            if 'images' in coco_data:
                self.dataset_info.media_stats['total_images'] = len(coco_data['images'])
                
                # Calculate image size statistics
                widths = [img.get('width', 0) for img in coco_data['images']]
                heights = [img.get('height', 0) for img in coco_data['images']]
                
                if widths and heights:
                    self.dataset_info.media_stats['avg_width'] = np.mean(widths)
                    self.dataset_info.media_stats['avg_height'] = np.mean(heights)
                    self.dataset_info.media_stats['min_width'] = min(widths)
                    self.dataset_info.media_stats['max_width'] = max(widths)
                    self.dataset_info.media_stats['min_height'] = min(heights)
                    self.dataset_info.media_stats['max_height'] = max(heights)
            
            logger.info(f"Parsed COCO annotations: {len(self.dataset_info.classes)} classes")
            
        except Exception as e:
            self.dataset_info.errors.append(f"Error parsing COCO annotations: {str(e)}")
            logger.error(f"Error parsing COCO annotations: {e}")
    
    def _parse_yolo_annotations(self):
        """Parse YOLO format annotations"""
        # Find classes file
        classes_files = list(self.current_dataset_path.glob('**/classes.txt')) + \
                       list(self.current_dataset_path.glob('**/obj.names'))
        
        if classes_files:
            try:
                with open(classes_files[0], 'r') as f:
                    self.dataset_info.classes = [line.strip() for line in f.readlines()]
            except Exception as e:
                self.dataset_info.errors.append(f"Error reading classes file: {str(e)}")
        
        # Parse annotation files
        txt_files = list(self.current_dataset_path.glob('**/*.txt'))
        class_counts = Counter()
        
        for txt_file in txt_files:
            # Skip classes file
            if txt_file.name in ['classes.txt', 'obj.names']:
                continue
            
            try:
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
            except:
                continue
        
        # Map class IDs to names
        if self.dataset_info.classes:
            self.dataset_info.class_distribution = {
                self.dataset_info.classes[i]: count
                for i, count in class_counts.items()
                if i < len(self.dataset_info.classes)
            }
        else:
            self.dataset_info.class_distribution = {
                f'class_{i}': count for i, count in class_counts.items()
            }
        
        logger.info(f"Parsed YOLO annotations: {len(class_counts)} classes found")
    
    def _parse_jsonl_annotations(self):
        """Parse JSONL format annotations"""
        jsonl_files = list(self.current_dataset_path.glob('**/*.jsonl'))
        
        if not jsonl_files:
            self.dataset_info.errors.append("No JSONL files found")
            return
        
        all_labels = []
        annotator_set = set()
        
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            
                            # Extract labels
                            if 'label' in data:
                                all_labels.append(data['label'])
                            elif 'labels' in data:
                                all_labels.extend(data['labels'])
                            
                            # Extract annotator info
                            if 'annotator' in data:
                                annotator_set.add(data['annotator'])
                            
            except Exception as e:
                self.dataset_info.errors.append(f"Error parsing {jsonl_file}: {str(e)}")
        
        # Update dataset info
        self.dataset_info.classes = list(set(all_labels))
        self.dataset_info.class_distribution = dict(Counter(all_labels))
        self.dataset_info.annotators = list(annotator_set)
        
        logger.info(f"Parsed JSONL annotations: {len(self.dataset_info.classes)} unique labels")
    
    def _parse_csv_annotations(self):
        """Parse CSV format annotations"""
        csv_files = list(self.current_dataset_path.glob('**/*.csv'))
        
        if not csv_files:
            self.dataset_info.errors.append("No CSV files found")
            return
        
        all_labels = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Look for label columns
                label_columns = [col for col in df.columns if 'label' in col.lower()]
                
                for col in label_columns:
                    all_labels.extend(df[col].dropna().tolist())
                
                # Check for annotator column
                if 'annotator' in df.columns:
                    self.dataset_info.annotators = df['annotator'].dropna().unique().tolist()
                
            except Exception as e:
                self.dataset_info.errors.append(f"Error parsing {csv_file}: {str(e)}")
        
        # Update dataset info
        self.dataset_info.classes = list(set(all_labels))
        self.dataset_info.class_distribution = dict(Counter(all_labels))
        
        logger.info(f"Parsed CSV annotations: {len(self.dataset_info.classes)} unique labels")
    
    def _calculate_statistics(self):
        """Calculate additional dataset statistics"""
        # Calculate class imbalance ratio
        if self.dataset_info.class_distribution:
            counts = list(self.dataset_info.class_distribution.values())
            if counts:
                max_count = max(counts)
                min_count = min(counts)
                if min_count > 0:
                    self.dataset_info.metadata['class_imbalance_ratio'] = max_count / min_count
                else:
                    self.dataset_info.metadata['class_imbalance_ratio'] = float('inf')
        
        # Calculate coverage statistics
        total_annotations = sum(self.dataset_info.class_distribution.values())
        if total_annotations > 0 and self.dataset_info.file_types.get('images', 0) > 0:
            self.dataset_info.metadata['annotations_per_image'] = (
                total_annotations / self.dataset_info.file_types['images']
            )
        
        # Add summary statistics
        self.dataset_info.metadata['total_annotations'] = total_annotations
        self.dataset_info.metadata['unique_classes'] = len(self.dataset_info.classes)
        self.dataset_info.metadata['unique_annotators'] = len(self.dataset_info.annotators)
    
    def get_sample_data(self, n_samples: int = 10) -> List[Dict]:
        """
        Get sample data entries for preview
        
        Args:
            n_samples: Number of samples to retrieve
            
        Returns:
            List of sample data entries
        """
        samples = []
        
        if self.dataset_info.annotation_format == 'coco':
            samples = self._get_coco_samples(n_samples)
        elif self.dataset_info.annotation_format == 'yolo':
            samples = self._get_yolo_samples(n_samples)
        elif self.dataset_info.annotation_format == 'jsonl':
            samples = self._get_jsonl_samples(n_samples)
        
        return samples
    
    def _get_coco_samples(self, n_samples: int) -> List[Dict]:
        """Get sample COCO annotations"""
        samples = []
        coco_files = list(self.current_dataset_path.glob('**/annotations*.json'))
        
        if coco_files:
            try:
                with open(coco_files[0], 'r') as f:
                    coco_data = json.load(f)
                
                # Create image ID to filename mapping
                image_map = {img['id']: img for img in coco_data.get('images', [])}
                
                # Get sample annotations
                annotations = coco_data.get('annotations', [])[:n_samples]
                
                for ann in annotations:
                    image_info = image_map.get(ann['image_id'], {})
                    sample = {
                        'image_id': ann['image_id'],
                        'image_file': image_info.get('file_name', 'unknown'),
                        'category_id': ann['category_id'],
                        'bbox': ann.get('bbox', []),
                        'area': ann.get('area', 0),
                        'segmentation': ann.get('segmentation', [])
                    }
                    samples.append(sample)
            except:
                pass
        
        return samples
    
    def _get_yolo_samples(self, n_samples: int) -> List[Dict]:
        """Get sample YOLO annotations"""
        samples = []
        txt_files = list(self.current_dataset_path.glob('**/*.txt'))
        
        for txt_file in txt_files[:n_samples]:
            if txt_file.name in ['classes.txt', 'obj.names']:
                continue
            
            try:
                # Find corresponding image
                image_path = None
                for ext in self.SUPPORTED_IMAGE_FORMATS:
                    potential_path = txt_file.with_suffix(ext)
                    if potential_path.exists():
                        image_path = potential_path
                        break
                
                with open(txt_file, 'r') as f:
                    annotations = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            annotations.append({
                                'class_id': int(parts[0]),
                                'x_center': float(parts[1]),
                                'y_center': float(parts[2]),
                                'width': float(parts[3]),
                                'height': float(parts[4])
                            })
                
                sample = {
                    'annotation_file': str(txt_file),
                    'image_file': str(image_path) if image_path else 'not_found',
                    'annotations': annotations
                }
                samples.append(sample)
            except:
                continue
        
        return samples
    
    def _get_jsonl_samples(self, n_samples: int) -> List[Dict]:
        """Get sample JSONL entries"""
        samples = []
        jsonl_files = list(self.current_dataset_path.glob('**/*.jsonl'))
        
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= n_samples:
                            break
                        if line.strip():
                            samples.append(json.loads(line))
            except:
                continue
            
            if len(samples) >= n_samples:
                break
        
        return samples
    
    def validate_schema(self, schema: Optional[Dict] = None) -> List[str]:
        """
        Validate dataset against a schema
        
        Args:
            schema: Optional schema definition
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Basic validation
        if self.dataset_info.total_files == 0:
            errors.append("Dataset is empty")
        
        if not self.dataset_info.classes:
            errors.append("No classes/labels found")
        
        if self.dataset_info.annotation_format == "unknown":
            errors.append("Could not detect annotation format")
        
        # Format-specific validation
        if self.dataset_info.annotation_format == 'coco':
            errors.extend(self._validate_coco_schema())
        elif self.dataset_info.annotation_format == 'yolo':
            errors.extend(self._validate_yolo_schema())
        
        # Custom schema validation
        if schema:
            errors.extend(self._validate_custom_schema(schema))
        
        return errors
    
    def _validate_coco_schema(self) -> List[str]:
        """Validate COCO format specific requirements"""
        errors = []
        coco_files = list(self.current_dataset_path.glob('**/annotations*.json'))
        
        if coco_files:
            try:
                with open(coco_files[0], 'r') as f:
                    coco_data = json.load(f)
                
                # Check required fields
                required_fields = ['images', 'annotations', 'categories']
                for field in required_fields:
                    if field not in coco_data:
                        errors.append(f"Missing required COCO field: {field}")
                
                # Validate image-annotation consistency
                if 'images' in coco_data and 'annotations' in coco_data:
                    image_ids = {img['id'] for img in coco_data['images']}
                    orphan_annotations = [
                        ann for ann in coco_data['annotations']
                        if ann['image_id'] not in image_ids
                    ]
                    if orphan_annotations:
                        errors.append(f"Found {len(orphan_annotations)} annotations without corresponding images")
                
            except Exception as e:
                errors.append(f"Error validating COCO schema: {str(e)}")
        
        return errors
    
    def _validate_yolo_schema(self) -> List[str]:
        """Validate YOLO format specific requirements"""
        errors = []
        
        # Check for classes file
        classes_files = list(self.current_dataset_path.glob('**/classes.txt')) + \
                       list(self.current_dataset_path.glob('**/obj.names'))
        
        if not classes_files:
            errors.append("No classes file found (classes.txt or obj.names)")
        
        # Check annotation format
        txt_files = list(self.current_dataset_path.glob('**/*.txt'))
        invalid_annotations = 0
        
        for txt_file in txt_files:
            if txt_file.name in ['classes.txt', 'obj.names']:
                continue
            
            try:
                with open(txt_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            invalid_annotations += 1
                        else:
                            # Validate normalized coordinates
                            values = [float(p) for p in parts[1:5]]
                            if not all(0 <= v <= 1 for v in values):
                                invalid_annotations += 1
            except:
                invalid_annotations += 1
        
        if invalid_annotations > 0:
            errors.append(f"Found {invalid_annotations} invalid YOLO annotations")
        
        return errors
    
    def _validate_custom_schema(self, schema: Dict) -> List[str]:
        """Validate against custom schema"""
        errors = []
        
        # Example schema validation
        if 'required_classes' in schema:
            missing_classes = set(schema['required_classes']) - set(self.dataset_info.classes)
            if missing_classes:
                errors.append(f"Missing required classes: {missing_classes}")
        
        if 'min_samples_per_class' in schema:
            min_samples = schema['min_samples_per_class']
            for class_name, count in self.dataset_info.class_distribution.items():
                if count < min_samples:
                    errors.append(f"Class '{class_name}' has only {count} samples (minimum: {min_samples})")
        
        return errors
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.current_dataset_path and self.current_dataset_path.parent == self.temp_dir:
            try:
                shutil.rmtree(self.current_dataset_path)
                logger.info(f"Cleaned up temporary directory: {self.current_dataset_path}")
            except Exception as e:
                logger.error(f"Error cleaning up: {e}")


def create_dataset_summary_df(dataset_info: DatasetInfo) -> pd.DataFrame:
    """
    Create a summary DataFrame from dataset info
    
    Args:
        dataset_info: DatasetInfo object
        
    Returns:
        Summary DataFrame
    """
    summary_data = {
        'Metric': [
            'Total Files',
            'Annotation Format',
            'Number of Classes',
            'Number of Annotators',
            'Class Imbalance Ratio',
            'Images',
            'Audio Files',
            'Video Files',
            'Text Files',
            'Errors Found'
        ],
        'Value': [
            dataset_info.total_files,
            dataset_info.annotation_format,
            len(dataset_info.classes),
            len(dataset_info.annotators),
            dataset_info.metadata.get('class_imbalance_ratio', 'N/A'),
            dataset_info.file_types.get('images', 0),
            dataset_info.file_types.get('audio', 0),
            dataset_info.file_types.get('video', 0),
            dataset_info.file_types.get('text', 0),
            len(dataset_info.errors)
        ]
    }
    
    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Example usage
    loader = DatasetLoader()
    
    # Load a dataset
    dataset_info = loader.load_dataset("path/to/dataset.zip")
    
    # Print summary
    print(f"Dataset: {dataset_info.name}")
    print(f"Format: {dataset_info.annotation_format}")
    print(f"Classes: {dataset_info.classes}")
    print(f"Files: {dataset_info.file_types}")
    
    # Validate schema
    errors = loader.validate_schema()
    if errors:
        print(f"Validation errors: {errors}")
    
    # Get samples
    samples = loader.get_sample_data(5)
    print(f"Sample data: {samples}")
    
    # Cleanup
    loader.cleanup()
