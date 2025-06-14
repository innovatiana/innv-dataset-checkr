"""
Quality Control Rules Module
Implements rule-based quality checks for various annotation types
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
from datetime import datetime
import logging
from enum import Enum
import pandas as pd
from scipy import stats
import Levenshtein

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IssueType(Enum):
    """Types of quality issues"""
    INVALID_BOUNDS = "invalid_bounds"
    TOO_SMALL = "too_small"
    OUT_OF_FRAME = "out_of_frame"
    DUPLICATE = "duplicate"
    OVERLAP = "overlap"
    LABEL_MISMATCH = "label_mismatch"
    MISSING_LABEL = "missing_label"
    INVALID_FORMAT = "invalid_format"
    TEMPORAL_GAP = "temporal_gap"
    TEMPORAL_OVERLAP = "temporal_overlap"
    CLASS_IMBALANCE = "class_imbalance"
    INCONSISTENT_LABEL = "inconsistent_label"
    INVALID_SPAN = "invalid_span"
    ENCODING_ERROR = "encoding_error"
    METADATA_MISMATCH = "metadata_mismatch"
    LOW_CONSENSUS = "low_consensus"


@dataclass
class QualityIssue:
    """Container for quality issues"""
    issue_type: IssueType
    severity: str  # "critical", "warning", "info"
    description: str
    affected_items: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QCConfig:
    """Configuration for quality control thresholds"""
    # Bounding box thresholds
    min_box_area: float = 100  # pixels squared
    min_box_width: float = 5
    min_box_height: float = 5
    max_box_overlap_ratio: float = 0.95
    
    # Temporal thresholds
    max_temporal_gap: float = 1.0  # seconds
    min_temporal_overlap: float = 0.1  # seconds
    
    # Text/NLP thresholds
    min_entity_length: int = 1
    max_entity_length: int = 100
    min_label_consensus: float = 0.7
    
    # Dataset thresholds
    max_class_imbalance_ratio: float = 10.0
    min_samples_per_class: int = 10
    min_annotator_agreement: float = 0.8
    
    # Duplicate detection
    duplicate_iou_threshold: float = 0.95
    text_similarity_threshold: float = 0.95


class QualityChecker:
    """Main class for rule-based quality checking"""
    
    def __init__(self, config: Optional[QCConfig] = None):
        """
        Initialize quality checker
        
        Args:
            config: QC configuration object
        """
        self.config = config or QCConfig()
        self.issues = []
        
    def check_dataset(self, annotations: List[Dict], 
                     media_info: Optional[Dict] = None,
                     annotation_type: str = "bbox") -> List[QualityIssue]:
        """
        Run all quality checks on dataset
        
        Args:
            annotations: List of annotation dictionaries
            media_info: Optional media information (dimensions, duration, etc.)
            annotation_type: Type of annotations ("bbox", "segmentation", "text", "temporal")
            
        Returns:
            List of quality issues found
        """
        self.issues = []
        
        if annotation_type == "bbox":
            self._check_bounding_boxes(annotations, media_info)
        elif annotation_type == "segmentation":
            self._check_segmentation(annotations, media_info)
        elif annotation_type == "temporal":
            self._check_temporal_annotations(annotations, media_info)
        elif annotation_type == "text":
            self._check_text_annotations(annotations)
        
        # Common checks for all types
        self._check_label_consistency(annotations)
        self._check_class_distribution(annotations)
        self._check_annotator_agreement(annotations)
        
        return self.issues
    
    def _check_bounding_boxes(self, annotations: List[Dict], 
                            media_info: Optional[Dict] = None):
        """Check bounding box annotations"""
        # Group by image
        image_annotations = defaultdict(list)
        for ann in annotations:
            image_id = ann.get('image_id', ann.get('filename', 'unknown'))
            image_annotations[image_id].append(ann)
        
        for image_id, image_anns in image_annotations.items():
            # Get image dimensions if available
            img_width = media_info.get(image_id, {}).get('width', None) if media_info else None
            img_height = media_info.get(image_id, {}).get('height', None) if media_info else None
            
            for i, ann in enumerate(image_anns):
                bbox = ann.get('bbox', ann.get('box', []))
                
                if not bbox or len(bbox) < 4:
                    self.issues.append(QualityIssue(
                        issue_type=IssueType.INVALID_FORMAT,
                        severity="critical",
                        description="Missing or invalid bounding box format",
                        affected_items=[ann]
                    ))
                    continue
                
                # Check box validity
                x, y, w, h = bbox[:4]
                
                # Check dimensions
                if w <= 0 or h <= 0:
                    self.issues.append(QualityIssue(
                        issue_type=IssueType.INVALID_BOUNDS,
                        severity="critical",
                        description=f"Invalid box dimensions: width={w}, height={h}",
                        affected_items=[ann]
                    ))
                
                # Check minimum size
                area = w * h
                if area < self.config.min_box_area:
                    self.issues.append(QualityIssue(
                        issue_type=IssueType.TOO_SMALL,
                        severity="warning",
                        description=f"Box area too small: {area} < {self.config.min_box_area}",
                        affected_items=[ann]
                    ))
                
                # Check if out of frame
                if img_width and img_height:
                    if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                        self.issues.append(QualityIssue(
                            issue_type=IssueType.OUT_OF_FRAME,
                            severity="critical",
                            description=f"Box extends outside image bounds",
                            affected_items=[ann],
                            metadata={'box': bbox, 'image_size': (img_width, img_height)}
                        ))
                
                # Check for duplicates and overlaps
                for j, other_ann in enumerate(image_anns[i+1:], i+1):
                    other_bbox = other_ann.get('bbox', other_ann.get('box', []))
                    if len(other_bbox) >= 4:
                        iou = self._calculate_iou(bbox, other_bbox)
                        
                        if iou > self.config.duplicate_iou_threshold:
                            self.issues.append(QualityIssue(
                                issue_type=IssueType.DUPLICATE,
                                severity="warning",
                                description=f"Duplicate boxes detected (IoU={iou:.2f})",
                                affected_items=[ann, other_ann]
                            ))
                        elif iou > self.config.max_box_overlap_ratio:
                            self.issues.append(QualityIssue(
                                issue_type=IssueType.OVERLAP,
                                severity="info",
                                description=f"High overlap between boxes (IoU={iou:.2f})",
                                affected_items=[ann, other_ann]
                            ))
    
    def _check_segmentation(self, annotations: List[Dict], 
                          media_info: Optional[Dict] = None):
        """Check segmentation annotations"""
        for ann in annotations:
            segmentation = ann.get('segmentation', [])
            
            if not segmentation:
                self.issues.append(QualityIssue(
                    issue_type=IssueType.MISSING_LABEL,
                    severity="critical",
                    description="Missing segmentation data",
                    affected_items=[ann]
                ))
                continue
            
            # Check polygon validity
            if isinstance(segmentation, list):
                for poly in segmentation:
                    if isinstance(poly, list) and len(poly) < 6:
                        self.issues.append(QualityIssue(
                            issue_type=IssueType.INVALID_FORMAT,
                            severity="warning",
                            description=f"Polygon has too few points: {len(poly)//2}",
                            affected_items=[ann]
                        ))
    
    def _check_temporal_annotations(self, annotations: List[Dict], 
                                  media_info: Optional[Dict] = None):
        """Check temporal annotations (video/audio)"""
        # Group by media file
        media_annotations = defaultdict(list)
        for ann in annotations:
            media_id = ann.get('media_id', ann.get('filename', 'unknown'))
            media_annotations[media_id].append(ann)
        
        for media_id, media_anns in media_annotations.items():
            # Sort by start time
            sorted_anns = sorted(media_anns, key=lambda x: x.get('start_time', 0))
            
            # Get media duration if available
            duration = media_info.get(media_id, {}).get('duration', None) if media_info else None
            
            for i, ann in enumerate(sorted_anns):
                start_time = ann.get('start_time', ann.get('start', None))
                end_time = ann.get('end_time', ann.get('end', None))
                
                if start_time is None or end_time is None:
                    self.issues.append(QualityIssue(
                        issue_type=IssueType.INVALID_FORMAT,
                        severity="critical",
                        description="Missing start or end time",
                        affected_items=[ann]
                    ))
                    continue
                
                # Check temporal validity
                if start_time >= end_time:
                    self.issues.append(QualityIssue(
                        issue_type=IssueType.INVALID_BOUNDS,
                        severity="critical",
                        description=f"Invalid time range: start={start_time}, end={end_time}",
                        affected_items=[ann]
                    ))
                
                # Check if out of media duration
                if duration and end_time > duration:
                    self.issues.append(QualityIssue(
                        issue_type=IssueType.OUT_OF_FRAME,
                        severity="warning",
                        description=f"Annotation extends beyond media duration",
                        affected_items=[ann],
                        metadata={'end_time': end_time, 'duration': duration}
                    ))
                
                # Check for gaps and overlaps
                if i > 0:
                    prev_ann = sorted_anns[i-1]
                    prev_end = prev_ann.get('end_time', prev_ann.get('end', 0))
                    
                    gap = start_time - prev_end
                    if gap > self.config.max_temporal_gap:
                        self.issues.append(QualityIssue(
                            issue_type=IssueType.TEMPORAL_GAP,
                            severity="info",
                            description=f"Large temporal gap: {gap:.2f} seconds",
                            affected_items=[prev_ann, ann]
                        ))
                    elif gap < -self.config.min_temporal_overlap:
                        self.issues.append(QualityIssue(
                            issue_type=IssueType.TEMPORAL_OVERLAP,
                            severity="warning",
                            description=f"Temporal overlap: {-gap:.2f} seconds",
                            affected_items=[prev_ann, ann]
                        ))
    
    def _check_text_annotations(self, annotations: List[Dict]):
        """Check text/NLP annotations"""
        for ann in annotations:
            text = ann.get('text', '')
            label = ann.get('label', ann.get('entity', ''))
            span = ann.get('span', ann.get('indices', []))
            
            # Check label format
            if not label:
                self.issues.append(QualityIssue(
                    issue_type=IssueType.MISSING_LABEL,
                    severity="critical",
                    description="Missing label for text annotation",
                    affected_items=[ann]
                ))
            
            # Check span validity
            if span and len(span) >= 2:
                start, end = span[0], span[1]
                
                if start < 0 or end <= start:
                    self.issues.append(QualityIssue(
                        issue_type=IssueType.INVALID_SPAN,
                        severity="critical",
                        description=f"Invalid text span: [{start}, {end}]",
                        affected_items=[ann]
                    ))
                
                # Check entity length
                entity_length = end - start
                if entity_length < self.config.min_entity_length:
                    self.issues.append(QualityIssue(
                        issue_type=IssueType.TOO_SMALL,
                        severity="warning",
                        description=f"Entity too short: {entity_length} characters",
                        affected_items=[ann]
                    ))
                elif entity_length > self.config.max_entity_length:
                    self.issues.append(QualityIssue(
                        issue_type=IssueType.INVALID_SPAN,
                        severity="warning",
                        description=f"Entity too long: {entity_length} characters",
                        affected_items=[ann]
                    ))
            
            # Check for encoding issues
            if text and not text.isprintable():
                self.issues.append(QualityIssue(
                    issue_type=IssueType.ENCODING_ERROR,
                    severity="warning",
                    description="Text contains non-printable characters",
                    affected_items=[ann]
                ))
    
    def _check_label_consistency(self, annotations: List[Dict]):
        """Check label consistency across dataset"""
        # Group similar labels
        label_groups = defaultdict(list)
        all_labels = []
        
        for ann in annotations:
            label = ann.get('label', ann.get('class', ann.get('category', '')))
            if label:
                all_labels.append(label)
                label_lower = label.lower().strip()
                label_groups[label_lower].append(label)
        
        # Check for inconsistent casing/formatting
        for label_lower, label_list in label_groups.items():
            unique_labels = list(set(label_list))
            if len(unique_labels) > 1:
                self.issues.append(QualityIssue(
                    issue_type=IssueType.INCONSISTENT_LABEL,
                    severity="warning",
                    description=f"Inconsistent label formatting: {unique_labels}",
                    affected_items=unique_labels,
                    metadata={'count': len(label_list)}
                ))
        
        # Check for very similar labels (possible typos)
        unique_labels = list(set(all_labels))
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                similarity = Levenshtein.ratio(label1.lower(), label2.lower())
                if 0.8 < similarity < 1.0:
                    self.issues.append(QualityIssue(
                        issue_type=IssueType.LABEL_MISMATCH,
                        severity="warning",
                        description=f"Very similar labels found: '{label1}' vs '{label2}'",
                        affected_items=[label1, label2],
                        metadata={'similarity': similarity}
                    ))
    
    def _check_class_distribution(self, annotations: List[Dict]):
        """Check class balance in dataset"""
        class_counts = Counter()
        
        for ann in annotations:
            label = ann.get('label', ann.get('class', ann.get('category', '')))
            if label:
                class_counts[label] += 1
        
        if not class_counts:
            return
        
        # Check for class imbalance
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        
        if min_count > 0:
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > self.config.max_class_imbalance_ratio:
                self.issues.append(QualityIssue(
                    issue_type=IssueType.CLASS_IMBALANCE,
                    severity="warning",
                    description=f"High class imbalance: ratio={imbalance_ratio:.2f}",
                    affected_items=list(class_counts.items()),
                    metadata={'distribution': dict(class_counts)}
                ))
        
        # Check for underrepresented classes
        for label, count in class_counts.items():
            if count < self.config.min_samples_per_class:
                self.issues.append(QualityIssue(
                    issue_type=IssueType.TOO_SMALL,
                    severity="warning",
                    description=f"Class '{label}' has too few samples: {count}",
                    affected_items=[label],
                    metadata={'count': count, 'minimum': self.config.min_samples_per_class}
                ))
    
    def _check_annotator_agreement(self, annotations: List[Dict]):
        """Check agreement between multiple annotators"""
        # Group by item and annotator
        item_annotations = defaultdict(lambda: defaultdict(list))
        
        for ann in annotations:
            item_id = ann.get('item_id', ann.get('image_id', ann.get('id', None)))
            annotator = ann.get('annotator', ann.get('user', 'unknown'))
            
            if item_id:
                item_annotations[item_id][annotator].append(ann)
        
        # Calculate agreement for items with multiple annotators
        low_agreement_items = []
        
        for item_id, annotator_data in item_annotations.items():
            if len(annotator_data) < 2:
                continue
            
            # Simple agreement: check if labels match
            labels_by_annotator = {}
            for annotator, anns in annotator_data.items():
                labels = [ann.get('label', ann.get('class', '')) for ann in anns]
                labels_by_annotator[annotator] = labels
            
            # Calculate pairwise agreement
            annotators = list(labels_by_annotator.keys())
            if len(annotators) >= 2:
                agreements = []
                for i in range(len(annotators)):
                    for j in range(i+1, len(annotators)):
                        labels1 = set(labels_by_annotator[annotators[i]])
                        labels2 = set(labels_by_annotator[annotators[j]])
                        
                        if labels1 and labels2:
                            agreement = len(labels1.intersection(labels2)) / len(labels1.union(labels2))
                            agreements.append(agreement)
                
                if agreements:
                    avg_agreement = np.mean(agreements)
                    if avg_agreement < self.config.min_annotator_agreement:
                        low_agreement_items.append({
                            'item_id': item_id,
                            'agreement': avg_agreement,
                            'annotators': annotators
                        })
        
        if low_agreement_items:
            self.issues.append(QualityIssue(
                issue_type=IssueType.LOW_CONSENSUS,
                severity="warning",
                description=f"Low annotator agreement on {len(low_agreement_items)} items",
                affected_items=low_agreement_items[:10],  # Show first 10
                metadata={'total_affected': len(low_agreement_items)}
            ))
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union for two boxes"""
        x1, y1, w1, h1 = box1[:4]
        x2, y2, w2, h2 = box2[:4]
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of quality issues"""
        issue_counts = Counter()
        severity_counts = Counter()
        
        for issue in self.issues:
            issue_counts[issue.issue_type.value] += 1
            severity_counts[issue.severity] += 1
        
        return {
            'total_issues': len(self.issues),
            'issue_types': dict(issue_counts),
            'severity_distribution': dict(severity_counts),
            'critical_issues': [i for i in self.issues if i.severity == 'critical']
        }
    
    def generate_report(self) -> pd.DataFrame:
        """Generate a detailed report of quality issues"""
        report_data = []
        
        for issue in self.issues:
            report_data.append({
                'Type': issue.issue_type.value,
                'Severity': issue.severity,
                'Description': issue.description,
                'Affected Items': len(issue.affected_items),
                'Details': str(issue.metadata)
            })
        
        return pd.DataFrame(report_data)


class ConsensusCalculator:
    """Calculate consensus metrics between annotators"""
    
    @staticmethod
    def calculate_fleiss_kappa(annotations: List[Dict], 
                             categories: List[str]) -> float:
        """
        Calculate Fleiss' Kappa for inter-annotator agreement
        
        Args:
            annotations: List of annotations with 'item_id', 'annotator', 'label'
            categories: List of possible categories
            
        Returns:
            Fleiss' Kappa score
        """
        # Create rating matrix
        items = defaultdict(lambda: defaultdict(int))
        annotators = set()
        
        for ann in annotations:
            item_id = ann.get('item_id')
            annotator = ann.get('annotator')
            label = ann.get('label')
            
            if item_id and annotator and label:
                items[item_id][label] += 1
                annotators.add(annotator)
        
        if not items or len(annotators) < 2:
            return 0.0
        
        # Build matrix
        n_items = len(items)
        n_annotators = len(annotators)
        n_categories = len(categories)
        
        # Create category counts matrix
        matrix = np.zeros((n_items, n_categories))
        
        item_ids = list(items.keys())
        for i, item_id in enumerate(item_ids):
            for j, category in enumerate(categories):
                matrix[i, j] = items[item_id].get(category, 0)
        
        # Calculate Fleiss' Kappa
        n = n_annotators
        N = n_items
        k = n_categories
        
        # Calculate P_i (proportion of agreement for each item)
        P_i = np.sum(matrix * matrix, axis=1) - n
        P_i = P_i / (n * (n - 1))
        
        # Calculate P_bar (mean of P_i)
        P_bar = np.mean(P_i)
        
        # Calculate P_e (chance agreement)
        p_j = np.sum(matrix, axis=0) / (N * n)
        P_e = np.sum(p_j * p_j)
        
        # Calculate Kappa
        if P_e == 1:
            return 1.0 if P_bar == 1 else 0.0
        
        kappa = (P_bar - P_e) / (1 - P_e)
        return float(kappa)
    
    @staticmethod
    def calculate_pairwise_f1(annotations: List[Dict], 
                            annotator1: str, 
                            annotator2: str) -> float:
        """
        Calculate pairwise F1 score between two annotators
        
        Args:
            annotations: List of annotations
            annotator1: First annotator ID
            annotator2: Second annotator ID
            
        Returns:
            F1 score
        """
        # Group annotations by item
        ann1_items = defaultdict(set)
        ann2_items = defaultdict(set)
        
        for ann in annotations:
            item_id = ann.get('item_id')
            annotator = ann.get('annotator')
            label = ann.get('label')
            
            if item_id and label:
                if annotator == annotator1:
                    ann1_items[item_id].add(label)
                elif annotator == annotator2:
                    ann2_items[item_id].add(label)
        
        # Calculate F1
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        all_items = set(ann1_items.keys()) | set(ann2_items.keys())
        
        for item_id in all_items:
            labels1 = ann1_items.get(item_id, set())
            labels2 = ann2_items.get(item_id, set())
            
            true_positives += len(labels1 & labels2)
            false_positives += len(labels1 - labels2)
            false_negatives += len(labels2 - labels1)
        
        if true_positives == 0:
            return 0.0
        
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return float(f1)


def create_qc_summary_plot(issues: List[QualityIssue]) -> Dict[str, Any]:
    """
    Create summary visualization data for quality issues
    
    Args:
        issues: List of quality issues
        
    Returns:
        Dictionary with plot data
    """
    import matplotlib.pyplot as plt
    
    # Count issues by type and severity
    issue_type_counts = Counter()
    severity_counts = Counter()
    type_severity = defaultdict(Counter)
    
    for issue in issues:
        issue_type_counts[issue.issue_type.value] += 1
        severity_counts[issue.severity] += 1
        type_severity[issue.issue_type.value][issue.severity] += 1
    
    # Prepare data for plotting
    plot_data = {
        'issue_types': dict(issue_type_counts),
        'severities': dict(severity_counts),
        'type_severity_matrix': {
            issue_type: dict(severities)
            for issue_type, severities in type_severity.items()
        }
    }
    
    return plot_data


if __name__ == "__main__":
    # Example usage
    checker = QualityChecker()
    
    # Example bounding box annotations
    bbox_annotations = [
        {'image_id': 'img1', 'bbox': [10, 10, 100, 100], 'label': 'cat'},
        {'image_id': 'img1', 'bbox': [15, 15, 95, 95], 'label': 'cat'},  # Duplicate
        {'image_id': 'img1', 'bbox': [-10, 10, 50, 50], 'label': 'dog'},  # Out of bounds
        {'image_id': 'img2', 'bbox': [0, 0, 2, 2], 'label': 'mouse'},  # Too small
    ]
    
    media_info = {
        'img1': {'width': 640, 'height': 480},
        'img2': {'width': 640, 'height': 480}
    }
    
    issues = checker.check_dataset(bbox_annotations, media_info, 'bbox')
    
    # Print summary
    summary = checker.get_summary_statistics()
    print(f"Total issues found: {summary['total_issues']}")
    print(f"Issue types: {summary['issue_types']}")
    print(f"Severity distribution: {summary['severity_distribution']}")
    
    # Generate report
    report_df = checker.generate_report()
    print("\nDetailed Report:")
    print(report_df)