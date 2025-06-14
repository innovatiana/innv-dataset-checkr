"""
AI-Powered Quality Control Module
Integrates Mistral API for intelligent annotation validation
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from .mistral_client import MistralClient
from .qc_rules import QualityIssue, IssueType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AIValidationResult:
    """Container for AI validation results"""
    annotation_id: str
    is_valid: bool
    confidence: float
    reasoning: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    corrected_annotation: Optional[Dict] = None
    ai_agreement_score: float = 0.0


@dataclass
class AIQCConfig:
    """Configuration for AI-powered QC"""
    confidence_threshold: float = 0.7
    batch_size: int = 10
    max_workers: int = 4
    enable_correction_suggestions: bool = True
    validate_edge_cases: bool = True
    cross_validate_with_rules: bool = True
    cache_results: bool = True


class AIQualityChecker:
    """AI-powered quality checker using Mistral API"""
    
    def __init__(self, 
                 mistral_client: Optional[MistralClient] = None,
                 config: Optional[AIQCConfig] = None):
        """
        Initialize AI quality checker
        
        Args:
            mistral_client: MistralClient instance
            config: AI QC configuration
        """
        self.client = mistral_client or MistralClient()
        self.config = config or AIQCConfig()
        self.validation_results = []
        self.aggregate_stats = {}
        
    def validate_dataset(self, 
                        annotations: List[Dict],
                        media_content: Optional[Dict[str, Any]] = None,
                        context: Optional[Dict] = None,
                        progress_callback: Optional[callable] = None) -> List[AIValidationResult]:
        """
        Validate entire dataset using AI
        
        Args:
            annotations: List of annotations to validate
            media_content: Optional mapping of media IDs to content/descriptions
            context: Additional context (taxonomy, rules, etc.)
            progress_callback: Function to call with progress updates
            
        Returns:
            List of AI validation results
        """
        self.validation_results = []
        total_annotations = len(annotations)
        
        # Prepare validation items
        validation_items = []
        for i, ann in enumerate(annotations):
            # Get media content for this annotation
            media_id = ann.get('media_id', ann.get('image_id', ann.get('id', str(i))))
            content = None
            
            if media_content and media_id in media_content:
                content = media_content[media_id]
            elif 'text' in ann:
                content = ann['text']
            elif 'description' in ann:
                content = ann['description']
            else:
                content = f"Media item {media_id}"
            
            validation_items.append({
                'annotation_id': media_id,
                'content': content,
                'annotation': ann,
                'context': context
            })
        
        # Validate in batches
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for i in range(0, len(validation_items), self.config.batch_size):
                batch = validation_items[i:i + self.config.batch_size]
                future = executor.submit(self._validate_batch, batch, progress_callback)
                futures.append(future)
            
            # Collect results
            for future in futures:
                batch_results = future.result()
                self.validation_results.extend(batch_results)
        
        # Calculate aggregate statistics
        self._calculate_aggregate_stats()
        
        return self.validation_results
    
    def _validate_batch(self, 
                       batch: List[Dict],
                       progress_callback: Optional[callable] = None) -> List[AIValidationResult]:
        """Validate a batch of annotations"""
        results = []
        
        for item in batch:
            try:
                # Call Mistral API
                mistral_result = self.client.validate_annotation(
                    media_content=item['content'],
                    annotation=item['annotation'],
                    context=item['context']
                )
                
                # Convert to AIValidationResult
                result = AIValidationResult(
                    annotation_id=item['annotation_id'],
                    is_valid=mistral_result.get('is_valid', True),
                    confidence=mistral_result.get('confidence', 0.5),
                    reasoning=mistral_result.get('reasoning', ''),
                    issues=mistral_result.get('issues', []),
                    suggestions=mistral_result.get('suggestions', []),
                    corrected_annotation=mistral_result.get('corrected_annotation')
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error validating annotation {item['annotation_id']}: {e}")
                # Create error result
                results.append(AIValidationResult(
                    annotation_id=item['annotation_id'],
                    is_valid=None,
                    confidence=0.0,
                    reasoning=f"Validation error: {str(e)}",
                    issues=["Validation failed"]
                ))
            
            if progress_callback:
                progress_callback(len(results) / len(batch))
        
        return results
    
    def _calculate_aggregate_stats(self):
        """Calculate aggregate statistics from validation results"""
        if not self.validation_results:
            return
        
        valid_count = sum(1 for r in self.validation_results if r.is_valid)
        invalid_count = sum(1 for r in self.validation_results if r.is_valid is False)
        error_count = sum(1 for r in self.validation_results if r.is_valid is None)
        
        confidences = [r.confidence for r in self.validation_results if r.confidence > 0]
        
        self.aggregate_stats = {
            'total_validated': len(self.validation_results),
            'valid_annotations': valid_count,
            'invalid_annotations': invalid_count,
            'validation_errors': error_count,
            'validity_rate': valid_count / len(self.validation_results) if self.validation_results else 0,
            'average_confidence': np.mean(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 1,
            'confidence_std': np.std(confidences) if confidences else 0
        }
    
    def get_low_confidence_annotations(self, 
                                     threshold: Optional[float] = None) -> List[AIValidationResult]:
        """
        Get annotations with low confidence scores
        
        Args:
            threshold: Confidence threshold (uses config default if not provided)
            
        Returns:
            List of low confidence validation results
        """
        threshold = threshold or self.config.confidence_threshold
        return [r for r in self.validation_results if r.confidence < threshold]
    
    def get_invalid_annotations(self) -> List[AIValidationResult]:
        """Get all invalid annotations"""
        return [r for r in self.validation_results if r.is_valid is False]
    
    def get_correction_suggestions(self) -> Dict[str, Dict]:
        """
        Get all correction suggestions grouped by annotation ID
        
        Returns:
            Dictionary mapping annotation IDs to correction suggestions
        """
        suggestions = {}
        
        for result in self.validation_results:
            if result.corrected_annotation or result.suggestions:
                suggestions[result.annotation_id] = {
                    'original_valid': result.is_valid,
                    'confidence': result.confidence,
                    'suggestions': result.suggestions,
                    'corrected_annotation': result.corrected_annotation
                }
        
        return suggestions
    
    def compare_with_human_annotations(self, 
                                     human_annotations: List[Dict]) -> Dict[str, float]:
        """
        Compare AI validation results with human annotations
        
        Args:
            human_annotations: List of human-validated annotations
            
        Returns:
            Dictionary with comparison metrics
        """
        # Create mapping of annotation IDs to human validation
        human_valid = {}
        for ann in human_annotations:
            ann_id = ann.get('annotation_id', ann.get('id'))
            human_valid[ann_id] = ann.get('is_valid', ann.get('valid', True))
        
        # Compare with AI results
        agreements = []
        disagreements = []
        
        for result in self.validation_results:
            if result.annotation_id in human_valid:
                human_says_valid = human_valid[result.annotation_id]
                ai_says_valid = result.is_valid
                
                if ai_says_valid is not None:
                    if human_says_valid == ai_says_valid:
                        agreements.append(result)
                    else:
                        disagreements.append({
                            'annotation_id': result.annotation_id,
                            'human_valid': human_says_valid,
                            'ai_valid': ai_says_valid,
                            'ai_confidence': result.confidence,
                            'ai_reasoning': result.reasoning
                        })
        
        total_compared = len(agreements) + len(disagreements)
        
        return {
            'total_compared': total_compared,
            'agreements': len(agreements),
            'disagreements': len(disagreements),
            'agreement_rate': len(agreements) / total_compared if total_compared > 0 else 0,
            'disagreement_details': disagreements[:10]  # First 10 disagreements
        }
    
    def generate_ai_issues(self) -> List[QualityIssue]:
        """
        Convert AI validation results to QualityIssue objects
        
        Returns:
            List of quality issues identified by AI
        """
        issues = []
        
        for result in self.validation_results:
            if not result.is_valid or result.confidence < self.config.confidence_threshold:
                # Determine severity based on confidence
                if result.confidence < 0.3:
                    severity = "critical"
                elif result.confidence < 0.6:
                    severity = "warning"
                else:
                    severity = "info"
                
                # Create quality issue
                issue = QualityIssue(
                    issue_type=IssueType.LABEL_MISMATCH,
                    severity=severity,
                    description=f"AI validation: {result.reasoning}",
                    affected_items=[result.annotation_id],
                    metadata={
                        'confidence': result.confidence,
                        'ai_issues': result.issues,
                        'suggestions': result.suggestions,
                        'has_correction': result.corrected_annotation is not None
                    }
                )
                issues.append(issue)
        
        return issues
    
    def export_validation_report(self) -> Dict[str, Any]:
        """
        Export comprehensive validation report
        
        Returns:
            Dictionary with full validation report
        """
        report = {
            'summary': self.aggregate_stats,
            'low_confidence': {
                'count': len(self.get_low_confidence_annotations()),
                'annotations': [
                    {
                        'id': r.annotation_id,
                        'confidence': r.confidence,
                        'reasoning': r.reasoning
                    }
                    for r in self.get_low_confidence_annotations()[:20]
                ]
            },
            'invalid_annotations': {
                'count': len(self.get_invalid_annotations()),
                'annotations': [
                    {
                        'id': r.annotation_id,
                        'confidence': r.confidence,
                        'issues': r.issues,
                        'suggestions': r.suggestions
                    }
                    for r in self.get_invalid_annotations()[:20]
                ]
            },
            'correction_suggestions': self.get_correction_suggestions(),
            'issue_distribution': self._get_issue_distribution()
        }
        
        return report
    
    def _get_issue_distribution(self) -> Dict[str, int]:
        """Get distribution of issues found"""
        issue_counts = defaultdict(int)
        
        for result in self.validation_results:
            for issue in result.issues:
                issue_counts[issue] += 1
        
        return dict(issue_counts)


class HybridQualityChecker:
    """
    Combines rule-based and AI-powered quality checking
    """
    
    def __init__(self, 
                 rule_checker,
                 ai_checker: AIQualityChecker):
        """
        Initialize hybrid quality checker
        
        Args:
            rule_checker: Rule-based quality checker instance
            ai_checker: AI-powered quality checker instance
        """
        self.rule_checker = rule_checker
        self.ai_checker = ai_checker
        
    def check_dataset(self, 
                     annotations: List[Dict],
                     media_info: Optional[Dict] = None,
                     annotation_type: str = "bbox",
                     use_ai: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive quality check using both rules and AI
        
        Args:
            annotations: List of annotations
            media_info: Media information
            annotation_type: Type of annotations
            use_ai: Whether to use AI validation
            
        Returns:
            Combined quality check results
        """
        results = {
            'rule_based_issues': [],
            'ai_issues': [],
            'combined_issues': [],
            'statistics': {}
        }
        
        # Run rule-based checks
        rule_issues = self.rule_checker.check_dataset(
            annotations, media_info, annotation_type
        )
        results['rule_based_issues'] = rule_issues
        
        # Run AI-powered checks if enabled
        if use_ai:
            ai_results = self.ai_checker.validate_dataset(annotations)
            ai_issues = self.ai_checker.generate_ai_issues()
            results['ai_issues'] = ai_issues
            results['ai_validation_results'] = ai_results
        
        # Combine and deduplicate issues
        results['combined_issues'] = self._merge_issues(rule_issues, ai_issues)
        
        # Calculate statistics
        results['statistics'] = {
            'total_annotations': len(annotations),
            'rule_based_issues': len(rule_issues),
            'ai_issues': len(ai_issues) if use_ai else 0,
            'combined_issues': len(results['combined_issues']),
            'ai_statistics': self.ai_checker.aggregate_stats if use_ai else {}
        }
        
        return results
    
    def _merge_issues(self, 
                     rule_issues: List[QualityIssue],
                     ai_issues: List[QualityIssue]) -> List[QualityIssue]:
        """Merge and deduplicate issues from both sources"""
        # Simple merge - in production, implement smarter deduplication
        all_issues = rule_issues + ai_issues
        
        # Sort by severity
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        all_issues.sort(key=lambda x: severity_order.get(x.severity, 3))
        
        return all_issues


if __name__ == "__main__":
    # Example usage
    ai_checker = AIQualityChecker()
    
    # Example annotations
    annotations = [
        {
            'annotation_id': '1',
            'text': 'The cat is sleeping on the couch',
            'label': 'dog',  # Wrong label
            'bbox': [100, 100, 200, 200]
        },
        {
            'annotation_id': '2',
            'text': 'A red car parked on the street',
            'label': 'vehicle',
            'bbox': [50, 50, 300, 200]
        }
    ]
    
    # Validate with AI
    results = ai_checker.validate_dataset(
        annotations,
        context={'taxonomy': ['cat', 'dog', 'vehicle', 'person']}
    )
    
    # Print results
    print(f"Validation complete: {ai_checker.aggregate_stats}")
    print(f"Invalid annotations: {len(ai_checker.get_invalid_annotations())}")
    print(f"Low confidence: {len(ai_checker.get_low_confidence_annotations())}")
    
    # Get suggestions
    suggestions = ai_checker.get_correction_suggestions()
    print(f"Correction suggestions: {len(suggestions)}")
