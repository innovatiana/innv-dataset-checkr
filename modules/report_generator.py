"""
Report Generator Module
Generates comprehensive quality assurance reports in various formats
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import xlsxwriter
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Main class for generating QA reports"""
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceBefore=20,
            spaceAfter=12
        ))
    
    def generate_pdf_report(self, 
                           report_data: Dict[str, Any],
                           filename: Optional[str] = None) -> str:
        """
        Generate PDF report
        
        Args:
            report_data: Dictionary containing all report data
            filename: Optional custom filename
            
        Returns:
            Path to generated PDF file
        """
        if not filename:
            filename = f"qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        filepath = self.output_dir / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build content
        story = []
        
        # Title page
        story.append(Paragraph("Dataset Quality Assurance Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Report metadata
        metadata = [
            f"<b>Dataset:</b> {report_data.get('dataset_info', {}).get('name', 'Unknown')}",
            f"<b>Generated:</b> {report_data.get('generation_date', datetime.now().strftime('%Y-%m-%d %H:%M'))}",
            f"<b>Format:</b> {report_data.get('dataset_info', {}).get('annotation_format', 'Unknown')}"
        ]
        
        for item in metadata:
            story.append(Paragraph(item, self.styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))
        
        story.append(PageBreak())
        
        # Executive Summary
        if 'summary' in report_data:
            story.append(Paragraph("Executive Summary", self.styles['CustomSubtitle']))
            summary_data = report_data['summary']
            
            summary_table_data = [
                ['Metric', 'Value'],
                ['Total Files', summary_data.get('total_files', 0)],
                ['Annotation Format', summary_data.get('annotation_format', 'Unknown')],
                ['Number of Classes', len(summary_data.get('classes', []))],
                ['File Types', ', '.join(f"{k}: {v}" for k, v in summary_data.get('file_types', {}).items())]
            ]
            
            summary_table = Table(summary_table_data, colWidths=[3*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 0.5 * inch))
        
        # Quality Issues Section
        if 'quality_issues' in report_data:
            story.append(Paragraph("Quality Check Results", self.styles['CustomSubtitle']))
            
            issues = report_data['quality_issues']
            if issues:
                # Group by severity
                severity_groups = {}
                for issue in issues:
                    severity = issue.get('severity', 'info')
                    if severity not in severity_groups:
                        severity_groups[severity] = []
                    severity_groups[severity].append(issue)
                
                # Create issues table
                for severity in ['critical', 'warning', 'info']:
                    if severity in severity_groups:
                        story.append(Paragraph(f"{severity.capitalize()} Issues ({len(severity_groups[severity])})", 
                                             self.styles['Heading3']))
                        
                        issue_data = [['Type', 'Description']]
                        for issue in severity_groups[severity][:10]:  # Show first 10
                            issue_data.append([
                                issue.get('type', 'Unknown'),
                                issue.get('description', '')[:100] + '...' if len(issue.get('description', '')) > 100 else issue.get('description', '')
                            ])
                        
                        issue_table = Table(issue_data, colWidths=[2*inch, 4*inch])
                        issue_table.setStyle(self._get_issue_table_style(severity))
                        story.append(issue_table)
                        story.append(Spacer(1, 0.3 * inch))
            else:
                story.append(Paragraph("No quality issues found.", self.styles['Normal']))
        
        # AI Validation Section
        if 'ai_validation' in report_data:
            story.append(PageBreak())
            story.append(Paragraph("AI Validation Results", self.styles['CustomSubtitle']))
            
            ai_data = report_data['ai_validation']
            if 'summary' in ai_data:
                ai_summary = ai_data['summary']
                
                ai_table_data = [
                    ['Metric', 'Value'],
                    ['Total Validated', ai_summary.get('total_validated', 0)],
                    ['Valid Annotations', ai_summary.get('valid_annotations', 0)],
                    ['Invalid Annotations', ai_summary.get('invalid_annotations', 0)],
                    ['Average Confidence', f"{ai_summary.get('average_confidence', 0):.2%}"],
                    ['Low Confidence Count', ai_data.get('low_confidence_count', 0)]
                ]
                
                ai_table = Table(ai_table_data, colWidths=[3*inch, 3*inch])
                ai_table.setStyle(self._get_standard_table_style())
                story.append(ai_table)
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF report generated: {filepath}")
        return str(filepath)
    
    def generate_excel_report(self,
                            report_data: Dict[str, Any],
                            filename: Optional[str] = None) -> str:
        """
        Generate Excel report with multiple sheets
        
        Args:
            report_data: Dictionary containing all report data
            filename: Optional custom filename
            
        Returns:
            Path to generated Excel file
        """
        if not filename:
            filename = f"qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        filepath = self.output_dir / filename
        
        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#1f77b4',
                'font_color': 'white',
                'border': 1
            })
            
            # Summary sheet
            if 'summary' in report_data:
                summary_data = report_data['summary']
                summary_df = pd.DataFrame([
                    {'Metric': 'Total Files', 'Value': summary_data.get('total_files', 0)},
                    {'Metric': 'Annotation Format', 'Value': summary_data.get('annotation_format', 'Unknown')},
                    {'Metric': 'Number of Classes', 'Value': len(summary_data.get('classes', []))},
                ])
                
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format the summary sheet
                worksheet = writer.sheets['Summary']
                for col_num, value in enumerate(summary_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    worksheet.set_column(col_num, col_num, 20)
            
            # Quality Issues sheet
            if 'quality_issues' in report_data:
                issues_data = []
                for issue in report_data['quality_issues']:
                    issues_data.append({
                        'Type': issue.get('type', 'Unknown'),
                        'Severity': issue.get('severity', 'info'),
                        'Description': issue.get('description', ''),
                        'Affected Items': len(issue.get('affected_items', []))
                    })
                
                if issues_data:
                    issues_df = pd.DataFrame(issues_data)
                    issues_df.to_excel(writer, sheet_name='Quality Issues', index=False)
                    
                    # Format issues sheet
                    worksheet = writer.sheets['Quality Issues']
                    for col_num, value in enumerate(issues_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                    
                    # Conditional formatting for severity
                    worksheet.conditional_format('B2:B1000', {
                        'type': 'text',
                        'criteria': 'containing',
                        'value': 'critical',
                        'format': workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
                    })
                    worksheet.conditional_format('B2:B1000', {
                        'type': 'text',
                        'criteria': 'containing',
                        'value': 'warning',
                        'format': workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
                    })
            
            # AI Validation sheet
            if 'ai_validation' in report_data and 'details' in report_data['ai_validation']:
                ai_details = []
                for item in report_data['ai_validation']['details']:
                    ai_details.append({
                        'Annotation ID': item.get('id', ''),
                        'Valid': item.get('is_valid', ''),
                        'Confidence': item.get('confidence', 0),
                        'Issues': ', '.join(item.get('issues', [])),
                        'Suggestions': ', '.join(item.get('suggestions', []))
                    })
                
                if ai_details:
                    ai_df = pd.DataFrame(ai_details)
                    ai_df.to_excel(writer, sheet_name='AI Validation', index=False)
                    
                    # Format AI sheet
                    worksheet = writer.sheets['AI Validation']
                    for col_num, value in enumerate(ai_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
            
            # Class Distribution sheet
            if 'class_distribution' in report_data:
                class_dist = report_data['class_distribution']
                if class_dist:
                    class_df = pd.DataFrame(
                        list(class_dist.items()),
                        columns=['Class', 'Count']
                    )
                    class_df = class_df.sort_values('Count', ascending=False)
                    class_df.to_excel(writer, sheet_name='Class Distribution', index=False)
                    
                    # Add chart
                    worksheet = writer.sheets['Class Distribution']
                    chart = workbook.add_chart({'type': 'column'})
                    chart.add_series({
                        'categories': ['Class Distribution', 1, 0, len(class_df), 0],
                        'values': ['Class Distribution', 1, 1, len(class_df), 1],
                        'name': 'Count'
                    })
                    chart.set_title({'name': 'Class Distribution'})
                    chart.set_x_axis({'name': 'Class'})
                    chart.set_y_axis({'name': 'Count'})
                    worksheet.insert_chart('D2', chart)
        
        logger.info(f"Excel report generated: {filepath}")
        return str(filepath)
    
    def generate_csv_report(self,
                          report_data: Dict[str, Any],
                          filename: Optional[str] = None) -> List[str]:
        """
        Generate CSV reports (multiple files for different sections)
        
        Args:
            report_data: Dictionary containing all report data
            filename: Optional base filename
            
        Returns:
            List of paths to generated CSV files
        """
        if not filename:
            filename = f"qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        generated_files = []
        
        # Quality Issues CSV
        if 'quality_issues' in report_data:
            issues_data = []
            for issue in report_data['quality_issues']:
                issues_data.append({
                    'Type': issue.get('type', 'Unknown'),
                    'Severity': issue.get('severity', 'info'),
                    'Description': issue.get('description', ''),
                    'Affected Items': len(issue.get('affected_items', []))
                })
            
            if issues_data:
                issues_df = pd.DataFrame(issues_data)
                issues_file = self.output_dir / f"{filename}_quality_issues.csv"
                issues_df.to_csv(issues_file, index=False)
                generated_files.append(str(issues_file))
        
        # AI Validation CSV
        if 'ai_validation' in report_data and 'details' in report_data['ai_validation']:
            ai_details = []
            for item in report_data['ai_validation']['details']:
                ai_details.append({
                    'Annotation ID': item.get('id', ''),
                    'Valid': item.get('is_valid', ''),
                    'Confidence': item.get('confidence', 0),
                    'Issues': '|'.join(item.get('issues', [])),
                    'Suggestions': '|'.join(item.get('suggestions', []))
                })
            
            if ai_details:
                ai_df = pd.DataFrame(ai_details)
                ai_file = self.output_dir / f"{filename}_ai_validation.csv"
                ai_df.to_csv(ai_file, index=False)
                generated_files.append(str(ai_file))
        
        # Summary CSV
        if 'summary' in report_data:
            summary_data = report_data['summary']
            summary_rows = []
            for key, value in summary_data.items():
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                summary_rows.append({'Metric': key, 'Value': value})
            
            summary_df = pd.DataFrame(summary_rows)
            summary_file = self.output_dir / f"{filename}_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            generated_files.append(str(summary_file))
        
        logger.info(f"CSV reports generated: {len(generated_files)} files")
        return generated_files
    
    def generate_visualizations(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate visualization plots for the report
        
        Args:
            report_data: Dictionary containing report data
            
        Returns:
            Dictionary mapping visualization names to base64-encoded images
        """
        visualizations = {}
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Class Distribution Plot
        if 'class_distribution' in report_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            class_dist = report_data['class_distribution']
            
            if class_dist:
                classes = list(class_dist.keys())
                counts = list(class_dist.values())
                
                # Sort by count
                sorted_data = sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
                classes, counts = zip(*sorted_data[:20])  # Top 20 classes
                
                bars = ax.bar(range(len(classes)), counts)
                ax.set_xticks(range(len(classes)))
                ax.set_xticklabels(classes, rotation=45, ha='right')
                ax.set_xlabel('Class')
                ax.set_ylabel('Count')
                ax.set_title('Top 20 Classes by Frequency')
                
                # Color bars by count
                colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                plt.tight_layout()
                visualizations['class_distribution'] = self._fig_to_base64(fig)
                plt.close()
        
        # 2. Issue Severity Distribution
        if 'quality_issues' in report_data:
            severity_counts = {'critical': 0, 'warning': 0, 'info': 0}
            for issue in report_data['quality_issues']:
                severity = issue.get('severity', 'info')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if sum(severity_counts.values()) > 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                colors_map = {'critical': '#d62728', 'warning': '#ff7f0e', 'info': '#2ca02c'}
                
                wedges, texts, autotexts = ax.pie(
                    severity_counts.values(),
                    labels=severity_counts.keys(),
                    colors=[colors_map[k] for k in severity_counts.keys()],
                    autopct='%1.1f%%',
                    startangle=90
                )
                
                ax.set_title('Quality Issues by Severity')
                plt.setp(autotexts, size=12, weight="bold")
                
                visualizations['severity_distribution'] = self._fig_to_base64(fig)
                plt.close()
        
        # 3. AI Confidence Distribution
        if 'ai_validation' in report_data and 'confidence_distribution' in report_data['ai_validation']:
            fig, ax = plt.subplots(figsize=(10, 6))
            confidences = report_data['ai_validation']['confidence_distribution']
            
            if confidences:
                ax.hist(confidences, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                ax.axvline(x=0.7, color='red', linestyle='--', label='Threshold (0.7)')
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Count')
                ax.set_title('AI Validation Confidence Distribution')
                ax.legend()
                
                visualizations['confidence_distribution'] = self._fig_to_base64(fig)
                plt.close()
        
        # 4. Annotation Coverage Heatmap
        if 'coverage_data' in report_data:
            fig, ax = plt.subplots(figsize=(12, 8))
            coverage = report_data['coverage_data']
            
            # Create heatmap data
            sns.heatmap(coverage, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
            ax.set_title('Annotation Coverage by Category')
            
            visualizations['coverage_heatmap'] = self._fig_to_base64(fig)
            plt.close()
        
        return visualizations
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return image_base64
    
    def _get_standard_table_style(self) -> TableStyle:
        """Get standard table style"""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
    
    def _get_issue_table_style(self, severity: str) -> TableStyle:
        """Get table style based on issue severity"""
        # Define colors for each severity
        severity_colors = {
            'critical': colors.HexColor('#FFC7CE'),
            'warning': colors.HexColor('#FFEB9C'),
            'info': colors.HexColor('#C7FFCE')
        }
        
        bg_color = severity_colors.get(severity, colors.beige)
        
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), bg_color),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ])
    
    def generate_comprehensive_report(self,
                                    report_data: Dict[str, Any],
                                    formats: List[str] = ['pdf', 'excel']) -> Dict[str, str]:
        """
        Generate reports in multiple formats
        
        Args:
            report_data: Dictionary containing all report data
            formats: List of formats to generate ('pdf', 'excel', 'csv')
            
        Returns:
            Dictionary mapping format to file path
        """
        generated_reports = {}
        
        # Add visualizations to report data
        visualizations = self.generate_visualizations(report_data)
        report_data['visualizations'] = visualizations
        
        # Generate requested formats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"qa_report_{timestamp}"
        
        if 'pdf' in formats:
            pdf_path = self.generate_pdf_report(report_data, f"{base_filename}.pdf")
            generated_reports['pdf'] = pdf_path
        
        if 'excel' in formats:
            excel_path = self.generate_excel_report(report_data, f"{base_filename}.xlsx")
            generated_reports['excel'] = excel_path
        
        if 'csv' in formats:
            csv_paths = self.generate_csv_report(report_data, base_filename)
            generated_reports['csv'] = csv_paths
        
        return generated_reports


def create_sample_report_data() -> Dict[str, Any]:
    """Create sample report data for testing"""
    return {
        'dataset_info': {
            'name': 'Sample Dataset',
            'annotation_format': 'COCO',
            'total_files': 1000
        },
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_files': 1000,
            'annotation_format': 'COCO',
            'classes': ['cat', 'dog', 'bird', 'car', 'person'],
            'file_types': {'images': 800, 'annotations': 200}
        },
        'quality_issues': [
            {
                'type': 'out_of_bounds',
                'severity': 'critical',
                'description': 'Bounding box extends outside image boundaries',
                'affected_items': ['img_001', 'img_002']
            },
            {
                'type': 'duplicate',
                'severity': 'warning',
                'description': 'Duplicate annotations detected',
                'affected_items': ['img_003']
            },
            {
                'type': 'class_imbalance',
                'severity': 'info',
                'description': 'Significant class imbalance detected',
                'affected_items': []
            }
        ],
        'ai_validation': {
            'summary': {
                'total_validated': 100,
                'valid_annotations': 85,
                'invalid_annotations': 15,
                'average_confidence': 0.82
            },
            'low_confidence_count': 12,
            'details': [
                {
                    'id': 'ann_001',
                    'is_valid': False,
                    'confidence': 0.45,
                    'issues': ['Label mismatch'],
                    'suggestions': ['Change label to "cat"']
                }
            ],
            'confidence_distribution': np.random.beta(8, 2, 100).tolist()
        },
        'class_distribution': {
            'cat': 250,
            'dog': 200,
            'bird': 150,
            'car': 300,
            'person': 100
        }
    }


if __name__ == "__main__":
    # Example usage
    generator = ReportGenerator()
    
    # Create sample data
    sample_data = create_sample_report_data()
    
    # Generate reports
    reports = generator.generate_comprehensive_report(
        sample_data,
        formats=['pdf', 'excel', 'csv']
    )
    
    print("Generated reports:")
    for format_type, path in reports.items():
        print(f"- {format_type}: {path}")