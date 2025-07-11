"""
AI-Powered Dataset QA Platform
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import yaml
import tempfile
import os
from datetime import datetime
import zipfile
from typing import Dict, List, Optional, Any

# Import custom modules
from modules.dataset_loader import DatasetLoader, create_dataset_summary_df
from modules.qc_rules import QualityChecker, ConsensusCalculator, QCConfig
from modules.mistral_client import MistralClient
from modules.qc_ai import AIQualityChecker, HybridQualityChecker, AIQCConfig
from modules.manual_review import ManualReviewInterface
from modules.report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="AI Dataset QA Platform",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .severity-critical { color: #d62728; font-weight: bold; }
    .severity-warning { color: #ff7f0e; font-weight: bold; }
    .severity-info { color: #2ca02c; }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
    st.session_state.dataset_info = None
    st.session_state.dataset_loader = None
    st.session_state.quality_issues = []
    st.session_state.ai_results = []
    st.session_state.annotations = []
    st.session_state.current_page = 'upload'
    st.session_state.language = 'en'

# Language translations
translations = {
    'en': {
        'title': 'AI-Powered Dataset QA Platform',
        'upload': 'Dataset Upload',
        'overview': 'Dataset Overview',
        'quality_checks': 'Quality Checks',
        'ai_validation': 'AI Validation',
        'manual_review': 'Manual Review',
        'reports': 'Reports',
        'settings': 'Settings',
        'upload_instructions': 'Upload your dataset (ZIP file or select folder)',
        'processing': 'Processing...',
        'success': 'Success!',
        'error': 'Error',
        'generate_report': 'Generate Report'
    },
    'fr': {
        'title': 'Plateforme QA de Dataset avec IA',
        'upload': 'Téléchargement du Dataset',
        'overview': 'Aperçu du Dataset',
        'quality_checks': 'Contrôles Qualité',
        'ai_validation': 'Validation IA',
        'manual_review': 'Révision Manuelle',
        'reports': 'Rapports',
        'settings': 'Paramètres',
        'upload_instructions': 'Téléchargez votre dataset (fichier ZIP ou dossier)',
        'processing': 'Traitement...',
        'success': 'Succès!',
        'error': 'Erreur',
        'generate_report': 'Générer le Rapport'
    }
}

def t(key: str) -> str:
    """Get translation for current language"""
    return translations[st.session_state.language].get(key, key)


def load_config():
    """Load configuration from file"""
    config_path = Path("config/thresholds.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    temp_dir = tempfile.mkdtemp()
    file_path = Path(temp_dir) / uploaded_file.name
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def initialize_components():
    """Initialize all components"""
    if 'quality_checker' not in st.session_state:
        config = load_config()
        qc_config = QCConfig(**config.get('qc_rules', {}))
        st.session_state.quality_checker = QualityChecker(qc_config)
    
    if 'mistral_client' not in st.session_state:
        try:
            st.session_state.mistral_client = MistralClient()
            st.session_state.ai_checker = AIQualityChecker(st.session_state.mistral_client)
        except Exception as e:
            st.warning(f"Could not initialize Mistral client: {e}")
            st.session_state.mistral_client = None
            st.session_state.ai_checker = None


def main():
    """Main application logic"""
    initialize_components()
    
    # Header
    st.markdown(f'<h1 class="main-header">{t("title")}</h1>', unsafe_allow_html=True)
    
    # Language selector
    col1, col2, col3 = st.columns([1, 6, 1])
    with col3:
        st.session_state.language = st.selectbox(
            "Language",
            options=['en', 'fr'],
            index=0 if st.session_state.language == 'en' else 1,
            label_visibility="collapsed"
        )
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        
        pages = {
            'upload': '📤 ' + t('upload'),
            'overview': '📊 ' + t('overview'),
            'quality_checks': '✅ ' + t('quality_checks'),
            'ai_validation': '🤖 ' + t('ai_validation'),
            'manual_review': '👁️ ' + t('manual_review'),
            'reports': '📝 ' + t('reports'),
            'settings': '⚙️ ' + t('settings')
        }
        
        for page_id, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_id}"):
                st.session_state.current_page = page_id
        
        # Dataset status
        if st.session_state.dataset_loaded:
            st.divider()
            st.success("Dataset loaded!")
            if st.session_state.dataset_info:
                st.write(f"**Name:** {st.session_state.dataset_info.name}")
                st.write(f"**Files:** {st.session_state.dataset_info.total_files}")
                st.write(f"**Format:** {st.session_state.dataset_info.annotation_format}")
    
    # Main content area
    if st.session_state.current_page == 'upload':
        show_upload_page()
    elif st.session_state.current_page == 'overview':
        show_overview_page()
    elif st.session_state.current_page == 'quality_checks':
        show_quality_checks_page()
    elif st.session_state.current_page == 'ai_validation':
        show_ai_validation_page()
    elif st.session_state.current_page == 'manual_review':
        show_manual_review_page()
    elif st.session_state.current_page == 'reports':
        show_reports_page()
    elif st.session_state.current_page == 'settings':
        show_settings_page()


def show_upload_page():
    """Dataset upload page"""
    st.header(t('upload'))
    
    st.info(t('upload_instructions'))
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['zip'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Save file
        file_path = save_uploaded_file(uploaded_file)
        
        # Load dataset
        with st.spinner(t('processing')):
            try:
                loader = DatasetLoader()
                dataset_info = loader.load_dataset(file_path)
                
                # Store in session state
                st.session_state.dataset_loaded = True
                st.session_state.dataset_info = dataset_info
                st.session_state.dataset_loader = loader
                
                st.success(t('success'))
                
                # Show summary
                st.subheader("Dataset Summary")
                summary_df = create_dataset_summary_df(dataset_info)
                st.dataframe(summary_df, use_container_width=True)
                
                # Show errors if any
                if dataset_info.errors:
                    st.warning(f"Found {len(dataset_info.errors)} errors during loading:")
                    for error in dataset_info.errors:
                        st.error(error)
                
                # Auto-navigate to overview
                st.session_state.current_page = 'overview'
                st.rerun()
                
            except Exception as e:
                st.error(f"{t('error')}: {str(e)}")


def show_overview_page():
    """Dataset overview page"""
    if not st.session_state.dataset_loaded:
        st.warning("Please upload a dataset first")
        return
    
    st.header(t('overview'))
    
    dataset_info = st.session_state.dataset_info
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Files", dataset_info.total_files)
    with col2:
        st.metric("Classes", len(dataset_info.classes))
    with col3:
        st.metric("Annotators", len(dataset_info.annotators))
    with col4:
        imbalance = dataset_info.metadata.get('class_imbalance_ratio', 0)
        st.metric("Class Imbalance", f"{imbalance:.2f}")
    
    # File type distribution
    st.subheader("File Type Distribution")
    if dataset_info.file_types:
        fig_files = px.pie(
            values=list(dataset_info.file_types.values()),
            names=list(dataset_info.file_types.keys()),
            title="File Types"
        )
        st.plotly_chart(fig_files, use_container_width=True)
    
    # Class distribution
    st.subheader("Class Distribution")
    if dataset_info.class_distribution:
        fig_classes = px.bar(
            x=list(dataset_info.class_distribution.keys()),
            y=list(dataset_info.class_distribution.values()),
            title="Annotations per Class"
        )
        fig_classes.update_xaxis(title="Class")
        fig_classes.update_yaxis(title="Count")
        st.plotly_chart(fig_classes, use_container_width=True)
    
    # Sample data preview
    st.subheader("Sample Data")
    if st.session_state.dataset_loader:
        samples = st.session_state.dataset_loader.get_sample_data(5)
        if samples:
            st.json(samples)


def show_quality_checks_page():
    """Quality checks page"""
    if not st.session_state.dataset_loaded:
        st.warning("Please upload a dataset first")
        return
    
    st.header(t('quality_checks'))
    
    # Run quality checks button
    if st.button("Run Quality Checks", type="primary"):
        with st.spinner("Running quality checks..."):
            # Get annotations (simplified - in real implementation, parse from dataset)
            dataset_loader = st.session_state.dataset_loader
            samples = dataset_loader.get_sample_data(1000)  # Get more samples
            
            # Determine annotation type
            if dataset_info := st.session_state.dataset_info:
                if dataset_info.annotation_format == 'coco':
                    annotation_type = 'bbox'
                elif dataset_info.annotation_format == 'yolo':
                    annotation_type = 'bbox'
                else:
                    annotation_type = 'text'
            
            # Run checks
            quality_checker = st.session_state.quality_checker
            issues = quality_checker.check_dataset(
                samples,
                annotation_type=annotation_type
            )
            
            st.session_state.quality_issues = issues
            
            # Show summary
            summary = quality_checker.get_summary_statistics()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Issues", summary['total_issues'])
            with col2:
                st.metric("Critical Issues", 
                         summary['severity_distribution'].get('critical', 0))
            with col3:
                st.metric("Warnings", 
                         summary['severity_distribution'].get('warning', 0))
    
    # Display issues
    if st.session_state.quality_issues:
        st.subheader("Quality Issues Found")
        
        # Filter by severity
        severity_filter = st.multiselect(
            "Filter by severity",
            options=['critical', 'warning', 'info'],
            default=['critical', 'warning']
        )
        
        # Create issues dataframe
        issues_data = []
        for issue in st.session_state.quality_issues:
            if issue.severity in severity_filter:
                issues_data.append({
                    'Type': issue.issue_type.value,
                    'Severity': issue.severity,
                    'Description': issue.description,
                    'Affected Items': len(issue.affected_items)
                })
        
        if issues_data:
            issues_df = pd.DataFrame(issues_data)
            
            # Apply styling
            def style_severity(val):
                if val == 'critical':
                    return 'color: red; font-weight: bold'
                elif val == 'warning':
                    return 'color: orange; font-weight: bold'
                else:
                    return 'color: green'
            
            styled_df = issues_df.style.applymap(
                style_severity, 
                subset=['Severity']
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Issue type distribution
            issue_counts = issues_df['Type'].value_counts()
            fig_issues = px.bar(
                x=issue_counts.index,
                y=issue_counts.values,
                title="Issues by Type"
            )
            st.plotly_chart(fig_issues, use_container_width=True)


def show_ai_validation_page():
    """AI validation page"""
    if not st.session_state.dataset_loaded:
        st.warning("Please upload a dataset first")
        return
    
    st.header(t('ai_validation'))
    
    if not st.session_state.mistral_client:
        st.error("Mistral API not configured. Please set MISTRAL_API_KEY environment variable.")
        return
    
    selected_checks = st.multiselect(
        "Select the AI validation checks to perform",
        options=[
            "Check label correctness",
            "Check content consistency",
            "Detect bias",
            "Check annotation completeness",
            "Suggest corrections"
        ],
        default=["Check label correctness"]
    )
    
    st.markdown("### Description of selected checks")
    if "Check label correctness" in selected_checks:
        st.info("✅ Checks whether the assigned label matches the actual media or text content.")
    if "Check content consistency" in selected_checks:
        st.warning("🧠 Analyzes for logical inconsistencies or hallucinated text.")
    if "Detect bias" in selected_checks:
        st.error("⚖️ Detects potential bias in annotation distribution or wording.")
    if "Suggest corrections" in selected_checks:
        st.success("🛠 Suggests a corrected version of the annotation or label.")
    if "Check annotation completeness" in selected_checks:
        st.info("🔍 Checks whether the annotation fully covers the expected content.")


with st.expander("🧠 Test a single AI validation criterion", expanded=False):
    st.markdown("Select a criterion and run it on a sample from your dataset:")

    # Library of prompt criteria
    ai_validation_checks = {
        "Label correctness": "Given this text and its label, does the label accurately describe the content? Justify your answer.",
        "Label consistency": "Are the labels across similar texts consistent and coherent?",
        "Label completeness": "Does the provided annotation cover all relevant aspects of this text?",
        "Redundant labels": "Is the annotation overly detailed or repetitive for this type of content?",
        "Spelling and grammar": "Does this text contain spelling or grammatical errors?",
        "Logical consistency": "Does the text contain logical inconsistencies or contradictions?",
        "Formatting consistency": "Does this dataset follow a consistent text formatting style?",
        "Duplicates": "Identify if the following texts are exact or near duplicates.",
        "Stereotypes or bias": "Does this text contain any gender, racial, or cultural stereotypes?",
        "Label imbalance": "Analyze the label distribution. Are any classes underrepresented or overrepresented?",
        "Subjectivity": "Is this annotation subjective or emotionally biased?",
        "Toxicity": "Does this text contain any offensive, abusive, or inappropriate language?",
        "PII detection": "Does this text include any personally identifiable information (PII) such as names, addresses, or IDs?",
        "Dangerous content": "Does the text promote violence, hate speech, or misinformation?",
        "Linguistic diversity": "Does this dataset include a diverse range of language styles, dialects, and expressions?",
        "Topic variety": "Are there a broad range of topics represented in the dataset?",
        "Length variation": "Are texts in the dataset varied in length, or are they overly uniform?",
        "Text completeness": "Is this text complete or does it appear to be cut off or missing parts?",
        "Relevance": "Is this text meaningful and relevant for the dataset’s purpose?",
        "Readability": "Is the text easy to read and understand? Rate its clarity on a scale from 1 to 10.",
        "Annotation guideline match": "Does this annotation conform to the project’s annotation guidelines?"
    }

    selected_criterion = st.selectbox("📝 Choose a validation check to run:", list(ai_validation_checks.keys()))

    if st.button("▶️ Run test on a sample"):
        with st.spinner("Analyzing sample with Mistral..."):
            try:
                # Extract sample text
                samples = st.session_state.dataset_loader.get_sample_data(5)
                sample_text = None
                for s in samples:
                    if isinstance(s, dict):
                        sample_text = s.get("text") or s.get("content")
                    elif isinstance(s, str):
                        try:
                            parsed = json.loads(s)
                            sample_text = parsed.get("text") or parsed.get("content")
                        except:
                            sample_text = s
                    if sample_text and len(sample_text.strip()) > 20:
                        break

                if not sample_text:
                    st.warning("No suitable text found in your dataset.")
                else:
                    prompt = ai_validation_checks[selected_criterion] + "\n\nExample:\n" + sample_text[:3000]
                    response = st.session_state.mistral_client.query(prompt=prompt)
                    st.success(f"✅ AI response for: {selected_criterion}")
                    st.text_area("📥 Mistral's Response", value=response.strip(), height=300)

            except Exception as e:
                st.error(f"❌ Error while calling Mistral: {e}")



    
    # Run AI validation
    if st.button("Run AI Validation", type="primary"):
        with st.spinner("Running AI validation..."):
            # Get samples
            samples = st.session_state.dataset_loader.get_sample_data(100)
            
            # Configure AI checker
            ai_checker = st.session_state.ai_checker
            ai_checker.config.confidence_threshold = confidence_threshold
            ai_checker.config.batch_size = batch_size
            
            # Progress bar
            progress_bar = st.progress(0)
            
            def update_progress(progress):
                progress_bar.progress(progress)
            
            # Run validation
            ai_results = ai_checker.validate_dataset(
                samples,
                context={"selected_checks": selected_checks},
                progress_callback=update_progress
            )
            
            st.session_state.ai_results = ai_results
            
            # Show results
            st.success("AI validation complete!")
            
            # Summary metrics
            stats = ai_checker.aggregate_stats or {}
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Validated", stats.get('total_validated', 0))
            with col2:
                st.metric("Valid", stats.get('valid_annotations', 0))
            with col3:
                st.metric("Invalid", stats.get('invalid_annotations', 0))
            with col4:
                avg_conf = stats.get('average_confidence', 0.0)
                st.metric("Avg Confidence", f"{avg_conf:.2%}")


    
    # Section de test pour prompt libre avec Mistral
    st.markdown("### 🧪 Test Mistral Prompt")
    test_prompt = st.text_area("📝 Enter your prompt for Mistral", "Can you explain what AI is in simple terms?")
    send_prompt = st.button("🚀 Send Prompt")
    
    if send_prompt:
        try:
            client = MistralClient()  # Assumes API key is in env or .streamlit/secrets.toml
            response = client.query(prompt=test_prompt)
    
            st.success("✅ Response received from Mistral")
            st.markdown("#### 🤖 Mistral's Response")
            st.markdown(f"> {response.strip()}")
            
            # Optionally: Show raw response in expandable panel
            with st.expander("🔍 Raw Response"):
                st.code(response, language="text")
    
        except Exception as e:
            st.error(f"❌ Error while querying Mistral: {e}")

    st.subheader("🧪 Mistral Text Analysis Test")
    if st.button("Send Dataset Text to Mistral"):
        with st.spinner("Reading dataset and sending to Mistral..."):
            dataset_path = st.session_state.dataset_loader.current_dataset_path  # ZIP extracted folder
            text_content = []
    
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(".txt") or file.endswith(".json"):  # optionally add .csv etc.
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                if len(content.strip()) > 20:  # only keep non-empty files
                                    text_content.append(content[:3000])  # truncate large files
                        except Exception as e:
                            st.warning(f"Error reading {file}: {e}")
    
            if not text_content:
                st.error("❌ No readable text found in dataset files.")
            else:
                # Assemble a prompt
                prompt = (
                    "You are a dataset reviewer AI. Analyze the following dataset content "
                    "and identify potential annotation issues, inconsistencies, or bias:\n\n"
                    + "\n\n---\n\n".join(text_content[:3])  # Send first few texts
                )
    
                mistral = st.session_state.mistral_client
                try:
                    response = mistral.query(prompt)
                    st.text_area("💬 Mistral Response", value=response, height=400)
                except Exception as e:
                    st.error(f"Error calling Mistral: {e}")

    # NEW SECTION: Define available AI validation checks
    ai_validation_checks = {
        "Label correctness": "Given this text and its label, does the label accurately describe the content? Justify your answer.",
        "Label consistency": "Are the labels across similar texts consistent and coherent?",
        "Label completeness": "Does the provided annotation cover all relevant aspects of this text?",
        "Redundant labels": "Is the annotation overly detailed or repetitive for this type of content?",
        "Spelling and grammar": "Does this text contain spelling or grammatical errors?",
        "Logical consistency": "Does the text contain logical inconsistencies or contradictions?",
        "Formatting consistency": "Does this dataset follow a consistent text formatting style?",
        "Duplicates": "Identify if the following texts are exact or near duplicates.",
        "Stereotypes or bias": "Does this text contain any gender, racial, or cultural stereotypes?",
        "Label imbalance": "Analyze the label distribution. Are any classes underrepresented or overrepresented?",
        "Subjectivity": "Is this annotation subjective or emotionally biased?",
        "Toxicity": "Does this text contain any offensive, abusive, or inappropriate language?",
        "PII detection": "Does this text include any personally identifiable information (PII) such as names, addresses, or IDs?",
        "Dangerous content": "Does the text promote violence, hate speech, or misinformation?",
        "Linguistic diversity": "Does this dataset include a diverse range of language styles, dialects, and expressions?",
        "Topic variety": "Are there a broad range of topics represented in the dataset?",
        "Length variation": "Are texts in the dataset varied in length, or are they overly uniform?",
        "Text completeness": "Is this text complete or does it appear to be cut off or missing parts?",
        "Relevance": "Is this text meaningful and relevant for the dataset’s purpose?",
        "Readability": "Is the text easy to read and understand? Rate its clarity on a scale from 1 to 10.",
        "Annotation guideline match": "Does this annotation conform to the project’s annotation guidelines?"
    }

    selected_checks = st.multiselect(
        "Select validation checks to run:",
        options=list(ai_validation_checks.keys()),
        default=["Label correctness"]
    )

    # Display explanation table
    if selected_checks:
        st.markdown("### 🧾 Selected Checks and Their Purpose")
        for check in selected_checks:
            st.markdown(f"- **{check}**: {ai_validation_checks[check]}")

    # Prompt and response testing button
    if st.button("🧠 Run selected AI validation prompts"):
        with st.spinner("Sending selected prompts to Mistral..."):
            mistral = st.session_state.mistral_client
            loader = st.session_state.dataset_loader
            samples = loader.get_sample_data(3)  # limited sample

            results = []
            for check in selected_checks:
                prompt = ai_validation_checks[check] + "\n\nExample:\n" + samples[0].get("text", "<no text found>")
                try:
                    response = mistral.query(prompt=prompt)
                    results.append((check, response.strip()))
                except Exception as e:
                    results.append((check, f"❌ Error: {str(e)}"))

            # Display results
            for check, res in results:
                with st.expander(f"🔍 {check}"):
                    st.markdown(res)

    # (rest of show_ai_validation_page unchanged)





    
    # Display AI results
    if st.session_state.ai_results:
        st.subheader("AI Validation Results")
    
        for result in st.session_state.ai_results[:10]:
            with st.expander(f"🔍 Annotation {result.annotation_id} – Confidence: {result.confidence:.2%}"):
                st.markdown(f"**Is Valid:** {'✅ Yes' if result.is_valid else '❌ No'}")
                st.markdown(f"**Reasoning:** {result.reasoning}")
    
                if hasattr(result, "issues") and result.issues:
                    st.warning("⚠️ **Detected Issues:**")
                    for issue in result.issues:
                        st.write(f"- {issue}")
    
                if hasattr(result, "bias_flagged") and result.bias_flagged:
                    st.error("🚨 **Potential Bias Detected**")
                    st.write(f"> {result.bias_notes}")
    
                if hasattr(result, "corrected_annotation") and result.corrected_annotation:
                    st.success("🔧 **Suggested Correction:**")
                    st.json(result.corrected_annotation)
    
                if hasattr(result, "suggestions") and result.suggestions:
                    st.info("💡 **Suggestions:**")
                    for suggestion in result.suggestions:
                        st.write(f"- {suggestion}")
    
        # Global histogram
        st.markdown("### Confidence Score Distribution")
        confidences = [r.confidence for r in st.session_state.ai_results]
        fig_conf = px.histogram(
            x=confidences,
            nbins=20,
            title="Confidence Scores"
        )
        fig_conf.update_xaxis(title="Confidence")
        fig_conf.update_yaxis(title="Number of Annotations")
        st.plotly_chart(fig_conf, use_container_width=True)


def show_manual_review_page():
    """Manual review page"""
    if not st.session_state.dataset_loaded:
        st.warning("Please upload a dataset first")
        return
    
    st.header(t('manual_review'))
    
    # Get items for review
    if not hasattr(st.session_state, 'review_items'):
        # Get low confidence or problematic items
        review_items = []
        
        # Add items from quality checks
        for issue in st.session_state.quality_issues[:10]:
            if issue.severity in ['critical', 'warning']:
                review_items.append({
                    'id': f"issue_{len(review_items)}",
                    'type': 'quality_issue',
                    'data': issue
                })
        
        # Add items from AI validation
        if st.session_state.ai_results:
            ai_checker = st.session_state.ai_checker
            for result in ai_checker.get_low_confidence_annotations()[:10]:
                review_items.append({
                    'id': result.annotation_id,
                    'type': 'ai_validation',
                    'data': result
                })
        
        st.session_state.review_items = review_items
        st.session_state.current_review_index = 0
    
    if not st.session_state.review_items:
        st.info("No items to review")
        return
    
    # Review interface
    total_items = len(st.session_state.review_items)
    current_index = st.session_state.current_review_index
    
    # Progress
    st.progress((current_index + 1) / total_items)
    st.write(f"Reviewing item {current_index + 1} of {total_items}")
    
    # Current item
    current_item = st.session_state.review_items[current_index]
    
    with st.container():
        st.subheader(f"Item: {current_item['id']}")
        
        if current_item['type'] == 'quality_issue':
            issue = current_item['data']
            st.write(f"**Issue Type:** {issue.issue_type.value}")
            st.write(f"**Severity:** {issue.severity}")
            st.write(f"**Description:** {issue.description}")
            
        elif current_item['type'] == 'ai_validation':
            result = current_item['data']
            st.write(f"**Confidence:** {result.confidence:.2%}")
            st.write(f"**AI Assessment:** {'Valid' if result.is_valid else 'Invalid'}")
            st.write(f"**Reasoning:** {result.reasoning}")
        
        # Review actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Accept", type="primary"):
                current_item['review_status'] = 'accepted'
                next_item()
        
        with col2:
            if st.button("✏️ Edit"):
                current_item['review_status'] = 'edited'
                # In real implementation, show edit interface
                st.info("Edit functionality would be shown here")
        
        with col3:
            if st.button("❌ Reject"):
                current_item['review_status'] = 'rejected'
                next_item()
        
        # Comments
        comment = st.text_area("Comments (optional)")
        if comment:
            current_item['comment'] = comment
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous") and current_index > 0:
                st.session_state.current_review_index -= 1
                st.rerun()
        
        with col2:
            if st.button("Next →") and current_index < total_items - 1:
                st.session_state.current_review_index += 1
                st.rerun()


def next_item():
    """Move to next review item"""
    if st.session_state.current_review_index < len(st.session_state.review_items) - 1:
        st.session_state.current_review_index += 1
        st.rerun()
    else:
        st.success("Review complete!")


def show_reports_page():
    """Reports page"""
    st.header(t('reports'))
    
    if not st.session_state.dataset_loaded:
        st.warning("Please upload a dataset first")
        return
    
    # Report options
    st.subheader("Report Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        report_format = st.selectbox(
            "Report Format",
            options=['PDF', 'Excel', 'CSV']
        )
    
    with col2:
        include_visualizations = st.checkbox("Include Visualizations", value=True)
    
    # Report sections
    st.subheader("Report Sections")
    include_summary = st.checkbox("Dataset Summary", value=True)
    include_quality = st.checkbox("Quality Check Results", value=True)
    include_ai = st.checkbox("AI Validation Results", value=True)
    include_review = st.checkbox("Manual Review Results", value=True)
    
    # Generate report button
    if st.button(t('generate_report'), type="primary"):
        with st.spinner("Generating report..."):
            # Initialize report generator
            report_gen = ReportGenerator()
            
            # Prepare report data
            report_data = {
                'dataset_info': st.session_state.dataset_info,
                'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'format': report_format.lower()
            }
            
            if include_summary:
                report_data['summary'] = {
                    'total_files': st.session_state.dataset_info.total_files,
                    'annotation_format': st.session_state.dataset_info.annotation_format,
                    'classes': st.session_state.dataset_info.classes,
                    'file_types': st.session_state.dataset_info.file_types
                }
            
            if include_quality and st.session_state.quality_issues:
                report_data['quality_issues'] = [
                    {
                        'type': issue.issue_type.value,
                        'severity': issue.severity,
                        'description': issue.description
                    }
                    for issue in st.session_state.quality_issues
                ]
            
            if include_ai and st.session_state.ai_results:
                ai_checker = st.session_state.ai_checker
                report_data['ai_validation'] = {
                    'summary': ai_checker.aggregate_stats,
                    'low_confidence_count': len(ai_checker.get_low_confidence_annotations()),
                    'invalid_count': len(ai_checker.get_invalid_annotations())
                }
            
            # Generate report
            if report_format == 'PDF':
                # In real implementation, generate PDF
                st.success("PDF report generated!")
                st.download_button(
                    label="Download PDF Report",
                    data=json.dumps(report_data, indent=2),  # Placeholder
                    file_name=f"dataset_qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            elif report_format == 'Excel':
                # Create Excel report
                output_file = f"dataset_qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                
                # In real implementation, create proper Excel file
                st.success("Excel report generated!")
                
            elif report_format == 'CSV':
                # Create CSV report
                if st.session_state.quality_issues:
                    issues_df = pd.DataFrame([
                        {
                            'Type': issue.issue_type.value,
                            'Severity': issue.severity,
                            'Description': issue.description
                        }
                        for issue in st.session_state.quality_issues
                    ])
                    
                    csv = issues_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV Report",
                        data=csv,
                        file_name=f"quality_issues_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )


def show_settings_page():
    """Settings page"""
    st.header(t('settings'))
    
    # API Configuration
    st.subheader("API Configuration")
    
    api_key = st.text_input(
        "Mistral API Key",
        type="password",
        help="Enter your Mistral API key"
    )
    
    if api_key and st.button("Save API Key"):
        os.environ['MISTRAL_API_KEY'] = api_key
        st.success("API key saved!")
        # Reinitialize Mistral client
        try:
            st.session_state.mistral_client = MistralClient(api_key=api_key)
            st.session_state.ai_checker = AIQualityChecker(st.session_state.mistral_client)
        except Exception as e:
            st.error(f"Error initializing Mistral client: {e}")
    
    # Quality Check Thresholds
    st.subheader("Quality Check Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_box_area = st.number_input(
            "Minimum Box Area (pixels²)",
            min_value=1,
            max_value=10000,
            value=100
        )
        
        min_samples_per_class = st.number_input(
            "Minimum Samples per Class",
            min_value=1,
            max_value=1000,
            value=10
        )
    
    with col2:
        max_class_imbalance = st.number_input(
            "Maximum Class Imbalance Ratio",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=0.5
        )
        
        min_annotator_agreement = st.slider(
            "Minimum Annotator Agreement",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05
        )
    
    if st.button("Save Settings"):
        # Update QC config
        if hasattr(st.session_state, 'quality_checker'):
            st.session_state.quality_checker.config.min_box_area = min_box_area
            st.session_state.quality_checker.config.min_samples_per_class = min_samples_per_class
            st.session_state.quality_checker.config.max_class_imbalance_ratio = max_class_imbalance
            st.session_state.quality_checker.config.min_annotator_agreement = min_annotator_agreement
        
        st.success("Settings saved!")
    
    # Export/Import Configuration
    st.subheader("Configuration Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Configuration"):
            config = {
                'qc_rules': {
                    'min_box_area': min_box_area,
                    'min_samples_per_class': min_samples_per_class,
                    'max_class_imbalance_ratio': max_class_imbalance,
                    'min_annotator_agreement': min_annotator_agreement
                }
            }
            
            st.download_button(
                label="Download Configuration",
                data=yaml.dump(config),
                file_name="qa_config.yaml",
                mime="text/yaml"
            )
    
    with col2:
        uploaded_config = st.file_uploader(
            "Import Configuration",
            type=['yaml', 'yml']
        )
        
        if uploaded_config:
            config_data = yaml.safe_load(uploaded_config)
            st.success("Configuration loaded!")
            st.json(config_data)


if __name__ == "__main__":
    main()
