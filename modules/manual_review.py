"""
Manual Review Interface Module
Provides UI components for manual annotation review and editing
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional, Tuple, Any
import json
import cv2
import base64
from io import BytesIO
from dataclasses import dataclass, field
from datetime import datetime
import plotly.graph_objects as go
from collections import defaultdict

@dataclass
class ReviewSession:
    """Container for a review session"""
    session_id: str
    annotator: str
    start_time: datetime
    items_reviewed: List[Dict] = field(default_factory=list)
    decisions: Dict[str, str] = field(default_factory=dict)
    comments: Dict[str, str] = field(default_factory=dict)
    edits: Dict[str, Any] = field(default_factory=dict)
    
    def add_review(self, item_id: str, decision: str, 
                   comment: Optional[str] = None, 
                   edit: Optional[Dict] = None):
        """Add a review decision"""
        self.items_reviewed.append(item_id)
        self.decisions[item_id] = decision
        if comment:
            self.comments[item_id] = comment
        if edit:
            self.edits[item_id] = edit
    
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary"""
        return {
            'session_id': self.session_id,
            'annotator': self.annotator,
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'total_reviewed': len(self.items_reviewed),
            'decisions': dict(pd.Series(list(self.decisions.values())).value_counts()),
            'edits_made': len(self.edits)
        }


class ManualReviewInterface:
    """Main class for manual review interface"""
    
    def __init__(self):
        """Initialize manual review interface"""
        self.current_session = None
        self.review_queue = []
        self.current_index = 0
        
    def create_session(self, annotator: str) -> ReviewSession:
        """Create a new review session"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session = ReviewSession(
            session_id=session_id,
            annotator=annotator,
            start_time=datetime.now()
        )
        return self.current_session
    
    def render_image_review(self, image_path: str, annotations: List[Dict], 
                           container) -> Dict[str, Any]:
        """
        Render image review interface
        
        Args:
            image_path: Path to image file
            annotations: List of annotations for the image
            container: Streamlit container to render in
            
        Returns:
            Review decision and edits
        """
        with container:
            # Load and display image
            try:
                image = Image.open(image_path)
                
                # Create figure with annotations
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(image)
                
                # Draw annotations
                colors = plt.cm.rainbow(np.linspace(0, 1, len(annotations)))
                
                for idx, (ann, color) in enumerate(zip(annotations, colors)):
                    if 'bbox' in ann:
                        x, y, w, h = ann['bbox']
                        rect = patches.Rectangle(
                            (x, y), w, h,
                            linewidth=2,
                            edgecolor=color,
                            facecolor='none'
                        )
                        ax.add_patch(rect)
                        
                        # Add label
                        label = ann.get('label', ann.get('class', f'Object {idx}'))
                        ax.text(x, y-5, label, 
                               color=color, 
                               fontsize=10, 
                               weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', 
                                       alpha=0.7))
                
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Error loading image: {e}")
                return {}
            
            # Annotation list
            st.subheader("Annotations")
            
            # Create editable dataframe
            ann_data = []
            for idx, ann in enumerate(annotations):
                ann_data.append({
                    'ID': idx,
                    'Label': ann.get('label', ''),
                    'Confidence': ann.get('confidence', 1.0),
                    'BBox': str(ann.get('bbox', [])),
                    'Valid': True
                })
            
            ann_df = pd.DataFrame(ann_data)
            
            # Allow editing
            edited_df = st.data_editor(
                ann_df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Valid": st.column_config.CheckboxColumn(
                        "Valid",
                        help="Is this annotation correct?",
                        default=True,
                    ),
                    "Confidence": st.column_config.NumberColumn(
                        "Confidence",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        format="%.2f"
                    )
                }
            )
            
            # Review actions
            col1, col2, col3 = st.columns(3)
            
            decision = None
            with col1:
                if st.button("✅ Accept All", key=f"accept_{self.current_index}"):
                    decision = "accepted"
            
            with col2:
                if st.button("✏️ Save Edits", key=f"edit_{self.current_index}"):
                    decision = "edited"
            
            with col3:
                if st.button("❌ Reject", key=f"reject_{self.current_index}"):
                    decision = "rejected"
            
            # Comments
            comment = st.text_area("Comments", key=f"comment_{self.current_index}")
            
            # Prepare edits if dataframe was modified
            edits = None
            if not edited_df.equals(ann_df):
                edits = {
                    'original': ann_df.to_dict('records'),
                    'edited': edited_df.to_dict('records')
                }
            
            return {
                'decision': decision,
                'comment': comment,
                'edits': edits
            }
    
    def render_text_review(self, text: str, annotations: List[Dict], 
                          container) -> Dict[str, Any]:
        """
        Render text/NLP annotation review interface
        
        Args:
            text: Text content
            annotations: List of annotations
            container: Streamlit container
            
        Returns:
            Review decision and edits
        """
        with container:
            # Display text with highlights
            st.subheader("Text Content")
            
            # Sort annotations by start position
            sorted_anns = sorted(annotations, 
                               key=lambda x: x.get('span', [0])[0] if x.get('span') else 0)
            
            # Create highlighted text
            highlighted_text = self._create_highlighted_text(text, sorted_anns)
            st.markdown(highlighted_text, unsafe_allow_html=True)
            
            # Annotation editor
            st.subheader("Entity Annotations")
            
            ann_data = []
            for idx, ann in enumerate(sorted_anns):
                span = ann.get('span', [])
                if len(span) >= 2:
                    entity_text = text[span[0]:span[1]]
                else:
                    entity_text = ann.get('text', '')
                
                ann_data.append({
                    'ID': idx,
                    'Entity': entity_text,
                    'Label': ann.get('label', ''),
                    'Start': span[0] if span else 0,
                    'End': span[1] if len(span) >= 2 else 0,
                    'Valid': True
                })
            
            ann_df = pd.DataFrame(ann_data)
            
            # Allow editing
            edited_df = st.data_editor(
                ann_df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Valid": st.column_config.CheckboxColumn(
                        "Valid",
                        help="Is this annotation correct?",
                        default=True,
                    )
                }
            )
            
            # Review actions
            col1, col2, col3 = st.columns(3)
            
            decision = None
            with col1:
                if st.button("✅ Accept All", key=f"accept_text_{self.current_index}"):
                    decision = "accepted"
            
            with col2:
                if st.button("✏️ Save Edits", key=f"edit_text_{self.current_index}"):
                    decision = "edited"
            
            with col3:
                if st.button("❌ Reject", key=f"reject_text_{self.current_index}"):
                    decision = "rejected"
            
            # Comments
            comment = st.text_area("Comments", key=f"comment_text_{self.current_index}")
            
            # Prepare edits
            edits = None
            if not edited_df.equals(ann_df):
                edits = {
                    'original': ann_df.to_dict('records'),
                    'edited': edited_df.to_dict('records')
                }
            
            return {
                'decision': decision,
                'comment': comment,
                'edits': edits
            }
    
    def _create_highlighted_text(self, text: str, annotations: List[Dict]) -> str:
        """Create HTML with highlighted entities"""
        # Define colors for different entity types
        colors = {
            'person': '#ff9999',
            'organization': '#99ccff',
            'location': '#99ff99',
            'date': '#ffcc99',
            'default': '#ffff99'
        }
        
        # Build highlighted text
        html_parts = []
        last_end = 0
        
        for ann in annotations:
            span = ann.get('span', [])
            if len(span) >= 2:
                start, end = span[0], span[1]
                label = ann.get('label', '')
                
                # Add text before annotation
                if start > last_end:
                    html_parts.append(text[last_end:start])
                
                # Add highlighted annotation
                color = colors.get(label.lower(), colors['default'])
                entity_text = text[start:end]
                html_parts.append(
                    f'<span style="background-color: {color}; padding: 2px 4px; '
                    f'border-radius: 3px;" title="{label}">{entity_text}</span>'
                )
                
                last_end = end
        
        # Add remaining text
        if last_end < len(text):
            html_parts.append(text[last_end:])
        
        return ''.join(html_parts)
    
    def render_audio_review(self, audio_path: str, annotations: List[Dict],
                           container) -> Dict[str, Any]:
        """
        Render audio annotation review interface
        
        Args:
            audio_path: Path to audio file
            annotations: List of temporal annotations
            container: Streamlit container
            
        Returns:
            Review decision and edits
        """
        with container:
            # Audio player
            st.subheader("Audio Content")
            st.audio(audio_path)
            
            # Timeline visualization
            st.subheader("Annotation Timeline")
            
            # Create timeline plot
            fig = go.Figure()
            
            # Sort annotations by start time
            sorted_anns = sorted(annotations, key=lambda x: x.get('start_time', 0))
            
            # Add annotations to timeline
            y_pos = 0
            for ann in sorted_anns:
                start = ann.get('start_time', 0)
                end = ann.get('end_time', start + 1)
                label = ann.get('label', 'Unknown')
                
                fig.add_trace(go.Scatter(
                    x=[start, end],
                    y=[y_pos, y_pos],
                    mode='lines+markers',
                    name=label,
                    line=dict(width=10),
                    marker=dict(size=8),
                    hovertemplate=f"{label}<br>Start: {start:.2f}s<br>End: {end:.2f}s"
                ))
                
                y_pos += 1
            
            fig.update_layout(
                title="Temporal Annotations",
                xaxis_title="Time (seconds)",
                yaxis_title="Annotations",
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Annotation editor
            st.subheader("Edit Annotations")
            
            ann_data = []
            for idx, ann in enumerate(sorted_anns):
                ann_data.append({
                    'ID': idx,
                    'Label': ann.get('label', ''),
                    'Start Time': ann.get('start_time', 0),
                    'End Time': ann.get('end_time', 0),
                    'Duration': ann.get('end_time', 0) - ann.get('start_time', 0),
                    'Valid': True
                })
            
            ann_df = pd.DataFrame(ann_data)
            
            # Allow editing
            edited_df = st.data_editor(
                ann_df,
                use_container_width=True,
                column_config={
                    "Valid": st.column_config.CheckboxColumn(
                        "Valid",
                        help="Is this annotation correct?",
                        default=True,
                    ),
                    "Start Time": st.column_config.NumberColumn(
                        "Start Time",
                        min_value=0.0,
                        step=0.1,
                        format="%.2f"
                    ),
                    "End Time": st.column_config.NumberColumn(
                        "End Time",
                        min_value=0.0,
                        step=0.1,
                        format="%.2f"
                    )
                }
            )
            
            # Review actions
            col1, col2, col3 = st.columns(3)
            
            decision = None
            with col1:
                if st.button("✅ Accept All", key=f"accept_audio_{self.current_index}"):
                    decision = "accepted"
            
            with col2:
                if st.button("✏️ Save Edits", key=f"edit_audio_{self.current_index}"):
                    decision = "edited"
            
            with col3:
                if st.button("❌ Reject", key=f"reject_audio_{self.current_index}"):
                    decision = "rejected"
            
            # Comments
            comment = st.text_area("Comments", key=f"comment_audio_{self.current_index}")
            
            # Prepare edits
            edits = None
            if not edited_df.equals(ann_df):
                edits = {
                    'original': ann_df.to_dict('records'),
                    'edited': edited_df.to_dict('records')
                }
            
            return {
                'decision': decision,
                'comment': comment,
                'edits': edits
            }
    
    def render_comparison_view(self, original_annotations: List[Dict],
                             ai_suggestions: List[Dict],
                             container) -> Dict[str, Any]:
        """
        Render side-by-side comparison of original vs AI-suggested annotations
        
        Args:
            original_annotations: Original human annotations
            ai_suggestions: AI-suggested corrections
            container: Streamlit container
            
        Returns:
            Review decision
        """
        with container:
            st.subheader("Annotation Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Annotations**")
                for ann in original_annotations:
                    st.write(f"- {ann.get('label', 'Unknown')}: {ann}")
            
            with col2:
                st.write("**AI Suggestions**")
                for ann in ai_suggestions:
                    st.write(f"- {ann.get('label', 'Unknown')}: {ann}")
            
            # Differences
            st.subheader("Differences")
            differences = self._compute_differences(original_annotations, ai_suggestions)
            
            if differences:
                for diff in differences:
                    if diff['type'] == 'added':
                        st.success(f"➕ AI suggests adding: {diff['annotation']}")
                    elif diff['type'] == 'removed':
                        st.error(f"➖ AI suggests removing: {diff['annotation']}")
                    elif diff['type'] == 'modified':
                        st.warning(f"✏️ AI suggests modifying: {diff['original']} → {diff['suggested']}")
            else:
                st.info("No differences found")
            
            # Decision
            decision = st.radio(
                "Select action:",
                options=['Keep Original', 'Accept AI Suggestions', 'Manual Merge'],
                key=f"comparison_{self.current_index}"
            )
            
            return {'decision': decision}
    
    def _compute_differences(self, original: List[Dict], 
                           suggested: List[Dict]) -> List[Dict]:
        """Compute differences between annotation sets"""
        differences = []
        
        # Simple comparison - in production, use more sophisticated matching
        orig_labels = {ann.get('label', '') for ann in original}
        sugg_labels = {ann.get('label', '') for ann in suggested}
        
        # Added
        for label in sugg_labels - orig_labels:
            differences.append({
                'type': 'added',
                'annotation': label
            })
        
        # Removed
        for label in orig_labels - sugg_labels:
            differences.append({
                'type': 'removed',
                'annotation': label
            })
        
        return differences
    
    def export_review_session(self, session: ReviewSession) -> Dict[str, Any]:
        """
        Export review session data
        
        Args:
            session: Review session to export
            
        Returns:
            Exported session data
        """
        return {
            'session_info': session.get_summary(),
            'decisions': session.decisions,
            'comments': session.comments,
            'edits': session.edits,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_reviewer_metrics(self, sessions: List[ReviewSession]) -> Dict[str, Any]:
        """
        Calculate metrics for reviewer performance
        
        Args:
            sessions: List of review sessions
            
        Returns:
            Reviewer metrics
        """
        metrics = {
            'total_sessions': len(sessions),
            'total_items_reviewed': sum(len(s.items_reviewed) for s in sessions),
            'average_items_per_session': np.mean([len(s.items_reviewed) for s in sessions]),
            'decision_distribution': defaultdict(int),
            'edit_rate': 0,
            'average_review_time': 0
        }
        
        total_decisions = 0
        total_edits = 0
        total_time = 0
        
        for session in sessions:
            summary = session.get_summary()
            
            # Aggregate decisions
            for decision, count in summary['decisions'].items():
                metrics['decision_distribution'][decision] += count
                total_decisions += count
            
            total_edits += summary['edits_made']
            total_time += summary['duration']
        
        if total_decisions > 0:
            metrics['edit_rate'] = total_edits / total_decisions
        
        if len(sessions) > 0:
            metrics['average_review_time'] = total_time / len(sessions)
        
        return metrics


def create_review_dashboard(review_sessions: List[ReviewSession]) -> None:
    """
    Create a dashboard showing review statistics
    
    Args:
        review_sessions: List of completed review sessions
    """
    st.subheader("Review Dashboard")
    
    if not review_sessions:
        st.info("No review sessions completed yet")
        return
    
    # Calculate metrics
    interface = ManualReviewInterface()
    metrics = interface.calculate_reviewer_metrics(review_sessions)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", metrics['total_items_reviewed'])
    
    with col2:
        st.metric("Sessions", metrics['total_sessions'])
    
    with col3:
        st.metric("Avg Items/Session", f"{metrics['average_items_per_session']:.1f}")
    
    with col4:
        st.metric("Edit Rate", f"{metrics['edit_rate']:.1%}")
    
    # Decision distribution
    if metrics['decision_distribution']:
        fig = go.Figure(data=[
            go.Pie(
                labels=list(metrics['decision_distribution'].keys()),
                values=list(metrics['decision_distribution'].values()),
                hole=0.3
            )
        ])
        fig.update_layout(title="Review Decisions Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Session timeline
    st.subheader("Review Timeline")
    
    timeline_data = []
    for session in review_sessions:
        timeline_data.append({
            'Session': session.session_id,
            'Annotator': session.annotator,
            'Start Time': session.start_time,
            'Items Reviewed': len(session.items_reviewed),
            'Duration (min)': session.get_summary()['duration'] / 60
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(timeline_df, use_container_width=True)


if __name__ == "__main__":
    # Example usage
    interface = ManualReviewInterface()
    
    # Create a session
    session = interface.create_session("reviewer_1")
    
    # Example annotations
    annotations = [
        {'label': 'cat', 'bbox': [100, 100, 200, 200], 'confidence': 0.95},
        {'label': 'dog', 'bbox': [300, 150, 150, 250], 'confidence': 0.87}
    ]
    
    # Simulate review
    session.add_review(
        item_id="img_001",
        decision="accepted",
        comment="Looks good"
    )
    
    print(session.get_summary())