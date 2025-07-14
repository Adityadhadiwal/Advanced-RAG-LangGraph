
"""
Advanced RAG System with LangGraph
Streamlit application for intelligent document search and analysis
"""
import streamlit as st

# Local imports
from config import QUESTION_PLACEHOLDER
from utils import clear_chroma_db, initialize_session_state
from ui_components import (
    setup_page_config, render_header, render_sidebar, 
    render_upload_section, render_upload_placeholder,
    render_question_section, render_answer_section
)
from document_loader import MultiModalDocumentLoader
from document_processor import DocumentProcessor
from rag_workflow import RAGWorkflow

# Initialize components
document_loader = MultiModalDocumentLoader()
document_processor = DocumentProcessor(document_loader)
rag_workflow = RAGWorkflow()


def handle_question_processing(question):
    """Handle the Q&A processing workflow"""
    st.markdown("### 🤖 LLM Processing")
    
    # Debug info
    print(f"Processing question: {question}")
    
    with st.container():
        with st.spinner('🧠 Analyzing your question and retrieving relevant information...'):
            # Process the question - workflow will handle retriever automatically
            result = rag_workflow.process_question(question)
        
        render_answer_section(result)


def handle_user_interaction(user_file):
    """Handle user interactions for Q&A"""
    if user_file is None:
        render_upload_placeholder()
        return
    
    # Render question section
    question, ask_button = render_question_section(user_file)
    
    # Process question if submitted
    if ask_button and question.strip():
        handle_question_processing(question)
    elif ask_button and not question.strip():
        st.warning("Please enter a question before clicking Ask.")


def main():
    """Main application function"""
    # Initialize session state and clear DB only once
    initialize_session_state()
    
    # Clear ChromaDB only on first run
    if 'db_cleared' not in st.session_state:
        clear_chroma_db()
        st.session_state.db_cleared = True
        print("ChromaDB cleared on app startup")
    
    # Setup page and render UI
    setup_page_config()
    render_header()
    render_sidebar(document_loader)
    
    # Handle file upload
    user_file = render_upload_section(document_loader)
    
    # Process uploaded file
    if user_file:
        retriever = document_processor.process_file(user_file)
        if retriever:
            st.session_state.retriever = retriever
            print(f"File processed, retriever stored in session state")
        else:
            print(f"File processing failed - no retriever created")
    
    # Handle user interactions
    handle_user_interaction(user_file)


if __name__ == "__main__":
    main()
