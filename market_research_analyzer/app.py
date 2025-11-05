import streamlit as st
import os
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.chat_engine import ChatEngine
from src.utils import WebScraper
import tempfile

st.set_page_config(page_title="Market Research Analyzer", layout="wide")

def initialize_session_state():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStoreManager()
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "quick_action_result" not in st.session_state:
        st.session_state.quick_action_result = None
    if "quick_action_type" not in st.session_state:
        st.session_state.quick_action_type = None

def load_documents(documents):
    with st.spinner("Processing documents and building knowledge base..."):
        vector_store = st.session_state.vector_store
        vector_store.add_documents(documents)
        retriever = vector_store.get_retriever()
        st.session_state.chat_engine = ChatEngine(retriever)
        st.session_state.documents_loaded = True
    st.success("Documents processed successfully!")

def main():
    st.title("📊 Market Research Analyzer")
    st.markdown("Upload market reports and ask questions about competitors, industries, and trends.")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("📁 Data Ingestion")
        
        uploaded_files = st.file_uploader(
            "Upload Market Reports",
            type=["pdf", "csv", "txt"],
            accept_multiple_files=True,
            help="Upload PDF reports, CSV files, or text documents"
        )
        
        st.header("🌐 Web Data")
        url = st.text_input("Enter URL to scrape market data:")
        if st.button("Scrape URL") and url:
            with st.spinner("Scraping website..."):
                scraper = WebScraper()
                web_content = scraper.scrape_url(url)
                if web_content and not web_content.startswith("Error"):
                    processor = DocumentProcessor()
                    documents = processor.process_text(web_content, source=url)
                    if documents:
                        load_documents(documents)
                        st.success("Web content processed!")
                    else:
                        st.error("No readable content found on this page.")
                else:
                    st.error(web_content)
        
        if uploaded_files:
            processor = DocumentProcessor()
            all_documents = []
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    if uploaded_file.type == "application/pdf":
                        documents = processor.process_pdf(tmp_path)
                    elif uploaded_file.type == "text/csv":
                        documents = processor.process_csv(tmp_path)
                    else:
                        content = uploaded_file.getvalue().decode("utf-8")
                        documents = processor.process_text(content, source=uploaded_file.name)
                    
                    if documents:
                        all_documents.extend(documents)
                        st.success(f"✅ Processed {uploaded_file.name} ({len(documents)} chunks)")
                    else:
                        st.warning(f"⚠️ No content extracted from {uploaded_file.name}")
                
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                finally:
                    os.unlink(tmp_path)
            
            if all_documents:
                load_documents(all_documents)
            else:
                st.error("No documents were successfully processed.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat with Market Data")
        
        if not st.session_state.documents_loaded:
            st.info("📂 Please upload market reports or provide a URL to start analyzing.")
            st.info("💡 The system uses local AI embeddings - no API keys required!")
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("Ask about market trends, competitors, or industry insights..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Searching through documents..."):
                        response = st.session_state.chat_engine.ask_question(prompt)
                    st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.header("🚀 Quick Insights")
        
        if st.session_state.documents_loaded:
            st.markdown("Generate instant analysis reports:")
            
            
            action_options = {
                "Select an analysis type...": None,
                "📊 Industry Overview": "Provide a comprehensive summary of the main industry trends and key players mentioned across all reports.",
                "🏢 Competitor Analysis": "Identify and compare the main competitors mentioned in the reports, highlighting their strengths, weaknesses, and market positions.",
                "📈 Market Trends": "What are the emerging market trends, technological advancements, and consumer behavior changes mentioned in the reports?",
                "💡 Key Insights": "Extract the top 5 most important insights and actionable takeaways from all the reports."
            }
            
            selected_action = st.selectbox(
                "Choose analysis:",
                options=list(action_options.keys()),
                key="action_selector"
            )
            
            col_btn1, col_btn2 = st.columns([1, 1])
            
            with col_btn1:
                if st.button("🔍 Generate", use_container_width=True, type="primary"):
                    if action_options[selected_action]:
                        with st.spinner("Analyzing documents..."):
                            result = st.session_state.chat_engine.ask_question(
                                action_options[selected_action]
                            )
                            st.session_state.quick_action_result = result
                            st.session_state.quick_action_type = selected_action
                    else:
                        st.warning("Please select an analysis type first.")
            
            with col_btn2:
                if st.button("🔄 Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    if st.session_state.chat_engine:
                        st.session_state.chat_engine.memory.clear()
                    st.rerun()
            
            
            if st.session_state.quick_action_result:
                st.markdown("---")
                st.subheader(st.session_state.quick_action_type)
                
                
                with st.container():
                    st.markdown(st.session_state.quick_action_result)
                    
                    
                    st.download_button(
                        label="📋 Copy to Clipboard",
                        data=st.session_state.quick_action_result,
                        file_name=f"{st.session_state.quick_action_type.replace(' ', '_').lower()}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                    if st.button("✖️ Close", use_container_width=True):
                        st.session_state.quick_action_result = None
                        st.session_state.quick_action_type = None
                        st.rerun()
        else:
            st.info("📤 Upload documents to access quick insights")
            st.markdown("""
            **Available analyses:**
            - 📊 Industry Overview
            - 🏢 Competitor Analysis  
            - 📈 Market Trends
            - 💡 Key Insights
            """)

if __name__ == "__main__":
    main()