"""
Pharma Intelligence AI - Main Streamlit Application
Multi-Agent System for Drug Repurposing and Market Intelligence
"""

import streamlit as st
from dotenv import load_dotenv
import os
import sys
from datetime import datetime
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import agents
from agents.master_agent import create_master_agent, run_research_crew
from agents.clinical_trials_agent import create_clinical_trials_agent
from agents.drug_info_agent import create_drug_info_agent
from agents.literature_agent import create_literature_agent
from agents.market_agent import create_market_agent

# Import utilities
from utils.logger import setup_logger
from utils.cache_manager import cache
from config import (
    OPENAI_API_KEY,
    APP_TITLE,
    APP_ICON,
    APP_VERSION,
    STREAMLIT_CONFIG,
    LLM_MODEL
)

# Set up logger
logger = setup_logger("pharma_ai.app")

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(**STREAMLIT_CONFIG)

# Custom CSS
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4 0%, #2e8bc0 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Result box styling */
    .result-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 6px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        color: #000 !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Info boxes */
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    /* Success boxes */
    .success-box {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    /* Warning boxes */
    .warning-box {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Expander styling */
    .streamlit-expander {
        background: white;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False

if 'cache_stats' not in st.session_state:
    st.session_state.cache_stats = cache.get_stats()


def initialize_llm():
    """Initialize OpenAI ChatGPT model for CrewAI."""
    if not OPENAI_API_KEY:
        st.error("⚠️ **OpenAI API Key not found!**")
        st.info("Please set OPENAI_API_KEY in your .env file")
        st.stop()

    try:
        # Ensure OPENAI_API_KEY is set in environment for CrewAI
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

        # Use model from config in CrewAI provider syntax
        model = LLM_MODEL  # e.g., 'openai/gpt-4o-mini'
        logger.info("LLM (OpenAI) initialized successfully for CrewAI")
        return model
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        st.error(f"Error initializing AI model: {e}")
        st.stop()


def display_header():
    """Display application header."""
    st.markdown(
        f'<h1 class="main-header">{APP_ICON} {APP_TITLE}</h1>', 
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">AI-Powered Drug Repurposing & Market Intelligence Platform</p>', 
        unsafe_allow_html=True
    )


def display_sidebar():
    """Display sidebar with configuration and examples."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Status
        st.markdown("### 🔌 API Status")
        if OPENAI_API_KEY:
            st.success("✅ OpenAI API Connected")
        else:
            st.error("❌ OpenAI API Not Connected")
        
        st.markdown("---")
        
        # Example Queries
        st.subheader("📋 Example Queries")
        
        example_queries = {
            "Drug Repurposing": [
                "Find respiratory drugs with potential for rare disease repurposing",
                "Analyze metformin for non-diabetes indications",
                "Identify cardiovascular drugs that could treat neurodegenerative diseases"
            ],
            "Clinical Research": [
                "What are the latest clinical trials for Alzheimer's disease?",
                "Find trials testing immunotherapy for autoimmune diseases",
                "Search for Phase 3 trials in oncology"
            ],
            "Market Analysis": [
                "Analyze the diabetes drug market opportunity",
                "What is the competitive landscape in oncology?",
                "Evaluate market potential for rare disease treatments"
            ],
            "Drug Investigation": [
                "Get comprehensive information about aspirin",
                "What are the properties of pembrolizumab?",
                "Find scientific literature on GLP-1 agonists"
            ]
        }
        
        selected_category = st.selectbox(
            "Category:",
            ["Select a category..."] + list(example_queries.keys())
        )
        
        if selected_category != "Select a category...":
            selected_example = st.selectbox(
                "Example query:",
                [""] + example_queries[selected_category]
            )
            
            if selected_example and st.button("Use This Example"):
                st.session_state.current_query = selected_example
                st.rerun()
        
        st.markdown("---")
        
        # Cache Management
        st.subheader("💾 Cache Management")
        
        cache_stats = cache.get_stats()
        st.metric("Cached Items", cache_stats['total_files'])
        st.metric("Cache Size", f"{cache_stats['total_size_mb']} MB")
        
        if st.button("Clear Cache"):
            deleted = cache.clear()
            st.success(f"Cleared {deleted} cached items")
            st.rerun()
        
        st.markdown("---")
        
        # About Section
        st.subheader("ℹ️ About")
        st.info(f"""
        **Version:** {APP_VERSION}
        
        This AI system uses specialized agents to:
        - 🔬 Search clinical trials
        - 💊 Analyze drug properties  
        - 📚 Review scientific literature
        - 📊 Assess market opportunities
        
        **Data Sources:**
        - ClinicalTrials.gov
        - PubChem
        - PubMed
        - OpenFDA
        """)
        
        # Statistics
        st.markdown("---")
        st.subheader("📈 Session Stats")
        st.metric("Queries This Session", len(st.session_state.research_history))


def main():
    """Main application logic."""
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔬 Research Query")
        
        # Get query from session state if set by example
        default_query = st.session_state.get('current_query', '')
        
        user_query = st.text_area(
            "Enter your pharmaceutical research question:",
            value=default_query,
            height=120,
            placeholder="Example: Find drugs for diabetes that could be repurposed for Alzheimer's disease",
            help="Be specific about the therapeutic area, drug, or condition you want to research"
        )
        
        # Clear the session state query after using it
        if 'current_query' in st.session_state:
            del st.session_state.current_query
        
        analyze_button = st.button("🚀 Start AI Analysis", type="primary")
    
    with col2:
        st.markdown("### 📊 System Overview")
        
        # Metrics in a grid
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🤖 AI Agents", "5")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📡 Data Sources", "4")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metrics_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💾 Cache Items", cache.get_stats()['total_files'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📝 History", len(st.session_state.research_history))
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis execution
    if analyze_button and user_query:
        
        st.markdown("---")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Initialize LLM
            status_text.text("🔧 Initializing AI models...")
            progress_bar.progress(10)
            llm = initialize_llm()
            
            # Step 2: Create agents
            status_text.text("🤖 Creating specialized AI agents...")
            progress_bar.progress(25)
            
            clinical_agent = create_clinical_trials_agent(llm)
            drug_info_agent = create_drug_info_agent(llm)
            literature_agent = create_literature_agent(llm)
            market_agent = create_market_agent(llm)
            master_agent = create_master_agent(llm)
            
            # Step 3: Run research
            status_text.text("🔍 AI agents are conducting research... This may take 60-90 seconds")
            progress_bar.progress(40)
            
            with st.spinner("🧠 Analyzing data from multiple sources..."):
                result = run_research_crew(
                    user_query=user_query,
                    master_agent=master_agent,
                    worker_agents=[
                        clinical_agent,
                        drug_info_agent,
                        literature_agent,
                        market_agent
                    ],
                    llm=llm
                )
            
            # Step 4: Display results
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            st.success("✅ **Analysis Complete!**")
            
            # Display results in formatted box
            st.markdown("### 📄 Research Findings")
            st.markdown(f'<div class="result-box">{result}</div>', unsafe_allow_html=True)
            
            # Save to history
            st.session_state.research_history.append({
                "timestamp": datetime.now(),
                "query": user_query,
                "result": result
            })
            
            # Download options
            st.markdown("### 💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📥 Download as TXT",
                    data=f"Pharma Intelligence AI - Research Report\n\n"
                         f"Query: {user_query}\n"
                         f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                         f"{'='*80}\n\n{result}",
                    file_name=f"pharma_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Create markdown version
                md_content = f"""# Pharma Intelligence AI - Research Report

**Query:** {user_query}  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

{result}

---

*Generated by Pharma Intelligence AI v{APP_VERSION}*
"""
                st.download_button(
                    label="📥 Download as MD",
                    data=md_content,
                    file_name=f"pharma_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            with col3:
                if st.button("🔄 New Query"):
                    st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            
            st.error("❌ **Error during analysis**")
            st.error(str(e))
            
            with st.expander("🔍 View Error Details"):
                st.code(traceback.format_exc())
            
            logger.error(f"Analysis error: {e}\n{traceback.format_exc()}")
    
    elif analyze_button:
        st.warning("⚠️ Please enter a research question before starting analysis")
    
    # Research History Section
    if st.session_state.research_history:
        st.markdown("---")
        st.markdown("### 📚 Research History")
        
        # Show most recent 5 queries
        for idx, item in enumerate(reversed(st.session_state.research_history[-5:])):
            timestamp = item['timestamp'].strftime('%Y-%m-%d %H:%M')
            
            with st.expander(f"🕐 {timestamp} - {item['query'][:70]}..."):
                st.markdown("**Query:**")
                st.info(item['query'])
                
                st.markdown("**Results:**")
                st.markdown(item['result'])
                
                # Download button for historical query
                st.download_button(
                    label="📥 Download This Report",
                    data=f"Query: {item['query']}\n\nDate: {timestamp}\n\n{item['result']}",
                    file_name=f"report_{timestamp.replace(':', '').replace(' ', '_')}.txt",
                    mime="text/plain",
                    key=f"download_{idx}"
                )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}\n{traceback.format_exc()}")