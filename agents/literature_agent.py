"""
Scientific Literature Agent.
Specializes in searching and analyzing published research.
"""

from crewai import Agent
from tools.pubmed_tools import search_pubmed_literature
import logging

logger = logging.getLogger("pharma_ai.agents.literature")

def create_literature_agent(llm):
    """
    Create Literature Research Agent.
    
    Args:
        llm: Language model instance
    
    Returns:
        Configured Agent instance
    """
    logger.info("Creating Literature Agent")
    
    return Agent(
        role="Scientific Literature Analyst",
        goal="Search and analyze scientific publications to find research evidence supporting drug repurposing opportunities",
        backstory="""You are a medical researcher and literature review expert with a PhD in 
        pharmacology. You have published over 50 peer-reviewed papers and excel at:
        
        - Conducting systematic literature reviews
        - Identifying key research findings and trends
        - Analyzing preclinical and clinical study results
        - Evaluating the strength of scientific evidence
        - Finding connections between different areas of research
        
        Your ability to synthesize information from thousands of publications has been 
        instrumental in discovering new therapeutic applications for existing drugs. You 
        can quickly identify promising research directions and gaps in current knowledge 
        that represent opportunities for drug repurposing.""",
        tools=[
            search_pubmed_literature
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    )