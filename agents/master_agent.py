"""
Master Orchestrator Agent.
Coordinates research across specialist agents.
"""

from crewai import Agent, Task, Crew, Process
from typing import List
import logging

logger = logging.getLogger("pharma_ai.agents.master")

def create_master_agent(llm):
    """
    Create Master Orchestrator Agent.
    
    Args:
        llm: Language model instance
    
    Returns:
        Configured Agent instance
    """
    logger.info("Creating Master Agent")
    
    return Agent(
        role="Pharmaceutical Research Coordinator",
        goal="Synthesize and coordinate research findings from multiple specialist agents into comprehensive reports",
        backstory="""You are a senior pharmaceutical strategist and research director with 
        20+ years of experience leading drug development programs. You excel at:
        
        - Synthesizing complex information from multiple sources
        - Identifying high-value drug repurposing opportunities
        - Evaluating opportunities from clinical, regulatory, and commercial perspectives
        - Making strategic recommendations based on comprehensive analysis
        
        You coordinate specialist teams to gather information, then synthesize their findings 
        into clear, actionable recommendations. Your reports are known for being thorough 
        yet concise, highlighting key opportunities and potential risks.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,  # Disabled to prevent recursion issues
        max_iter=5  # Reduced to prevent infinite loops
    )


def run_research_crew(
    user_query: str,
    master_agent: Agent,
    worker_agents: List[Agent],
    llm
) -> str:
    """
    Run the multi-agent research crew.
    
    Args:
        user_query: User's research question
        master_agent: Orchestrator agent
        worker_agents: List of specialist agents
        llm: Language model
    
    Returns:
        Research findings as a formatted string
    """
    logger.info(f"Starting research crew for query: {user_query[:100]}...")
    
    # Create separate tasks for each specialist agent to avoid delegation issues
    clinical_task = Task(
        description=f"""
        Research Question: "{user_query}"
        
        Conduct clinical trials research:
        - Search for relevant clinical trials related to this question
        - Identify drugs being tested for related conditions
        - Find gaps or opportunities in current research
        - List key trial information including NCT IDs, phases, and statuses
        
        Focus on finding specific, actionable information about clinical trials.
        """,
        agent=worker_agents[0],  # Clinical trials agent
        expected_output="A detailed report on relevant clinical trials with specific trial IDs, drugs tested, and key findings"
    )
    
    drug_info_task = Task(
        description=f"""
        Research Question: "{user_query}"
        
        Gather comprehensive drug information:
        - Get detailed properties of relevant drugs mentioned or found in research
        - Review FDA approval status and indications
        - Assess chemical and pharmacological characteristics
        - Identify key drug properties relevant to the research question
        
        Focus on specific drugs relevant to the research question.
        """,
        agent=worker_agents[1],  # Drug info agent
        expected_output="Detailed drug information including properties, FDA status, and relevant characteristics"
    )
    
    literature_task = Task(
        description=f"""
        Research Question: "{user_query}"
        
        Search scientific literature:
        - Find recent research publications related to this question
        - Identify emerging evidence for new applications
        - Highlight key scientific findings and research trends
        - Note publication dates and authors when relevant
        
        Focus on finding recent, relevant scientific publications.
        """,
        agent=worker_agents[2],  # Literature agent
        expected_output="A summary of relevant scientific literature with key findings and publication details"
    )
    
    market_task = Task(
        description=f"""
        Research Question: "{user_query}"
        
        Analyze market intelligence:
        - Analyze market size and growth potential for relevant therapeutic areas
        - Evaluate competitive landscape
        - Assess commercial viability
        - Identify market opportunities
        
        Focus on market insights relevant to the research question.
        """,
        agent=worker_agents[3],  # Market agent
        expected_output="Market analysis including market size, competition, and commercial viability assessment"
    )
    
    # Final synthesis task for master agent (depends on all worker tasks)
    synthesis_task = Task(
        description=f"""
        Research Question: "{user_query}"
        
        Synthesize all research findings from the specialist agents into a comprehensive report.
        
        Review the findings from:
        1. Clinical Trials Specialist - who searched for relevant trials
        2. Drug Information Specialist - who gathered drug properties
        3. Scientific Literature Analyst - who found research publications
        4. Market Intelligence Analyst - who analyzed market opportunities
        
        Create a comprehensive report that includes:
        - Executive Summary with key findings and recommendations
        - Clinical Trials Insights from the trials specialist
        - Drug Information Analysis from the drug information specialist
        - Scientific Evidence Review from the literature analyst
        - Market Analysis from the market intelligence analyst
        - Final Recommendations with supporting rationale
        
        Format your response clearly and professionally for pharmaceutical executives.
        """,
        agent=master_agent,
        expected_output="""A comprehensive research report with:
        
        EXECUTIVE SUMMARY
        - Brief overview of key findings and recommendations
        
        CLINICAL TRIALS INSIGHTS
        - Relevant trials found with details
        
        DRUG INFORMATION
        - Properties and regulatory status
        
        SCIENTIFIC EVIDENCE
        - Key publications and findings
        
        MARKET ANALYSIS
        - Market size, competition, and viability
        
        RECOMMENDATIONS
        - Prioritized opportunities with rationale""",
        context=[clinical_task, drug_info_task, literature_task, market_task]  # Depends on worker tasks
    )
    
    # Create crew with all agents and tasks
    # Tasks execute in order: workers first, then synthesis
    crew = Crew(
        agents=[master_agent] + worker_agents,
        tasks=[clinical_task, drug_info_task, literature_task, market_task, synthesis_task],
        verbose=True,
        max_iter=20,  # Increased for multiple sequential tasks
        max_rpm=15,  # Rate limiting to prevent API exhaustion
        process=Process.sequential  # Run tasks sequentially to avoid conflicts
    )
    
    try:
        # Execute research
        logger.info("Executing research crew...")
        result = crew.kickoff()
        
        logger.info("Research crew completed successfully")
        return str(result)
    
    except Exception as e:
        logger.error(f"Error in research crew execution: {e}")
        return f"Error during research: {str(e)}\n\nPlease try rephrasing your query or contact support."