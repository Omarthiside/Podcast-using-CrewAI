from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
from dotenv import load_dotenv
import os
from tools import PodcastAudioGenerator, PodcastMixer, VoiceConfig


def setup_directories():
    """Set up organized directory structure"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    dirs = {
        'BASE': f'outputs/{timestamp}',
        'SEGMENTS': f'outputs/{timestamp}/segments',  # Individual voice segments
        'FINAL': f'outputs/{timestamp}/podcast',      # Final podcast file
        'DATA': f'outputs/{timestamp}/data'          # Metadata/JSON files
    }
    
    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    return dirs


load_dotenv()

research_paper = PDFKnowledgeSource(file_paths="workplace-prod.pdf")


class PaperSummary(BaseModel):
    """Summary of a research paper."""
    title: str = Field(..., description="Title of the research paper")                   
    main_findings: List[str] = Field(..., description="Key findings as a list of strings")
    methodology: str = Field(..., description="Research methodology as a single text block")
    key_implications: List[str] = Field(..., description="Implications as a list of strings")
    limitations: List[str] = Field(..., description="Limitations as a list of strings")
    future_work: List[str] = Field(..., description="Future research directions as a list")
    summary_date: datetime = Field(..., description="Timestamp of summary creation")

class DialogueLine(BaseModel):
    """Dialogue line for a podcast script."""
    speaker: str = Field(..., description="Name of the speaker (Julia or Guido)")
    text: str = Field(..., description="The actual dialogue line")

class PodcastScript(BaseModel):
    """Podcast script with dialogue lines."""
    dialogue: List[DialogueLine] = Field(..., description="Ordered list of dialogue lines")

class AudioGeneration(BaseModel):
    """Audio generation result with metadata."""
    segment_files: List[str] = Field(..., description="List of generated audio segment files")
    final_podcast: str = Field(..., description="Path to the final mixed podcast file")

# --- LLM Setup ---

summary_llm = LLM(
    model="openai/gpt-4.1",
    temperature=0.0,
)

script_llm = LLM(
    model="openai/gpt-4.1-mini", 
    temperature=0.3,
)

script_enhancer_llm = LLM(
    model="openai/gpt-4.1",
    temperature=0.7,
)
audio_llm = LLM(
    model="cerebras/llama3.3-70b",
    temperature=0.0,
)


dirs = setup_directories()
audio_generator = PodcastAudioGenerator(output_dir=dirs['SEGMENTS'])


audio_generator.add_voice(
    "Julia", 
    os.getenv("LAURA_VOICE_ID"),
    VoiceConfig(
        stability=0.35, 
        similarity_boost=0.75,  
        style=0.65,  
        use_speaker_boost=True
    )
)


audio_generator.add_voice(
    "Guido", 
    os.getenv("LIAM_VOICE_ID"),
    VoiceConfig(
        stability=0.4,
        similarity_boost=0.75,
        style=0.6, 
        use_speaker_boost=True
    )
)

podcast_mixer = PodcastMixer(output_dir=dirs['FINAL'])
search_tool = SerperDevTool()


# --- Agents ---
researcher = Agent(
    role="Research Analyst",
    goal="Create comprehensive yet accessible research paper summaries",
    backstory="""You're a PhD researcher with a talent for breaking down complex
    academic papers into clear, understandable summaries. You excel at identifying
    key findings and their real-world implications.""",
    verbose=True,
    llm=summary_llm
)

research_support = Agent(
    role="Research Support Specialist",
    goal="Find current context and supporting materials relevant to the paper's topic",
    backstory="""You're a versatile research assistant who excels at finding 
    supplementary information across academic fields. You have a talent for 
    connecting academic research with real-world applications, current events, 
    and practical examples, regardless of the field. You know how to find 
    credible sources and relevant discussions across various domains.""",
    verbose=True,
    tools=[search_tool],
    llm=script_enhancer_llm
)

script_writer = Agent(
    role="Podcast Script Writer",
    goal="Create engaging and educational podcast scripts about technical topics",
    backstory="""You're a skilled podcast writer who specializes in making technical 
    content engaging and accessible. You create natural dialogue between two hosts: 
    Julia (a knowledgeable expert who explains concepts clearly) and Guido (an informed 
    co-host who asks thoughtful questions and helps guide the discussion).""",
    verbose=True,
    llm=script_llm
)

script_enhancer = Agent(
    role="Podcast Script Enhancer",
    goal="Enhance podcast scripts to be more engaging while maintaining educational value",
    backstory="""You're a veteran podcast producer who specializes in making technical 
    content both entertaining and informative. You excel at adding natural humor, 
    relatable analogies, and engaging banter while ensuring the core technical content 
    remains accurate and valuable. You've worked on shows like Lex Fridman's podcast, 
    Hardcore History, and the Joe Rogan Experience, bringing their signature blend of 
    entertainment and education.""",
    verbose=True,
    llm=script_enhancer_llm 
)

audio_generator_agent = Agent(
    role="Audio Generation Specialist",
    goal="Generate high-quality podcast audio with natural-sounding voices",
    backstory="""You are an expert in audio generation and processing. You understand 
    how to generate natural-sounding voices and create professional podcast audio. You 
    consider pacing, tone, and audio quality in your productions.""",
    verbose=True,
    allow_delegation=False,
    tools=[audio_generator, podcast_mixer],
    llm=audio_llm
)

# --- Tasks ---
summary_task = Task(
    description="""Read and analyze the provided research paper: {paper}.
    
    Create a comprehensive summary that includes:
    1. Main findings and conclusions
    2. Methodology overview
    3. Key implications for the field
    4. Limitations of the study
    5. Suggested future research directions
    
    Make the summary accessible to an educated general audience while maintaining accuracy.""",
    expected_output="A structured summary of the research paper with all key components.",
    agent=researcher,
    output_pydantic=PaperSummary,
    output_file="output/metadata/paper_summary.json"
)

supporting_research_task = Task(
    description="""After analyzing the paper summary, find recent and relevant supporting 
    materials that add context and real-world perspective to the topic.
    
    Research Approach:
    1. Topic Analysis:
        • Identify key themes and concepts from the paper
        • Determine related fields and applications
        • Note any specific claims or findings to verify
    
    2. Current Context:
        • Recent developments in the field
        • Latest practical applications
        • Industry or field-specific news
        • Related ongoing research
    
    3. Supporting Evidence:
        • Academic discussions and debates
        • Industry reports and white papers
        • Professional forum discussions
        • Conference presentations
        • Expert opinions and analyses
    
    4. Real-world Impact:
        • Practical implementations
        • Case studies
        • Success stories or challenges
        • Market or field adoption
    
    5. Different Perspectives:
        • Alternative approaches
        • Critical viewpoints
        • Cross-disciplinary applications
        • Regional or cultural variations
    
    Focus on finding information that:
    • Is recent (preferably within last 2 years)
    • Comes from credible sources
    • Adds valuable context to the paper's topic
    • Provides concrete examples or applications
    • Offers different viewpoints or approaches""",
    expected_output="A structured collection of relevant supporting materials and examples",
    agent=research_support,
    context=[summary_task],
    output_file="output/metadata/supporting_research.json"
)

podcast_task = Task(
    description="""Using the paper summary and supporting research, create an engaging and informative podcast conversation 
    between Julia and Guido. Make it feel natural while clearly distinguishing between paper findings and supplementary research.

    Source Attribution Guidelines:
    • For Paper Content:
        - "According to the paper..."
        - "The researchers found that..."
        - "In their study, they discovered..."
        - "The paper's methodology showed..."
    
    • For Supporting Research:
        - "I recently read about..."
        - "There's some interesting related work by..."
        - "This reminds me of a recent case study..."
        - "Building on this, other researchers have found..."

    Host Dynamics:
    - Julia: A knowledgeable but relatable expert who:
        • Explains technical concepts with enthusiasm
        • Sometimes playfully challenges Guido's assumptions
        • Clearly distinguishes between paper findings and broader context
        • Occasionally plays devil's advocate on certain points
        • Admits when she's uncertain about specific aspects
        • Shares relevant personal experiences with AI and tech
        • Can connect the research to broader tech trends
        • Uses casual expressions and shows genuine excitement
    
    - Guido: An engaged and curious co-host who:
        • Asks insightful questions and follows interesting threads
        • Occasionally disagrees based on his practical experience
        • Brings up relevant external examples and research
        • Respectfully pushes back on theoretical claims with real-world examples
        • Helps find middle ground in discussions
        • Helps make connections to practical applications
        • Naturally guides the conversation back to main topics

    Example Flow with Attribution:
    Julia: "The paper's findings show that RAG is superior for factual queries."
    Guido: "That's interesting, because I recently read about a case study where..."
    Julia: "Oh, that's a great point! While the researchers found X, these real-world examples show Y..."

    Disagreement Guidelines:
    • Keep disagreements friendly and constructive
    • Use phrases like:
        - "I see what the paper suggests, but in practice..."
        - "While the study found X, other research shows..."
        - "That's an interesting finding, though recent developments suggest..."
    • Always find common ground or learning points
    • Use disagreements to explore nuances
    • Resolve differences with mutual understanding

    Conversation Flow:
    1. Core Discussion: Focus on the research and findings
    2. Natural Tangents with Clear Attribution:
        • "Building on the paper's findings..."
        • "This relates to some recent developments..."
        • "While not covered in the paper, there's interesting work on..."
    3. Smooth Returns: Natural ways to bring the conversation back:
        • "Coming back to what the researchers found..."
        • "This actually connects to the paper's methodology..."
        • "That's a great example of what the study was trying to solve..."

    Writing Guidelines:
    1. Clearly distinguish paper findings from supplementary research
    2. Use attribution phrases naturally within the conversation
    3. Connect different sources of information meaningfully
    4. Keep technical content accurate but conversational
    5. Maintain engagement through relatable stories
    6. Include occasional friendly disagreements
    7. Show how different perspectives and sources enrich understanding
    
    Note: Convey reactions through natural language rather than explicit markers like *laughs*.""",
    expected_output="A well-balanced podcast script that clearly distinguishes between paper content and supplementary research.",
    agent=script_writer,
    context=[summary_task, supporting_research_task],
    output_pydantic=PodcastScript,
    output_file="output/metadata/podcast_script.json"
)

enhance_script_task = Task(
    description="""Take the initial podcast script and enhance it to be more engaging 
    and conversational while maintaining its educational value.
    
    IMPORTANT RULES:
    1. NEVER change the host names - always keep Julia and Guido exactly as they are
    2. NEVER add explicit reaction markers like *chuckles*, *laughs*, etc.
    3. NEVER add new hosts or characters
    
    Enhancement Guidelines:
    1. Add Natural Elements:
        • Include natural verbal reactions ("Oh that's fascinating", "Wow", etc.)
        • Keep all dialogue between Julia and Guido only
        • Add relevant personal anecdotes or examples that fit their established roles:
            - Julia as the knowledgeable expert
            - Guido as the engaged and curious co-host
        • Express reactions through words rather than action markers
    
    2. Improve Flow:
        • Ensure smooth transitions between topics
        • Add brief casual exchanges that feel natural
        • Include moments of reflection or connection-making
        • Balance technical depth with accessibility
    
    3. Maintain Quality:
        • Keep all technical information accurate
        • Ensure added content supports rather than distracts
        • Preserve the core findings and insights
        • Keep the overall length reasonable
    
    4. Add Engagement Techniques:
        • Include thought-provoking analogies by both hosts
        • Add relatable real-world examples
        • Express enthusiasm through natural dialogue
        • Include collaborative problem-solving moments
        • Inject humor where appropriate and it has to be funny

    Natural Reaction Examples:
    ✓ RIGHT: "Oh, that's fascinating!"
    ✓ RIGHT: "Wait, that doesn't make sense!"
    ✓ RIGHT: "Wait, really? I hadn't thought of it that way."
    ✓ RIGHT: "That's such a great point."
    ✗ WRONG: *chuckles* or *laughs* or any other action markers
    ✗ WRONG: Adding new speakers or changing host names
    
    The goal is to make the content feel like a conversation between Julia and Guido
    who are genuinely excited about the topic, while ensuring listeners learn 
    something valuable.""",
    expected_output="An enhanced version of the podcast script that's more engaging and natural",
    agent=script_enhancer,
    context=[summary_task, podcast_task],
    output_pydantic=PodcastScript,
    output_file="output/metadata/enhanced_podcast_script.json"
)

audio_task = Task(
    description="""Generate high-quality audio for the podcast script and create the final podcast.
    
    The script will be provided in the context as a list of dialogue entries, each with:
    - speaker: Either "Julia" or "Guido"
    - text: The line to be spoken
    
    Process:
    1. Generate natural-sounding audio for each line of dialogue using appropriate voices
    2. Apply audio processing for professional quality:
       - Normalize audio levels
       - Add subtle fade effects between segments
       - Apply appropriate pacing and pauses
    3. Mix all segments into a cohesive final podcast
    
    Voice Assignments:
    - For Julia's lines: Use configured Julia voice
    - For Guido's lines: Use configured Guido voice
    
    Quality Guidelines:
    - Ensure consistent audio levels across all segments
    - Maintain natural pacing and flow
    - Create smooth transitions between speakers
    - Verify audio clarity and quality""",
    expected_output="A professional-quality podcast audio file with natural-sounding voices and smooth transitions",
    agent=audio_generator_agent,
    context=[enhance_script_task],
    output_pydantic=AudioGeneration,
    output_file="output/metadata/audio_generation_meta.json"
)

# --- Crew and Process ---
crew = Crew(
    agents=[researcher, research_support, script_writer, script_enhancer, audio_generator_agent],
    tasks=[summary_task, supporting_research_task, podcast_task, enhance_script_task, audio_task],
    process=Process.sequential,
    knowledge_sources=[research_paper],
    verbose=True
)

if __name__ == "__main__":    
    # Update task output files
    summary_task.output_file = os.path.join(dirs['DATA'], "paper_summary.json")
    supporting_research_task.output_file = os.path.join(dirs['DATA'], "supporting_research.json")
    podcast_task.output_file = os.path.join(dirs['DATA'], "podcast_script.json")
    enhance_script_task.output_file = os.path.join(dirs['DATA'], "enhanced_podcast_script.json")
    audio_task.output_file = os.path.join(dirs['DATA'], "audio_generation_meta.json")
    
    # Run the podcast generation process
    results = crew.kickoff(inputs={"paper": "workplace-prod.pdf"})
