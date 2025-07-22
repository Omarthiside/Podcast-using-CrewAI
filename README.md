# Podcast-using-CrewAI
# AI Podcast Generator

An automated podcast generation system using [CrewAI](https://crewai.com/) and [ElevenLabs](https://elevenlabs.io/). The system reads research papers and creates engaging, conversational podcasts with natural-sounding voices.

You all can watch my video i uploaded on Youtube - (https://www.youtube.com/watch?v=H_OxQnuSmR0)


Generated using AI agents for research analysis, script writing, and voice synthesis.

## Features
- Research paper analysis and summarization
- Natural conversational script generation
- Enhanced script refinement for engagement
- High-quality voice synthesis using ElevenLabs
- Professional audio mixing and processing

## Setup
1. Clone this repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   ELEVENLABS_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   ```

## Usage
1. Place your research paper in the `knowledge/` directory or use the one provided in the repository
2. Run the agents script:
   ```bash
   python podcast_generator.py
   ```
3. Find outputs in the `outputs/` directory:
   - Generated scripts
   - Audio segments
   - Final podcast

## Configuration
- Voice settings can be adjusted in `tools.py`
- Agent behaviors configured in `podcast_generator.py`

## TODO
- [ ] Add Docling to perform RAG on various file types
- [ ] Add multilingual support
- [ ] Add search functionality to reaserch supporting materials for the podcast

## Acknowledgements
- [CrewAI](https://crewai.com/)
- [ElevenLabs](https://elevenlabs.io/)
