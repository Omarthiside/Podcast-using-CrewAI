import os
import traceback
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
from pydub import AudioSegment
from crewai.tools import BaseTool
from pydantic import Field, BaseModel, ConfigDict
from elevenlabs.client import ElevenLabs

class VoiceConfig(BaseModel):
    """Voice configuration settings."""
    stability: float = 0.45  # Slightly lower for more natural variation
    similarity_boost: float = 0.85  # Higher to maintain consistent voice character
    style: float = 0.65  # Balanced expressiveness
    use_speaker_boost: bool = True
    model_id: str = "eleven_multilingual_v2"
    output_format: str = "mp3_44100_128"
    apply_text_normalization: str = "auto"  # 'auto', 'on', or 'off'

class AudioConfig(BaseModel):
    """Audio processing configuration."""
    format: str = "mp3"
    sample_rate: int = 48000  # Higher for better quality
    channels: int = 2
    bitrate: str = "256k"     # Higher bitrate for clearer audio
    normalize: bool = True    # Normalize audio levels
    target_loudness: float = -14.0  # Standard podcast loudness (LUFS)
    compression_ratio: float = 2.0   # Light compression for voice

class Dialogue(BaseModel):
    """Dialogue for the podcast audio generation tool."""
    speaker: str
    text: str

class PodcastAudioGeneratorInput(BaseModel):
    """Input for the podcast audio generation tool."""
    dialogue: List[Dialogue]

class PodcastAudioGenerator(BaseTool):
    """Enhanced podcast audio generation tool."""
    
    name: str = "PodcastAudioGenerator"
    description: str = "Synthesizes podcast voices using ElevenLabs API."
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    api_key: str = Field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY"))
    voice_configs: Dict[str, Dict] = Field(default_factory=dict)
    audio_config: AudioConfig = Field(default_factory=AudioConfig)
    output_dir: str = Field(default="output/audio-files")
    client: Any = Field(default=None)
    args_schema: Type[BaseModel] = PodcastAudioGeneratorInput

    def __init__(self, **data):
        super().__init__(**data)
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")
        self.client = ElevenLabs(api_key=self.api_key)

    def add_voice(self, name: str, voice_id: str, config: Optional[VoiceConfig] = None) -> None:
        """Add a voice configuration."""
        self.voice_configs[name] = {
            "voice_id": voice_id,
            "config": config or VoiceConfig()
        }

    def _run(self, dialogue: List[Dialogue]) -> List[str]:
        """Generate audio files for each script segment."""
        os.makedirs(self.output_dir, exist_ok=True)

        audio_files = []
        for index, segment in enumerate(dialogue):
            speaker = segment.get('speaker', '').strip()
            text = segment.get('text', '').strip()
            
            if not speaker or not text:
                print(f"Skipping segment {index}: missing speaker or text")
                continue

            voice_config = self.voice_configs.get(speaker)
            if not voice_config:
                print(f"Skipping unknown speaker: {speaker}")
                continue

            try:
                audio_generator = self.client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_config["voice_id"],
                    model_id=voice_config['config'].model_id,
                    output_format=voice_config['config'].output_format,
                    voice_settings={
                        "stability": voice_config['config'].stability,
                        "similarity_boost": voice_config['config'].similarity_boost,
                        "style": voice_config['config'].style,
                        "use_speaker_boost": voice_config['config'].use_speaker_boost
                    }
                )

                # Convert generator to bytes
                audio_bytes = b''.join(chunk for chunk in audio_generator)

                filename = f"{self.output_dir}/{index:03d}_{speaker}.{self.audio_config.format}"
                with open(filename, "wb") as out:
                    out.write(audio_bytes)

                # Basic audio normalization
                if self.audio_config.normalize:
                    audio = AudioSegment.from_file(filename)
                    normalized = audio.normalize()  # Simple normalization
                    normalized = normalized + 4  # Slight boost
                    
                    # Use context manager to ensure file is closed
                    with normalized.export(
                        filename,
                        format=self.audio_config.format,
                        bitrate=self.audio_config.bitrate,
                        parameters=["-ar", str(self.audio_config.sample_rate)]
                    ) as f:
                        f.close()

                audio_files.append(filename)
                print(f'Audio content written to file "{filename}"')

            except Exception as e:
                print(f"Error processing segment {index}: {str(e)}")
                continue

        return sorted(audio_files)

class PodcastMixer(BaseTool):
    """Enhanced audio mixing tool for podcast production."""
    
    name: str = "PodcastMixer"
    description: str = "Mixes multiple audio files with effects into final podcast."
    
    audio_config: AudioConfig = Field(default_factory=AudioConfig)
    output_dir: str = Field(default="output/podcast")

    def _run(
        self,
        audio_files: List[str],
        crossfade: int = 50
    ) -> str:
        if not audio_files:
            print("DEBUG: No audio files provided to mix.")
            return "" # Return empty string instead of raising, so crewai can continue

        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"DEBUG: Output directory confirmed: {self.output_dir}")
            
            # --- IMPORTANT DEBUGGING ADDITION ---
            # Verify if segment files exist before attempting to load them
            validated_audio_files = []
            for af in audio_files:
                # Construct an absolute path for checking, assuming segments are relative to CWD if not absolute
                segment_path = af if os.path.isabs(af) else os.path.join(os.getcwd(), af)
                if not os.path.exists(segment_path):
                    print(f"ERROR: Segment file does not exist at path: {segment_path}. (Original received: {af})")
                    return "" # If a segment is missing, mixing cannot proceed
                validated_audio_files.append(segment_path)
            # --- END DEBUGGING ADDITION ---

            mixed = AudioSegment.from_file(validated_audio_files[0])
            print(f"DEBUG: Initial segment loaded: {validated_audio_files[0]}")

            for i, audio_file in enumerate(validated_audio_files[1:]):
                next_segment = AudioSegment.from_file(audio_file)
                # Add silence and use crossfade
                silence = AudioSegment.silent(duration=200)
                next_segment = silence + next_segment
                mixed = mixed.append(next_segment, crossfade=crossfade)
                print(f"DEBUG: Appended segment {i+2}: {audio_file}")

            # Define the final output file path using the configured output_dir
            output_file = os.path.join(self.output_dir, "podcast_final.mp3")
            print(f"DEBUG: Attempting to export to: {output_file}")
            
            mixed.export(
                output_file,
                format="mp3",
                parameters=[
                    "-q:a", "0",  # Highest quality
                    "-ar", "48000"  # Professional sample rate
                ]
            )

            print(f"DEBUG: Export completed successfully.")
            print(f"Successfully mixed podcast to: {output_file}")
            return output_file

        except Exception as e:
            # Print a prominent error message and the full traceback
            print(f"CRITICAL ERROR IN PODCASTMIXER: {str(e)}")
            traceback.print_exc() # This will show you exactly where the error occurred in the stack
            return ""
