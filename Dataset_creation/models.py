"""
Pydantic models for STT evaluation dataset generation.
"""
from typing import List, Optional, Dict, Any, Literal, Union
from enum import Enum
from pydantic import BaseModel, Field
from pathlib import Path
import datetime


class TTSProvider(str, Enum):
    """TTS provider options."""
    ELEVENLABS = "elevenlabs"
    GEMINI = "gemini"


class ConversationTurn(BaseModel):
    """Represents a single turn in a conversation."""
    speaker: str = Field(description="Name or ID of the speaker")
    text: str = Field(description="The spoken text content")
    duration_estimate: Optional[float] = Field(None, description="Estimated duration in seconds")
    voice_characteristics: Optional[str] = Field(None, description="Description of voice characteristics")


class ConversationScenario(BaseModel):
    """Defines a conversation scenario to be generated."""
    scenario_id: str = Field(description="Unique identifier for the scenario")
    title: str = Field(description="Title of the conversation scenario")
    description: str = Field(description="Brief description of the scenario")
    context: str = Field(description="Context or setting for the conversation")
    participants: List[str] = Field(description="List of participant names/roles")
    target_duration: Optional[int] = Field(60, description="Target duration in seconds")
    difficulty_level: Literal["easy", "medium", "hard"] = Field("medium", description="Complexity level")
    language: str = Field("en", description="Language code")
    domain: Optional[str] = Field(None, description="Specific domain (medical, legal, etc.)")


class ConversationMetadata(BaseModel):
    """Metadata for a generated conversation."""
    model_config = {"extra": "forbid"}  # Explicitly forbid additional properties
    
    word_count: Optional[int] = Field(None, description="Total word count")
    turn_count: Optional[int] = Field(None, description="Number of turns")
    avg_turn_length: Optional[float] = Field(None, description="Average turn length in words")


class GeneratedConversation(BaseModel):
    """A complete generated conversation."""
    scenario_id: str
    title: str
    context: str
    turns: List[ConversationTurn]
    metadata: Optional[ConversationMetadata] = Field(default=None, description="Conversation metadata")
    estimated_total_duration: Optional[float] = Field(None)
    generated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)


class VoiceMapping(BaseModel):
    """Maps speakers to voice IDs (ElevenLabs or Gemini voice names)."""
    speaker_name: str
    voice_id: str = Field(description="ElevenLabs voice ID or Gemini voice name")
    voice_name: Optional[str] = None
    voice_description: Optional[str] = None


class ElevenLabsAudioConfiguration(BaseModel):
    """Configuration for audio generation with ElevenLabs v3."""
    model_id: str = Field("eleven_v3", description="ElevenLabs model ID")
    output_format: str = Field("mp3_44100_128", description="Audio output format")
    stability: Optional[float] = Field(0.5, description="Voice stability (0.0-1.0) - Creative: 0.3, Natural: 0.5, Robust: 0.8")
    use_speaker_boost: Optional[bool] = Field(True, description="Boost similarity to original speaker")
    apply_text_normalization: str = Field("auto", description="Text normalization: auto, on, off")
    use_audio_tags: bool = Field(True, description="Enable v3 audio tags for enhanced expression")
    language_code: Optional[str] = Field(None, description="Language code (ISO 639-1)")


class GeminiAudioConfiguration(BaseModel):
    """Configuration for audio generation with Gemini 2.5 TTS."""
    model: str = Field("gemini-2.5-flash-preview-tts", description="Gemini model for TTS")
    output_format: str = Field("wav", description="Audio output format (wav for Gemini)")
    voice_name: Optional[str] = Field(None, description="Default voice name for single-speaker")
    language_code: Optional[str] = Field(None, description="Language code (ISO 639-1)")
    speech_style_prompt: Optional[str] = Field(None, description="Natural language prompt for controlling speech style, tone, accent, pace")


class AudioConfiguration(BaseModel):
    """Unified configuration for audio generation with multiple TTS providers."""
    provider: TTSProvider = Field(TTSProvider.ELEVENLABS, description="TTS provider to use")
    elevenlabs_config: Optional[ElevenLabsAudioConfiguration] = Field(default_factory=ElevenLabsAudioConfiguration, description="ElevenLabs-specific configuration")
    gemini_config: Optional[GeminiAudioConfiguration] = Field(default_factory=GeminiAudioConfiguration, description="Gemini-specific configuration")


class DatasetEntry(BaseModel):
    """A complete dataset entry with transcription and audio."""
    entry_id: str = Field(description="Unique identifier for this dataset entry")
    conversation: GeneratedConversation
    voice_mappings: List[VoiceMapping]
    audio_config: AudioConfiguration
    audio_file_path: Optional[Path] = Field(None, description="Path to generated audio file")
    transcript_file_path: Optional[Path] = Field(None, description="Path to transcript file")
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    stt_evaluation_ready: bool = Field(False, description="Whether this entry is ready for STT evaluation")


class GenerationBatch(BaseModel):
    """Batch of conversations to generate."""
    batch_id: str
    scenarios: List[ConversationScenario]
    voice_mappings: List[VoiceMapping]
    audio_config: AudioConfiguration
    output_directory: Path
    status: Literal["pending", "processing", "completed", "failed"] = Field("pending")
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    completed_entries: List[str] = Field(default_factory=list)
    failed_entries: List[str] = Field(default_factory=list)
