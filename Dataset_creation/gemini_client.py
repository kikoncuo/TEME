"""
Gemini client integration for converting text to speech using Gemini 2.5 TTS.
"""
import os
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
import wave
import io
from google import genai
from google.genai import types

from models import GeneratedConversation, VoiceMapping, GeminiAudioConfiguration, ConversationTurn


class GeminiAudioGenerator:
    """Generates audio from text using Google Gemini 2.5 TTS API."""

    # Gemini's supported voice options
    SUPPORTED_VOICES = {
        "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
        "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
        "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
        "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
        "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required (set GOOGLE_API_KEY environment variable)")

        self.client = genai.Client(api_key=self.api_key)

    def _build_style_prompt(self, voice_mappings: List[VoiceMapping], language_code: Optional[str]) -> Optional[str]:
        """Build a speech style prompt from voice descriptions in mappings and language accent hints."""
        lines = []
        if language_code:
            if language_code.lower().startswith("es"):
                lines.append("Overall: European Spanish (es-ES), peninsular accent.")
            elif language_code.lower().startswith("en"):
                lines.append("Overall: English (en-US) neutral broadcast accent.")

        for mapping in voice_mappings:
            if getattr(mapping, "voice_description", None):
                lines.append(f"- {mapping.speaker_name}: {mapping.voice_description}")
        if not lines:
            return None
        return "Use the following speech styles per speaker (tone, accent, pace, emotion):\n" + "\n".join(lines)

    def _create_wave_file(self, filename: Path, pcm_data: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2):
        """Create a WAV file from PCM data."""
        with wave.open(str(filename), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)

    def _format_multi_speaker_prompt(self, conversation: GeneratedConversation, voice_mappings: List[VoiceMapping]) -> str:
        """Format conversation turns into a multi-speaker prompt for Gemini."""
        voice_map = {mapping.speaker_name: mapping.voice_id for mapping in voice_mappings}

        # Build the conversation prompt
        prompt_parts = ["TTS the following conversation:"]

        for turn in conversation.turns:
            speaker = turn.speaker
            voice_name = voice_map.get(speaker, "Kore")  # Default to Kore if not found

            # Validate voice name
            if voice_name not in self.SUPPORTED_VOICES:
                print(f"Warning: Voice '{voice_name}' not supported by Gemini, using 'Kore'")
                voice_name = "Kore"

            text = turn.text.strip()
            if text:
                prompt_parts.append(f"{speaker}: {text}")

        return "\n".join(prompt_parts)

    def _get_speaker_voice_configs(self, voice_mappings: List[VoiceMapping], speakers_in_conversation: set) -> List[types.SpeakerVoiceConfig]:
        """Create speaker voice configurations for multi-speaker TTS."""
        configs = []

        for mapping in voice_mappings:
            # Only create config for speakers that are actually in the conversation
            if mapping.speaker_name not in speakers_in_conversation:
                continue

            voice_name = mapping.voice_id
            if voice_name not in self.SUPPORTED_VOICES:
                print(f"Warning: Voice '{voice_name}' not supported by Gemini, using 'Kore'")
                voice_name = "Kore"

            config = types.SpeakerVoiceConfig(
                speaker=mapping.speaker_name,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            )
            configs.append(config)

        return configs

    def generate_conversation_audio(
        self,
        conversation: GeneratedConversation,
        voice_mappings: List[VoiceMapping],
        config: GeminiAudioConfiguration,
        output_path: Path
    ) -> Path:
        """Generate complete conversation audio using Gemini 2.5 TTS API."""

        # Create voice mapping dictionary
        voice_map = {mapping.speaker_name: mapping.voice_id for mapping in voice_mappings}

        # Validate voices
        for mapping in voice_mappings:
            if mapping.voice_id not in self.SUPPORTED_VOICES:
                print(f"Warning: Voice '{mapping.voice_id}' not supported by Gemini, will use fallback")

        # Determine if this is single-speaker or multi-speaker
        unique_speakers = set(turn.speaker for turn in conversation.turns)

        # Prefer explicit speech_style_prompt; otherwise derive from voice descriptions and language
        effective_style_prompt = config.speech_style_prompt or self._build_style_prompt(voice_mappings, config.language_code)
        if effective_style_prompt:
            # Clone config with injected style prompt to avoid mutating caller's instance
            config = GeminiAudioConfiguration(**config.model_dump())
            config.speech_style_prompt = effective_style_prompt

        if len(unique_speakers) == 1:
            # Single-speaker conversation
            return self._generate_single_speaker_audio(conversation, voice_mappings, config, output_path)
        elif len(unique_speakers) == 2:
            # Multi-speaker conversation with exactly 2 speakers
            return self._generate_multi_speaker_audio(conversation, voice_mappings, config, output_path, unique_speakers)
        else:
            # More than 2 speakers - Gemini only supports exactly 2 speakers
            # Fall back to individual single-speaker generations and combine
            print(f"Warning: Gemini multi-speaker mode requires exactly 2 speakers, but found {len(unique_speakers)}. Using fallback method.")
            return self._generate_multi_speaker_fallback(conversation, voice_mappings, config, output_path)

    def _generate_multi_speaker_fallback(
        self,
        conversation: GeneratedConversation,
        voice_mappings: List[VoiceMapping],
        config: GeminiAudioConfiguration,
        output_path: Path
    ) -> Path:
        """Fallback method for conversations with more than 2 speakers using individual single-speaker generations."""
        from pydub import AudioSegment
        import tempfile
        import os

        voice_map = {mapping.speaker_name: mapping.voice_id for mapping in voice_mappings}
        audio_segments = []

        # Group turns by speaker
        speaker_turns = {}
        for turn in conversation.turns:
            speaker = turn.speaker
            if speaker not in speaker_turns:
                speaker_turns[speaker] = []
            speaker_turns[speaker].append(turn)

        # Generate audio for each speaker's turns separately
        with tempfile.TemporaryDirectory() as temp_dir:
            for speaker, turns in speaker_turns.items():
                if speaker not in voice_map:
                    print(f"Warning: No voice mapping found for speaker '{speaker}', skipping")
                    continue

                # Combine all turns for this speaker into one text
                combined_text = " ".join(turn.text for turn in turns)

                # Create a temporary single-speaker conversation
                temp_conversation = GeneratedConversation(
                    scenario_id=conversation.scenario_id,
                    title=conversation.title,
                    context=conversation.context,
                    turns=[ConversationTurn(speaker=speaker, text=combined_text)],
                    metadata=conversation.metadata,
                    estimated_total_duration=conversation.estimated_total_duration,
                    generated_at=conversation.generated_at
                )

                # Generate single-speaker audio
                temp_output = Path(temp_dir) / f"{speaker}_audio.wav"
                try:
                    self._generate_single_speaker_audio(
                        temp_conversation,
                        [VoiceMapping(speaker_name=speaker, voice_id=voice_map[speaker])],
                        config,
                        temp_output
                    )

                    # Load and add to segments
                    if temp_output.exists():
                        segment = AudioSegment.from_wav(str(temp_output))
                        audio_segments.append(segment)

                        # Add pause between speakers (except for the last one)
                        if speaker != list(speaker_turns.keys())[-1]:
                            pause = AudioSegment.silent(duration=800)  # 800ms pause
                            audio_segments.append(pause)

                except Exception as e:
                    print(f"Error generating audio for speaker '{speaker}': {e}")
                    continue

        if not audio_segments:
            raise ValueError("No audio segments were generated for any speakers")

        # Combine all segments
        final_audio = AudioSegment.empty()
        for segment in audio_segments:
            final_audio += segment

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export as WAV first
        wav_path = output_path.with_suffix('.wav')
        final_audio.export(str(wav_path), format="wav")

        # Convert to MP3 if needed for compatibility
        if config.output_format == "mp3":
            final_audio.export(str(output_path), format="mp3")
            return output_path
        else:
            return wav_path

    def _generate_single_speaker_audio(
        self,
        conversation: GeneratedConversation,
        voice_mappings: List[VoiceMapping],
        config: GeminiAudioConfiguration,
        output_path: Path
    ) -> Path:
        """Generate audio for single-speaker conversation."""

        # Combine all turns into one text
        full_text = " ".join(turn.text for turn in conversation.turns)

        # Get voice name (use first mapping or config default)
        voice_name = config.voice_name
        if not voice_name and voice_mappings:
            voice_name = voice_mappings[0].voice_id

        if not voice_name or voice_name not in self.SUPPORTED_VOICES:
            voice_name = "Kore"  # Default fallback
            print(f"Warning: Using default voice 'Kore' for single-speaker audio")

        # Build the prompt with style instructions if provided
        if config.speech_style_prompt:
            prompt = f"{config.speech_style_prompt}\n\n{full_text}"
        else:
            prompt = full_text

        try:
            debug_cfg = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    language_code=config.language_code,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
            )
            print("[Gemini][Single] model=", config.model)
            print("[Gemini][Single] language_code=", config.language_code)
            print("[Gemini][Single] voice_name=", voice_name)
            if config.speech_style_prompt:
                print("[Gemini][Single] speech_style_prompt=", (config.speech_style_prompt[:300] + '...') if len(config.speech_style_prompt) > 300 else config.speech_style_prompt)
            print("[Gemini][Single] prompt=", (prompt[:500] + '...') if len(prompt) > 500 else prompt)
            response = self.client.models.generate_content(
                model=config.model,
                contents=prompt,
                config=debug_cfg,
            )

            # Extract audio data
            audio_data = response.candidates[0].content.parts[0].inline_data.data

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as WAV file
            self._create_wave_file(output_path.with_suffix('.wav'), audio_data)

            # Convert to MP3 if needed for compatibility
            if config.output_format == "mp3":
                self._convert_wav_to_mp3(output_path.with_suffix('.wav'), output_path)

            return output_path

        except Exception as e:
            print(f"Error generating single-speaker audio with Gemini: {e}")
            raise

    def _generate_multi_speaker_audio(
        self,
        conversation: GeneratedConversation,
        voice_mappings: List[VoiceMapping],
        config: GeminiAudioConfiguration,
        output_path: Path,
        speakers_in_conversation: set
    ) -> Path:
        """Generate audio for multi-speaker conversation."""

        # Create the prompt
        prompt = self._format_multi_speaker_prompt(conversation, voice_mappings)

        # Add style prompt if provided
        if config.speech_style_prompt:
            prompt = f"{config.speech_style_prompt}\n\n{prompt}"

        # Get speaker configurations
        speaker_configs = self._get_speaker_voice_configs(voice_mappings, speakers_in_conversation)
        if config.speech_style_prompt:
            print("[Gemini][Multi] speech_style_prompt=", (config.speech_style_prompt[:300] + '...') if len(config.speech_style_prompt) > 300 else config.speech_style_prompt)

        try:
            print("[Gemini][Multi] model=", config.model)
            print("[Gemini][Multi] language_code=", config.language_code)
            print("[Gemini][Multi] speakers=", sorted(list(speakers_in_conversation)))
            # Log voice mapping summary
            mapping_summary = {m.speaker_name: m.voice_id for m in voice_mappings if m.speaker_name in speakers_in_conversation}
            print("[Gemini][Multi] voice_mappings=", mapping_summary)
            print("[Gemini][Multi] prompt=", (prompt[:500] + '...') if len(prompt) > 500 else prompt)
            response = self.client.models.generate_content(
                model=config.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        language_code=config.language_code,
                        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                            speaker_voice_configs=speaker_configs
                        )
                    ),
                )
            )

            # Extract audio data
            audio_data = response.candidates[0].content.parts[0].inline_data.data

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as WAV file
            self._create_wave_file(output_path.with_suffix('.wav'), audio_data)

            # Convert to MP3 if needed for compatibility
            if config.output_format == "mp3":
                self._convert_wav_to_mp3(output_path.with_suffix('.wav'), output_path)

            return output_path

        except Exception as e:
            print(f"Error generating multi-speaker audio with Gemini: {e}")
            raise

    async def generate_conversation_audio_async(
        self,
        conversation: GeneratedConversation,
        voice_mappings: List[VoiceMapping],
        config: GeminiAudioConfiguration,
        output_path: Path
    ) -> Path:
        """Async version of conversation audio generation."""
        # For now, run the sync version in a thread pool since Gemini SDK is synchronous
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate_conversation_audio,
            conversation,
            voice_mappings,
            config,
            output_path
        )

    def _convert_wav_to_mp3(self, wav_path: Path, mp3_path: Path):
        """Convert WAV file to MP3 format."""
        try:
            from pydub import AudioSegment

            # Load WAV file
            audio = AudioSegment.from_wav(str(wav_path))

            # Export as MP3
            audio.export(str(mp3_path), format="mp3")

            # Remove temporary WAV file
            wav_path.unlink()

        except ImportError:
            print("Warning: pydub not available, keeping WAV format")
            # Rename WAV to MP3 extension for compatibility (not actual conversion)
            wav_path.rename(mp3_path)
        except Exception as e:
            print(f"Warning: Could not convert WAV to MP3: {e}")
            # Keep WAV format
            if mp3_path != wav_path:
                wav_path.rename(mp3_path)

    @classmethod
    def validate_voice_name(cls, voice_name: str) -> bool:
        """Validate if a voice name is supported by Gemini."""
        return voice_name in cls.SUPPORTED_VOICES

    @classmethod
    def get_supported_voices(cls) -> set:
        """Get the set of supported voice names."""
        return cls.SUPPORTED_VOICES.copy()


if __name__ == "__main__":
    # Test the Gemini client
    from dotenv import load_dotenv
    load_dotenv()

    try:
        generator = GeminiAudioGenerator()
        print("✓ Gemini client initialized successfully")
        print(f"✓ Supported voices: {len(generator.get_supported_voices())}")
        print("✓ Sample voices:", sorted(list(generator.get_supported_voices())[:5]), "...")
    except Exception as e:
        print(f"✗ Failed to initialize Gemini client: {e}")
        print("Make sure GOOGLE_API_KEY is set in your .env file")
