"""
ElevenLabs client integration for converting text to speech.
"""
import os
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
import tempfile
from elevenlabs import ElevenLabs, AsyncElevenLabs
from models import GeneratedConversation, VoiceMapping, AudioConfiguration, ConversationTurn
import io
import re


class ElevenLabsAudioGenerator:
    """Generates audio from text using ElevenLabs API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ELEVEN_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")
        
        self.client = ElevenLabs(api_key=self.api_key)
        self.async_client = AsyncElevenLabs(api_key=self.api_key)
    
    
    def generate_conversation_audio(
        self, 
        conversation: GeneratedConversation, 
        voice_mappings: List[VoiceMapping],
        audio_config: AudioConfiguration,
        output_path: Path,
        add_pauses: bool = True
    ) -> Path:
        """Generate complete conversation audio using ElevenLabs v3 Text to Dialogue API."""
        
        # Create voice mapping dictionary
        voice_map = {mapping.speaker_name: mapping.voice_id for mapping in voice_mappings}
        
        # Prepare dialogue inputs for v3 API
        dialogue_inputs = []
        
        for turn in conversation.turns:
            if turn.speaker not in voice_map:
                print(f"Warning: No voice mapping found for speaker '{turn.speaker}', skipping turn")
                continue
            
            # Enhance text with v3 audio tags if enabled
            enhanced_text = self._enhance_text_with_v3_tags(
                text=turn.text, 
                turn=turn,
                audio_config=audio_config
            )
            
            dialogue_inputs.append({
                "text": enhanced_text,
                "voice_id": voice_map[turn.speaker]
            })
            
            # Add natural pauses between speakers if requested
            if add_pauses and turn != conversation.turns[-1]:
                # Add a brief pause turn (using same speaker for consistency)
                pause_text = "[pause]" if audio_config.use_audio_tags else "..."
                dialogue_inputs.append({
                    "text": pause_text,
                    "voice_id": voice_map[turn.speaker]
                })
        
        if not dialogue_inputs:
            raise ValueError("No dialogue inputs were created")
        
        try:
            # Use ElevenLabs v3 Text to Dialogue API
            audio = self.client.text_to_dialogue.convert(
                inputs=dialogue_inputs,
                model_id=audio_config.model_id,
                output_format=audio_config.output_format,
                settings={
                    "stability": audio_config.stability,
                    "use_speaker_boost": audio_config.use_speaker_boost
                },
                apply_text_normalization=audio_config.apply_text_normalization,
                language_code=audio_config.language_code
            )
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write audio to file
            with open(output_path, 'wb') as f:
                if hasattr(audio, '__iter__'):
                    # Handle streaming audio
                    for chunk in audio:
                        f.write(chunk)
                else:
                    # Handle direct bytes
                    f.write(audio)
            
            return output_path
            
        except Exception as e:
            print(f"Error generating conversation audio with v3 API: {e}")
            # Fallback to legacy method if v3 fails
            return self._generate_conversation_audio_legacy(
                conversation, voice_mappings, audio_config, output_path, add_pauses
            )
    
    async def generate_conversation_audio_async(
        self,
        conversation: GeneratedConversation,
        voice_mappings: List[VoiceMapping],
        audio_config: AudioConfiguration,
        output_path: Path,
        add_pauses: bool = True
    ) -> Path:
        """Async version using ElevenLabs v3 Text to Dialogue API."""
        
        voice_map = {mapping.speaker_name: mapping.voice_id for mapping in voice_mappings}
        
        # Prepare dialogue inputs for v3 API
        dialogue_inputs = []
        
        for turn in conversation.turns:
            if turn.speaker not in voice_map:
                print(f"Warning: No voice mapping found for speaker '{turn.speaker}', skipping turn")
                continue
            
            # Enhance text with v3 audio tags if enabled
            enhanced_text = self._enhance_text_with_v3_tags(
                text=turn.text, 
                turn=turn,
                audio_config=audio_config
            )
            
            dialogue_inputs.append({
                "text": enhanced_text,
                "voice_id": voice_map[turn.speaker]
            })
            
            # Add natural pauses between speakers if requested
            if add_pauses and turn != conversation.turns[-1]:
                pause_text = "[pause]" if audio_config.use_audio_tags else "..."
                dialogue_inputs.append({
                    "text": pause_text,
                    "voice_id": voice_map[turn.speaker]
                })
        
        if not dialogue_inputs:
            raise ValueError("No dialogue inputs were created")
        
        try:
            # Use async ElevenLabs v3 Text to Dialogue API
            audio = await self.async_client.text_to_dialogue.convert(
                inputs=dialogue_inputs,
                model_id=audio_config.model_id,
                output_format=audio_config.output_format,
                settings={
                    "stability": audio_config.stability,
                    "use_speaker_boost": audio_config.use_speaker_boost
                },
                apply_text_normalization=audio_config.apply_text_normalization,
                language_code=audio_config.language_code
            )
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write audio to file
            with open(output_path, 'wb') as f:
                if hasattr(audio, '__aiter__'):  # Check if it's an async iterator
                    async for chunk in audio:
                        f.write(chunk)
                elif hasattr(audio, '__iter__'):  # Check if it's a sync iterator
                    for chunk in audio:
                        f.write(chunk)
                else:  # Assume it's bytes
                    f.write(audio)
            
            return output_path
            
        except Exception as e:
            print(f"Error generating conversation audio with v3 async API: {e}")
            # Fallback to legacy method if v3 fails
            return await self._generate_conversation_audio_async_legacy(
                conversation, voice_mappings, audio_config, output_path, add_pauses
            )
    
    
    def _calculate_pause_duration(self, turn: ConversationTurn, conversation: GeneratedConversation) -> float:
        """Calculate appropriate pause duration between turns."""
        # Base pause duration
        base_pause = 0.5  # 500ms
        
        # Adjust based on turn length (longer turns might need longer pauses)
        text_length_factor = min(len(turn.text) / 100, 1.0)  # Normalize to 0-1
        
        # Add some randomness to make it more natural
        import random
        randomness = random.uniform(0.8, 1.2)
        
        return (base_pause + text_length_factor * 0.3) * randomness
    
    def _enhance_text_with_v3_tags(
        self,
        text: str,
        turn: ConversationTurn,
        audio_config: AudioConfiguration
    ) -> str:
        """Enhance text with ElevenLabs v3 audio tags for better expression."""
        if not audio_config.use_audio_tags:
            return text

        enhanced_text = text

        # Add emotional tags based on voice characteristics
        if hasattr(turn, 'voice_characteristics') and turn.voice_characteristics:
            characteristics = turn.voice_characteristics.lower()

            # Direct mapping of voice characteristics to v3 audio tags
            tag_mappings = {
                # Primary emotional tags
                "anxious": "[nervous]",
                "nervous": "[nervous]",
                "whispering": "[whispers]",
                "excited": "[excited]",
                "questioning": "[questioning]",
                "professional": "[professional]",
                "warm": "[warm]",
                "curious": "[curious]",
                "frustrated": "[frustrated]",
                "reassuring": "[reassuring]",
                "authoritative": "[authoritative]",
                "emphatic": "[emphatic]",
                "soft-spoken": "[whispers]",

                # Additional emotional context
                "cheerful": "[cheerfully]",
                "happy": "[cheerfully]",
                "surprised": "[excited]",
                "worried": "[nervous]",
                "calm": "[warm]",
                "urgent": "[excited]",
                "concerned": "[reassuring]",
                "confident": "[professional]",
                "hesitant": "[nervous]"
            }

            # Apply the first matching tag
            for characteristic, tag in tag_mappings.items():
                if characteristic in characteristics:
                    enhanced_text = f"{tag} {enhanced_text}"
                    break  # Only apply one primary tag per turn

        # Add contextual tags based on content
        text_lower = text.lower()

        # Add question intonation (if not already tagged)
        if text.strip().endswith('?'):
            if not any(tag in enhanced_text for tag in ['[curious]', '[questioning]']):
                enhanced_text = enhanced_text.replace(text, f"[questioning] {text}")

        # Add emotional context for exclamations (if not already tagged)
        if '!' in text and not any(tag in enhanced_text for tag in ['[excited]', '[surprised]']):
            enhanced_text = enhanced_text.replace(text, f"[excited] {text}")

        # Add pauses for ellipses and dashes
        if '...' in text or ' - ' in text:
            enhanced_text = enhanced_text.replace('...', '... [pause]')
            enhanced_text = enhanced_text.replace(' - ', ' [pause] ')

        # Add emphasis for capitalized words (technical terms, drug names, etc.)
        enhanced_text = re.sub(r'\b([A-Z][A-Z]+)\b', r'[emphasis] \1', enhanced_text)

        # Add breathing/sighing for hesitation markers
        if 'well,' in text_lower or 'um,' in text_lower or 'uh,' in text_lower:
            enhanced_text = re.sub(r'\b(well|um|uh),\s*', r'[sighs] \1, ', enhanced_text, flags=re.IGNORECASE)

        return enhanced_text
    
    def _generate_conversation_audio_legacy(
        self,
        conversation: GeneratedConversation,
        voice_mappings: List[VoiceMapping],
        audio_config: AudioConfiguration,
        output_path: Path,
        add_pauses: bool = True
    ) -> Path:
        """Legacy method using individual TTS calls (fallback for v3 API failures)."""
        from pydub import AudioSegment
        
        voice_map = {mapping.speaker_name: mapping.voice_id for mapping in voice_mappings}
        audio_segments = []
        
        for turn in conversation.turns:
            if turn.speaker not in voice_map:
                print(f"Warning: No voice mapping found for speaker '{turn.speaker}', skipping turn")
                continue
            
            # Generate audio for this turn using legacy method
            audio_bytes = self._generate_turn_audio_legacy(
                text=turn.text,
                voice_id=voice_map[turn.speaker],
                audio_config=audio_config
            )
            
            if audio_bytes:
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                audio_segments.append(audio_segment)
                
                if add_pauses and turn != conversation.turns[-1]:
                    pause_duration = self._calculate_pause_duration(turn, conversation)
                    pause = AudioSegment.silent(duration=int(pause_duration * 1000))
                    audio_segments.append(pause)
        
        if not audio_segments:
            raise ValueError("No audio segments were generated")
        
        final_audio = AudioSegment.empty()
        for segment in audio_segments:
            final_audio += segment
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_audio.export(str(output_path), format="mp3")
        
        return output_path
    
    async def _generate_conversation_audio_async_legacy(
        self,
        conversation: GeneratedConversation,
        voice_mappings: List[VoiceMapping],
        audio_config: AudioConfiguration,
        output_path: Path,
        add_pauses: bool = True
    ) -> Path:
        """Async legacy method using individual TTS calls."""
        from pydub import AudioSegment
        
        voice_map = {mapping.speaker_name: mapping.voice_id for mapping in voice_mappings}
        
        # Generate all audio turns concurrently
        tasks = []
        for turn in conversation.turns:
            if turn.speaker in voice_map:
                task = self._generate_turn_audio_async_legacy(
                    text=turn.text,
                    voice_id=voice_map[turn.speaker],
                    audio_config=audio_config
                )
                tasks.append(task)
            else:
                print(f"Warning: No voice mapping found for speaker '{turn.speaker}'")
                tasks.append(None)
        
        audio_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        audio_segments = []
        for i, (turn, audio_result) in enumerate(zip(conversation.turns, audio_results)):
            if isinstance(audio_result, Exception):
                print(f"Error generating audio for turn {i}: {audio_result}")
                continue
            
            if audio_result and turn.speaker in {mapping.speaker_name for mapping in voice_mappings}:
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_result))
                audio_segments.append(audio_segment)
                
                if add_pauses and i < len(conversation.turns) - 1:
                    pause_duration = self._calculate_pause_duration(turn, conversation)
                    pause = AudioSegment.silent(duration=int(pause_duration * 1000))
                    audio_segments.append(pause)
        
        if not audio_segments:
            raise ValueError("No audio segments were generated successfully")
        
        final_audio = AudioSegment.empty()
        for segment in audio_segments:
            final_audio += segment
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_audio.export(str(output_path), format="mp3")
        
        return output_path
    
    def _generate_turn_audio_legacy(
        self, 
        text: str, 
        voice_id: str, 
        audio_config: AudioConfiguration
    ) -> Optional[bytes]:
        """Legacy method for generating audio for a single turn."""
        try:
            # Use the older text_to_speech API
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",  # Fallback to v2
                output_format=audio_config.output_format
            )
            
            if hasattr(audio, '__iter__'):
                audio_bytes = b''.join(audio)
            else:
                audio_bytes = audio
            
            return audio_bytes
            
        except Exception as e:
            print(f"Error generating legacy audio for text '{text[:50]}...': {e}")
            return None
    
    async def _generate_turn_audio_async_legacy(
        self,
        text: str,
        voice_id: str,
        audio_config: AudioConfiguration
    ) -> Optional[bytes]:
        """Async legacy method for generating audio for a single turn."""
        try:
            audio = await self.async_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",  # Fallback to v2
                output_format=audio_config.output_format
            )

            if hasattr(audio, '__aiter__'):  # Check if it's an async iterator
                audio_chunks = []
                async for chunk in audio:
                    audio_chunks.append(chunk)
                audio_bytes = b''.join(audio_chunks)
            elif hasattr(audio, '__iter__'):  # Check if it's a sync iterator
                audio_bytes = b''.join(audio)
            else:  # Assume it's bytes
                audio_bytes = audio

            return audio_bytes

        except Exception as e:
            print(f"Error generating async legacy audio for text '{text[:50]}...': {e}")
            return None
if __name__ == "__main__":
    # Test the audio generator
    from dotenv import load_dotenv
    load_dotenv()

    generator = ElevenLabsAudioGenerator()
    print("ElevenLabs client initialized successfully")
    print("Voice mappings are now loaded from JSON files")
