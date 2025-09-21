"""
Main dataset generation pipeline combining OpenAI and ElevenLabs.
"""
import os
import json
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid
from datetime import datetime

from openai_client import OpenAIConversationGenerator
from elevenlabs_client import ElevenLabsAudioGenerator
from gemini_client import GeminiAudioGenerator
from models import (
    ConversationScenario, GeneratedConversation, DatasetEntry,
    GenerationBatch, VoiceMapping, AudioConfiguration, TTSProvider,
    ElevenLabsAudioConfiguration, GeminiAudioConfiguration
)


class STTDatasetGenerator:
    """Main class for generating STT evaluation datasets."""

    def __init__(
        self,
        openai_api_key: str = None,
        elevenlabs_api_key: str = None,
        gemini_api_key: str = None,
        output_base_dir: Path = None
    ):
        self.openai_generator = OpenAIConversationGenerator(openai_api_key)
        self.elevenlabs_generator = ElevenLabsAudioGenerator(elevenlabs_api_key)

        # Initialize Gemini generator if API key is provided
        self.gemini_generator = None
        if gemini_api_key or os.getenv("GOOGLE_API_KEY"):
            try:
                self.gemini_generator = GeminiAudioGenerator(gemini_api_key)
                print("✓ Gemini TTS generator initialized")
            except Exception as e:
                print(f"⚠ Failed to initialize Gemini TTS generator: {e}")

        self.output_base_dir = output_base_dir or Path("./generated_datasets")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Default configurations
        self.default_audio_config = AudioConfiguration()
        self.english_voice_mappings = self._load_voice_mappings("en")
        self.spanish_voice_mappings = self._load_voice_mappings("es")
        self.english_gemini_voice_mappings = self._load_voice_mappings("en", "gemini")
        self.spanish_gemini_voice_mappings = self._load_voice_mappings("es", "gemini")

    def _load_voice_mappings(self, language: str, provider: str = "elevenlabs") -> List[VoiceMapping]:
        """Load voice mappings from JSON file for the specified language and provider."""
        if provider == "gemini":
            mappings_path = Path(__file__).parent / f"voice_mappings_gemini_{language}.json"
        else:
            mappings_path = Path(__file__).parent / f"voice_mappings_{language}.json"

        if mappings_path.exists():
            try:
                with open(mappings_path, 'r', encoding='utf-8') as f:
                    mappings_data = json.load(f)
                return [VoiceMapping(**mapping) for mapping in mappings_data]
            except Exception as e:
                print(f"Warning: Could not load {language} {provider} voice mappings: {e}")
                return []
        else:
            print(f"Warning: Voice mappings file for {language} ({provider}) not found")
            return []

    def _get_voice_mappings_for_scenario(self, scenario: ConversationScenario, provider: TTSProvider = TTSProvider.ELEVENLABS) -> List[VoiceMapping]:
        """Get appropriate voice mappings based on scenario language and provider."""
        if provider == TTSProvider.GEMINI:
            if scenario.language and scenario.language.lower() in ['es', 'spanish', 'espanol']:
                return self.spanish_gemini_voice_mappings if self.spanish_gemini_voice_mappings else []
            else:
                return self.english_gemini_voice_mappings if self.english_gemini_voice_mappings else []
        else:
            if scenario.language and scenario.language.lower() in ['es', 'spanish', 'espanol']:
                return self.spanish_voice_mappings if self.spanish_voice_mappings else []
            else:
                return self.english_voice_mappings if self.english_voice_mappings else []

    def generate_single_dataset_entry(
        self,
        scenario: ConversationScenario,
        voice_mappings: List[VoiceMapping] = None,
        audio_config: AudioConfiguration = None,
        output_subdir: str = None
    ) -> DatasetEntry:
        """Generate a single complete dataset entry."""

        # Use default audio config if none provided
        audio_config = audio_config or self.default_audio_config

        # Use language-specific voice mappings if none provided
        if voice_mappings is None:
            voice_mappings = self._get_voice_mappings_for_scenario(scenario, audio_config.provider)
        
        # Create output directory
        if output_subdir:
            output_dir = self.output_base_dir / output_subdir
        else:
            output_dir = self.output_base_dir / f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating conversation for scenario: {scenario.title}")
        
        # Step 1: Generate conversation using OpenAI
        try:
            conversation = self.openai_generator.generate_conversation(scenario)
            print(f"Generated {len(conversation.turns)} conversation turns")
        except Exception as e:
            print(f"Failed to generate conversation: {e}")
            raise
        
        # Step 2: Create dataset entry
        entry_id = f"{scenario.scenario_id}_{uuid.uuid4().hex[:8]}"
        
        # Step 3: Generate audio using the specified TTS provider
        if audio_config.provider == TTSProvider.GEMINI:
            if not self.gemini_generator:
                raise ValueError("Gemini TTS generator not initialized. Please provide GOOGLE_API_KEY.")
            audio_filename = f"{entry_id}_conversation.wav"  # Gemini outputs WAV by default
            audio_path = output_dir / audio_filename

            try:
                # Ensure language_code is set for accent control
                if audio_config.gemini_config and not audio_config.gemini_config.language_code:
                    lang = (scenario.language or "en").lower()
                    audio_config.gemini_config.language_code = "es-ES" if lang.startswith("es") else "en-US"
                print(f"Generating audio for conversation using Gemini TTS...")
                final_audio_path = self.gemini_generator.generate_conversation_audio(
                    conversation=conversation,
                    voice_mappings=voice_mappings,
                    config=audio_config.gemini_config,
                    output_path=audio_path
                )
                print(f"Audio generated with Gemini: {final_audio_path}")
            except Exception as e:
                print(f"Failed to generate audio with Gemini: {e}")
                raise
        else:
            # Use ElevenLabs as default/fallback
            audio_filename = f"{entry_id}_conversation.mp3"
            audio_path = output_dir / audio_filename

            try:
                print(f"Generating audio for conversation using ElevenLabs...")
                final_audio_path = self.elevenlabs_generator.generate_conversation_audio(
                    conversation=conversation,
                    voice_mappings=voice_mappings,
                    audio_config=audio_config.elevenlabs_config,
                    output_path=audio_path
                )
                print(f"Audio generated with ElevenLabs: {final_audio_path}")
            except Exception as e:
                print(f"Failed to generate audio with ElevenLabs: {e}")
                raise
        
        # Step 4: Save transcript
        transcript_filename = f"{entry_id}_transcript.json"
        transcript_path = output_dir / transcript_filename
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(conversation.model_dump(mode='json'), f, indent=2, ensure_ascii=False, default=str)
        
        # Step 5: Create dataset entry
        dataset_entry = DatasetEntry(
            entry_id=entry_id,
            conversation=conversation,
            voice_mappings=voice_mappings,
            audio_config=audio_config,
            audio_file_path=final_audio_path,
            transcript_file_path=transcript_path,
            stt_evaluation_ready=True
        )
        
        # Step 6: Save dataset entry metadata
        entry_metadata_path = output_dir / f"{entry_id}_metadata.json"
        with open(entry_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_entry.model_dump(mode='json'), f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Dataset entry completed: {entry_id}")
        print(f"  - Audio: {final_audio_path}")
        print(f"  - Transcript: {transcript_path}")
        print(f"  - Metadata: {entry_metadata_path}")
        
        return dataset_entry
    
    def generate_batch_sync(
        self,
        batch: GenerationBatch,
        max_concurrent: int = 3
    ) -> GenerationBatch:
        """Generate multiple dataset entries concurrently."""
        
        batch.status = "processing"
        batch_output_dir = self.output_base_dir / f"batch_{batch.batch_id}"
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting batch generation: {batch.batch_id}")
        print(f"Scenarios to process: {len(batch.scenarios)}")
        print(f"Output directory: {batch_output_dir}")
        
        # Process scenarios sequentially for now (async was causing ElevenLabs issues)
        results = []
        for i, scenario in enumerate(batch.scenarios):
            print(f"Processing scenario {i+1}/{len(batch.scenarios)}: {scenario.title}")
            try:
                entry = self.generate_single_dataset_entry(
                    scenario,
                    batch.voice_mappings,
                    batch.audio_config,
                    f"batch_{scenario.scenario_id}"
                )
                results.append(entry.entry_id)
                print(f"✓ Completed: {scenario.scenario_id}")
            except Exception as e:
                print(f"✗ Failed: {scenario.scenario_id} - {e}")
                results.append(None)
        
        # Update batch status
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                batch.failed_entries.append(batch.scenarios[i].scenario_id)
            elif result:
                batch.completed_entries.append(result)
            else:
                batch.failed_entries.append(batch.scenarios[i].scenario_id)
        
        batch.status = "completed" if not batch.failed_entries else "completed"
        
        # Save batch metadata
        batch_metadata_path = batch_output_dir / f"batch_{batch.batch_id}_metadata.json"
        with open(batch_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(batch.model_dump(mode='json'), f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Batch processing completed:")
        print(f"  - Successful: {len(batch.completed_entries)}")
        print(f"  - Failed: {len(batch.failed_entries)}")
        print(f"  - Batch metadata: {batch_metadata_path}")
        
        return batch
    
    async def _generate_single_entry_async(
        self,
        scenario: ConversationScenario,
        voice_mappings: List[VoiceMapping],
        audio_config: AudioConfiguration,
        output_dir: Path
    ) -> DatasetEntry:
        """Async version of generating a single dataset entry."""
        
        print(f"Processing scenario: {scenario.title}")
        
        # Generate conversation
        conversation = await self.openai_generator._generate_conversation_async(scenario)
        print(f"Generated conversation for {scenario.scenario_id}")
        
        # Create entry ID and paths
        entry_id = f"{scenario.scenario_id}_{uuid.uuid4().hex[:8]}"
        audio_path = output_dir / f"{entry_id}_conversation.mp3"
        transcript_path = output_dir / f"{entry_id}_transcript.json"
        
        # Generate audio
        await self.elevenlabs_generator.generate_conversation_audio_async(
            conversation=conversation,
            voice_mappings=voice_mappings,
            audio_config=audio_config,
            output_path=audio_path
        )
        
        # Save transcript
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(conversation.model_dump(mode='json'), f, indent=2, ensure_ascii=False, default=str)
        
        # Create dataset entry
        dataset_entry = DatasetEntry(
            entry_id=entry_id,
            conversation=conversation,
            voice_mappings=voice_mappings,
            audio_config=audio_config,
            audio_file_path=audio_path,
            transcript_file_path=transcript_path,
            stt_evaluation_ready=True
        )
        
        # Save metadata
        metadata_path = output_dir / f"{entry_id}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_entry.model_dump(mode='json'), f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Completed dataset entry: {entry_id}")
        return dataset_entry
    
    def create_batch_from_scenarios(
        self,
        scenarios: List[ConversationScenario],
        voice_mappings: List[VoiceMapping] = None,
        audio_config: AudioConfiguration = None,
        batch_id: str = None
    ) -> GenerationBatch:
        """Create a generation batch from a list of scenarios."""

        batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Use default audio config if none provided
        audio_config = audio_config or self.default_audio_config

        # Use language-specific voice mappings if none provided and all scenarios have the same language
        if voice_mappings is None and scenarios:
            first_scenario = scenarios[0]
            # Check if all scenarios have the same language
            if all(s.language == first_scenario.language for s in scenarios):
                voice_mappings = self._get_voice_mappings_for_scenario(first_scenario, audio_config.provider)
            else:
                # Mixed languages - try English as fallback
                if audio_config.provider == TTSProvider.GEMINI:
                    voice_mappings = self.english_gemini_voice_mappings
                else:
                    voice_mappings = self.english_voice_mappings

        # Fallback to default voice mappings based on provider
        if voice_mappings is None:
            if audio_config.provider == TTSProvider.GEMINI:
                voice_mappings = self.english_gemini_voice_mappings
            else:
                voice_mappings = self.english_voice_mappings

        return GenerationBatch(
            batch_id=batch_id,
            scenarios=scenarios,
            voice_mappings=voice_mappings,
            audio_config=audio_config,
            output_directory=self.output_base_dir / f"batch_{batch_id}"
        )
    
    def load_scenarios_from_json(self, json_path: Path) -> List[ConversationScenario]:
        """Load conversation scenarios from a JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            scenarios_data = json.load(f)
        
        scenarios = []
        for scenario_data in scenarios_data:
            scenario = ConversationScenario(**scenario_data)
            scenarios.append(scenario)
        
        return scenarios
    
    def save_scenarios_template(self, output_path: Path):
        """Save a template JSON file for defining conversation scenarios."""
        from openai_client import create_sample_scenarios
        
        sample_scenarios = create_sample_scenarios()
        scenarios_data = [scenario.model_dump() for scenario in sample_scenarios]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scenarios_data, f, indent=2, ensure_ascii=False)
        
        print(f"Sample scenarios template saved to: {output_path}")
        print("You can edit this file to define your own conversation scenarios.")


if __name__ == "__main__":
    # Test the dataset generator
    from dotenv import load_dotenv
    load_dotenv()
    
    generator = STTDatasetGenerator()
    
    # Create sample scenarios
    from openai_client import create_sample_scenarios
    scenarios = create_sample_scenarios()
    
    # Generate a single dataset entry for testing
    try:
        entry = generator.generate_single_dataset_entry(scenarios[0])
        print("Test completed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")
