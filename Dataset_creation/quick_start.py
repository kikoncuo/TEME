#!/usr/bin/env python3
"""
Quick start demo for STT Dataset Generator
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def demo_without_api():
    """Demonstrate functionality without API calls."""
    print("🎯 STT Dataset Generator - Demo Mode")
    print("=" * 50)
    
    try:
        from models import ConversationScenario, AudioConfiguration
        from openai_client import create_sample_scenarios
        from dataset_generator import STTDatasetGenerator
        
        print("📝 Creating sample scenarios...")
        scenarios = create_sample_scenarios()
        for i, scenario in enumerate(scenarios, 1):
            print(f"   {i}. {scenario.title}")
            print(f"      - Context: {scenario.context}")
            print(f"      - Participants: {', '.join(scenario.participants)}")
            print(f"      - Duration: {scenario.target_duration}s")
            print(f"      - Difficulty: {scenario.difficulty_level}")
            print()
        
        print("🎙️ Voice mapping system ready...")
        english_mappings_path = Path("voice_mappings_en.json")
        spanish_mappings_path = Path("voice_mappings_es.json")

        if english_mappings_path.exists():
            print("   - English voice mappings available (voice_mappings_en.json)")
        if spanish_mappings_path.exists():
            print("   - Spanish voice mappings available (voice_mappings_es.json)")
        
        print(f"\n🔧 Audio configuration...")
        audio_config = AudioConfiguration()
        print(f"   - Model: {audio_config.model_id}")
        print(f"   - Format: {audio_config.output_format}")
        print(f"   - Stability: {audio_config.stability}")
        print(f"   - Speaker Boost: {audio_config.use_speaker_boost}")
        print(f"   - Audio Tags: {audio_config.use_audio_tags}")
        
        print("\n✅ Demo completed successfully!")
        print("\nTo generate actual datasets:")
        print("1. Configure your API keys in .env file")
        print("2. Run: python cli.py generate --scenarios example_scenarios.json --single")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def demo_with_api():
    """Demonstrate with API calls (requires valid API keys)."""
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("ELEVEN_API_KEY"):
        print("⚠️  API keys not found. Running demo without API calls.")
        demo_without_api()
        return
    
    print("🚀 STT Dataset Generator - Live Demo")
    print("=" * 50)
    
    try:
        from dataset_generator import STTDatasetGenerator
        from models import ConversationScenario
        
        # Initialize generator
        print("🔧 Initializing generators...")
        generator = STTDatasetGenerator()
        
        # List voices
        print("\n🎙️ Available ElevenLabs voices:")
        generator.elevenlabs_generator.print_available_voices()
        
        # Create a simple test scenario
        print("\n📝 Creating test scenario...")
        test_scenario = ConversationScenario(
            scenario_id="quick_demo",
            title="Quick Demo Conversation",
            description="A simple test conversation for demonstration",
            context="Two people having a brief chat about the weather",
            participants=["Alex", "Sam"],
            target_duration=30,
            difficulty_level="easy",
            language="en",
            domain="casual"
        )
        
        print(f"   Scenario: {test_scenario.title}")
        print(f"   Participants: {', '.join(test_scenario.participants)}")
        
        # Generate conversation (this will use OpenAI API)
        print("\n🤖 Generating conversation with OpenAI...")
        try:
            conversation = generator.openai_generator.generate_conversation(test_scenario)
            print(f"   Generated {len(conversation.turns)} conversation turns:")
            for i, turn in enumerate(conversation.turns, 1):
                print(f"   {i}. {turn.speaker}: {turn.text[:50]}{'...' if len(turn.text) > 50 else ''}")
        except Exception as e:
            print(f"   ❌ OpenAI generation failed: {e}")
            return
        
        # Note: We skip actual audio generation in demo to save on API costs
        print("\n🎵 Audio generation (skipped in demo to save API costs)")
        print("   In full mode, this would generate MP3 audio using ElevenLabs")
        
        print("\n✅ Live demo completed!")
        print("\nTo generate full datasets with audio:")
        print("1. Run: python cli.py generate --scenarios example_scenarios.json --single")
        
    except Exception as e:
        print(f"❌ Live demo failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main demo function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        demo_with_api()
    else:
        demo_without_api()

if __name__ == "__main__":
    main()
