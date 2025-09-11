"""
OpenAI client integration for generating structured conversation datasets.
"""
import os
import asyncio
from typing import List, Dict, Any
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from models import ConversationScenario, GeneratedConversation, ConversationTurn
import json


class OpenAIConversationGenerator:
    """Generates structured conversations using OpenAI's API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
    
    def generate_conversation(self, scenario: ConversationScenario) -> GeneratedConversation:
        """Generate a single conversation based on a scenario."""
        
        # Create the system prompt
        system_prompt = self._create_system_prompt(scenario)
        
        # Create the user prompt
        user_prompt = self._create_user_prompt(scenario)
        
        try:
            # Use structured output with Pydantic model
            completion = self.client.chat.completions.parse(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=GeneratedConversation,
                temperature=0.7,
            )
            
            conversation = completion.choices[0].message.parsed
            if conversation:
                # Ensure scenario_id matches
                conversation.scenario_id = scenario.scenario_id
                
                # Calculate and set metadata
                from models import ConversationMetadata
                word_count = sum(len(turn.text.split()) for turn in conversation.turns)
                turn_count = len(conversation.turns)
                avg_turn_length = word_count / turn_count if turn_count > 0 else 0
                
                conversation.metadata = ConversationMetadata(
                    word_count=word_count,
                    turn_count=turn_count,
                    avg_turn_length=avg_turn_length
                )
                
                # Calculate estimated total duration based on word count
                # Average speaking rate is ~150 words per minute
                conversation.estimated_total_duration = word_count / 150 * 60  # Convert to seconds
                
                return conversation
            else:
                raise ValueError("Failed to parse conversation from OpenAI response")
                
        except Exception as e:
            print(f"Error generating conversation for scenario {scenario.scenario_id}: {e}")
            raise
    
    async def generate_conversations_batch(self, scenarios: List[ConversationScenario]) -> List[GeneratedConversation]:
        """Generate multiple conversations concurrently."""
        tasks = [self._generate_conversation_async(scenario) for scenario in scenarios]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        conversations = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Failed to generate conversation for scenario {scenarios[i].scenario_id}: {result}")
            else:
                conversations.append(result)
        
        return conversations
    
    async def _generate_conversation_async(self, scenario: ConversationScenario) -> GeneratedConversation:
        """Async version of generate_conversation."""
        system_prompt = self._create_system_prompt(scenario)
        user_prompt = self._create_user_prompt(scenario)
        
        try:
            completion = await self.async_client.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=GeneratedConversation,
                temperature=0.7,
            )
            
            conversation = completion.choices[0].message.parsed
            if conversation:
                conversation.scenario_id = scenario.scenario_id
                
                # Calculate and set metadata
                from models import ConversationMetadata
                word_count = sum(len(turn.text.split()) for turn in conversation.turns)
                turn_count = len(conversation.turns)
                avg_turn_length = word_count / turn_count if turn_count > 0 else 0
                
                conversation.metadata = ConversationMetadata(
                    word_count=word_count,
                    turn_count=turn_count,
                    avg_turn_length=avg_turn_length
                )
                
                # Calculate estimated total duration
                conversation.estimated_total_duration = word_count / 150 * 60  # Convert to seconds
                
                return conversation
            else:
                raise ValueError("Failed to parse conversation from OpenAI response")
                
        except Exception as e:
            print(f"Error generating conversation for scenario {scenario.scenario_id}: {e}")
            raise
    
    def _create_system_prompt(self, scenario: ConversationScenario) -> str:
        """Create system prompt for conversation generation."""
        return f"""You are an expert conversation generator for speech-to-text (STT) evaluation datasets.

Your task is to create realistic, natural conversations that will be used to test speech recognition systems using ElevenLabs v3 with advanced audio tags.

Guidelines:
- Generate natural, flowing conversations with realistic dialogue
- Include voice_characteristics for each turn that can be enhanced with ElevenLabs v3 audio tags
- Vary sentence length and complexity based on difficulty level
- Use domain-specific vocabulary when specified
- Ensure conversations are appropriate for the given context
- Make speakers distinct through their speech patterns, vocabulary, and emotional delivery
- Include realistic conversational elements like "um", "aha", "well" and very technical words for harder difficulties
- For medical/technical domains, include relevant terminology
- Keep conversations engaging and realistic
- Add emotional context and voice characteristics that work well with v3 audio tags

Voice Characteristics (CRITICAL - choose ONE primary characteristic per turn that maps to v3 audio tags):
- Emotional: "warm", "professional", "cheerful", "curious", "anxious", "excited", "frustrated", "confident", "nervous"
- Delivery: "soft-spoken", "authoritative", "conversational", "whispering", "emphatic", "questioning", "reassuring"
- Technical: Include specific technical terms and complex vocabulary for harder difficulties like specific medical terms, drugs, technical jargon, etc.

IMPORTANT: The voice_characteristics you provide will be automatically mapped to ElevenLabs v3 audio tags:
- "anxious" → [nervous]
- "whispering" → [whispers]
- "excited" → [excited]
- "questioning" → [questioning]
- "professional" → [professional]
- "warm" → [warm]
- "curious" → [curious]
- "frustrated" → [frustrated]
- "reassuring" → [reassuring]

Difficulty levels:
- Easy: Clear, simple sentences, minimal overlaps, formal speech, basic emotional range
- Medium: Natural speech with some informal elements, occasional overlaps, moderate emotional variety
- Hard: Complex sentences, technical words, informal speech, interruptions, background noise references, wide emotional range

Language: {scenario.language}
Domain: {scenario.domain if scenario.domain else 'General conversation'}
Target duration: Approximately {scenario.target_duration} seconds of speech
"""
    
    def _create_user_prompt(self, scenario: ConversationScenario) -> str:
        """Create user prompt for specific scenario."""
        participants_str = ", ".join(scenario.participants)
        
        return f"""Generate a conversation for the following scenario:

Title: {scenario.title}
Description: {scenario.description}
Context: {scenario.context}
Participants: {participants_str}
Difficulty: {scenario.difficulty_level}
Target Duration: {scenario.target_duration} seconds

Requirements:
1. Create a natural conversation between the specified participants
2. Each turn should be realistic and appropriate for the context
3. Estimate duration for each turn (typical speaking rate is ~150 words per minute)
4. Include natural speech characteristics appropriate for the difficulty level
5. Ensure the total estimated duration is close to the target duration
6. Make each speaker's voice distinct through their choice of words and speaking style
7. **IMPORTANT**: Include voice_characteristics for each turn that describe the emotional delivery and tone (e.g., "warm professional tone", "anxious soft-spoken", "excited emphatic", "curious questioning")
8. Voice characteristics should be realistic and appropriate for the speaker and context
9. Vary emotional delivery throughout the conversation to create natural dynamics

Return the conversation as a structured format with turns, including speaker identification, estimated durations, and voice characteristics for each turn."""


def create_sample_scenarios() -> List[ConversationScenario]:
    """Create sample conversation scenarios for testing."""
    scenarios = [
        ConversationScenario(
            scenario_id="medical_consultation_01",
            title="Medical Consultation",
            description="A doctor consulting with a patient about symptoms",
            context="A medical office where a doctor is examining a patient with flu-like symptoms",
            participants=["Doctor", "Patient"],
            target_duration=90,
            difficulty_level="medium",
            language="en",
            domain="medical"
        ),
        ConversationScenario(
            scenario_id="business_meeting_01",
            title="Team Status Meeting",
            description="A brief team meeting discussing project progress",
            context="A conference room where team members discuss weekly progress",
            participants=["Manager", "Developer", "Designer"],
            target_duration=120,
            difficulty_level="medium",
            language="en",
            domain="business"
        ),
        ConversationScenario(
            scenario_id="casual_friends_01",
            title="Friends Planning Weekend",
            description="Two friends discussing weekend plans",
            context="A casual phone conversation between friends",
            participants=["Alex", "Sam"],
            target_duration=75,
            difficulty_level="easy",
            language="en",
            domain="casual"
        )
    ]
    return scenarios


if __name__ == "__main__":
    # Test the generator with sample scenarios
    from dotenv import load_dotenv
    load_dotenv()
    
    generator = OpenAIConversationGenerator()
    scenarios = create_sample_scenarios()
    
    print("Testing OpenAI conversation generation...")
    for scenario in scenarios[:1]:  # Test with first scenario
        try:
            conversation = generator.generate_conversation(scenario)
            print(f"Generated conversation: {conversation.title}")
            print(f"Number of turns: {len(conversation.turns)}")
            print(f"Estimated duration: {conversation.estimated_total_duration} seconds")
            print("---")
        except Exception as e:
            print(f"Error: {e}")
