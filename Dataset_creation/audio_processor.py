"""
Audio post-processing module for STT evaluation dataset enhancement.
Adds noise, speed variations, and volume changes to create diverse audio conditions.
"""
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Optional
import uuid

try:
    from pydub import AudioSegment
    from pydub.generators import WhiteNoise
    # Note: PinkNoise and BrownNoise may not be available in all pydub versions
    try:
        from pydub.generators import PinkNoise, BrownNoise
        PINK_NOISE_AVAILABLE = True
        BROWN_NOISE_AVAILABLE = True
    except ImportError:
        PINK_NOISE_AVAILABLE = False
        BROWN_NOISE_AVAILABLE = False
        print("Note: PinkNoise and BrownNoise not available, using white noise alternatives")

    PYDUB_AVAILABLE = True
except ImportError as e:
    PYDUB_AVAILABLE = False
    print(f"Warning: pydub not available. Audio processing will be limited. Error: {e}")
    AudioSegment = None
    WhiteNoise = None


class AudioProcessor:
    """Processes audio files with random effects for STT evaluation diversity."""

    def __init__(
        self,
        noise_level: float = 0.05,
        speed_variation: float = 0.1,
        volume_variation: float = 0.2,
        noise_types: List[str] = None
    ):
        """
        Initialize the audio processor.

        Args:
            noise_level: Background noise level (0.0-0.2)
            speed_variation: Speed variation range (±percentage)
            volume_variation: Volume variation range (±dB)
            noise_types: Types of noise to use ['white', 'pink', 'brown']
        """
        if not PYDUB_AVAILABLE:
            raise ImportError("pydub is required for audio processing. Install with: pip install pydub")

        self.noise_level = max(0.0, min(0.2, noise_level))
        self.speed_variation = max(0.0, min(0.5, speed_variation))
        self.volume_variation = max(0.0, min(1.0, volume_variation))
        self.noise_types = noise_types or ['white']

    def process_file(self, input_file: Path, output_dir: Path) -> Optional[Path]:
        """
        Process a single audio file with random effects.

        Args:
            input_file: Path to input MP3 file
            output_dir: Directory to save processed file

        Returns:
            Path to processed file or None if failed
        """
        try:
            # Load audio file
            audio = AudioSegment.from_mp3(str(input_file))

            # Apply random effects
            processed_audio = self._apply_random_effects(audio)

            # Generate output filename
            original_name = input_file.stem
            suffix = f"_processed_{uuid.uuid4().hex[:6]}"
            output_filename = f"{original_name}{suffix}.mp3"
            output_path = output_dir / output_filename

            # Export processed audio
            processed_audio.export(str(output_path), format="mp3", bitrate="128k")

            return output_path

        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            return None

    def _apply_random_effects(self, audio: AudioSegment) -> AudioSegment:
        """
        Apply random audio effects to create diverse conditions.

        Args:
            audio: Input audio segment

        Returns:
            Processed audio segment
        """
        processed = audio

        # Apply speed variation
        if random.random() < 0.7:  # 70% chance of speed change
            processed = self._apply_speed_variation(processed)

        # Apply volume variation
        if random.random() < 0.8:  # 80% chance of volume change
            processed = self._apply_volume_variation(processed)

        # Apply background noise
        if random.random() < 0.6:  # 60% chance of adding noise
            processed = self._apply_background_noise(processed)

        return processed

    def _apply_speed_variation(self, audio: AudioSegment) -> AudioSegment:
        """Apply random speed variation."""
        # Random speed factor between (1 - variation) and (1 + variation)
        speed_factor = 1.0 + random.uniform(-self.speed_variation, self.speed_variation)

        # Ensure speed factor stays within reasonable bounds
        speed_factor = max(0.5, min(2.0, speed_factor))

        # Apply speed change
        if speed_factor != 1.0:
            new_sample_rate = int(audio.frame_rate * speed_factor)
            processed = audio._spawn(audio.raw_data, overrides={
                'frame_rate': new_sample_rate
            }).set_frame_rate(audio.frame_rate)

            return processed

        return audio

    def _apply_volume_variation(self, audio: AudioSegment) -> AudioSegment:
        """Apply random volume variation."""
        # Random volume change in dB
        volume_change = random.uniform(-self.volume_variation, self.volume_variation)

        if abs(volume_change) > 0.1:  # Only apply if change is significant
            return audio + volume_change

        return audio

    def _apply_background_noise(self, audio: AudioSegment) -> AudioSegment:
        """Apply background noise."""
        if not self.noise_types:
            return audio

        # Select random noise type
        noise_type = random.choice(self.noise_types)

        try:
            # Generate noise
            noise_duration = len(audio)  # Match audio duration

            if noise_type == 'white':
                noise = WhiteNoise().to_audio_segment(duration=noise_duration)
            elif noise_type == 'pink' and PINK_NOISE_AVAILABLE:
                noise = PinkNoise().to_audio_segment(duration=noise_duration)
            elif noise_type == 'brown' and BROWN_NOISE_AVAILABLE:
                noise = BrownNoise().to_audio_segment(duration=noise_duration)
            else:
                # Default to white noise for unsupported types
                noise = WhiteNoise().to_audio_segment(duration=noise_duration)

            # Adjust noise level
            noise = noise - (1.0 / self.noise_level) if self.noise_level > 0 else noise

            # Mix original audio with noise
            # Reduce noise volume to make it background noise
            noise = noise - 20  # Reduce by 20dB to make it subtle

            return audio.overlay(noise, position=0)

        except Exception as e:
            print(f"Warning: Could not apply {noise_type} noise: {e}")
            return audio

    def create_noise_variations(
        self,
        input_file: Path,
        output_dir: Path,
        variations: int = 5
    ) -> List[Path]:
        """
        Create multiple noise variations of a single file.

        Args:
            input_file: Input audio file
            output_dir: Output directory
            variations: Number of variations to create

        Returns:
            List of paths to created variations
        """
        variations_paths = []

        for i in range(variations):
            # Force noise application for this variation
            original_noise_level = self.noise_level
            self.noise_level = max(0.02, self.noise_level)  # Ensure some noise

            variation_path = self.process_file(input_file, output_dir)
            if variation_path:
                variations_paths.append(variation_path)

            self.noise_level = original_noise_level

        return variations_paths

    def create_speed_variations(
        self,
        input_file: Path,
        output_dir: Path,
        variations: int = 5
    ) -> List[Path]:
        """
        Create multiple speed variations of a single file.

        Args:
            input_file: Input audio file
            output_dir: Output directory
            variations: Number of variations to create

        Returns:
            List of paths to created variations
        """
        variations_paths = []

        for i in range(variations):
            try:
                audio = AudioSegment.from_mp3(str(input_file))

                # Create variation with specific speed
                speed_factor = 0.8 + (i * 0.1)  # 0.8, 0.9, 1.0, 1.1, 1.2
                speed_factor = min(1.5, max(0.7, speed_factor))

                new_sample_rate = int(audio.frame_rate * speed_factor)
                processed = audio._spawn(audio.raw_data, overrides={
                    'frame_rate': new_sample_rate
                }).set_frame_rate(audio.frame_rate)

                # Save variation
                original_name = input_file.stem
                output_filename = f"{original_name}_speed_{speed_factor:.1f}_{uuid.uuid4().hex[:4]}.mp3"
                output_path = output_dir / output_filename

                processed.export(str(output_path), format="mp3", bitrate="128k")
                variations_paths.append(output_path)

            except Exception as e:
                print(f"Error creating speed variation {i+1}: {e}")

        return variations_paths


# Utility functions for batch processing
def process_audio_batch(
    input_dir: Path,
    output_dir: Path,
    noise_level: float = 0.05,
    speed_variation: float = 0.1,
    volume_variation: float = 0.2,
    noise_types: List[str] = None
) -> dict:
    """
    Process all MP3 files in a directory.

    Args:
        input_dir: Directory containing MP3 files
        output_dir: Output directory
        noise_level: Background noise level
        speed_variation: Speed variation range
        volume_variation: Volume variation range
        noise_types: Types of noise to use

    Returns:
        Dictionary with processing statistics
    """
    processor = AudioProcessor(
        noise_level=noise_level,
        speed_variation=speed_variation,
        volume_variation=volume_variation,
        noise_types=noise_types
    )

    stats = {
        'total_files': 0,
        'processed_files': 0,
        'failed_files': 0,
        'output_files': []
    }

    # Find all MP3 files
    mp3_files = list(input_dir.glob("**/*.mp3"))
    stats['total_files'] = len(mp3_files)

    for mp3_file in mp3_files:
        print(f"Processing: {mp3_file.name}")
        processed_file = processor.process_file(mp3_file, output_dir)

        if processed_file:
            stats['processed_files'] += 1
            stats['output_files'].append(str(processed_file))
            print(f"✓ {mp3_file.name} → {Path(processed_file).name}")
        else:
            stats['failed_files'] += 1
            print(f"✗ Failed: {mp3_file.name}")

    return stats


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    processor = AudioProcessor(
        noise_level=0.03,
        speed_variation=0.15,
        volume_variation=0.3,
        noise_types=['white', 'pink']
    )

    # Process a single file
    input_file = Path("example.mp3")
    output_dir = Path("processed_audio")

    if input_file.exists():
        result = processor.process_file(input_file, output_dir)
        print(f"Processed file: {result}")
    else:
        print("Example file not found. This is just a demonstration.")
