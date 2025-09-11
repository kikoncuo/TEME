#!/usr/bin/env python3
"""
Command-line interface for the STT Dataset Generator.
"""
import click
import asyncio
import json
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from dataset_generator import STTDatasetGenerator
from openai_client import create_sample_scenarios
from models import ConversationScenario, AudioConfiguration


@click.group()
@click.option('--env-file', default='.env', help='Path to environment file')
@click.pass_context
def cli(ctx, env_file):
    """STT Dataset Generator - Create speech-to-text evaluation datasets."""
    ctx.ensure_object(dict)
    
    # Load environment variables
    if Path(env_file).exists():
        load_dotenv(env_file)
        click.echo(f"Loaded environment from {env_file}")
    else:
        click.echo(f"Warning: Environment file {env_file} not found")
    
    # Initialize generator
    try:
        ctx.obj['generator'] = STTDatasetGenerator()
        click.echo("✓ Generator initialized successfully")
    except Exception as e:
        click.echo(f"✗ Failed to initialize generator: {e}")
        ctx.exit(1)


@cli.command()
@click.option('--output', '-o', default='./sample_scenarios.json', help='Output file path')
@click.pass_context
def create_sample_config(ctx, output):
    """Create a sample scenarios configuration file."""
    generator = ctx.obj['generator']
    output_path = Path(output)
    
    try:
        generator.save_scenarios_template(output_path)
        click.echo(f"✓ Sample configuration created: {output_path}")
        click.echo("\nEdit this file to define your conversation scenarios, then use:")
        click.echo(f"  python cli.py generate --scenarios {output_path}")
    except Exception as e:
        click.echo(f"✗ Failed to create sample config: {e}")




@cli.command()
@click.option('--scenarios', '-s', required=True, help='Path to scenarios JSON file')
@click.option('--output-dir', '-o', default='./generated_datasets', help='Output directory')
@click.option('--batch-id', '-b', help='Custom batch ID')
@click.option('--max-concurrent', '-c', default=3, help='Max concurrent generations')
@click.option('--single', '-1', is_flag=True, help='Generate only the first scenario (for testing)')
@click.pass_context
def generate(ctx, scenarios, output_dir, batch_id, max_concurrent, single):
    """Generate dataset from scenarios configuration file."""
    generator = ctx.obj['generator']
    scenarios_path = Path(scenarios)
    
    if not scenarios_path.exists():
        click.echo(f"✗ Scenarios file not found: {scenarios_path}")
        return
    
    try:
        # Load scenarios
        scenarios_list = generator.load_scenarios_from_json(scenarios_path)
        click.echo(f"✓ Loaded {len(scenarios_list)} scenarios")
        
        if single:
            # Generate single entry for testing
            click.echo("Generating single dataset entry for testing...")
            entry = generator.generate_single_dataset_entry(
                scenarios_list[0], 
                output_subdir=batch_id or "test"
            )
            click.echo(f"✓ Test entry generated: {entry.entry_id}")
        else:
            # Generate full batch
            click.echo(f"Generating batch with {len(scenarios_list)} scenarios...")
            
            # Set output directory
            generator.output_base_dir = Path(output_dir)
            
            # Create batch
            batch = generator.create_batch_from_scenarios(
                scenarios_list, 
                batch_id=batch_id
            )
            
            # Run sync batch generation
            completed_batch = generator.generate_batch_sync(batch, max_concurrent=max_concurrent)
            
            click.echo(f"✓ Batch completed: {completed_batch.batch_id}")
            click.echo(f"  - Successful: {len(completed_batch.completed_entries)}")
            click.echo(f"  - Failed: {len(completed_batch.failed_entries)}")
            
            if completed_batch.failed_entries:
                click.echo(f"  - Failed scenarios: {completed_batch.failed_entries}")
    
    except Exception as e:
        click.echo(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()


@cli.command()
@click.option('--title', '-t', required=True, help='Conversation title')
@click.option('--description', '-d', required=True, help='Conversation description')
@click.option('--context', '-c', required=True, help='Conversation context')
@click.option('--participants', '-p', required=True, help='Comma-separated list of participants')
@click.option('--duration', '-u', default=60, help='Target duration in seconds')
@click.option('--difficulty', '-f', type=click.Choice(['easy', 'medium', 'hard']), default='medium', help='Difficulty level')
@click.option('--language', '-l', default='en', help='Language code')
@click.option('--domain', help='Domain (medical, business, casual, etc.)')
@click.option('--output-dir', '-o', default='./generated_datasets', help='Output directory')
@click.pass_context
def quick_generate(ctx, title, description, context, participants, duration, difficulty, language, domain, output_dir):
    """Quickly generate a single conversation from command-line parameters."""
    generator = ctx.obj['generator']
    
    try:
        # Create scenario from parameters
        scenario = ConversationScenario(
            scenario_id=f"quick_{title.lower().replace(' ', '_')}",
            title=title,
            description=description,
            context=context,
            participants=[p.strip() for p in participants.split(',')],
            target_duration=duration,
            difficulty_level=difficulty,
            language=language,
            domain=domain
        )
        
        click.echo(f"Generating conversation: {title}")
        
        # Set output directory
        generator.output_base_dir = Path(output_dir)
        
        # Generate single entry (voice mappings will be selected based on language)
        entry = generator.generate_single_dataset_entry(
            scenario,
            output_subdir="quick_generation"
        )
        
        click.echo(f"✓ Generated: {entry.entry_id}")
        click.echo(f"  - Audio: {entry.audio_file_path}")
        click.echo(f"  - Transcript: {entry.transcript_file_path}")
        
    except Exception as e:
        click.echo(f"✗ Quick generation failed: {e}")
        import traceback
        traceback.print_exc()


@cli.command()
@click.option('--input', '-i', required=True, help='Input MP3 file or directory containing MP3 files')
@click.option('--output', '-o', help='Output directory (defaults to input directory with _processed suffix)')
@click.option('--noise-level', '-n', default=0.05, help='Background noise level (0.0-0.2, default: 0.05)')
@click.option('--speed-variation', '-s', default=0.1, help='Speed variation range (±percentage, default: 0.1)')
@click.option('--volume-variation', '-v', default=0.2, help='Volume variation range (±dB, default: 0.2)')
@click.option('--noise-types', help='Comma-separated noise types: white,pink,brown (default: white)')
@click.option('--seed', type=int, help='Random seed for reproducible results')
@click.pass_context
def process_audio(ctx, input, output, noise_level, speed_variation, volume_variation, noise_types, seed):
    """Apply random audio effects to MP3 files for STT evaluation diversity."""
    from audio_processor import AudioProcessor
    import os

    # Set random seed if provided
    if seed is not None:
        import random
        random.seed(seed)

    # Initialize processor
    processor = AudioProcessor(
        noise_level=noise_level,
        speed_variation=speed_variation,
        volume_variation=volume_variation,
        noise_types=noise_types.split(',') if noise_types else ['white']
    )

    input_path = Path(input)

    if input_path.is_file():
        # Process single file
        if output:
            output_path = Path(output)
        else:
            output_path = input_path.parent / f"{input_path.parent.name}_processed"
        output_path.mkdir(exist_ok=True)

        click.echo(f"Processing single file: {input_path.name}")
        processed_file = processor.process_file(input_path, output_path)
        if processed_file:
            click.echo(f"✓ Processed: {processed_file}")
        else:
            click.echo("✗ Failed to process file")

    elif input_path.is_dir():
        # Process directory
        if output:
            output_path = Path(output)
        else:
            output_path = input_path.parent / f"{input_path.name}_processed"
        output_path.mkdir(exist_ok=True)

        # Find all MP3 files
        mp3_files = list(input_path.glob("**/*.mp3"))
        if not mp3_files:
            click.echo(f"No MP3 files found in {input_path}")
            return

        click.echo(f"Processing {len(mp3_files)} MP3 files from {input_path.name}")

        processed_count = 0
        for mp3_file in mp3_files:
            click.echo(f"Processing: {mp3_file.name}")
            processed_file = processor.process_file(mp3_file, output_path)
            if processed_file:
                processed_count += 1
                click.echo(f"✓ {mp3_file.name} → {Path(processed_file).name}")

        click.echo(f"\nCompleted: {processed_count}/{len(mp3_files)} files processed")

    else:
        click.echo(f"Input path does not exist: {input_path}")


@cli.command()
@click.option('--directory', '-d', required=True, help='Dataset directory to validate')
@click.pass_context
def validate(ctx, directory):
    """Validate a generated dataset directory."""
    dataset_dir = Path(directory)
    
    if not dataset_dir.exists():
        click.echo(f"✗ Directory not found: {dataset_dir}")
        return
    
    # Find all metadata files
    metadata_files = list(dataset_dir.glob("*_metadata.json"))
    click.echo(f"Found {len(metadata_files)} dataset entries")
    
    valid_entries = 0
    invalid_entries = []
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            entry_id = metadata.get('entry_id')
            audio_path = Path(metadata.get('audio_file_path', ''))
            transcript_path = Path(metadata.get('transcript_file_path', ''))
            
            # Check if files exist
            if audio_path.exists() and transcript_path.exists():
                valid_entries += 1
                click.echo(f"  ✓ {entry_id}")
            else:
                invalid_entries.append(entry_id)
                click.echo(f"  ✗ {entry_id} - Missing files")
        
        except Exception as e:
            invalid_entries.append(metadata_file.name)
            click.echo(f"  ✗ {metadata_file.name} - Error: {e}")
    
    click.echo(f"\nValidation complete:")
    click.echo(f"  Valid entries: {valid_entries}")
    click.echo(f"  Invalid entries: {len(invalid_entries)}")
    
    if invalid_entries:
        click.echo(f"  Invalid: {invalid_entries}")


@cli.command()
@click.option('--directory', '-d', required=True, help='Dataset directory')
@click.pass_context
def info(ctx, directory):
    """Display information about a dataset directory."""
    dataset_dir = Path(directory)
    
    if not dataset_dir.exists():
        click.echo(f"✗ Directory not found: {dataset_dir}")
        return
    
    # Find batch metadata
    batch_metadata_files = list(dataset_dir.glob("batch_*_metadata.json"))
    if batch_metadata_files:
        with open(batch_metadata_files[0], 'r') as f:
            batch_info = json.load(f)
        
        click.echo(f"Batch Information:")
        click.echo(f"  ID: {batch_info['batch_id']}")
        click.echo(f"  Status: {batch_info['status']}")
        click.echo(f"  Created: {batch_info['created_at']}")
        click.echo(f"  Scenarios: {len(batch_info['scenarios'])}")
        click.echo(f"  Completed: {len(batch_info['completed_entries'])}")
        click.echo(f"  Failed: {len(batch_info['failed_entries'])}")
    
    # Count dataset entries
    metadata_files = list(dataset_dir.glob("*_metadata.json"))
    metadata_files = [f for f in metadata_files if not f.name.startswith("batch_")]
    
    if metadata_files:
        click.echo(f"\nDataset Entries: {len(metadata_files)}")
        
        total_duration = 0
        languages = set()
        domains = set()
        difficulties = set()
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                conversation = metadata['conversation']
                total_duration += conversation.get('estimated_total_duration', 0) or 0
                
                # Extract info from scenarios
                for scenario in [metadata.get('conversation', {})]:
                    if 'language' in scenario:
                        languages.add(scenario['language'])
                    if 'domain' in scenario and scenario['domain']:
                        domains.add(scenario['domain'])
                    if 'difficulty_level' in scenario:
                        difficulties.add(scenario['difficulty_level'])
                        
            except Exception as e:
                continue
        
        click.echo(f"  Total estimated duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        click.echo(f"  Languages: {', '.join(sorted(languages)) if languages else 'N/A'}")
        click.echo(f"  Domains: {', '.join(sorted(domains)) if domains else 'N/A'}")
        click.echo(f"  Difficulties: {', '.join(sorted(difficulties)) if difficulties else 'N/A'}")


if __name__ == '__main__':
    cli()
