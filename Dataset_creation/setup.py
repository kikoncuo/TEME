#!/usr/bin/env python3
"""
Setup script for STT Dataset Generator
"""
from pathlib import Path
import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages."""
    try:
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_env_file():
    """Check if .env file exists and has required keys."""
    env_path = Path(".env")
    if not env_path.exists():
        print("⚠️  .env file not found")
        print("   Creating template .env file...")
        with open(".env", "w") as f:
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("ELEVEN_API_KEY=your_elevenlabs_api_key_here\n")
        print("✅ Created .env template file")
        print("   Please edit .env file and add your actual API keys")
        return False
    
    # Read .env file and check for keys
    with open(".env", "r") as f:
        content = f.read()
    
    has_openai = "OPENAI_API_KEY=" in content and "your_openai_api_key_here" not in content
    has_eleven = "ELEVEN_API_KEY=" in content and "your_elevenlabs_api_key_here" not in content
    
    if has_openai and has_eleven:
        print("✅ API keys found in .env file")
        return True
    else:
        print("⚠️  API keys not configured in .env file")
        if not has_openai:
            print("   - Missing or invalid OPENAI_API_KEY")
        if not has_eleven:
            print("   - Missing or invalid ELEVEN_API_KEY")
        return False

def test_basic_functionality():
    """Test basic functionality without API calls."""
    try:
        print("🧪 Testing basic functionality...")
        
        # Test imports
        from models import ConversationScenario, AudioConfiguration
        from openai_client import create_sample_scenarios
        
        # Test model creation
        scenarios = create_sample_scenarios()
        # Voice mappings are now loaded from JSON files automatically
        audio_config = AudioConfiguration()
        
        print(f"   - Created {len(scenarios)} sample scenarios")
        print("   - Audio configuration ready")
        print("   - Voice mappings loaded from JSON files")
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ["generated_datasets", "temp"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Created necessary directories")

def main():
    """Main setup function."""
    print("🚀 STT Dataset Generator Setup")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    success &= check_python_version()
    
    # Install requirements
    success &= install_requirements()
    
    # Create directories
    create_directories()
    
    # Check environment file
    env_ready = check_env_file()
    
    # Test basic functionality
    success &= test_basic_functionality()
    
    print("\n" + "=" * 50)
    
    if success and env_ready:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Test the CLI: python cli.py quick-generate --title 'Test' --description 'Test conversation' --participants 'Alice,Bob' --duration 30")
        print("2. Create sample config: python cli.py create-sample-config")
        print("3. Generate a test conversation: python cli.py generate --scenarios example_scenarios.json --single")
    elif success and not env_ready:
        print("⚠️  Setup mostly complete, but API keys need configuration")
        print("\nNext steps:")
        print("1. Edit .env file with your actual API keys")
        print("2. Test the CLI: python cli.py info --directory ./generated_datasets")
        print("3. Generate a test conversation: python cli.py generate --scenarios example_scenarios.json --single")
    else:
        print("❌ Setup failed. Please resolve the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
