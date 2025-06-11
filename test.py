#!/usr/bin/env python3
"""
Basic test script for Opi Voice Assistant
Tests components individually before full integration
"""

import sys
import os
import traceback
from pathlib import Path
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables at the start
load_dotenv()

def test_imports():
    """Test that all required modules can be imported."""
    cprint("🧪 Testing imports...", "cyan")
    
    tests = [
        ("sounddevice", "Audio I/O"),
        ("numpy", "Numerical processing"),
        ("scipy", "Audio processing"),
        ("termcolor", "Terminal colors"),
        ("faster_whisper", "Speech recognition"),
        ("google.generativeai", "LLM API"),
        ("langchain", "LLM framework"),
    ]
    
    passed = 0
    for module, description in tests:
        try:
            __import__(module)
            cprint(f"  ✅ {module} - {description}", "green")
            passed += 1
        except ImportError as e:
            cprint(f"  ❌ {module} - {description}: {e}", "red")
    
    cprint(f"\n📊 Import test: {passed}/{len(tests)} passed", "yellow")
    return passed == len(tests)

def test_audio_system():
    """Test audio system."""
    cprint("\n🎵 Testing audio system...", "cyan")
    
    try:
        from voice.audio_utils import list_audio_devices
        list_audio_devices()
        cprint("✅ Audio system accessible", "green")
        return True
    except Exception as e:
        cprint(f"❌ Audio system error: {e}", "red")
        traceback.print_exc()
        return False

def test_speech_recognition():
    """Test speech recognition setup."""
    cprint("\n🎤 Testing speech recognition...", "cyan")
    
    try:
        from faster_whisper import WhisperModel
        cprint("  Loading Whisper model (this may take a moment)...", "yellow")
        model = WhisperModel("tiny.en", compute_type="int8")
        cprint("  ✅ Whisper model loaded successfully", "green")
        return True
    except Exception as e:
        cprint(f"  ❌ Speech recognition error: {e}", "red")
        return False

def test_tts_backends():
    """Test available TTS backends."""
    cprint("\n🔊 Testing TTS backends...", "cyan")
    
    backends = []
    
    # Test espeak
    try:
        import subprocess
        result = subprocess.run(['which', 'espeak'], capture_output=True, text=True)
        if result.returncode == 0:
            backends.append("espeak")
            cprint("  ✅ espeak available", "green")
    except:
        pass
    
    # Test piper
    try:
        result = subprocess.run(['which', 'piper'], capture_output=True, text=True)
        if result.returncode == 0:
            backends.append("piper")
            cprint("  ✅ piper available", "green")
    except:
        pass
    
    if not backends:
        cprint("  ❌ No TTS backends found", "red")
        return False
    else:
        cprint(f"  📊 Found {len(backends)} TTS backend(s): {', '.join(backends)}", "cyan")
        return True

def test_llm_api():
    """Test LLM API access."""
    cprint("\n🤖 Testing LLM API...", "cyan")
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
    
    if not api_key:
        cprint("  ❌ No API key found", "red")
        cprint("  💡 Set GOOGLE_API_KEY environment variable", "yellow")
        cprint("  💡 Or create .env file with: GOOGLE_API_KEY=your_key_here", "yellow")
        return False
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Test API with a simple request
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Hello, respond with just 'API working'")
        
        cprint(f"  ✅ API response: {response.text.strip()}", "green")
        return True
        
    except Exception as e:
        cprint(f"  ❌ LLM API error: {e}", "red")
        return False

def test_directory_structure():
    """Test and create required directories."""
    cprint("\n📁 Testing directory strucure...", "cyan")
    
    required_dirs = [
        "data",
        "data/audio_chunks", 
        "cache",
        "logs",
        "output"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                cprint(f"  ✅ Created: {dir_path}", "green")
            except Exception as e:
                cprint(f"  ❌ Failed to create {dir_path}: {e}", "red")
        else:
            cprint(f"  ✅ Exists: {dir_path}", "green")
    
    return True

def test_config_loading():
    """Test configuration loading."""
    cprint("\n⚙️  Testing configuration...", "cyan")
    
    try:
        from config.settings import OpiConfig
        config = OpiConfig.load()
        cprint("  ✅ Configuration loaded successfully", "green")
        
        # Validate config
        issues = config.validate()
        if issues:
            cprint("  ⚠️  Configuration issues found:", "yellow")
            for issue in issues:
                cprint(f"    - {issue}", "yellow")
        else:
            cprint("  ✅ Configuration validation passed", "green")
        
        return True
        
    except Exception as e:
        cprint(f"  ❌ Configuration error: {e}", "red")
        traceback.print_exc()
        return False

def test_simple_tts():
    """Test simple TTS without complex synthesis."""
    cprint("\n🔊 Testing simple TTS...", "cyan")
    
    try:
        # Test espeak directly
        import subprocess
        result = subprocess.run([
            'espeak', '-s', '150', 'Hello, this is a test'
        ], capture_output=True, timeout=10)
        
        if result.returncode == 0:
            cprint("  ✅ TTS test successful", "green")
            return True
        else:
            cprint(f"  ❌ TTS test failed: {result.stderr}", "red")
            return False
            
    except Exception as e:
        cprint(f"  ❌ TTS test error: {e}", "red")
        return False

def print_summary_and_next_steps(results):
    """Print test summary and next steps."""
    cprint("\n" + "="*50, "cyan")
    cprint("📋 TEST SUMMARY", "cyan", attrs=['bold'])
    cprint("="*50, "cyan")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        cprint(f"  {status} {test_name}", "green" if passed else "red")
    
    cprint(f"\n📊 Overall: {passed_tests}/{total_tests} tests passed", 
           "green" if passed_tests == total_tests else "yellow")
    
    if passed_tests >= 6:  # Most tests passing
        cprint("\n🎉 Most tests passed! You can try running Opi!", "green", attrs=['bold'])
        cprint("\n🚀 Next steps:", "cyan")
        cprint("  1. python main.py --list-devices  # Check audio devices", "white")
        cprint("  2. espeak 'Hello Opi'             # Test TTS directly", "white")
        cprint("  3. python main.py --verbose       # Try voice assistant", "white")
    else:
        cprint("\n⚠️  Some tests failed. Please fix the issues above.", "yellow")

def main():
    """Run all tests."""
    cprint("🧪 Opi Voice Assistant - System Test", "cyan", attrs=['bold'])
    cprint("="*50, "cyan")
    
    results = {}
    
    # Run all tests
    results["Imports"] = test_imports()
    results["Audio System"] = test_audio_system()
    results["Speech Recognition"] = test_speech_recognition()
    results["TTS Backends"] = test_tts_backends()
    results["LLM API"] = test_llm_api()
    results["Directory Structure"] = test_directory_structure()
    results["Configuration"] = test_config_loading()
    results["Simple TTS"] = test_simple_tts()
    
    print_summary_and_next_steps(results)
    
    # Exit with error code if critical tests failed
    critical_failures = not all([
        results.get("Imports", False),
        results.get("Audio System", False),
        results.get("Configuration", False)
    ])
    
    if critical_failures:
        sys.exit(1)

if __name__ == "__main__":
    main()
