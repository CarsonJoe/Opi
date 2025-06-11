# voice/audio_utils.py
"""
Audio utilities for Opi Voice Assistant
"""

import sounddevice as sd
from termcolor import cprint
import sys


def list_audio_devices():
    """List available audio input and output devices."""
    cprint("🎵 Available Audio Devices:", "yellow")
    try:
        devices = sd.query_devices()
        has_input_devices = False
        has_output_devices = False
        
        cprint("\n📥 Input Devices (Microphones):", "cyan")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                has_input_devices = True
                default_marker = " (DEFAULT)" if i == sd.default.device[0] else ""
                cprint(f"  [{i:2d}] {device['name']}{default_marker}", "green")
                cprint(f"       Channels: {device['max_input_channels']}, "
                      f"Sample Rate: {device['default_samplerate']:.0f}Hz", "white")
        
        if not has_input_devices:
            cprint("  ❌ No input devices found.", "red")
        
        cprint("\n📤 Output Devices (Speakers):", "cyan")
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                has_output_devices = True
                default_marker = " (DEFAULT)" if i == sd.default.device[1] else ""
                cprint(f"  [{i:2d}] {device['name']}{default_marker}", "green")
                cprint(f"       Channels: {device['max_output_channels']}, "
                      f"Sample Rate: {device['default_samplerate']:.0f}Hz", "white")
        
        if not has_output_devices:
            cprint("  ❌ No output devices found.", "red")
        
        # Show current defaults
        try:
            default_input = sd.query_devices(sd.default.device[0])
            default_output = sd.query_devices(sd.default.device[1])
            cprint(f"\n🎯 Current Defaults:", "yellow")
            cprint(f"  Input:  [{sd.default.device[0]:2d}] {default_input['name']}", "white")
            cprint(f"  Output: [{sd.default.device[1]:2d}] {default_output['name']}", "white")
        except Exception as e:
            cprint(f"\n❌ Error getting default devices: {e}", "red")
            
    except Exception as e:
        cprint(f"❌ Could not query audio devices: {e}", "red")
        cprint("💡 Try: sudo apt install python3-dev portaudio19-dev", "yellow")


def test_audio_recording(duration=3, device=None):
    """Test audio recording functionality."""
    cprint(f"🎤 Testing audio recording for {duration} seconds...", "yellow")
    
    try:
        import numpy as np
        
        # Record audio
        sample_rate = 44100
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device
        )
        
        cprint("🔴 Recording... (speak into microphone)", "red")
        sd.wait()  # Wait until recording is finished
        
        # Analyze the recording
        rms = np.sqrt(np.mean(recording.flatten()**2))
        max_amplitude = np.max(np.abs(recording.flatten()))
        
        cprint("✅ Recording complete!", "green")
        cprint(f"📊 Audio Analysis:", "cyan")
        cprint(f"  RMS Level: {rms:.4f}", "white")
        cprint(f"  Max Amplitude: {max_amplitude:.4f}", "white")
        cprint(f"  Duration: {len(recording) / sample_rate:.1f}s", "white")
        
        if rms > 0.01:
            cprint("🎉 Audio input is working well!", "green")
        elif rms > 0.001:
            cprint("⚠️  Audio detected but level is low. Check microphone volume.", "yellow")
        else:
            cprint("❌ No audio detected. Check microphone connection.", "red")
            
        return recording, sample_rate
        
    except Exception as e:
        cprint(f"❌ Audio recording test failed: {e}", "red")
        return None, None


def test_audio_playback(frequency=440, duration=2, device=None):
    """Test audio playback with a sine wave."""
    cprint(f"🔊 Testing audio playback ({frequency}Hz tone for {uration}s)...", "yellow")
    
    try:
        import numpy as np
        
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate sine wave
        sine_wave = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        cprint("🔊 Playing test tone...", "blue")
        sd.play(sine_wave, samplerate=sample_rate, device=device)
        sd.wait()  # Wait until playback is finished
        
        cprint("✅ Playback test complete!", "green")
        cprint("💡 Did you hear the tone? If not, check speaker volume and connections.", "cyan")
        
        return True
        
    except Exception as e:
        cprint(f"❌ Audio playback test failed: {e}", "red")
        return False


def get_recommended_audio_settings():
    """Get recommended audio settings for the current system."""
    cprint("🔧 Analyzing audio system for optimal settings...", "yellow")
    
    try:
        devices = sd.query_devices()
        
        # Find best input device
        best_input = None
        best_input_score = 0
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                score = 0
                
                # Prefer higher sample rates
                if device['default_samplerate'] >= 44100:
                    score += 2
                elif device['default_samplerate'] >= 22050:
                    score += 1
                
                # Prefer USB/external devices over built-in
                if 'usb' in device['name'].lower():
                    score += 3
                elif 'external' in device['name'].lower():
                    score += 2
                
                # Prefer devices with "mic" in the name
                if 'mic' in device['name'].lower():
                    score += 1
                
                if score > best_input_score:
                    best_input_score = score
                    best_input = (i, device)
        
        # Find best output device
        best_output = None
        best_output_score = 0
        
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                score = 0
                
                # Prefer higher sample rates
                if device['default_samplerate'] >= 44100:
                    score += 2
                elif device['default_samplerate'] >= 22050:
                    score += 1
                
                # Prefer USB/external devices
                if 'usb' in device['name'].lower():
                    score += 3
                elif 'hdmi' in device['name'].lower():
                    score += 2
                elif 'analog' in device['name'].lower() or '3.5mm' in device['name'].lower():
                    score += 1
                
                if score > best_output_score:
                    best_output_score = score
                    best_output = (i, device)
        
        cprint("🎯 Recommended Audio Settings:", "green")
        
        if best_input:
            cprint(f"  Input Device: [{best_input[0]:2d}] {best_input[1]['name']}", "cyan")
            cprint(f"    Sample Rate: {best_input[1]['default_samplerate']:.0f}Hz", "white")
        else:
            cprint("  ❌ No suitable input device found", "red")
        
        if best_output:
            cprint(f"  Output Device: [{best_output[0]:2d}] {best_output[1]['name']}", "cyan")
            cprint(f"    Sample Rate: {best_output[1]['default_samplerate']:.0f}Hz", "white")
        else:
            cprint("  ❌ No suitable output device found", "red")
        
        # Return configuration
        config = {
            "input_device": best_input[0] if best_input else None,
            "output_device": best_output[0] if best_output else None,
            "sample_rate": int(min(
                best_input[1]['default_samplerate'] if best_input else 44100,
                best_output[1]['default_samplerate'] if best_output else 44100
            ))
        }
        
        cprint(f"\n📋 Suggested config.json settings:", "yellow")
        cprint(f'  "voice": {{', "white")
        cprint(f'    "audio_input_device": {config["input_device"]},', "white")
        cprint(f'    "audio_output_device": {config["output_device"]},', "white")
        cprint(f'    "sample_rate": {config["sample_rate"]}', "white")
        cprint(f'  }}', "white")
        
        return config
        
    except Exception as e:
        cprint(f"❌ Error analyzing audio system: {e}", "red")
        return None


def check_audio_permissions():
    """Check if the current user has audio permissions."""
    cprint("🔐 Checking audio permissions...", "yellow")
    
    import os
    import grp
    
    try:
        # Check if user is in audio group
        user_groups = [grp.getgrgid(gid).gr_name for gid in os.getgroups()]
        
        if 'audio' in user_groups:
            cprint("✅ User is in 'audio' group", "green")
        else:
            cprint("❌ User is NOT in 'audio' group", "red")
            cprint("💡 Fix: sudo usermod -a -G audio $USER", "yellow")
            cprint("   Then logout and login again", "yellow")
        
        # Check device permissions
        audio_devices = ['/dev/snd/', '/dev/dsp', '/dev/audio']
        
        for device in audio_devices:
            if os.path.exists(device):
                if os.access(device, os.R_OK | os.W_OK):
                    cprint(f"✅ Can access {device}", "green")
                else:
                    cprint(f"❌ Cannot access {device}", "red")
        
        return 'audio' in user_groups
        
    except Exception as e:
        cprint(f"❌ Error checking permissions: {e}", "red")
        return False


if __name__ == "__main__":
    """Run audio diagnostics if script is called directly."""
    cprint("🎵 Opi Audio Diagnostics", "cyan", attrs=['bold'])
    cprint("=" * 50, "cyan")
    
    # Check permissions first
    check_audio_permissions()
    print()
    
    # List devices
    list_audio_devices()
    print()
    
    # Get recommendations
    get_recommended_audio_settings()
    print()
    
    # Offer interactive tests
    try:
        cprint("🧪 Interactive Tests Available:", "yellow")
        cprint("  r - Test recording (3 seconds)", "white")
        cprint("  p - Test playback (tone)", "white")
        cprint("  q - Quit", "white")
        
        while True:
            choice = input("\nEnter choice (r/p/q): ").lower().strip()
            
            if choice == 'r':
                test_audio_recording()
            elif choice == 'p':
                test_audio_playback()
            elif choice == 'q':
                break
            else:
                cprint("Invalid choice. Use r, p, or q.", "red")
                
    except KeyboardInterrupt:
        cprint("\n👋 Audio diagnostics complete!", "cyan")
