# voice/text_input_worker.py
"""
Text input worker for Opi Voice Assistant
Handles CLI text input as alternative to speech
"""

import threading
import queue
import time
import sys
from termcolor import cprint


class TextInputWorker:
    """Worker for handling text input from CLI."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.running = False
        
    def process_text_input(self, text_queue, stop_event, timings, text_only_mode=False):
        """Process text input from CLI."""
        try:
            if text_only_mode:
                prompt = ""
            else:
                prompt = "Type: "
            
            while not stop_event.is_set():
                try:
                    # Use a non-blocking approach to check for input
                    if text_only_mode:
                        # In text-only mode, we can block on input
                        user_input = input(prompt).strip()
                    else:
                        # In hybrid mode, check periodically for input
                        import select
                        import sys
                        
                        # Check if input is available (Unix-like systems)
                        if hasattr(select, 'select'):
                            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                            if ready:
                                user_input = input(prompt).strip()
                            else:
                                continue
                        else:
                            # Fallback for Windows - just use input with a timeout simulation
                            try:
                                cprint(f"\n{prompt}", "white", end="", flush=True)
                                user_input = input().strip()
                            except KeyboardInterrupt:
                                break
                    
                    if not user_input:
                        continue
                        
                    # Check for exit commands
                    if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                        if text_only_mode:
                            cprint("[Text] 👋 Goodbye!", "yellow")
                            stop_event.set()
                            break
                        else:
                            cprint("[Text] 📝 Text input stopped (voice still active)", "yellow")
                            break
                    
                    # Add to queue with current timestamp
                    text_data = {
                        'text': user_input,
                        'user_speech_end_time': time.time(),
                        'source': 'text'
                    }
                    
                    text_queue.put(text_data)
                    
                    if self.verbose:
                        cprint(f"[Text] Queued: \"{user_input}\"", "cyan")
                        
                except EOFError:
                    # Handle Ctrl+D
                    cprint("\n[Text] 📝 Text input ended", "yellow")
                    if text_only_mode:
                        stop_event.set()
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C
                    cprint("\n[Text] 📝 Text input interrupted", "yellow")
                    if text_only_mode:
                        stop_event.set()
                    break
                except Exception as e:
                    if self.verbose:
                        cprint(f"[Text] Input error: {e}", "red")
                    continue
                    
        except Exception as e:
            cprint(f"[Text] Worker error: {e}", "red")
            if text_only_mode:
                stop_event.set()
        finally:
            if self.verbose:
                cprint("[Text] Text input worker finished", "green")


class AsyncTextInputWorker:
    """Non-blocking text input worker for hybrid mode."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.input_thread = None
        self.running = False
        
    def start_async_input(self, text_queue, stop_event):
        """Start non-blocking text input in a separate thread."""
        if self.running:
            return
            
        self.running = True
        self.input_thread = threading.Thread(
            target=self._async_input_loop,
            args=(text_queue, stop_event),
            daemon=True,
            name="AsyncTextInput"
        )
        self.input_thread.start()
        
        cprint("[Text] 💬 Async text input started (type messages anytime)", "green")
    
    def _async_input_loop(self, text_queue, stop_event):
        """Async input loop that doesn't block the main thread."""
        try:
            import sys
            
            cprint("\n" + "="*50, "cyan")
            cprint("📝 TEXT INPUT AVAILABLE", "cyan", attrs=['bold'])
            cprint("You can type messages at any time!", "green")
            cprint("Commands: 'quit' = stop text input, 'exit' = stop everything", "yellow")
            cprint("="*50, "cyan")
            
            while self.running and not stop_event.is_set():
                try:
                    # Platform-specific non-blocking input
                    user_input = self._get_non_blocking_input()
                    
                    if user_input is None:
                        continue
                        
                    user_input = user_input.strip()
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.lower() == 'quit':
                        cprint("[Text] 📝 Text input stopped (voice continues)", "yellow")
                        self.running = False
                        break
                    elif user_input.lower() in ['exit', 'stop', 'goodbye']:
                        cprint("[Text] 🛑 Stopping entire system", "yellow")
                        stop_event.set()
                        break
                    
                    # Queue the text input
                    text_data = {
                        'text': user_input,
                        'user_speech_end_time': time.time(),
                        'source': 'text'
                    }
                    
                    text_queue.put(text_data)
                    cprint(f"[Text] 📝 Sent: \"{user_input}\"", "blue")
                    
                except Exception as e:
                    if self.verbose:
                        cprint(f"[Text] Async input error: {e}", "red")
                    time.sleep(0.1)
                    
        except Exception as e:
            cprint(f"[Text] Async worker error: {e}", "red")
        finally:
            self.running = False
            if self.verbose:
                cprint("[Text] Async text input worker finished", "green")
    
    def _get_non_blocking_input(self):
        """Get non-blocking input (platform-specific)."""
        import sys
        import platform
        
        if platform.system() == "Windows":
            # Windows-specific non-blocking input
            import msvcrt
            
            if msvcrt.kbhit():
                line = ""
                while True:
                    char = msvcrt.getch().decode('utf-8')
                    if char == '\r':  # Enter key
                        print()  # New line
                        return line
                    elif char == '\b':  # Backspace
                        if line:
                            line = line[:-1]
                            sys.stdout.write('\b \b')
                            sys.stdout.flush()
                    else:
                        line += char
                        sys.stdout.write(char)
                        sys.stdout.flush()
            else:
                time.sleep(0.05)
                return None
        else:
            # Unix-like systems
            import select
            
            if select.select([sys.stdin], [], [], 0.05)[0]:
                return input("Type: ")
            else:
                return None
    
    def stop(self):
        """Stop the async text input."""
        self.running = False
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
