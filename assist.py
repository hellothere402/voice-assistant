import asyncio
import logging
from typing import Optional
from dataclasses import dataclass
import yaml
import signal
import sys
import time
from response import Query
import numpy as np
import os
import pygame

from sym import SystemManager
from voice import VoiceProcessingSystem
from audio import AudioInputSystem
from response import ResponseGenerator

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

@dataclass
class SystemConfig:
    openai_api_key: str
    sample_rate: int = 16000
    device_name: str = "VoiceAssistant"
    debug_mode: bool = False
    db_path: str = "system.db"
    voice_id: str = "nova"
       

class IntegratedVoiceAssistant:
    def __init__(self, config_path: str = "config.yaml"):
     print("Initializing IntegratedVoiceAssistant...")
     try:
        # Load configuration
        self.config = self._load_config(config_path)
        print("Config loaded successfully")
        
        # Initialize system components
        print("Initializing components...")
        self._init_components()
        print("Components initialized")
        
        # Setup signal handlers
        print("Setting up signal handlers...")
        self._setup_signal_handlers()
        print("Signal handlers set up")
        
        # Initialize state
        self.is_running = False
        self.start_time = time.time()
        print("Initialization complete")
     except Exception as e:
        print(f"Error during initialization: {e}")
        raise

    def _load_config(self, config_path: str) -> SystemConfig:
      """Load system configuration"""
      print(f"Loading config from {config_path}")
      try:
         print("Checking if config file exists...")
         if not os.path.exists(config_path):  # Fixed typo here, was 'exits'
            print(f"Config file {config_path} does not exist")
            raise FileNotFoundError(f"Config file {config_path} not found")
        
         print("Config file exists, attempting to load...")
         with open(config_path, 'r') as f:
            print("Reading config file...")
            config_data = yaml.safe_load(f)
            print(f"Loaded raw config data: {config_data}")
            
         # Extract voice_id from voice_settings if present
         if 'voice_settings' in config_data:
            config_data['voice_id'] = config_data['voice_settings'].get('voice_id', 'nova')
            del config_data['voice_settings']
            
         if not config_data.get('openai_api_key'):
            raise ValueError("OpenAI API key not found in config")
        
         print("Creating SystemConfig object...")
         return SystemConfig(**config_data)
        
      except FileNotFoundError:
        print("Config file not found, creating default config...")
        default_config = {
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'sample_rate': 16000,
            'device_name': "VoiceAssistant",
            'debug_mode': False,
            'db_path': "system.db",
            'voice_id': "nova"
        }
        
        if not default_config['openai_api_key']:
            print("No OpenAI API key found in environment variables")
            raise ValueError("OpenAI API key not found in environment variables")
            
        # Save default config
        print("Saving default config...")
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
            
        print("Default config saved, returning SystemConfig object...")
        return SystemConfig(**default_config)
        
      except Exception as e:
        print(f"Unexpected error loading config: {e}")
        print(f"Error type: {type(e)}")
        print("Using default configuration...")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        return SystemConfig(openai_api_key=api_key)

    def _init_components(self):
     """Initialize all system components"""
     try:
         # Initialize system manager first
         self.system_manager = SystemManager(self.config.db_path)
        
         # Initialize audio input system
         self.audio_system = AudioInputSystem()
        
         # Initialize voice processing
         self.voice_processor = VoiceProcessingSystem(
            self.config.openai_api_key
         )
        
         # Initialize response generator with voice_id
         self.response_generator = ResponseGenerator(
            self.config.openai_api_key,
            cache_file="response_cache.json",
            voice_id=self.config.voice_id  # Pass the voice_id from config
         )
        
         logging.info("All components initialized successfully")
        
     except Exception as e:
        logging.error(f"Error initializing components: {e}")
        raise

    async def setup_voice_profile(self, name: str) -> bool:
        """Create initial voice profile"""
        try:
            print("\nPreparing to create voice profile...")
            print("Please speak for 5 seconds to create your voice profile...")
            print("Recording will start in 3 seconds...")
            await asyncio.sleep(3)  # Give user time to prepare
            
            print("\n🎤 Recording NOW - Please speak normally...")
            audio_data = await self._record_audio_sample(5)
            
            if audio_data is not None:
                print("\nRecording complete. Processing voice profile...")
                profile = self.voice_processor.create_voice_profile(audio_data, name)
                
                if profile:
                    success = await self.add_voice_profile(audio_data, name)
                    if success:
                        print(f"\n✅ Voice profile created successfully for {name}")
                        return True
                    else:
                        print("\n❌ Failed to save voice profile")
                else:
                    print("\n❌ Failed to create voice profile")
            else:
                print("\n❌ No audio data recorded")
                
            return False
        except Exception as e:
            print(f"\n❌ Error in setup_voice_profile: {e}")
            return False
        
    async def select_voice(self) -> str:
     """Allow user to select assistant voice"""
     voices = {
        "1": ("nova", "Female, warm and professional"),
        "2": ("shimmer", "Female, young and energetic"),
        "3": ("alloy", "Male, neutral and balanced"),
        "4": ("echo", "Male, deep and clear"),
        "5": ("fable", "Male, British accent"),
        "6": ("onyx", "Male, deep and authoritative")
      }  # Just close the dictionary with }
    

     print("\n🗣️ Please select a voice for your assistant:")
     for key, (voice, description) in voices.items():
        print(f"{key}. {description}")
     
     while True:
        try:
            choice = input("\nEnter number (1-6) [default=1]: ").strip() or "1"
            if choice in voices:
                voice_id = voices[choice][0]
                await self.update_voice_settings(voice_id)
                print(f"\n✅ Voice set to: {voices[choice][1]}")
                return voice_id
            else:
                print("❌ Invalid choice. Please enter a number between 1 and 6.")
        except Exception as e:
            print(f"❌ Error selecting voice: {e}")
            return "nova"

          
    async def update_voice_settings(self, voice_id: str) -> bool:
       """Update voice settings in config"""
       try:
          with open("config.yaml", 'r') as f:
            config_data = yaml.safe_load(f)
        
          config_data['voice_id'] = voice_id
        
          with open("config.yaml", 'w') as f:
            yaml.dump(config_data, f)
            
          self.config.voice_id = voice_id
          return True
       except Exception as e:
        print(f"Error updating voice settings: {e}")
        return False

    async def _record_audio_sample(self, duration: int) -> Optional[np.ndarray]:
     """Record an audio sample for the specified duration"""
     try:
        if not self.audio_system.is_running:
            self.audio_system.start_audio_stream()
        
        # Calculate required samples
        required_samples = int(self.config.sample_rate * duration)
        collected_samples = []
        
        print("Recording...")
        start_time = time.time()
        
        while len(collected_samples) < required_samples:
            if time.time() - start_time > duration + 2:  # Add timeout
                break
            audio_data = self.audio_system.get_audio_buffer()
            if len(audio_data) > 0:
                collected_samples.extend(audio_data)
            await asyncio.sleep(0.1)
        
        if len(collected_samples) >= required_samples:
            # Trim to exact length
            audio_data = np.array(collected_samples[:required_samples], dtype=np.float32)
            audio_data = audio_data.reshape(1, -1)
            print(f"Recorded audio shape: {audio_data.shape}")
            return audio_data
        else:
            print("❌ Insufficient audio data recorded")
            return None
            
     except Exception as e:
        print(f"❌ Error recording audio sample: {e}")
        return None

    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        print("\nInitiating graceful shutdown...")
        self.stop()
        sys.exit(0)

    async def start(self):
      """Start the voice assistant"""
      if self.is_running:
        return

      self.is_running = True
      print("\n✨ Voice assistant is now active and listening!")
      print("Speak clearly and I'll respond to you.")
      print("Press Ctrl+C to exit.\n")

      try:
         # Start audio input system
         self.audio_system.start_audio_stream()
        
         # Main processing loop
         while self.is_running:
            await self._process_cycle()
            
      except Exception as e:
        logging.error(f"Error in main loop: {e}")
        self.stop()

    async def _process_cycle(self):
     """Main processing cycle"""
     try:
        # Only process new input if we're not already handling a query
        if (hasattr(self, 'is_processing') and self.is_processing) or \
           self.response_generator.is_speaking:  # Check if assistant is speaking
            await asyncio.sleep(0.1)
            return

        # Get audio from buffer
        audio_data = self.audio_system.get_audio_buffer()
        
        if len(audio_data) > 0 and len(audio_data) >= self.config.sample_rate * 2:
            # Set processing flag
            self.is_processing = True
            self.audio_system.pause()  # Pause audio input
            
            try:
                print("\nProcessing audio input...")
                
                # Process voice and generate response
                audio_data = audio_data.reshape(1, -1)
                voice_result = self.voice_processor.process_voice(
                    audio_data,
                    self.config.sample_rate
                )
                
                if voice_result and voice_result.text:
                    print(f"\n👤 You said: {voice_result.text}")
                    
                    query_obj = Query(
                        text=voice_result.text,
                        intent=voice_result.intent,
                        speaker_id=voice_result.speaker_id
                    )
                    
                    print("💭 Generating response...")
                    response = await self.response_generator.generate_response(query_obj)
                    
                    if response and response.text:
                        print(f"🤖 Assistant: {response.text}\n")
                        
                        # Wait for TTS audio to complete
                        if response.audio:
                            # Wait for the playback to fully complete
                            while self.response_generator.is_speaking or \
                                  self.response_generator.audio_queue.qsize() > 0:
                                await asyncio.sleep(0.1)
                            
                            # Additional wait to ensure everything is done
                            await asyncio.sleep(0.2)
                            
                            # Now clear the buffer and resume listening
                            self.audio_system.audio_buffer.clear()
                            print("\n👂 Listening for your voice...")
                
            finally:
                self.is_processing = False
                self.audio_system.resume()
                
        await asyncio.sleep(0.1)

     except Exception as e:
        print(f"❌ Error in processing cycle: {e}")
        print(f"Error details: {str(e)}")
        self.system_manager.performance_monitor.log_error()
        self.is_processing = False
        self.audio_system.resume()

    async def _wait_for_audio_completion(self):
      """Wait for TTS audio playback to complete"""
      while self.response_generator.audio_queue.qsize() > 0:
        await asyncio.sleep(0.1)
      # Additional small delay after audio completes
      await asyncio.sleep(0.2)

    def stop(self):
        """Stop the voice assistant"""
        logging.info("Stopping voice assistant...")
        self.is_running = False
        
        # Cleanup components
        try:
            self.audio_system.stop()
            self.response_generator.cleanup()
            self.system_manager.cleanup()
            
            logging.info("Voice assistant stopped successfully")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")

    async def add_voice_profile(self, audio_data: np.ndarray, name: str) -> bool:
        """Add a new voice profile"""
        try:
            profile = self.voice_processor.create_voice_profile(
                audio_data,
                name
            )
            return self.system_manager.voice_manager.add_profile(profile)
            
        except Exception as e:
            logging.error(f"Error adding voice profile: {e}")
            return False

    def get_system_status(self) -> dict:
        """Get current system status"""
        return {
            "metrics": self.system_manager.performance_monitor.collect_metrics(),
            "active_profiles": len(self.system_manager.voice_manager.active_profiles),
            "cache_size": self.system_manager.cache_manager.get_cache_size(),
            "uptime": time.time() - self.start_time
        }


if __name__ == "__main__":
    print("Starting program...")
    async def main():
        print("Entered Main Function")
        try:
            print("\n🚀 Initializing Voice Assistant...")
            print("Creating IntegratedVoiceAssistant...")
            assistant = IntegratedVoiceAssistant()
            print("Assistant created successfully")
            
            # Voice selection on first run
            print("Checking for voice settings...")
            try:
                with open("config.yaml", 'r') as f:
                    config = yaml.safe_load(f)
                    print("Config loaded:", config)
                if 'voice_id' not in config:
                    print("\n🗣️ Initial setup - Voice selection")
                    voice_id = await assistant.select_voice()
                    assistant.config.voice_id = voice_id
                    assistant.response_generator = ResponseGenerator(
                        assistant.config.openai_api_key,
                        cache_file="response_cache.json",
                        voice_id=voice_id
                    )
            except FileNotFoundError:
                print("Config file not found, starting voice selection...")
                voice_id = await assistant.select_voice()
                assistant.config.voice_id = voice_id
                assistant.response_generator = ResponseGenerator(
                    assistant.config.openai_api_key,
                    cache_file="response_cache.json",
                    voice_id=voice_id
                )
            
            print("Setting up voice profile...")
            profile_created = await assistant.setup_voice_profile("User1")
            
            if profile_created or os.path.exists("voice_profiles.json"):
                print("\n🎯 Starting voice assistant... Press Ctrl+C to stop")
                print("👂 Listening for your voice...")
                await assistant.start()
            else:
                print("\n❌ Could not create voice profile. System will now exit.")
                return

        except KeyboardInterrupt:
            print("\n\n👋 Shutting down by user request...")
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            print(f"Exception type: {type(e)}")
            print(f"Full exception details: {str(e)}")
            raise
        finally:
            if 'assistant' in locals():
                print("Stopping assistant...")
                assistant.stop()
            print("\n✨ System shutdown complete")

    # Run the async main function
    import asyncio
    asyncio.run(main())