import pyaudio
import numpy as np
import wave
from collections import deque
import webrtcvad
from queue import Queue
from typing import Optional
import queue

class AudioInputSystem:
    def __init__(self):
        # Audio parameters
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000  # Required for WebRTC VAD
        self.CHUNK_DURATION_MS = 30  # Duration of each chunk in milliseconds
        self.CHUNK = int(self.RATE * self.CHUNK_DURATION_MS / 1000)  # Samples per chunk
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Initialize VAD (Voice Activity Detection)
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (0-3)
        
        # Buffer setup (3 seconds of audio)
        self.buffer_duration_seconds = 3
        self.buffer_size = int(self.RATE * self.buffer_duration_seconds)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Queue for processing detected voice segments
        self.voice_segments_queue = Queue()
        
        # Control flags
        self.is_running = False
        self.stream: Optional[pyaudio.Stream] = None

        self.is_paused = False

    def start_audio_stream(self):
      """Start the audio input stream"""
      if self.is_running:
         return

      self.is_running = True
      self.audio_buffer.clear()  # Clear any old data
    
      # Open audio stream
      self.stream = self.audio.open(
        format=self.FORMAT,
        channels=self.CHANNELS,
        rate=self.RATE,
        input=True,
        frames_per_buffer=self.CHUNK,
        stream_callback=self._audio_callback
      )
    
      print("Audio input system started")
    
    def pause(self):
        """Pause audio input processing"""
        self.is_paused = True
        self.audio_buffer.clear()  # Clear any buffered audio
        
    def resume(self):
        """Resume audio input processing"""
        self.is_paused = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        try:
            # Don't process audio if paused
            if self.is_paused:
                return (in_data, pyaudio.paContinue)
                
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Add to buffer only if not paused
            if not self.is_paused:
                self.audio_buffer.extend(audio_data)
            
            # Convert to format suitable for VAD (16-bit PCM)
            audio_for_vad = (audio_data * 32767).astype(np.int16).tobytes()
            
            # Check for voice activity only if not paused
            if not self.is_paused and self.vad.is_speech(audio_for_vad, self.RATE):
                self.voice_segments_queue.put(audio_data)
            
            return (in_data, pyaudio.paContinue)
            
        except Exception as e:
            print(f"Error in audio callback: {e}")
            return (None, pyaudio.paAbort)
        
    def _process_audio(self):
      """Process detected voice segments"""
      while self.is_running:
        try:
            # Get voice segment from queue without timeout
            audio_segment = self.voice_segments_queue.get()
            print("🎤 Voice detected - Listening...")
            
            # Collect audio segments for a complete utterance
            segments = [audio_segment]
            silence_counter = 0
            
            # Keep collecting until definitive silence is detected
            # No timeout, just wait for clear end of speech
            while silence_counter < 15:  # Increased silence threshold
                try:
                    segment = self.voice_segments_queue.get(timeout=0.1)
                    segments.append(segment)
                    silence_counter = 0
                except queue.Empty:
                    silence_counter += 1
            
            # Combine all segments
            complete_utterance = np.concatenate(segments)
            self.audio_buffer.extend(complete_utterance)
            print("✨ Processing your speech...")
            
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
   
    def get_audio_buffer(self):
        """Get the current contents of the audio buffer"""
        return np.array(list(self.audio_buffer))

    def stop(self):
        """Stop the audio input system"""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        print("Audio input system stopped")

    def __del__(self):
        """Cleanup on deletion"""
        self.stop()


class AudioBufferManager:
    """Manages the audio buffer and provides utilities for audio manipulation"""
    
    def __init__(self, max_duration_seconds: int = 3, sample_rate: int = 16000):
        self.max_duration = max_duration_seconds
        self.sample_rate = sample_rate
        self.max_samples = max_duration_seconds * sample_rate
        self.buffer = deque(maxlen=self.max_samples)
        
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to the buffer"""
        self.buffer.extend(audio_data)
    
    def get_latest_audio(self, duration_seconds: float = None) -> np.ndarray:
        """Get the latest audio from the buffer"""
        if duration_seconds is None:
            return np.array(list(self.buffer))
        
        samples = int(duration_seconds * self.sample_rate)
        return np.array(list(self.buffer)[-samples:])
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()

    def save_buffer_to_wav(self, filename: str):
        """Save current buffer contents to WAV file"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   
            wf.setframerate(self.sample_rate)
            audio_data = (np.array(list(self.buffer)) * 32767).astype(np.int16)
            wf.writeframes(audio_data.tobytes())


        