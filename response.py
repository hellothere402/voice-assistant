import openai
from typing import Dict, Optional, Union, List
from dataclasses import dataclass
import json
import time
from datetime import datetime
import pygame  
import os
import threading
from queue import Queue
import logging


@dataclass
class Query:
    text: str
    intent: str
    speaker_id: str
    context: Dict = None

@dataclass
class Response:
    text: str
    audio: bytes = None
    source: str = None
    cache_key: str = None

class ResponseGenerator:
    def __init__(self, openai_api_key: str, cache_file: str = "response_cache.json", voice_id: str = "nova"):
        self.api_key = openai_api_key
        self.cache_file = cache_file
        self.response_cache = self._load_cache()
        self.audio_queue = Queue()
        self.is_speaking = False

        # Initialize processors
        self.local_processor = LocalProcessor()
        self.cloud_processor = CloudProcessor(api_key=openai_api_key)
        self.response_merger = ResponseMerger()
        self.tts_engine = TTSEngine(api_key=openai_api_key, voice_id=voice_id)

        # Start audio playback thread
        self._start_audio_playback_thread()

    def _generate_cache_key(self, query: Query) -> str:
        """Generate a unique cache key for the query"""
        try:
            # Create a cache key based on text and intent
            key_parts = [
                query.text.lower().strip(),
                query.intent,
                query.speaker_id if query.speaker_id else "unknown"
            ]
            return "_".join(key_parts)
        except Exception as e:
            print(f"❌ Error generating cache key: {e}")
            return str(time.time())  # Fallback to timestamp
        
    def _check_cache(self, cache_key: str) -> Optional[Response]:
        """Check if response exists in cache"""
        try:
           cached_data = self.response_cache.get(cache_key)
           if cached_data:
            return Response(
                text=cached_data.get('text'),
                audio=cached_data.get('audio'),
                source=cached_data.get('source', 'cache'),
                cache_key=cache_key
            )
           return None
        except Exception as e:
         print(f"❌ Error checking cache: {e}")
        return None

    def _cache_response(self, cache_key: str, response: Response):
        """Cache a response"""
        try:
           self.response_cache[cache_key] = {
             'text': response.text,
             'audio': response.audio,
             'source': response.source,
             'timestamp': time.time()
           }
           self._save_cache(self.response_cache)
        except Exception as e:
         print(f"❌ Error caching response: {e}")

    def _should_cache(self, query: Query, response: Response) -> bool:
        """Determine if response should be cached"""
    # Don't cache error responses or simple queries
        if response.source == 'error' or self._is_simple_query(query):
          return False
        return True

    def _load_cache(self) -> dict:
        """Load response cache from file"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # If file doesn't exist, create an empty cache
            self._save_cache({})
            return {}

    def _save_cache(self, cache_data: dict):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logging.error(f"Error saving cache: {e}")

    async def generate_response(self, query: Query) -> Response:
        """Generate response for the given query"""
        try:
            response = None
            cache_key = self._generate_cache_key(query)

            # First try cache
            if not query.text.lower().startswith(('what', 'how', 'when')):  # Skip cache for time-sensitive queries
                response = self._check_cache(cache_key)
                if response:
                    return response

            # Try local processing
            if not response:
                response = await self.local_processor.process(query)

            # Try cloud processing
            if not response:
                response = await self.cloud_processor.process(query)

            # If response was successful and not a web search request
            if response and response.text and not response.source.startswith('web_required:'):
                # Cache non-web-search responses
                if self._should_cache(query, response):
                    self._cache_response(cache_key, response)

            return response

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return Response(
                text="I'm having trouble understanding. Could you please try again?",
                source="error"
            )

    def _is_simple_query(self, query: Query) -> bool:
        """Determine if query can be handled locally"""
        simple_intents = {'time', 'weather', 'reminder', 'alarm', 'volume'}
        return (
            query.intent in simple_intents or
            len(query.text.split()) < 5 or
            query.text.lower() in self.local_processor.command_templates
        )

    def cleanup(self):
        """Clean up resources"""
        try:
            # Signal audio playback thread to stop
            self.audio_queue.put(None)
            # Save cache
            self._save_cache(self.response_cache)
        except Exception as e:
            logging.error(f"Error in cleanup: {e}")

            
    def _start_audio_playback_thread(self):
      """Start thread for audio playback"""
      def playback_worker():
         try:
             pygame.mixer.init()
             while True:
                audio_data = self.audio_queue.get()
                if audio_data is None:
                    break
                self._play_audio(audio_data)
         except Exception as e:
            print(f"❌ Error in audio playback: {e}")
         finally:
            pygame.mixer.quit()

      self.playback_thread = threading.Thread(target=playback_worker)
      self.playback_thread.daemon = True
      self.playback_thread.start()

    class ResponseGenerator:
     def __init__(self, openai_api_key: str, cache_file: str = "response_cache.json", voice_id: str = "nova"):
        self.api_key = openai_api_key
        self.cache_file = cache_file
        self.response_cache = self._load_cache()
        self.audio_queue = Queue()
        self.is_speaking = False  
        

    def _play_audio(self, audio_data: bytes):
        """Play audio data"""
        temp_file = f"temp_audio_{int(time.time())}.mp3"
        try:
            self.is_speaking = True  # Set flag when starting to speak
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            pygame.mixer.music.unload()
            self.audio_queue.task_done()
            
        finally:
            self.is_speaking = False  
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"⚠️ Warning: Could not remove temporary file: {e}")

class LocalProcessor:
    def __init__(self):
        """Initialize LocalProcessor with command templates"""
        self.command_templates = self._load_command_templates()

    async def process(self, query: Query) -> Response:
        """Process simple queries locally"""
        if query.intent == 'time':
            current_time = datetime.now().strftime("%I:%M %p")
            return Response(
                text=f"It's currently {current_time}.",
                source="local"
            )
        elif query.intent == 'weather':
            return Response(
                text="I don't currently have access to weather data.",
                source="local"
            )
        else:
            
            template = self.command_templates.get(query.text.lower())
            if template:
                return Response(text=template, source="local")
            
            
            return None

    def _load_command_templates(self) -> Dict[str, str]:
        """Load command templates from file"""
        try:
            with open("command_templates.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "hello": "Hello! How can I help you today?",
                "goodbye": "Goodbye! Have a great day!",
                "thank you": "You're welcome! Is there anything else I can help you with?",
                "help": "I'm here to help. What can I assist you with?"
            }

    def _process_template(self, query: Query) -> Response:
        """Process query using templates"""
        response_text = self.command_templates.get(
            query.text.lower(),
            "I'm not sure about that. Would you like me to ask the cloud?"
        )
        return Response(text=response_text, source="local")


class CloudProcessor:
    def __init__(self, api_key: str):
        self.conversation_history = {}
        self.client = openai.OpenAI(api_key=api_key)

    async def process(self, query: Query) -> Response:
      """Process queries using OpenAI's GPT"""
      try:
        # First, determine if this query needs web search
        system_prompt = """You are a helpful home assistant with web access capabilities. For queries needing real-time data, respond with the exact format:
        [[SEARCH:type=<search_type>]]
        where <search_type> should be one of: news, scores, weather, price, general
        
        Examples:
        - For news queries: [[SEARCH:type=news]]
        - For sports scores: [[SEARCH:type=scores]]
        - For weather info: [[SEARCH:type=weather]]
        - For prices: [[SEARCH:type=price]]
        - For other real-time info: [[SEARCH:type=general]]
        
        Important: Respond ONLY with the search directive for queries needing real-time data.
        For other queries, respond normally with a helpful answer."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query.text}
        ]
        
        # Ask GPT if this needs web search
        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        response_text = completion.choices[0].message.content
        
        # Check for search directive
        if "[[SEARCH:" in response_text:
            # Extract search type
            search_type = response_text.split("type=")[1].split("]]")[0]
            # Return special response with search type
            return Response(
                text="",  # Empty text since we'll get it from web search
                source=f"web_required:{search_type}"
            )
        
        # Otherwise, proceed with normal conversation
        history = self.conversation_history.get(query.speaker_id, [])
        messages = self._prepare_messages(query, history)
        
        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        response_text = completion.choices[0].message.content
        
        # Update conversation history
        history.extend([
            {"role": "user", "content": query.text},
            {"role": "assistant", "content": response_text}
        ])
        self.conversation_history[query.speaker_id] = history[-20:]
        
        return Response(
            text=response_text,
            source="cloud"
        )
        
      except Exception as e:
        logging.error(f"Error in cloud processing: {e}")
        raise

    def _prepare_messages(self, query: Query, history: List) -> List[Dict]:
        """Prepare messages for GPT"""
        messages = [
            {"role": "system", "content": (
                "You are a helpful home assistant speaking with a person. "
                "Keep responses clear, concise, and friendly. "
                "If you're unsure about something, always err on the side of safety. "
                "Use natural conversational language."
            )}
        ]
        
        
        if query.context:
            messages.append({
                "role": "system",
                "content": f"Current context: {json.dumps(query.context)}"
            })
            
        # Add relevant history
        messages.extend(history[-4:])  # Last 2 exchanges
        
        # Add current query
        messages.append({"role": "user", "content": query.text})
        
        return messages


class ResponseMerger:
    def merge(self, response: Response) -> Response:
        """Merge responses if needed"""
        # For now, just return the response
        # In future, could merge multiple responses or add context
        return response


class TTSEngine:
    def __init__(self, api_key: str, voice_id: str = "nova"):
        self.client = openai.OpenAI(api_key=api_key)
        self.voice_id = voice_id
    
    async def generate_speech(self, text: str) -> bytes:
        """Generate speech from text using OpenAI's TTS"""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=self.voice_id,
                input=text
            )
            return response.content
            
        except Exception as e:
            print(f"Error in TTS: {e}")
            return None


