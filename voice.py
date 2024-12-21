import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import openai
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import os
import time
import wave

os.environ["TRANSFORMERS_OFFLINE"] = "1"


@dataclass
class VoiceProfile:
    user_id: str
    embeddings: List[np.ndarray]
    name: str
    created_at: float

@dataclass
class ProcessedSpeech:
    text: str
    confidence: float
    speaker_id: Optional[str]
    intent: Optional[str]

class IntentClassifier:
    def __init__(self):
        self.intents = {
            "query": ["what", "how", "why", "when", "where", "who"],
            "command": ["set", "turn", "play", "stop", "start", "open"],
            "conversation": ["tell", "chat", "talk", "discuss"],
            "emergency": ["help", "emergency", "urgent", "pain"]
        }

    def classify(self, text: str) -> str:
        """Classify the intent of the text"""
        try:
            text = text.lower()
            
            # Check emergency intent first
            if any(word in text for word in self.intents["emergency"]):
                return "emergency"
                
            # Check other intents
            for intent, keywords in self.intents.items():
                if any(word in text.split() for word in keywords):
                    return intent
                    
            # Default to conversation
            return "conversation"
        except Exception as e:
            print(f"❌ Error in intent classification: {e}")
            return "conversation"

class VoiceProcessingSystem:
    def __init__(self, openai_api_key: str):
        print("\n🔧 Initializing Voice Processing System...")
        # Initialize OpenAI with the new client structure
        openai.api_key = openai_api_key
        self.client = openai.OpenAI(api_key=openai_api_key)
    
        # Initialize speaker recognition model
        print("Loading speaker recognition model...")
        self.speaker_recognizer = SpeakerRecognition()
    
        
        self.intent_classifier = IntentClassifier()
    
        # Load voice profiles
        self.voice_profiles = self._load_voice_profiles()
        print("✅ Voice Processing System initialized")

    def process_voice(self, audio_data: np.ndarray, sample_rate: int) -> Optional[ProcessedSpeech]:
        """Process voice data through the pipeline"""
        try:
            
            if len(audio_data.shape) == 1:
               audio_data = audio_data.reshape(1, -1)
            elif audio_data.shape[0] != 1:
                audio_data = audio_data.reshape(1, -1)
            
            print(f"Processing audio data with shape: {audio_data.shape}")
        
        
            speaker_id = self.speaker_recognizer.identify_speaker(
               audio_data, 
               self.voice_profiles
            )
        
            if not speaker_id:
             print("👤 Speaker not recognized")
             return None

        
            print("🎯 Converting speech to text...")
            try:
               text, confidence = self._transcribe_audio(audio_data, sample_rate)
            except Exception as e:
             print(f"❌ Error in speech-to-text: {e}")
             return None
            
            if not text:
             print("📝 No text transcribed")
             return None
            
            print(f"📝 Transcribed text: {text}")
        
            
            intent = self.intent_classifier.classify(text)
            print(f"🎯 Detected intent: {intent}")
        
            return ProcessedSpeech(
                text=text,
                confidence=confidence,
                speaker_id=speaker_id,
                intent=intent
            )
        except Exception as e:
         print(f"❌ Error in voice processing: {e}")
        print(f"Audio data shape: {audio_data.shape}")
        return None

    def _transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, float]:
     """Transcribe audio using Whisper API"""
     temp_filename = None
     try:
        # Create a unique temporary filename
        temp_filename = f"temp_audio_{int(time.time())}.wav"
        self._save_audio_file(audio_data, sample_rate, temp_filename)
        
        # Open and read the audio file
        with open(temp_filename, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            
        return response, 1.0  
            
     except Exception as e:
        print(f"❌ Error in transcription: {e}")
        return "", 0.0
     finally:
        
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as e:
                print(f"⚠️ Warning: Could not remove temporary file: {e}")

    def create_voice_profile(self, audio_data: np.ndarray, name: str) -> Optional[VoiceProfile]:
     """Create new voice profile from audio data"""
     try:
        print("\n🎵 Generating voice embedding...")
        
        
        embeddings = []
        segment_length = 16000 * 2  
        
        for i in range(0, len(audio_data[0]), segment_length):
            segment = audio_data[:, i:i+segment_length]
            if len(segment[0]) == segment_length:
                embedding = self.speaker_recognizer.generate_embedding(segment)
                embeddings.append(embedding)
        
        if not embeddings:
            print("❌ No valid embeddings generated")
            return None
            
        user_id = f"user_{len(self.voice_profiles)}"
        
        profile = VoiceProfile(
            user_id=user_id,
            embeddings=embeddings,
            name=name,
            created_at=time.time()
        )
        
        self.voice_profiles[user_id] = profile
        success = self._save_voice_profiles()
        
        if success:
            print(f"✅ Created profile for {name} (ID: {user_id})")
            return profile
        else:
            print("❌ Failed to save profile")
            return None
            
     except Exception as e:
        print(f"❌ Error creating voice profile: {e}")
        return None
       
    
    def _load_voice_profiles(self) -> Dict[str, VoiceProfile]:
        """Load voice profiles from storage"""
        try:
            if not os.path.exists("voice_profiles.json"):
                return {}
                
            with open("voice_profiles.json", "r") as f:
                profiles_data = json.load(f)
                
            profiles = {}
            for profile in profiles_data:
                profiles[profile["user_id"]] = VoiceProfile(
                    user_id=profile["user_id"],
                    embeddings=[np.array(emb) for emb in profile["embeddings"]],
                    name=profile["name"],
                    created_at=profile["created_at"]
                )
            print(f"📚 Loaded {len(profiles)} voice profiles")
            return profiles
            
        except Exception as e:
            print(f"⚠️ Error loading voice profiles: {e}")
            return {}

    def _save_voice_profiles(self):
        """Save voice profiles to storage"""
        try:
            profiles_data = []
            for profile in self.voice_profiles.values():
                profiles_data.append({
                    "user_id": profile.user_id,
                    "embeddings": [emb.tolist() for emb in profile.embeddings],
                    "name": profile.name,
                    "created_at": profile.created_at
                })
            
            
            os.makedirs(os.path.dirname(os.path.abspath("voice_profiles.json")), exist_ok=True)
            
            with open("voice_profiles.json", "w") as f:
                json.dump(profiles_data, f)
            print("💾 Voice profiles saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving voice profiles: {e}")

    def _save_audio_file(self, audio_data: np.ndarray, sample_rate: int, filename: str):
        """Save audio data to WAV file"""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  
                wf.setframerate(sample_rate)
                
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
        except Exception as e:
            print(f"❌ Error saving audio file: {e}")


class SpeakerRecognition:
    def __init__(self, similarity_threshold: float = 0.6):  
        self.similarity_threshold = similarity_threshold
        self.model, self.processor = self._load_model()
        print("✅ Speaker recognition model loaded")

    def _load_model(self):
      """Load the speaker recognition model"""
      try:
          
          cache_dir = "models_cache"
          os.makedirs(cache_dir, exist_ok=True)
        
          processor = Wav2Vec2Processor.from_pretrained(
             "facebook/wav2vec2-base",
             cache_dir=cache_dir,
             local_files_only=True if os.path.exists(os.path.join(cache_dir, "wav2vec2-base")) else False
          )
         
          model = Wav2Vec2ForCTC.from_pretrained(
             "facebook/wav2vec2-base",
             cache_dir=cache_dir,
             local_files_only=True if os.path.exists(os.path.join(cache_dir, "wav2vec2-base")) else False
          )
        
          
          model.eval()
        
          print("✅ Successfully loaded wav2vec2 model and processor")
          return model, processor
        
      except Exception as e:
        print(f"❌ Error loading speaker recognition model: {e}")
        raise

    def generate_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """Generate voice embedding from audio data"""
        try:
            with torch.no_grad():
                # Process audio with the processor
                inputs = self.processor(
                    audio_data,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                ).input_values
                
                
                outputs = self.model(inputs, output_hidden_states=True)
                
                # Use the last hidden state as the embedding
                hidden_states = outputs.hidden_states[-1]
                # Average over the time dimension to get a fixed-size embedding
                embedding = torch.mean(hidden_states, dim=1).numpy()
                
                return embedding
                
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            raise

    def identify_speaker(self, audio_data: np.ndarray, profiles: Dict[str, VoiceProfile]) -> Optional[str]:
        """Identify speaker from voice profiles"""
        try:
            if not profiles:
                print("No voice profiles available for comparison")
                return None
                
            current_embedding = self.generate_embedding(audio_data)
            
            best_match = None
            best_similarity = -1
            similarities = {}  
            
            for user_id, profile in profiles.items():
                for stored_embedding in profile.embeddings:
                    similarity = self._calculate_similarity(
                        current_embedding.flatten(), 
                        stored_embedding.flatten()
                    )
                    similarities[f"{user_id}_{profile.name}"] = similarity
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = user_id
            
            print(f"Best similarity score: {best_similarity:.2%}")
            print(f"All similarity scores: {similarities}")
            
            if best_similarity >= self.similarity_threshold:
                print(f"👤 Speaker recognized as {profiles[best_match].name}")
                return best_match
                
            print(f"👤 Speaker not recognized (threshold: {self.similarity_threshold})")
            return None
            
        except Exception as e:
            print(f"❌ Error in speaker identification: {e}")
            print(f"Audio data shape: {audio_data.shape}")
            return None

    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate similarity between embeddings with normalization"""
        try:
            # Normalize embeddings
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)
            return 1 - cosine(emb1, emb2)
        except Exception as e:
            print(f"❌ Error calculating similarity: {e}")
            return 0.0