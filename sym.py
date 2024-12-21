import json
import time
import psutil
import logging
import sqlite3
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from pathlib import Path
import threading
import queue

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    response_time: float
    api_calls: int
    errors: int
    timestamp: datetime

@dataclass
class VoiceProfileData:
    user_id: str
    name: str
    embeddings: List[np.ndarray]
    last_updated: datetime
    active: bool

class SystemManager:
    def __init__(self, db_path: str = "system.db"):
        
        self.voice_manager = VoiceProfileManager(db_path)
        self.cache_manager = ResponseCacheManager(db_path)
        self.performance_monitor = PerformanceMonitor()
        
        
        self._setup_logging()
        
       
        self._init_system_checks()

    def _setup_logging(self):
        """Setup system logging"""
        logging.basicConfig(
            filename='system.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SystemManager')

    def _init_system_checks(self):
        """Initialize periodic system checks"""
        self.check_thread = threading.Thread(target=self._periodic_checks)
        self.check_thread.daemon = True
        self.check_thread.start()

    def _periodic_checks(self):
        """Run periodic system checks"""
        while True:
            try:
                
                metrics = self.performance_monitor.collect_metrics()
                
                
                self._check_system_health(metrics)
                
                
                self.cache_manager.clean_old_entries()
                
                
                self.logger.info(f"System metrics: {metrics}")
                
                time.sleep(300)  
                
            except Exception as e:
                self.logger.error(f"Error in periodic checks: {e}")

    def _check_system_health(self, metrics: SystemMetrics):
        """Check system health and respond to issues"""
        if metrics.cpu_usage > 80:
            self.logger.warning("High CPU usage detected")
            self._handle_high_cpu()
            
        if metrics.memory_usage > 80:
            self.logger.warning("High memory usage detected")
            self._handle_high_memory()
            
        if metrics.errors > 10:
            self.logger.error("High error rate detected")
            self._handle_high_errors()

    def cleanup(self):
        """Cleanup system resources"""
        self.voice_manager.cleanup()
        self.cache_manager.cleanup()
        self.performance_monitor.cleanup()

class VoiceProfileManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        
        
        self.active_profiles: Dict[str, VoiceProfileData] = {}
        self._load_active_profiles()  
    
    def _load_active_profiles(self):
        """Load active profiles from database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT user_id, name, embeddings, last_updated 
                FROM voice_profiles 
                WHERE active = 1
            ''')
            
            for row in cursor.fetchall():
                profile = VoiceProfileData(
                    user_id=row[0],
                    name=row[1],
                    embeddings=self._deserialize_embeddings(row[2]),
                    last_updated=row[3],
                    active=True
                )
                self.active_profiles[profile.user_id] = profile
                
        except Exception as e:
            logging.error(f"Error loading active profiles: {e}")

    def _init_db(self):
        """Initialize database tables"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                embeddings BLOB,
                last_updated TIMESTAMP,
                active BOOLEAN
            )
        ''')
        self.conn.commit()

    def add_profile(self, profile: VoiceProfileData) -> bool:
        """Add or update voice profile"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO voice_profiles 
                (user_id, name, embeddings, last_updated, active)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                profile.user_id,
                profile.name,
                self._serialize_embeddings(profile.embeddings),
                profile.last_updated,
                profile.active
            ))
            self.conn.commit()
            
            if profile.active:
                self.active_profiles[profile.user_id] = profile
            
            return True
            
        except Exception as e:
            logging.error(f"Error adding profile: {e}")
            return False

    def _serialize_embeddings(self, embeddings: List[np.ndarray]) -> bytes:
        """Serialize embeddings for storage"""
        return json.dumps([emb.tolist() for emb in embeddings]).encode()

    def _deserialize_embeddings(self, data: bytes) -> List[np.ndarray]:
        """Deserialize embeddings from storage"""
        return [np.array(emb) for emb in json.loads(data.decode())]


class ResponseCacheManager:
    def __init__(self, db_path: str, max_age_days: int = 30):
        self.db_path = db_path
        self.max_age_days = max_age_days
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize cache database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key TEXT PRIMARY KEY,
                response_text TEXT,
                response_audio BLOB,
                created_at TIMESTAMP,
                last_used TIMESTAMP,
                use_count INTEGER
            )
        ''')
        self.conn.commit()

    def get_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT response_text, response_audio, use_count 
            FROM response_cache 
            WHERE cache_key = ?
        ''', (cache_key,))
        
        result = cursor.fetchone()
        if result:
            # Update usage statistics
            self._update_usage(cache_key)
            return {
                'text': result[0],
                'audio': result[1],
                'use_count': result[2]
            }
        return None

    def _update_usage(self, cache_key: str):
        """Update cache entry usage statistics"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE response_cache 
            SET last_used = ?, use_count = use_count + 1
            WHERE cache_key = ?
        ''', (datetime.now(), cache_key))
        self.conn.commit()

    def clean_old_entries(self):
        """Remove old cache entries"""
        cursor = self.conn.cursor()
        cursor.execute('''
            DELETE FROM response_cache 
            WHERE julianday('now') - julianday(last_used) > ?
        ''', (self.max_age_days,))
        self.conn.commit()


class PerformanceMonitor:
    def __init__(self):
        self.metrics_queue = queue.Queue()
        self.start_time = time.time()
        self.api_calls = 0
        self.errors = 0

    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        return SystemMetrics(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            response_time=self._calculate_avg_response_time(),
            api_calls=self.api_calls,
            errors=self.errors,
            timestamp=datetime.now()
        )

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        try:
            times = []
            while not self.metrics_queue.empty():
                times.append(self.metrics_queue.get_nowait())
            
            if times:
                return sum(times) / len(times)
            return 0.0
            
        except queue.Empty:
            return 0.0

    def log_api_call(self):
        """Log an API call"""
        self.api_calls += 1

    def log_error(self):
        """Log an error"""
        self.errors += 1

    def log_response_time(self, duration: float):
        """Log response time"""
        self.metrics_queue.put(duration)



if __name__ == "__main__":
    # Initialize system manager
    system_manager = SystemManager()
    
    try:
        # Add test voice profile
        profile = VoiceProfileData(
            user_id="test_user",
            name="Test User",
            embeddings=[np.random.rand(512)],
            last_updated=datetime.now(),
            active=True
        )
        
        system_manager.voice_manager.add_profile(profile)
        
        # Run for a while to see metrics
        time.sleep(60)
        
    finally:
        system_manager.cleanup()