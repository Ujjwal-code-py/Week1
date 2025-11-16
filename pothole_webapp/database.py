import sqlite3
import os
import threading
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.db_path = os.getenv('DB_PATH', 'pothole_detection.db')
        self.thread_local = threading.local()
        
    def get_connection(self):
        """Get thread-specific database connection"""
        if not hasattr(self.thread_local, 'connection'):
            self.thread_local.connection = sqlite3.connect(self.db_path)
            self.thread_local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self.thread_local.connection.execute("PRAGMA foreign_keys = ON")
            self.create_tables()
        return self.thread_local.connection
    
    def connect(self):
        """Maintain compatibility with existing code"""
        return self.get_connection()
    
    def create_tables(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Locations Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS locations (
                location_id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_name TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                city TEXT,
                additional_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Media Files Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS media_files (
                media_id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                original_file_url TEXT,
                processed_file_url TEXT,
                file_size INTEGER,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Pothole Analysis Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pothole_analysis (
                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_id INTEGER,
                media_id INTEGER,
                total_potholes INTEGER NOT NULL,
                total_volume_liters REAL,
                average_width_cm REAL,
                average_depth_cm REAL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (location_id) REFERENCES locations (location_id),
                FOREIGN KEY (media_id) REFERENCES media_files (media_id)
            )
        ''')
        
        # Pothole Details Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pothole_details (
                pothole_detail_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                pothole_number INTEGER,
                width_cm REAL,
                depth_cm REAL,
                volume_liters REAL,
                confidence_score REAL,
                bounding_box TEXT,
                FOREIGN KEY (analysis_id) REFERENCES pothole_analysis (analysis_id)
            )
        ''')
        
        # Cost Analysis Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cost_analysis (
                cost_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                material_cost REAL,
                labor_cost REAL,
                equipment_cost REAL,
                transport_cost REAL,
                overhead_cost REAL,
                total_cost REAL,
                cost_parameters TEXT,
                FOREIGN KEY (analysis_id) REFERENCES pothole_analysis (analysis_id)
            )
        ''')
        
        # Time Estimation Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS time_estimation (
                time_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                total_hours REAL,
                setup_time REAL,
                prep_time REAL,
                fill_time REAL,
                compact_time REAL,
                cleanup_time REAL,
                FOREIGN KEY (analysis_id) REFERENCES pothole_analysis (analysis_id)
            )
        ''')
        
        conn.commit()
        print("✅ Database tables created/verified successfully")
    
    def get_cursor(self):
        """Get cursor from thread-specific connection"""
        conn = self.get_connection()
        return conn.cursor()
    
    def close(self):
        """Close thread-specific connection"""
        if hasattr(self.thread_local, 'connection'):
            self.thread_local.connection.close()
            del self.thread_local.connection
            print("✅ Database connection closed")

# Initialize database
db = Database()