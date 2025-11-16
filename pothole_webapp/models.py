from database import db
from datetime import datetime
import json

class Location:
    @staticmethod
    def create(location_data):
        cursor = db.get_cursor()
        try:
            query = '''
                INSERT INTO locations (location_name, latitude, longitude, city, additional_notes)
                VALUES (?, ?, ?, ?, ?)
            '''
            values = (
                location_data.get('location_name'),
                location_data.get('latitude'),
                location_data.get('longitude'),
                location_data.get('city'),
                location_data.get('additional_notes')
            )
            cursor.execute(query, values)
            db.get_connection().commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"❌ Location creation failed: {e}")
            db.get_connection().rollback()
            return None
        finally:
            cursor.close()

class MediaFile:
    @staticmethod
    def create(media_data):
        cursor = db.get_cursor()
        try:
            query = '''
                INSERT INTO media_files (original_filename, file_type, original_file_url, 
                                       processed_file_url, file_size)
                VALUES (?, ?, ?, ?, ?)
            '''
            values = (
                media_data.get('original_filename'),
                media_data.get('file_type'),
                media_data.get('original_file_url'),
                media_data.get('processed_file_url'),
                media_data.get('file_size')
            )
            cursor.execute(query, values)
            db.get_connection().commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"❌ Media file creation failed: {e}")
            db.get_connection().rollback()
            return None
        finally:
            cursor.close()

class PotholeAnalysis:
    @staticmethod
    def create(analysis_data):
        cursor = db.get_cursor()
        try:
            query = '''
                INSERT INTO pothole_analysis (location_id, media_id, total_potholes, 
                                            total_volume_liters, average_width_cm, average_depth_cm)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            values = (
                analysis_data.get('location_id'),
                analysis_data.get('media_id'),
                analysis_data.get('total_potholes'),
                analysis_data.get('total_volume_liters'),
                analysis_data.get('average_width_cm'),
                analysis_data.get('average_depth_cm')
            )
            cursor.execute(query, values)
            db.get_connection().commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"❌ Pothole analysis creation failed: {e}")
            db.get_connection().rollback()
            return None
        finally:
            cursor.close()

class PotholeDetails:
    @staticmethod
    def create_batch(analysis_id, potholes_data):
        cursor = db.get_cursor()
        try:
            query = '''
                INSERT INTO pothole_details (analysis_id, pothole_number, width_cm, 
                                           depth_cm, volume_liters, confidence_score, bounding_box)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            values = []
            for pothole in potholes_data:
                values.append((
                    analysis_id,
                    pothole.get('id'),
                    pothole.get('width_cm'),
                    pothole.get('depth_cm'),
                    pothole.get('volume_liters'),
                    pothole.get('confidence'),
                    json.dumps(pothole.get('bbox'))  # Convert list to JSON string
                ))
            
            cursor.executemany(query, values)
            db.get_connection().commit()
            return True
        except Exception as e:
            print(f"❌ Pothole details creation failed: {e}")
            db.get_connection().rollback()
            return False
        finally:
            cursor.close()

class CostAnalysis:
    @staticmethod
    def create(cost_data):
        cursor = db.get_cursor()
        try:
            query = '''
                INSERT INTO cost_analysis (analysis_id, material_cost, labor_cost, 
                                         equipment_cost, transport_cost, overhead_cost, 
                                         total_cost, cost_parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            '''
            values = (
                cost_data.get('analysis_id'),
                cost_data.get('material_cost'),
                cost_data.get('labor_cost'),
                cost_data.get('equipment_cost'),
                cost_data.get('transport_cost'),
                cost_data.get('overhead_cost'),
                cost_data.get('total_cost'),
                json.dumps(cost_data.get('cost_parameters'))
            )
            cursor.execute(query, values)
            db.get_connection().commit()
            return True
        except Exception as e:
            print(f"❌ Cost analysis creation failed: {e}")
            db.get_connection().rollback()
            return False
        finally:
            cursor.close()

class TimeEstimation:
    @staticmethod
    def create(time_data):
        cursor = db.get_cursor()
        try:
            query = '''
                INSERT INTO time_estimation (analysis_id, total_hours, setup_time, 
                                           prep_time, fill_time, compact_time, cleanup_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            values = (
                time_data.get('analysis_id'),
                time_data.get('total_hours'),
                time_data.get('setup_time'),
                time_data.get('prep_time'),
                time_data.get('fill_time'),
                time_data.get('compact_time'),
                time_data.get('cleanup_time')
            )
            cursor.execute(query, values)
            db.get_connection().commit()
            return True
        except Exception as e:
            print(f"❌ Time estimation creation failed: {e}")
            db.get_connection().rollback()
            return False
        finally:
            cursor.close()