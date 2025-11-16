from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import json
import tempfile
from werkzeug.utils import secure_filename
from utils.depth_estimation import PotholeDepthEstimator
from utils.cost_estimation import CostEstimator
from cloudinary_config import configure_cloudinary, upload_to_cloudinary, upload_annotated_image
from models import Location, MediaFile, PotholeAnalysis, PotholeDetails, CostAnalysis, TimeEstimation
from database import db
from datetime import datetime
import io  # For in-memory PDF buffer

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pothole-detection-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Allowed extensions - KEEP VIDEOS
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}

# Initialize estimators and services
depth_estimator = PotholeDepthEstimator()
cost_estimator = CostEstimator()
configure_cloudinary()
# Don't call db.connect() here - it will connect per thread automatically

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_temp_file_path(filename):
    """Get cross-platform temporary file path"""
    secure_name = secure_filename(filename)
    return os.path.join(tempfile.gettempdir(), secure_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/upload', methods=['POST'])

def upload_file():
    temp_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and allowed_file(file.filename):
            # Save uploaded file to temp directory (cross-platform)
            temp_path = get_temp_file_path(file.filename)
            file.save(temp_path)
            
            print(f"📁 File saved to temporary location: {temp_path}")
            
            # Get cost parameters from form
            material_cost = float(request.form.get('material_cost', 40.0))
            labor_cost = float(request.form.get('labor_cost', 300.0))
            team_size = int(request.form.get('team_size', 2))
            overhead = float(request.form.get('overhead', 15.0))
            
            # Get location data from form
            location_data = {
                'location_name': request.form.get('location_name', ''),
                'latitude': request.form.get('latitude', ''),
                'longitude': request.form.get('longitude', ''),
                'city': request.form.get('city', ''),
                'additional_notes': request.form.get('additional_notes', '')
            }
            
            # Step 1: Upload original file to Cloudinary
            file_ext = file.filename.lower().split('.')[-1]
            file_type = 'image' if file_ext in ['png', 'jpg', 'jpeg'] else 'video'
            
            print("☁️ Uploading to Cloudinary...")
            upload_result = upload_to_cloudinary(temp_path, 'uploads', file_type)
            
            if not upload_result['success']:
                return jsonify({'success': False, 'error': f'Cloudinary upload failed: {upload_result.get("error", "Unknown error")}'})
            
            print("✅ File uploaded to Cloudinary successfully")
            
            # Step 2: Store media file in database
            media_data = {
                'original_filename': file.filename,
                'file_type': file_type,
                'original_file_url': upload_result['url'],
                'processed_file_url': None,  # Will be updated after processing
                'file_size': upload_result['bytes']
            }
            media_id = MediaFile.create(media_data)
            
            if not media_id:
                return jsonify({'success': False, 'error': 'Failed to store media file in database'})
            
            # Step 3: Store location in database
            location_id = Location.create(location_data)
            
            if not location_id:
                return jsonify({'success': False, 'error': 'Failed to store location in database'})
            
            # Step 4: Process the file based on type (ORIGINAL LOGIC PRESERVED)
            print("🔍 Processing file for pothole detection...")
            if file_type == 'image':
                result = process_image(temp_path, material_cost, labor_cost, team_size, overhead, location_id, media_id, file.filename)
            else:
                result = process_video(temp_path, material_cost, labor_cost, team_size, overhead, location_id, media_id, file.filename)
            
            return jsonify(result)
        
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload images (PNG, JPG) or videos (MP4, AVI, MOV)'})
    
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'})
    
    finally:
        # Always clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"🧹 Cleaned up temporary file: {temp_path}")
            except Exception as e:
                print(f"⚠️ Could not delete temp file: {e}")
def debug_cloudinary_upload(file_path, folder):
    """Debug function to check Cloudinary upload"""
    try:
        print(f"🔍 Debug: Uploading {file_path} to folder {folder}")
        result = upload_to_cloudinary(file_path, folder)
        print(f"🔍 Debug: Upload result - {result}")
        return result
    except Exception as e:
        print(f"🔍 Debug: Upload failed - {e}")
        return None
def process_image(image_path, material_cost, labor_cost, team_size, overhead, location_id, media_id, filename):
    """Process single image and return results - ORIGINAL LOGIC PRESERVED"""
    try:
        print("🖼️ Processing image...")
        
        # Analyze potholes (ORIGINAL LOGIC)
        results = depth_estimator.calculate_pothole_dimensions(image_path)
        
        if not results:
            return {'success': False, 'error': 'No potholes detected in the image'}
        
        pothole_data, image = results
        print(f"✅ Found {len(pothole_data)} potholes")
        
        # Calculate costs (ORIGINAL LOGIC)
        cost_estimator.material_cost_per_liter = material_cost
        cost_estimator.labor_cost_per_hour = labor_cost
        cost_estimator.team_size = team_size
        cost_estimator.overhead_percentage = overhead
        
        cost_breakdown = cost_estimator.calculate_repair_cost(pothole_data)
        
        # Create annotated image (ORIGINAL LOGIC)
        result_image = image.copy()
        for pothole in pothole_data:
            x1, y1, x2, y2 = pothole['bbox']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add info text
            info_text = f"Pothole {pothole['id']}"
            cv2.putText(result_image, info_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Step 5: Upload annotated image to Cloudinary (NEW)
        print("☁️ Uploading annotated image to Cloudinary...")
        annotated_result = upload_annotated_image(result_image, filename, 'results')
        
        if not annotated_result['success']:
            return {'success': False, 'error': 'Annotated image upload failed'}
        
        print("✅ Annotated image uploaded to Cloudinary")
        
        # Update media file with processed URL (THREAD-SAFE)
        cursor = db.get_cursor()
        try:
            cursor.execute(
                "UPDATE media_files SET processed_file_url = ? WHERE media_id = ?",
                (annotated_result['url'], media_id)
            )
            db.get_connection().commit()
        finally:
            cursor.close()
        
        # Step 6: Store all analysis data in database (NEW)
        print("💾 Storing analysis data in database...")
        analysis_id = store_analysis_data(location_id, media_id, pothole_data, cost_breakdown, material_cost, labor_cost, team_size, overhead)
        
        if not analysis_id:
            return {'success': False, 'error': 'Failed to store analysis data'}
        
        print("✅ Analysis completed successfully")
        
        return {
            'success': True,
            'file_type': 'image',
            'potholes_detected': len(pothole_data),
            'pothole_data': pothole_data,
            'cost_breakdown': cost_breakdown,
            'result_image': annotated_result['url'],  # Cloudinary URL instead of local path
            'location_data': {
                'location_name': request.form.get('location_name', ''),
                'city': request.form.get('city', '')
            }
        }
        
    except Exception as e:
        print(f"❌ Image processing error: {str(e)}")
        return {'success': False, 'error': f'Image processing failed: {str(e)}'}

def process_video(video_path, material_cost, labor_cost, team_size, overhead, location_id, media_id, filename):
    """Process video file and return summary - ORIGINAL LOGIC PRESERVED with IoU deduplication"""
    try:
        print("🎥 Processing video...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'success': False, 'error': 'Could not open video file'}
        
        # Get video info (ORIGINAL LOGIC)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {total_video_frames} frames, {fps:.1f} FPS")
        
        frame_count = 0
        total_frames_analyzed = 0
        all_potholes = []
        
        # PROCESS ALL FRAMES - ABSOLUTELY NO LIMITS (ORIGINAL LOGIC)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Process EVERY frame (ORIGINAL LOGIC)
                results = depth_estimator.calculate_pothole_dimensions_from_array(frame)
                
                # ALWAYS count the frame, even if no potholes found
                total_frames_analyzed += 1
                
                if results and results[0]:  # If potholes were found
                    pothole_data, _ = results
                    all_potholes.extend(pothole_data)
                
            except Exception as e:
                print(f"⚠️ Error processing frame {frame_count}: {e}")
                total_frames_analyzed += 1  # Still count the frame even if error
            
            frame_count += 1
            
            # Progress update
            if frame_count % 50 == 0:
                print(f"📈 Processed {frame_count}/{total_video_frames} frames... Found {len(all_potholes)} potholes so far")
        
        cap.release()
        
        print(f"✅ Completed: Analyzed {total_frames_analyzed} frames out of {total_video_frames} total frames")
        
        if total_frames_analyzed == 0:
            return {'success': False, 'error': 'No frames could be processed from the video'}
        
        if not all_potholes:
            return {'success': False, 'error': 'No potholes detected in the video'}
        
        # IoU-based deduplication (ORIGINAL LOGIC)
        def calculate_iou(box1, box2):
            """Calculate Intersection over Union of two bounding boxes"""
            x11, y11, x21, y21 = box1
            x12, y12, x22, y22 = box2
            
            # Calculate intersection area
            xi1 = max(x11, x12)
            yi1 = max(y11, y12)
            xi2 = min(x21, x22)
            yi2 = min(y21, y22)
            intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            
            # Calculate union area
            area1 = (x21 - x11) * (y21 - y11)
            area2 = (x22 - x12) * (y22 - y12)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
        
        # Remove duplicates using IoU (ORIGINAL LOGIC)
        unique_potholes = []
        
        for pothole in all_potholes:
            is_duplicate = False
            
            for existing_pothole in unique_potholes:
                iou = calculate_iou(pothole['bbox'], existing_pothole['bbox'])
                if iou > 0.3:  # If overlap is more than 30%, consider it duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_potholes.append(pothole)
        
        print(f"🔄 Found {len(all_potholes)} total potholes, {len(unique_potholes)} unique potholes after IoU deduplication")
        
        # Calculate costs (ORIGINAL LOGIC)
        cost_estimator.material_cost_per_liter = material_cost
        cost_estimator.labor_cost_per_hour = labor_cost
        cost_estimator.team_size = team_size
        cost_estimator.overhead_percentage = overhead
        
        cost_breakdown = cost_estimator.calculate_repair_cost(unique_potholes)
        
        # For videos, create a summary frame from first frame
        result_image_url = None
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()
        
        if ret and unique_potholes:
            # Create annotated first frame
            result_frame = first_frame.copy()
            for i, pothole in enumerate(unique_potholes[:10]):  # Show first 10 potholes
                x1, y1, x2, y2 = pothole['bbox']
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                info_text = f"Pothole {i+1}"
                cv2.putText(result_frame, info_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Upload annotated frame to Cloudinary
            print("☁️ Uploading video summary image to Cloudinary...")
            annotated_result = upload_annotated_image(result_frame, f"video_summary_{filename}", 'results')
            if annotated_result['success']:
                result_image_url = annotated_result['url']
                
                # Update media file with processed URL (THREAD-SAFE)
                cursor = db.get_cursor()
                try:
                    cursor.execute(
                        "UPDATE media_files SET processed_file_url = ? WHERE media_id = ?",
                        (annotated_result['url'], media_id)
                    )
                    db.get_connection().commit()
                    print("✅ Video summary image uploaded to Cloudinary")
                finally:
                    cursor.close()
        
        # Store all analysis data in database
        print("💾 Storing video analysis data in database...")
        analysis_id = store_analysis_data(location_id, media_id, unique_potholes, cost_breakdown, material_cost, labor_cost, team_size, overhead)
        
        if not analysis_id:
            return {'success': False, 'error': 'Failed to store analysis data'}
        
        print("✅ Video analysis completed successfully")
        
        return {
            'success': True,
            'file_type': 'video',
            'potholes_detected': len(unique_potholes),
            'total_frames_analyzed': total_frames_analyzed,
            'total_video_frames': total_video_frames,
            'pothole_data': unique_potholes[:10],  # Send first 10 for display
            'cost_breakdown': cost_breakdown,
            'result_image': result_image_url,  # Cloudinary URL or None
            'location_data': {
                'location_name': request.form.get('location_name', ''),
                'city': request.form.get('city', '')
            }
        }
        
    except Exception as e:
        print(f"❌ Video processing error: {str(e)}")
        return {'success': False, 'error': f'Video processing failed: {str(e)}'}

def store_analysis_data(location_id, media_id, pothole_data, cost_breakdown, material_cost, labor_cost, team_size, overhead):
    """Store all analysis data in database"""
    try:
        # Calculate averages
        total_volume = sum(p['volume_liters'] for p in pothole_data) if pothole_data else 0
        avg_width = sum(p['width_cm'] for p in pothole_data) / len(pothole_data) if pothole_data else 0
        avg_depth = sum(p['depth_cm'] for p in pothole_data) / len(pothole_data) if pothole_data else 0
        
        # Store main analysis
        analysis_data = {
            'location_id': location_id,
            'media_id': media_id,
            'total_potholes': len(pothole_data),
            'total_volume_liters': total_volume,
            'average_width_cm': avg_width,
            'average_depth_cm': avg_depth
        }
        analysis_id = PotholeAnalysis.create(analysis_data)
        
        if not analysis_id:
            return None
        
        # Store pothole details
        if pothole_data:
            PotholeDetails.create_batch(analysis_id, pothole_data)
        
        # Store cost analysis
        cost_data = {
            'analysis_id': analysis_id,
            'material_cost': cost_breakdown.get('material_cost', 0),
            'labor_cost': cost_breakdown.get('labor_cost', 0),
            'equipment_cost': cost_breakdown.get('equipment_cost', 0),
            'transport_cost': cost_breakdown.get('transport_cost', 0),
            'overhead_cost': cost_breakdown.get('overhead_cost', 0),
            'total_cost': cost_breakdown.get('total_cost', 0),
            'cost_parameters': {
                'material_cost_per_liter': material_cost,
                'labor_cost_per_hour': labor_cost,
                'team_size': team_size,
                'overhead_percentage': overhead
            }
        }
        CostAnalysis.create(cost_data)
        
        # Store time estimation
        if 'time_breakdown' in cost_breakdown:
            time_data = {
                'analysis_id': analysis_id,
                'total_hours': cost_breakdown['time_breakdown'].get('total_hours', 0),
                'setup_time': cost_breakdown['time_breakdown'].get('setup_time', 0),
                'prep_time': cost_breakdown['time_breakdown'].get('prep_time', 0),
                'fill_time': cost_breakdown['time_breakdown'].get('fill_time', 0),
                'compact_time': cost_breakdown['time_breakdown'].get('compact_time', 0),
                'cleanup_time': cost_breakdown['time_breakdown'].get('cleanup_time', 0)
            }
            TimeEstimation.create(time_data)
        
        return analysis_id
        
    except Exception as e:
        print(f"❌ Error storing analysis data: {e}")
        return None

@app.route('/results/<filename>')
def get_result_image(filename):
    """Serve result images from Cloudinary - Redirect to Cloudinary URL"""
    # Since we're using Cloudinary, we don't serve local files anymore
    return jsonify({'success': False, 'error': 'Use Cloudinary URL directly'})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        print("📄 Generating PDF report...")
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,
            textColor=colors.HexColor('#2563eb')
        )
        elements.append(Paragraph("Pothole Inspection Report", title_style))
        elements.append(Spacer(1, 20))
        
        # Report Details
        details_data = [
            ['Report Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['File Type', data.get('file_type', 'N/A')],
            ['Potholes Detected', str(data.get('potholes_detected', 0))],
        ]
        
        # Add location information to report if available
        location_data = data.get('location_data', {})
        if location_data.get('location_name'):
            details_data.append(['Location', location_data['location_name']])
        if location_data.get('city'):
            details_data.append(['City', location_data['city']])
        
        details_table = Table(details_data, colWidths=[2*inch, 3*inch])
        details_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(details_table)
        elements.append(Spacer(1, 30))
        
        # Cost Breakdown
        cost_breakdown = data.get('cost_breakdown', {})
        
        # Format costs properly with rupee symbol
        material_cost = f"₹{cost_breakdown.get('material_cost', 0):.2f}"
        labor_cost = f"₹{cost_breakdown.get('labor_cost', 0):.2f}"
        equipment_transport = f"₹{(cost_breakdown.get('equipment_cost', 0) + cost_breakdown.get('transport_cost', 0)):.2f}"
        overhead_cost = f"₹{cost_breakdown.get('overhead_cost', 0):.2f}"
        total_cost = f"₹{cost_breakdown.get('total_cost', 0):.2f}"
        
        cost_data = [
            ['Cost Item', 'Amount (₹)'],
            ['Material Cost', material_cost],
            ['Labor Cost', labor_cost],
            ['Equipment & Transport', equipment_transport],
            ['Overhead', overhead_cost],
            ['TOTAL COST', total_cost]
        ]
        
        cost_table = Table(cost_data, colWidths=[3*inch, 2*inch])
        cost_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor('#f8fafc')),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (0, -1), (-1, -1), colors.whitesmoke),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(Paragraph("Cost Breakdown", styles['Heading2']))
        elements.append(Spacer(1, 10))
        elements.append(cost_table)
        
        # Pothole Details (if available)
        pothole_data = data.get('pothole_data', [])
        if pothole_data and len(pothole_data) > 0:
            elements.append(Spacer(1, 30))
            elements.append(Paragraph("Pothole Details", styles['Heading2']))
            elements.append(Spacer(1, 10))
            
            # Create pothole details table WITHOUT confidence column
            pothole_table_data = [['ID', 'Width (cm)', 'Depth (cm)', 'Volume (L)']]
            for pothole in pothole_data[:10]:  # Show first 10 potholes
                pothole_table_data.append([
                    str(pothole.get('id', '')),
                    f"{pothole.get('width_cm', 0):.1f}",
                    f"{pothole.get('depth_cm', 0):.1f}",
                    f"{pothole.get('volume_liters', 0):.2f}"
                ])
            
            pothole_table = Table(pothole_table_data, colWidths=[0.5*inch, 1.2*inch, 1.2*inch, 1.2*inch])
            pothole_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            elements.append(pothole_table)
        
        # Build PDF
        doc.build(elements)
        
        # Prepare response
        buffer.seek(0)
        
        if buffer.getbuffer().nbytes == 0:
            raise Exception("Generated PDF is empty")
            
        print(f"✅ PDF generated successfully, size: {buffer.getbuffer().nbytes} bytes")
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'pothole_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"❌ PDF generation error: {str(e)}")
        return jsonify({'success': False, 'error': f'PDF generation failed: {str(e)}'}), 500

@app.route('/history')
def get_history():
    """Retrieve analysis history from database"""
    try:
        cursor = db.get_cursor()
        query = """
            SELECT 
                pa.analysis_id,
                pa.total_potholes,
                pa.total_volume_liters,
                pa.analysis_date,
                l.location_name,
                l.city,
                l.latitude,
                l.longitude,
                mf.original_filename,
                mf.file_type,
                mf.processed_file_url as result_image_url,
                ca.total_cost
            FROM pothole_analysis pa
            LEFT JOIN locations l ON pa.location_id = l.location_id
            LEFT JOIN media_files mf ON pa.media_id = mf.media_id
            LEFT JOIN cost_analysis ca ON pa.analysis_id = ca.analysis_id
            ORDER BY pa.analysis_date DESC
            LIMIT 50
        """
        cursor.execute(query)
        history = cursor.fetchall()
        
        # Convert to list of dicts for JSON serialization
        history_list = []
        for row in history:
            history_list.append(dict(row))
        
        return jsonify({'success': True, 'history': history_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.teardown_appcontext
def close_db(error):
    """Close database connection at the end of request"""
    db.close()

if __name__ == '__main__':
    # Create necessary directories (for temporary processing only)

    
    # print("🚀 Pothole Detection Web App Starting...")
    # print("📊 Database: SQLite (thread-safe)")
    # print("☁️  Media Storage: Cloudinary")
    # print("📁 Temporary Processing: Local")
    # print("🌐 Access at: http://localhost:5000")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    
    # app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)