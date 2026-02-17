"""
Flask Application for Advanced Vehicle Tracking
Using testApp1.py backend for superior tracking quality
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash, session, abort
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import json
from datetime import datetime, timedelta
import threading
import uuid
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import sqlite3
import subprocess
import zipfile
import shutil
import io

# Matplotlib imports for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available - visualization features disabled")

# Import testApp1 components
sys.path.insert(0, os.path.dirname(__file__))
from testApp1 import VideoProcessorV1, ConfigV1, classify_tracks_from_df, logger as testapp_logger

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "77f493eed3adcbf6ea6e4fd1747083ca29970bbe161f10fe6c2cf43762dd1e58")
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'unified_output'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['DATABASE'] = 'vehicle_tracking.db'

# Unified counting constant (single source of truth for "near-line" checks)
LINE_HIT_TOLERANCE = 60  # pixels — fixed from backend

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store processing jobs
processing_jobs = {}

# ===================== DATABASE SETUP =====================

def get_db():
    """Get database connection"""
    db = sqlite3.connect(app.config['DATABASE'])
    db.row_factory = sqlite3.Row
    return db

def init_db():
    """Initialize database with tables"""
    db = get_db()
    cursor = db.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Analysis requests table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            csv_path TEXT NOT NULL,
            frame1_path TEXT,
            frame2_path TEXT,
            frame3_path TEXT,
            status TEXT DEFAULT 'pending',
            results_path TEXT,
            admin_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create default admin user if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        admin_password = generate_password_hash('admin123')
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ('admin', admin_password, 'admin')
        )
        print("[OK] Created default admin user (username: admin, password: admin123)")
    
    db.commit()
    db.close()

# Initialize database on startup
init_db()

# ===================== AUTHENTICATION =====================

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first', 'error')
            return redirect(url_for('login'))
        
        db = get_db()
        user = db.execute('SELECT role FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        db.close()
        
        if not user or user['role'] != 'admin':
            flash('You are not authorized to access this page', 'error')
            return redirect(url_for('index'))
        
        return f(*args, **kwargs)
    return decorated_function

# ===================== AUTH ROUTES =====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        db.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash(f'Welcome {username}!', 'success')
            
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('index'))
        
        flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register new user"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not password:
            flash('Please enter all required fields', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        db = get_db()
        existing = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        
        if existing:
            db.close()
            flash('Username already exists', 'error')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        db.execute(
            'INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
            (username, hashed_password, 'user')
        )
        db.commit()
        db.close()
        
        flash('Account created successfully! You can now log in', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('login'))

# ===================== USER REQUEST ROUTES =====================

@app.route('/request_analysis')
@login_required
def request_analysis_page():
    """Page for users to request analysis"""
    return render_template('request_analysis.html')

@app.route('/my_requests')
@login_required
def my_requests_page():
    """Page showing all user's analysis requests"""
    return render_template('my_requests.html')


@app.route('/user/line_drawing')
@login_required
def user_line_drawing():
    """User-facing line drawing page for a user's own request or for manual upload.

    - If `request_id` is provided we validate ownership and pass it to the template
      so the page auto-loads the request data.
    - If `request_id` is absent we render the page for manual file upload/drawing.
    """
    request_id = request.args.get('request_id')

    if request_id:
        db = get_db()
        req = db.execute('SELECT * FROM analysis_requests WHERE id = ?', (request_id,)).fetchone()
        db.close()

        if not req:
            abort(404)

        # Security: ensure the current user owns this request
        if req['user_id'] != session.get('user_id'):
            abort(403)

        return render_template('line_drawing.html', request_id=request_id)

    # No request_id -> allow manual usage (upload + draw) for logged-in users
    return render_template('line_drawing.html', request_id=None)

@app.route('/api/request_analysis_auto', methods=['POST'])
@login_required
def submit_analysis_request_auto():
    """Submit analysis request automatically from tracking results"""
    try:
        data = request.json
        folder_name = data.get('folder_name')
        
        if not folder_name:
            return jsonify({'error': 'Folder name required'}), 400
        
        # Get output directory from unified_output folder
        analysis_dir = os.path.join(app.config['OUTPUT_FOLDER'], folder_name)
        
        if not os.path.exists(analysis_dir):
            return jsonify({'error': 'Analysis folder not found'}), 404
        
        csv_path = os.path.join(analysis_dir, 'tracks.csv')
        
        if not os.path.exists(csv_path):
            return jsonify({'error': 'Tracking file not found'}), 404
        
        # Get frame paths - look in frames subfolder
        frames_dir = os.path.join(analysis_dir, 'frames')
        frames = []
        
        if os.path.exists(frames_dir):
            for file in os.listdir(frames_dir):
                if file.startswith('frame_') and file.endswith(('.jpg', '.png')):
                    frames.append(os.path.join(frames_dir, file))
        
        if len(frames) < 3:
            return jsonify({'error': f'Not enough images (found only {len(frames)})'}), 400
        
        # Sort frames and take first 3
        frames.sort()
        frame1_path = frames[0]
        frame2_path = frames[1] if len(frames) > 1 else frames[0]
        frame3_path = frames[2] if len(frames) > 2 else frames[0]
        
        # Insert into database
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO analysis_requests 
            (user_id, csv_path, frame1_path, frame2_path, frame3_path, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
        ''', (session['user_id'], csv_path, frame1_path, frame2_path, frame3_path))
        
        request_db_id = cursor.lastrowid
        db.commit()
        db.close()
        
        return jsonify({
            'success': True,
            'request_id': request_db_id,
            'message': 'Analysis request sent successfully'
        })
        
    except Exception as e:
        print(f"Error submitting auto request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/request_analysis', methods=['POST'])
@login_required
def submit_analysis_request():
    """Submit analysis request"""
    try:
        csv_file = request.files.get('csv_file')
        frame1 = request.files.get('frame1')
        frame2 = request.files.get('frame2')
        frame3 = request.files.get('frame3')
        
        if not all([csv_file, frame1, frame2, frame3]):
            return jsonify({'error': 'All files are required'}), 400
        
        # Create request folder in unified_output (same as video analysis)
        request_id = str(uuid.uuid4())
        request_folder = os.path.join(app.config['OUTPUT_FOLDER'], request_id)
        os.makedirs(request_folder, exist_ok=True)
        
        # Save files
        csv_path = os.path.join(request_folder, 'tracks.csv')
        frame1_path = os.path.join(request_folder, 'frame1.jpg')
        frame2_path = os.path.join(request_folder, 'frame2.jpg')
        frame3_path = os.path.join(request_folder, 'frame3.jpg')
        
        csv_file.save(csv_path)
        frame1.save(frame1_path)
        frame2.save(frame2_path)
        frame3.save(frame3_path)
        
        # Insert into database
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO analysis_requests 
            (user_id, csv_path, frame1_path, frame2_path, frame3_path, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?)
        ''', (session['user_id'], csv_path, frame1_path, frame2_path, frame3_path, datetime.now()))
        
        request_db_id = cursor.lastrowid
        db.commit()
        db.close()
        
        return jsonify({
            'success': True,
            'request_id': request_db_id,
            'message': 'Analysis request sent successfully'
        })
        
    except Exception as e:
        print(f"Error submitting request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis_status/<int:request_id>')
@login_required
def get_analysis_status(request_id):
    """Get status of analysis request"""
    try:
        db = get_db()
        request_data = db.execute('''
            SELECT * FROM analysis_requests 
            WHERE id = ? AND user_id = ?
        ''', (request_id, session['user_id'])).fetchone()
        db.close()
        
        if not request_data:
            return jsonify({'error': 'Request not found'}), 404
        
        response = {
            'status': request_data['status'],
            'created_at': request_data['created_at'],
            'completed_at': request_data['completed_at'],
            'admin_notes': request_data['admin_notes']
        }
        
        if request_data['status'] == 'completed' and request_data['results_path']:
            # Parse results files
            results_folder = request_data['results_path']
            if os.path.exists(results_folder):
                files = []
                file_types = {
                    'heatmap.png': {'name': 'Heatmap', 'icon': 'fa-fire'},
                    'overview.png': {'name': 'Overview', 'icon': 'fa-eye'},
                    'tracks.png': {'name': 'Tracks', 'icon': 'fa-route'},
                    'speed.png': {'name': 'Speeds', 'icon': 'fa-tachometer-alt'},
                    'direction.png': {'name': 'Directions', 'icon': 'fa-compass'},
                }
                
                for filename, info in file_types.items():
                    filepath = os.path.join(results_folder, filename)
                    if os.path.exists(filepath):
                        files.append({
                            'name': info['name'],
                            'icon': info['icon'],
                            'path': filepath.replace('\\', '/')
                        })
                
                response['results_files'] = files
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/check_existing_request')
@login_required
def check_existing_request():
    """Check if user has existing request for analysis"""
    analysis_id = request.args.get('analysis_id')
    
    db = get_db()
    existing = db.execute('''
        SELECT id FROM analysis_requests 
        WHERE user_id = ? AND csv_path LIKE ?
        ORDER BY created_at DESC LIMIT 1
    ''', (session['user_id'], f'%{analysis_id}%')).fetchone()
    db.close()
    
    if existing:
        return jsonify({'exists': True, 'request_id': existing['id']})
    return jsonify({'exists': False})

# ===================== ADMIN ROUTES =====================

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard to manage analysis requests"""
    return render_template('admin_dashboard.html')

@app.route('/api/admin/requests')
@admin_required
def get_all_requests():
    """Get all analysis requests for admin"""
    try:
        db = get_db()
        requests = db.execute('''
            SELECT ar.*, u.username
            FROM analysis_requests ar
            JOIN users u ON ar.user_id = u.id
            ORDER BY ar.created_at DESC
        ''').fetchall()
        db.close()
        
        return jsonify({
            'success': True,
            'requests': [dict(r) for r in requests]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/process_request/<int:request_id>', methods=['POST'])
@admin_required
def process_analysis_request(request_id):
    """Process analysis request - generate charts"""
    try:
        db = get_db()
        request_data = db.execute(
            'SELECT * FROM analysis_requests WHERE id = ?', (request_id,)
        ).fetchone()
        
        if not request_data:
            db.close()
            return jsonify({'error': 'Request not found'}), 404
        
        # Update status to processing
        db.execute(
            'UPDATE analysis_requests SET status = ? WHERE id = ?',
            ('processing', request_id)
        )
        db.commit()
        db.close()
        
        # Generate analysis charts
        csv_path = request_data['csv_path']
        results_folder = os.path.join(os.path.dirname(csv_path), 'results')
        os.makedirs(results_folder, exist_ok=True)
        
        # Read CSV
        df = pd.read_csv(csv_path)
        df = ensure_xy_columns(df)
        
        # Determine frame shape from data
        if 'x' in df.columns and 'y' in df.columns:
            max_x = int(df['x'].max() + 100)
            max_y = int(df['y'].max() + 100)
            frame_shape = (max_y, max_x, 3)
        else:
            frame_shape = (1080, 1920, 3)
        
        try:
            # Generate heatmap
            heatmap_path = os.path.join(results_folder, 'heatmap.png')
            generate_heatmap(df, frame_shape, heatmap_path, bins=120)
            
            # Generate track visualization
            tracks_path = os.path.join(results_folder, 'tracks.png')
            generate_track_visualization(df, frame_shape, tracks_path)
            
            # Generate speed histogram (if speed column exists)
            if 'speed' in df.columns:
                speed_path = os.path.join(results_folder, 'speed.png')
                generate_speed_histogram(df, speed_path)
            
            # Generate direction statistics
            direction_stats = {}
            for tid in df['track_id'].unique():
                track_data = df[df['track_id'] == tid]
                direction_info = analyze_track_direction(track_data)
                if direction_info and direction_info['overall_direction'] != 'Track too short':
                    dir_name = direction_info['overall_direction']
                    direction_stats[dir_name] = direction_stats.get(dir_name, 0) + 1
            
            # Generate direction chart
            if direction_stats:
                direction_path = os.path.join(results_folder, 'direction.png')
                generate_direction_chart(direction_stats, direction_path)
            
                print(f"[OK] Charts generated successfully")
                print(f"[RESULTS] Results folder: {results_folder}")
                print(f"[RESULTS] Generated files:")
                for file in os.listdir(results_folder):
                    print(f"   - {file}")
            
            # Update database
            db = get_db()
            db.execute('''
                UPDATE analysis_requests 
                SET status = ?, results_path = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', ('completed', results_folder, request_id))
            db.commit()
            db.close()
            
            print(f"[OK] Database updated - request #{request_id} completed")
            
            return jsonify({
                'success': True,
                'message': 'Analysis created successfully'
            })
            
        except Exception as e:
            db = get_db()
            db.execute(
                'UPDATE analysis_requests SET status = ? WHERE id = ?',
                ('pending', request_id)
            )
            db.commit()
            db.close()
            raise e
            
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/add_note/<int:request_id>', methods=['POST'])
@admin_required
def add_admin_note(request_id):
    """Add admin note to request"""
    try:
        note = request.json.get('note', '')
        
        db = get_db()
        db.execute(
            'UPDATE analysis_requests SET admin_notes = ? WHERE id = ?',
            (note, request_id)
        )
        db.commit()
        db.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/delete_request/<int:request_id>', methods=['DELETE'])
@admin_required
def delete_analysis_request(request_id):
    """Delete analysis request and its files"""
    try:
        db = get_db()
        request_data = db.execute(
            'SELECT * FROM analysis_requests WHERE id = ?', (request_id,)
        ).fetchone()
        
        if not request_data:
            db.close()
            return jsonify({'error': 'Request not found'}), 404
        
        # Delete files from disk
        try:
            # Delete CSV file
            if request_data['csv_path'] and os.path.exists(request_data['csv_path']):
                os.remove(request_data['csv_path'])
            
            # Delete frame files
            for col in ['frame1_path', 'frame2_path', 'frame3_path']:
                frame_path = request_data[col]
                if frame_path and os.path.exists(frame_path):
                    os.remove(frame_path)
            
            # Delete results folder
            if request_data['results_path'] and os.path.exists(request_data['results_path']):
                shutil.rmtree(request_data['results_path'])
            
            # Delete request folder
            request_dir = os.path.dirname(request_data['csv_path'])
            if os.path.exists(request_dir):
                shutil.rmtree(request_dir)
        
        except Exception as file_error:
            print(f"Warning: Could not delete all files: {file_error}")
        
        # Delete from database
        db.execute('DELETE FROM analysis_requests WHERE id = ?', (request_id,))
        db.commit()
        db.close()
        
        return jsonify({'success': True, 'message': 'Request and files deleted successfully'})
        
    except Exception as e:
        print(f"Error deleting request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/save_analysis_to_request', methods=['POST'])
@admin_required
def save_analysis_to_request():
    """Copy analysis results to request folder"""
    try:
        data = request.json
        request_id = data.get('request_id')
        analysis_id = data.get('analysis_id')
        
        if not request_id or not analysis_id:
            return jsonify({'error': 'Missing parameters'}), 400
        
        # Get request data
        db = get_db()
        request_data = db.execute(
            'SELECT * FROM analysis_requests WHERE id = ?', (request_id,)
        ).fetchone()
        
        if not request_data:
            db.close()
            return jsonify({'error': 'Request not found'}), 404
        
        # Get source and destination folders
        analysis_dir = os.path.join(app.config['OUTPUT_FOLDER'], analysis_id)
        analyzed_dir = os.path.join(analysis_dir, 'analyzed')
        charts_dir = os.path.join(analyzed_dir, 'charts')
        
        request_base_dir = os.path.dirname(request_data['csv_path'])
        results_base = os.path.join(request_base_dir, 'results')
        # Create analysis subfolder to separate from line_drawing
        results_dir = os.path.join(results_base, 'analysis')
        os.makedirs(results_dir, exist_ok=True)
        
        # Copy analysis charts
        if os.path.exists(charts_dir):
            for file in os.listdir(charts_dir):
                src = os.path.join(charts_dir, file)
                dst = os.path.join(results_dir, file)
                shutil.copy2(src, dst)
                print(f"Copied: {file}")
        
        # Copy CSV analysis files
        for csv_file in ['totals.csv', 'track_statistics.csv', 'direction_changes.csv']:
            src = os.path.join(analyzed_dir, csv_file)
            if os.path.exists(src):
                dst = os.path.join(results_dir, csv_file)
                shutil.copy2(src, dst)
                print(f"Copied: {csv_file}")
        
        # Update database - set to completed
        db.execute('''
            UPDATE analysis_requests 
            SET status = ?, results_path = ?, completed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', ('completed', results_base, request_id))
        db.commit()
        db.close()
        
        print(f"[OK] All results saved to request #{request_id}")
        print(f"[RESULTS] Line Drawing: {os.path.join(results_base, 'line_drawing')}")
        print(f"[RESULTS] Analysis: {results_dir}")
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error saving analysis to request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/request_status/<int:request_id>', methods=['GET'])
@login_required
def get_request_status(request_id):
    """Get status of analysis request for user"""
    try:
        db = get_db()
        request_data = db.execute('''
            SELECT ar.*, u.username 
            FROM analysis_requests ar
            LEFT JOIN users u ON ar.user_id = u.id
            WHERE ar.id = ?
        ''', (request_id,)).fetchone()
        
        db.close()
        
        if not request_data:
            return jsonify({'error': 'Request not found'}), 404
        
        # Check if user owns this request or is admin
        if request_data['user_id'] != session['user_id'] and session.get('role') != 'admin':
            return jsonify({'error': 'Unauthorized'}), 403
        
        result = {
            'id': request_data['id'],
            'status': request_data['status'],
            'created_at': request_data['created_at'],
            'completed_at': request_data['completed_at'],
            'admin_notes': request_data['admin_notes'],
            'csv_path': request_data['csv_path'],
            'frame1_path': request_data['frame1_path'],
            'frame2_path': request_data['frame2_path'],
            'frame3_path': request_data['frame3_path']
        }
        
        # Add results if completed
        if request_data['status'] == 'completed' and request_data['results_path']:
            results_folder = request_data['results_path']
            result['results'] = {}
            print(f"[INFO] Checking for charts in: {results_folder}")
            
            # Check for generated charts
            for chart_name in ['heatmap.png', 'tracks.png', 'speed.png', 'direction.png']:
                chart_path = os.path.join(results_folder, chart_name)
                if os.path.exists(chart_path):
                    # Convert absolute path to relative URL path
                    try:
                        relative_path = os.path.relpath(chart_path, 'unified_output').replace('\\', '/')
                        chart_url = f'/unified_output/{relative_path}'
                        result['results'][chart_name.replace('.png', '')] = chart_url
                        print(f"[OK] Found {chart_name}: {chart_url}")
                    except Exception as path_error:
                        print(f"[ERROR] Error processing path for {chart_name}: {path_error}")
                else:
                    print(f"[WARN] Chart not found: {chart_path}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error getting request status: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/my_requests', methods=['GET'])
@login_required
def get_my_requests():
    """Get all requests for current user"""
    try:
        db = get_db()
        requests = db.execute('''
            SELECT * FROM analysis_requests 
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (session['user_id'],)).fetchall()
        
        db.close()
        
        result = {
            'requests': [dict(req) for req in requests]
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error getting requests: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/my_requests/<int:request_id>', methods=['DELETE'])
@login_required
def delete_my_request(request_id):
    """Delete user's own analysis request"""
    try:
        db = get_db()
        request_data = db.execute(
            'SELECT * FROM analysis_requests WHERE id = ? AND user_id = ?',
            (request_id, session['user_id'])
        ).fetchone()
        
        if not request_data:
            db.close()
            return jsonify({'error': 'Request not found or you do not have permission to delete'}), 404
        
        # Delete files from disk
        try:
            # Delete CSV file
            if request_data['csv_path'] and os.path.exists(request_data['csv_path']):
                os.remove(request_data['csv_path'])
            
            # Delete frame files
            for col in ['frame1_path', 'frame2_path', 'frame3_path']:
                frame_path = request_data[col]
                if frame_path and os.path.exists(frame_path):
                    os.remove(frame_path)
            
            # Delete results folder
            if request_data['results_path'] and os.path.exists(request_data['results_path']):
                shutil.rmtree(request_data['results_path'])
        
        except Exception as file_error:
            print(f"Warning: Could not delete all files: {file_error}")
        
        # Delete from database
        db.execute('DELETE FROM analysis_requests WHERE id = ?', (request_id,))
        db.commit()
        db.close()
        
        return jsonify({'success': True, 'message': 'Request deleted successfully'})
        
    except Exception as e:
        print(f"Error deleting request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/unified_output/<path:filename>')
@login_required
def serve_unified_output(filename):
    """Serve files from unified_output folder with ownership check"""
    try:
        # Determine top-level folder from requested path
        folder = filename.split('/')[0].split('\\')[0]
        db = get_db()
        request_row = db.execute('''
            SELECT * FROM analysis_requests
            WHERE (results_path LIKE ? OR csv_path LIKE ?)
            LIMIT 1
        ''', (f'%{folder}%', f'%{folder}%')).fetchone()
        db.close()

        # If we found a matching request, ensure the user owns it or is admin
        if request_row:
            if request_row['user_id'] != session.get('user_id') and session.get('role') != 'admin':
                return jsonify({'error': 'Unauthorized'}), 403
        else:
            # No matching DB entry — deny access for non-admin users
            if session.get('role') != 'admin':
                return jsonify({'error': 'Unauthorized'}), 403

        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/download_results/<int:request_id>', methods=['GET'])
@login_required
def download_results_zip(request_id):
    """Download all analysis results as ZIP file"""
    try:
        db = get_db()
        request_data = db.execute(
            'SELECT * FROM analysis_requests WHERE id = ?', (request_id,)
        ).fetchone()
        db.close()
        
        if not request_data:
            return jsonify({'error': 'الطلب غير موجود'}), 404
        
        # Check permissions
        if request_data['user_id'] != session['user_id'] and session.get('role') != 'admin':
            return jsonify({'error': 'Unauthorized'}), 403
        
        if request_data['status'] != 'completed':
            return jsonify({'error': 'Analysis not completed yet'}), 400
        
        # Create ZIP in memory
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            results_folder = request_data['results_path']
            
            # Add all files from results folder
            if results_folder and os.path.exists(results_folder):
                for root, dirs, files in os.walk(results_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, results_folder)
                        zf.write(file_path, arcname)
            
            # Add CSV file
            if request_data['csv_path'] and os.path.exists(request_data['csv_path']):
                zf.write(request_data['csv_path'], 'tracks.csv')
            
            # Add frames
            for i, frame_col in enumerate(['frame1_path', 'frame2_path', 'frame3_path'], 1):
                frame_path = request_data[frame_col]
                if frame_path and os.path.exists(frame_path):
                    ext = os.path.splitext(frame_path)[1]
                    zf.write(frame_path, f'frame_{i}{ext}')
        
        memory_file.seek(0)
        
        from flask import send_file
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'analysis_results_{request_id}.zip'
        )
        
    except Exception as e:
        print(f"Error creating ZIP: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ===================== ORIGINAL ROUTES =====================


def calculate_direction_angle(dx, dy):
    """Calculate angle in degrees from dx, dy"""
    import math
    angle = math.atan2(dy, dx) * 180 / math.pi
    return angle

def get_direction_name(angle):
    """Convert angle to Arabic direction name"""
    # Normalize angle to 0-360
    angle = angle % 360
    
    if angle < 0:
        angle += 360
    
    # 8 directions
    if 337.5 <= angle or angle < 22.5:
        return "شرق"  # East (right)
    elif 22.5 <= angle < 67.5:
        return "جنوب شرقي"  # Southeast
    elif 67.5 <= angle < 112.5:
        return "جنوب"  # South (down)
    elif 112.5 <= angle < 157.5:
        return "جنوب غربي"  # Southwest
    elif 157.5 <= angle < 202.5:
        return "غرب"  # West (left)
    elif 202.5 <= angle < 247.5:
        return "شمال غربي"  # Northwest
    elif 247.5 <= angle < 292.5:
        return "شمال"  # North (up)
    elif 292.5 <= angle < 337.5:
        return "شمال شرقي"  # Northeast
    return "غير محدد"

def analyze_track_direction(track_df):
    """Analyze direction changes for a single track"""
    import math
    import numpy as np
    
    if len(track_df) < 2:
        return {
            'overall_direction': 'مسار قصير جداً',
            'direction_changes': [],
            'total_distance': 0,
            'dominant_direction': 'غير محدد'
        }
    
    # Sort by frame
    track_df = track_df.sort_values('frame_idx')
    
    # Get coordinates
    x_coords = track_df['x'].values
    y_coords = track_df['y'].values
    frames = track_df['frame_idx'].values
    
    # Calculate total distance
    total_distance = 0
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        total_distance += math.sqrt(dx**2 + dy**2)
    
    # Simple analysis: first, middle, last frames only
    first_idx = 0
    last_idx = len(x_coords) - 1
    middle_idx = len(x_coords) // 2
    
    direction_changes = []
    direction_counts = {}
    
    # First to Middle
    if middle_idx > first_idx:
        dx = x_coords[middle_idx] - x_coords[first_idx]
        dy = y_coords[middle_idx] - y_coords[first_idx]
        
        if abs(dx) > 1 or abs(dy) > 1:
            angle = calculate_direction_angle(dx, dy)
            direction = get_direction_name(angle)
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
            
            direction_changes.append({
                'frame': int(frames[first_idx]),
                'direction': direction,
                'angle': angle,
                'position': (float(x_coords[first_idx]), float(y_coords[first_idx]))
            })
            
            print(f"  First→Middle: dx={dx:.1f}, dy={dy:.1f}, angle={angle:.1f}° -> {direction}")
    
    # Middle to Last
    if last_idx > middle_idx:
        dx = x_coords[last_idx] - x_coords[middle_idx]
        dy = y_coords[last_idx] - y_coords[middle_idx]
        
        if abs(dx) > 1 or abs(dy) > 1:
            angle = calculate_direction_angle(dx, dy)
            direction = get_direction_name(angle)
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
            
            # Only add if direction changed significantly
            if not direction_changes or direction != direction_changes[-1]['direction']:
                direction_changes.append({
                    'frame': int(frames[middle_idx]),
                    'direction': direction,
                    'angle': angle,
                    'position': (float(x_coords[middle_idx]), float(y_coords[middle_idx]))
                })
            
            print(f"  Middle→Last: dx={dx:.1f}, dy={dy:.1f}, angle={angle:.1f}° -> {direction}")
    
    # Overall direction (first to last)
    dx_overall = x_coords[last_idx] - x_coords[first_idx]
    dy_overall = y_coords[last_idx] - y_coords[first_idx]
    
    overall_angle = calculate_direction_angle(dx_overall, dy_overall)
    overall_direction = get_direction_name(overall_angle)
    direction_counts[overall_direction] = direction_counts.get(overall_direction, 0) + 1
    
    print(f"  Overall (First→Last): dx={dx_overall:.1f}, dy={dy_overall:.1f}, angle={overall_angle:.1f}° -> {overall_direction}")
    
    # Dominant direction (most frequent)
    dominant_direction = max(direction_counts, key=direction_counts.get) if direction_counts else overall_direction
    
    print(f"  Dominant direction: {dominant_direction}")
    print(f"  Direction counts: {direction_counts}")
    
    return {
        'overall_direction': overall_direction,
        'dominant_direction': dominant_direction,
        'direction_changes': direction_changes,
        'total_distance': float(total_distance),
        'direction_counts': direction_counts
    }
    
    # Get coordinates (handle different column names)
    x_col = 'px' if 'px' in track_df.columns else ('x' if 'x' in track_df.columns else 'smoothed_x')
    y_col = 'py' if 'py' in track_df.columns else ('y' if 'y' in track_df.columns else 'smoothed_y')
    
    if x_col not in track_df.columns or y_col not in track_df.columns:
        return {
            'overall_direction': 'إحداثيات غير متوفرة',
            'direction_changes': [],
            'total_distance': 0,
            'dominant_direction': 'غير محدد'
        }
    
    x = track_df[x_col].values
    y = track_df[y_col].values
    frames = track_df['frame_idx'].values
    
    # Calculate overall direction (from start to end)
    dx_overall = x[-1] - x[0]
    dy_overall = y[-1] - y[0]
    overall_angle = calculate_direction_angle(dx_overall, dy_overall)
    overall_direction = get_direction_name(overall_angle)
    
    # Calculate total distance
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    total_distance = float(np.sum(distances))
    
    # Detect direction changes (using moving window)
    window_size = min(10, len(track_df) // 3)  # Adaptive window
    if window_size < 3:
        window_size = 3
    
    direction_changes = []
    direction_counts = {}
    
    for i in range(0, len(x) - window_size, window_size // 2):
        end_idx = min(i + window_size, len(x))
        
        dx = x[end_idx - 1] - x[i]
        dy = y[end_idx - 1] - y[i]
        
        # Ignore very small movements
        if abs(dx) < 5 and abs(dy) < 5:
            continue
        
        angle = calculate_direction_angle(dx, dy)
        direction = get_direction_name(angle)
        
        # Count directions
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        # Detect significant change
        if direction_changes:
            last_direction = direction_changes[-1]['direction']
            if last_direction != direction:
                # Calculate angle difference
                last_angle = direction_changes[-1]['angle']
                angle_diff = abs(angle - last_angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Only record if angle change is significant (> 30 degrees)
                if angle_diff > 30:
                    direction_changes.append({
                        'frame': int(frames[i]),
                        'direction': direction,
                        'angle': float(angle),
                        'position': (float(x[i]), float(y[i]))
                    })
        else:
            direction_changes.append({
                'frame': int(frames[i]),
                'direction': direction,
                'angle': float(angle),
                'position': (float(x[i]), float(y[i]))
            })
    
    # Get dominant direction
    if direction_counts:
        dominant_direction = max(direction_counts.items(), key=lambda item: item[1])[0]
    # Get dominant direction
    if direction_counts:
        dominant_direction = max(direction_counts.items(), key=lambda item: item[1])[0]
    else:
        dominant_direction = overall_direction
    
    return {
        'overall_direction': overall_direction,
        'dominant_direction': dominant_direction,
        'direction_changes': direction_changes,
        'total_distance': float(total_distance),
        'direction_counts': direction_counts
    }

# ====================== Visualization Functions from testApp2 =====================
def ensure_xy_columns(df):
    """Ensure px/py columns exist for visualization"""
    df = df.copy()
    
    # Use smoothed coordinates if available
    if 'smoothed_x' in df.columns and 'smoothed_y' in df.columns:
        df['px'] = df['smoothed_x'].astype(float)
        df['py'] = df['smoothed_y'].astype(float)
    elif 'x' in df.columns and 'y' in df.columns:
        df['px'] = df['x'].astype(float)
        df['py'] = df['y'].astype(float)
    else:
        raise ValueError("Missing required columns: need either (x, y) or (smoothed_x, smoothed_y)")
    
    return df

def generate_overview(frame_bgr, df, output_path, alpha=0.85):
    """Generate overview image with all tracks overlaid"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - skipping overview generation")
        return None
    
    try:
        df = ensure_xy_columns(df)
        img = frame_bgr.copy()
        overlay = img.copy()
        
        # Draw all tracks
        for tid, g in df.groupby("track_id"):
            pts = g[["px", "py"]].to_numpy(int)
            if len(pts) < 2:
                continue
            color = (0, 255, 255)  # Yellow in BGR
            for i in range(len(pts) - 1):
                cv2.line(overlay, tuple(pts[i]), tuple(pts[i + 1]), color, 2, cv2.LINE_AA)
        
        # Blend overlay with original
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
        
        # Save image
        cv2.imwrite(str(output_path), img)
        print(f"[OK] Overview image saved: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"[ERROR] Error generating overview: {e}")
        return None

def generate_heatmap(df, frame_shape, output_path, bins=120):
    """Generate heatmap of trajectory density"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - skipping heatmap generation")
        return None
    
    try:
        df = ensure_xy_columns(df)
        H, W = frame_shape[:2]
        pts = df[["px", "py"]].to_numpy(float)
        
        if pts.size == 0:
            print("No points for heatmap")
            return None
        
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Create 2D histogram
        heat, xedges, yedges = np.histogram2d(y, x, bins=bins, range=[[0, H], [0, W]])
        heat = heat.T
        
        # Plot heatmap
        fig = plt.figure(figsize=(12, 7), dpi=120)
        plt.imshow(heat, origin='lower', interpolation='bilinear', cmap='hot')
        plt.title("Trajectory Density Heatmap", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.colorbar(label='Density', shrink=0.8)
        plt.tight_layout()
        
        # Save figure
        fig.savefig(str(output_path), bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        print(f"[OK] Heatmap saved: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"[ERROR] Error generating heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_track_visualization(df, frame_shape, output_path):
    """Generate visualization showing individual track paths with colors"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - skipping track visualization")
        return None
    
    try:
        df = ensure_xy_columns(df)
        
        fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
        ax.set_xlim(0, frame_shape[1])
        ax.set_ylim(frame_shape[0], 0)  # Invert Y axis
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a1a')
        fig.patch.set_facecolor('#1a1a1a')
        
        # Get colormap
        from matplotlib import cm
        colors = cm.rainbow(np.linspace(0, 1, df['track_id'].nunique()))
        
        # Draw each track with different color
        for idx, (tid, g) in enumerate(df.groupby("track_id")):
            pts = g[["px", "py"]].to_numpy(float)
            if len(pts) < 2:
                continue
            
            color = colors[idx % len(colors)]
            ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.7, linewidth=2)
            
            # Mark start (green) and end (red)
            ax.scatter([pts[0, 0]], [pts[0, 1]], s=50, color='lime', marker='o', zorder=10)
            ax.scatter([pts[-1, 0]], [pts[-1, 1]], s=50, color='red', marker='s', zorder=10)
        
        ax.set_title('All Vehicle Trajectories', color='white', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()
        
        fig.savefig(str(output_path), bbox_inches='tight', dpi=150, facecolor='#1a1a1a')
        plt.close(fig)
        
        print(f"[OK] Track visualization saved: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"[ERROR] Error generating track visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_speed_histogram(df, output_path):
    """Generate histogram of vehicle speeds"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - skipping speed histogram")
        return None
    
    try:
        if 'speed' not in df.columns or df['speed'].isna().all():
            print("No speed data available")
            return None
        
        speeds = df[df['speed'] > 0]['speed'].values
        if len(speeds) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        ax.hist(speeds, bins=30, color='#667eea', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Speed (pixels/frame)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Vehicle Speed Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig.savefig(str(output_path), bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        print(f"[OK] Speed histogram saved: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"[ERROR] Error generating speed histogram: {e}")
        return None

def generate_direction_chart(direction_stats, output_path):
    """Generate pie chart of direction distribution"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available - skipping direction chart")
        return None
    
    try:
        if not direction_stats or len(direction_stats) == 0:
            print("No direction data available")
            return None
        
        labels = list(direction_stats.keys())
        sizes = list(direction_stats.values())
        
        # Define colors for each direction
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', 
                  '#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors[:len(labels)],
                                            autopct='%1.1f%%', startangle=90,
                                            textprops={'fontsize': 11, 'weight': 'bold'})
        
        # Make percentage text white
        for autotext in autotexts:
            autotext.set_color('white')
        
        ax.set_title('Vehicle Direction Distribution', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        fig.savefig(str(output_path), bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        print(f"[OK] Direction chart saved: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"[ERROR] Error generating direction chart: {e}")
        return None

def filter_track_data(df, config=None):
    """
    خوارزمية فلترة قوية لتنظيف بيانات التتبع
    Filter out low-quality tracks based on multiple criteria
    """
    import numpy as np
    
    if config is None:
        config = {
            'min_track_length': 10,          # الحد الأدنى لعدد الإطارات
            'max_track_length': 10000,       # الحد الأقصى لعدد الإطارات
            'min_movement': 20,              # الحد الأدنى للحركة (بكسل)
            'max_jump': 300,                 # أقصى قفزة مسموح بها بين إطارين
            'max_speed': 500,                # أقصى سرعة (بكسل/إطار)
            'min_bbox_size': 10,             # أقل حجم لـ bounding box
            'max_bbox_size': 2000,           # أكبر حجم لـ bounding box
            'max_frame_gap': 50,             # أقصى فجوة مسموح بها بين إطارات المسار
            'min_direction_consistency': 0.3, # نسبة الاتساق في الاتجاه
            'outlier_threshold': 3.0         # عتبة الـ Z-score للـ outliers
        }
    
    print(f"\n{'='*60}")
    print(f"[INFO] بدء فلترة البيانات")
    print(f"{'='*60}")
    print(f"عدد السجلات الأصلي: {len(df)}")
    print(f"عدد المسارات الأصلي: {df['track_id'].nunique()}")
    
    filtered_tracks = []
    filter_stats = {
        'too_short': 0,
        'too_long': 0,
        'no_movement': 0,
        'large_jumps': 0,
        'high_speed': 0,
        'invalid_bbox': 0,
        'large_gaps': 0,
        'inconsistent_direction': 0,
        'passed': 0
    }
    
    for track_id in df['track_id'].unique():
        track_df = df[df['track_id'] == track_id].copy()
        track_df = track_df.sort_values('frame_idx')
        
        track_len = len(track_df)
        passed = True
        reason = []
        
        # 1. فحص طول المسار
        if track_len < config['min_track_length']:
            filter_stats['too_short'] += 1
            reason.append(f"قصير جداً ({track_len} إطار)")
            passed = False
        
        if track_len > config['max_track_length']:
            filter_stats['too_long'] += 1
            reason.append(f"طويل جداً ({track_len} إطار)")
            passed = False
        
        if not passed:
            continue
        
        # 2. فحص الحركة الكلية
        x_coords = track_df['x'].values
        y_coords = track_df['y'].values
        
        total_movement = 0
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            total_movement += np.sqrt(dx**2 + dy**2)
        
        if total_movement < config['min_movement']:
            filter_stats['no_movement'] += 1
            reason.append(f"لا توجد حركة ({total_movement:.1f} بكسل)")
            passed = False
            continue
        
        # 3. فحص القفزات الكبيرة (Teleportation)
        max_jump = 0
        large_jump_count = 0
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            jump = np.sqrt(dx**2 + dy**2)
            max_jump = max(max_jump, jump)
            
            if jump > config['max_jump']:
                large_jump_count += 1
        
        # السماح بقفزة واحدة كبيرة (قد تكون إعادة ربط)
        if large_jump_count > 1:
            filter_stats['large_jumps'] += 1
            reason.append(f"قفزات كبيرة ({large_jump_count} قفزة)")
            passed = False
            continue
        
        # 4. فحص السرعة
        speeds = []
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            speed = np.sqrt(dx**2 + dy**2)
            speeds.append(speed)
        
        if speeds:
            max_speed = max(speeds)
            avg_speed = np.mean(speeds)
            
            if max_speed > config['max_speed']:
                filter_stats['high_speed'] += 1
                reason.append(f"سرعة عالية ({max_speed:.1f} بكسل/إطار)")
                passed = False
                continue
        
        # 5. فحص حجم bounding box
        if 'w' in track_df.columns and 'h' in track_df.columns:
            bbox_areas = track_df['w'].values * track_df['h'].values
            
            if (bbox_areas < config['min_bbox_size']).any():
                filter_stats['invalid_bbox'] += 1
                reason.append("صندوق صغير جداً")
                passed = False
                continue
            
            if (bbox_areas > config['max_bbox_size']).any():
                filter_stats['invalid_bbox'] += 1
                reason.append("صندوق كبير جداً")
                passed = False
                continue
        
        # 6. فحص الفجوات في الإطارات
        frames = track_df['frame_idx'].values
        max_gap = 0
        for i in range(1, len(frames)):
            gap = frames[i] - frames[i-1]
            max_gap = max(max_gap, gap)
        
        if max_gap > config['max_frame_gap']:
            filter_stats['large_gaps'] += 1
            reason.append(f"فجوة كبيرة ({max_gap} إطار)")
            passed = False
            continue
        
        # 7. فحص اتساق الاتجاه
        if len(x_coords) >= 5:
            # حساب الاتجاهات المحلية
            directions = []
            for i in range(len(x_coords) - 2):
                dx = x_coords[i+2] - x_coords[i]
                dy = y_coords[i+2] - y_coords[i]
                if abs(dx) > 1 or abs(dy) > 1:
                    angle = np.arctan2(dy, dx)
                    directions.append(angle)
            
            if directions:
                # حساب الانحراف المعياري للاتجاهات
                directions = np.array(directions)
                # معالجة الزوايا الدائرية
                mean_dir = np.arctan2(np.mean(np.sin(directions)), np.mean(np.cos(directions)))
                angular_diffs = np.abs(directions - mean_dir)
                angular_diffs = np.minimum(angular_diffs, 2*np.pi - angular_diffs)
                
                consistency = 1 - (np.std(angular_diffs) / np.pi)
                
                if consistency < config['min_direction_consistency']:
                    filter_stats['inconsistent_direction'] += 1
                    reason.append(f"اتجاه غير مستقر ({consistency:.2f})")
                    passed = False
                    continue
        
        # إذا نجح المسار في كل الفحوصات
        if passed:
            filter_stats['passed'] += 1
            filtered_tracks.append(track_id)
    
    # إنشاء DataFrame مفلتر
    filtered_df = df[df['track_id'].isin(filtered_tracks)].copy()
    
    print(f"\n[FILTER RESULTS]")
    print(f"  [OK] مسارات صالحة: {filter_stats['passed']}")
    print(f"  [REJECTED] مسارات مرفوضة:")
    print(f"     - قصيرة جداً: {filter_stats['too_short']}")
    print(f"     - طويلة جداً: {filter_stats['too_long']}")
    print(f"     - بدون حركة: {filter_stats['no_movement']}")
    print(f"     - قفزات كبيرة: {filter_stats['large_jumps']}")
    print(f"     - سرعة عالية: {filter_stats['high_speed']}")
    print(f"     - صندوق غير صالح: {filter_stats['invalid_bbox']}")
    print(f"     - فجوات كبيرة: {filter_stats['large_gaps']}")
    print(f"     - اتجاه غير مستقر: {filter_stats['inconsistent_direction']}")
    print(f"\n[INFO] عدد السجلات بعد الفلترة: {len(filtered_df)}")
    print(f"[INFO] عدد المسارات بعد الفلترة: {filtered_df['track_id'].nunique()}")
    print(f"{'='*60}\n")
    
    return filtered_df, filter_stats

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
@login_required
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/help')
@login_required
def help_page():
    """Help page"""
    return render_template('help.html')

@app.route('/admin/analysis')
@admin_required
def analysis_page():
    """Analysis page for uploading tracks.csv and analyzing data"""
    request_id = request.args.get('request_id')
    return render_template('analysis.html', request_id=request_id)


@app.route('/user/analysis')
@login_required
def user_analysis_page():
    """User-facing analysis page for an individual request (ownership enforced)"""
    request_id = request.args.get('request_id')
    if not request_id:
        abort(400)

    db = get_db()
    req = db.execute('SELECT * FROM analysis_requests WHERE id = ?', (request_id,)).fetchone()
    db.close()

    if not req:
        abort(404)

    # Security: allow only the owner or an admin to view this page
    if req['user_id'] != session.get('user_id') and session.get('role') != 'admin':
        abort(403)

    return render_template('analysis.html', request_id=request_id)

@app.route('/admin/line_drawing')
@admin_required
def line_drawing_page():
    """Line drawing page for drawing legs and counting vehicles"""
    request_id = request.args.get('request_id')
    return render_template('line_drawing.html', request_id=request_id)

@app.route('/api/models', methods=['GET'])
@login_required
def get_models():
    """Get available YOLO models"""
    model_dir = 'modal'
    models = []
    
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.pt'):
                models.append({
                    'name': file,
                    'path': os.path.join(model_dir, file),
                    'size': os.path.getsize(os.path.join(model_dir, file))
                })
    
    return jsonify({'models': models})

def _get_video_recording_metadata(path):
    """Try to extract recording start/end and duration from the video file.
    Strategy (in order):
      1. ffprobe (container tags: creation_time)
      2. OpenCV to get duration + filesystem mtime as best-effort start
    Returns dict with keys: recording_start (ISO str|None), recording_end (ISO str|None), duration (float|None), source
    """
    res = {'recording_start': None, 'recording_end': None, 'duration': None, 'source': None}

    # 1) Try ffprobe (fast and accurate when available)
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
        if proc.returncode == 0 and proc.stdout:
            try:
                info = json.loads(proc.stdout)
                duration = None
                if info.get('format') and info['format'].get('duration'):
                    duration = float(info['format'].get('duration'))
                # Look for creation_time in format tags or stream tags
                creation = None
                fmt_tags = info.get('format', {}).get('tags', {}) or {}
                creation = fmt_tags.get('creation_time') or fmt_tags.get('com.apple.quicktime.creationdate')
                if not creation:
                    for s in info.get('streams', []):
                        tags = s.get('tags', {}) or {}
                        if tags.get('creation_time'):
                            creation = tags.get('creation_time')
                            break
                if creation:
                    try:
                        dt = pd.to_datetime(creation)
                        start = dt.to_pydatetime()
                    except Exception:
                        start = None
                else:
                    start = None

                if start and duration:
                    end = start + timedelta(seconds=duration)
                else:
                    end = None

                if start or duration:
                    res.update({
                        'recording_start': start.isoformat() if start else None,
                        'recording_end': end.isoformat() if end else None,
                        'duration': duration,
                        'source': 'ffprobe'
                    })
                    return res
            except Exception:
                pass
    except Exception:
        # ffprobe missing or failed — fall through to OpenCV/filetime
        pass

    # 2) Fallback — use OpenCV for duration and filesystem mtime as a heuristic start
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        duration = None
        if fps > 0 and frames > 0:
            duration = float(frames) / float(fps)
    except Exception:
        duration = None

    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        mtime = None

    if mtime or duration:
        start = mtime
        end = (start + timedelta(seconds=duration)) if (start and duration) else None
        res.update({'recording_start': start.isoformat() if start else None,
                    'recording_end': end.isoformat() if end else None,
                    'duration': duration,
                    'source': 'filesystem'})

    return res


# Background helper: extract metadata and persist to a sidecar JSON so UI can read it later
def _async_persist_video_metadata(filepath):
    try:
        meta = _get_video_recording_metadata(filepath)
        meta_path = filepath + '.meta.json'
        try:
            with open(meta_path, 'w', encoding='utf-8') as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
            print(f"[INFO] Persisted async metadata for {filepath} -> {meta_path}")
        except Exception as e:
            print(f"[WARN] Failed to write meta sidecar for {filepath}: {e}")
    except Exception as e:
        print(f"[WARN] Async metadata extraction failed for {filepath}: {e}")


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video file (returns filepath plus best-effort recording metadata).

    If the form includes `process_now` (truthy) and the user is logged in, start
    background processing immediately and return the created `job_id`.
    The uploaded file will be deleted by the processing task after completion
    (same behaviour as `POST /api/process`).
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Invalid file type'}), 400

    # Save upload (timestamped to avoid collisions)
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Schedule async metadata extraction (unchanged)
    try:
        threading.Thread(target=_async_persist_video_metadata, args=(filepath,), daemon=True).start()
        print(f"[INFO] Uploaded {filename} — scheduled metadata extraction")
    except Exception as e:
        print(f"[WARN] Failed to start async metadata thread: {e}")

    # Basic response payload
    resp = {
        'success': True,
        'filename': filename,
        'filepath': filepath,
        'recording_start': None,
        'recording_end': None,
        'recording_duration': None,
        'recording_source': None
    }

    # Helper to interpret truthy form values
    def _is_truthy(v):
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in ('1', 'true', 'yes', 'on')

    # Check whether client asked to start processing immediately
    process_now = request.form.get('process_now') or request.form.get('process') or request.args.get('process_now')
    if _is_truthy(process_now):
        # Must be an authenticated user to start a processing job
        if 'user_id' not in session:
            resp['warning'] = 'process_now requested but user is not authenticated; upload saved only.'
            return jsonify(resp)

        # Read optional processing parameters from the form (fall back to same defaults as /api/process)
        model_name = request.form.get('model', 'best.pt')
        try:
            confidence = float(request.form.get('confidence', 0.25))
        except Exception:
            confidence = 0.25
        try:
            iou_threshold = float(request.form.get('iou_threshold', 0.5))
        except Exception:
            iou_threshold = 0.5
        try:
            max_age = int(request.form.get('max_age', 45))
        except Exception:
            max_age = 45
        try:
            min_hits = int(request.form.get('min_hits', 3))
        except Exception:
            min_hits = 3

        model_path = os.path.join('modal', model_name)
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found', 'model': model_name}), 400

        # Create job entry (mirror behaviour of /api/process)
        job_id = str(uuid.uuid4())
        out_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], f'tracking_{out_ts}')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)

        processing_jobs[job_id] = {
            'status': 'initializing',
            'progress': 0,
            'total_frames': 0,
            'processed_frames': 0,
            'output_dir': output_dir,
            'start_time': datetime.now().isoformat(),
            'roi_enabled': False,
            'user_id': session.get('user_id')
        }

        print(f"\n{'='*60}")
        print(f"[NEW JOB CREATED - upload/process_now]")
        print(f"Job ID: {job_id}")
        print(f"Video: {filepath}")
        print(f"Model: {model_path}")
        print(f"Total jobs in memory: {len(processing_jobs)}")
        print(f"{'='*60}\n")

        # Persist a minimal job-sidecar on disk so other server workers/processes can discover this job_id
        try:
            job_index_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
            os.makedirs(job_index_dir, exist_ok=True)
            meta = {
                'status': processing_jobs[job_id]['status'],
                'output_dir': processing_jobs[job_id]['output_dir'],
                'start_time': processing_jobs[job_id]['start_time'],
                'video_path': filepath,
                'user_id': processing_jobs[job_id].get('user_id')
            }
            with open(os.path.join(job_index_dir, 'meta.json'), 'w', encoding='utf-8') as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
        except Exception as _e:
            print(f"[WARN] could not write job index for {job_id}: {_e}")

        # Enqueue background processing via Celery (fallback to local thread if Celery unavailable)
        try:
            from celery_worker import run_process_video
            processing_jobs[job_id]['status'] = 'queued'
            run_process_video.delay(job_id, filepath, model_path, output_dir, confidence, iou_threshold, max_age, min_hits, None)
            print(f"[OK] Enqueued Celery task for job {job_id}")
        except Exception as _e:
            # fallback: start background thread
            thread = threading.Thread(
                target=process_video_task,
                args=(job_id, filepath, model_path, output_dir, confidence, iou_threshold, max_age, min_hits, None)
            )
            thread.daemon = True
            thread.start()
            print(f"[WARN] Celery unavailable, started background thread for job {job_id}: {_e}")

        resp.update({
            'job_id': job_id,
            'analysis_id': job_id,
            'output_dir': output_dir,
            'processing_started': True
        })

    # Return upload response (possibly with processing info)
    return jsonify(resp)


# Serve uploaded files (used for server-side converted files / thumbnails)
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving upload {filename}: {e}")
        return jsonify({'error': 'File not found'}), 404


@app.route('/api/transcode_upload', methods=['POST'])
def api_transcode_upload():
    """Server-side transcode of an uploaded file to H.264/AAC MP4.
    Expects JSON: { "filename": "20260215_...mp4" }
    Returns: { success: True, transcoded_filename, filepath, url }
    """
    data = request.get_json(force=True, silent=True) or {}
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'filename missing'}), 400

    safe_name = secure_filename(filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    if not os.path.exists(in_path):
        return jsonify({'error': 'uploaded file not found', 'filename': safe_name}), 404

    # ensure ffmpeg exists
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        return jsonify({'error': 'ffmpeg not installed on server'}), 500

    base, _ext = os.path.splitext(safe_name)
    out_name = f"{base}_h264.mp4"
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
    # avoid overwrite
    if os.path.exists(out_path):
        out_name = f"{base}_h264_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)

    cmd = [ffmpeg_path, '-y', '-i', in_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', '-b:a', '128k', out_path]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        if proc.returncode != 0:
            stderr = proc.stderr.decode('utf-8', errors='ignore')
            print(f"[ERROR] ffmpeg failed: {stderr[:400]}")
            return jsonify({'error': 'transcode_failed', 'stderr': stderr}), 500
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'transcode_timeout'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    url = url_for('uploaded_file', filename=out_name)
    return jsonify({'success': True, 'transcoded_filename': out_name, 'filepath': out_path, 'url': url})


@app.route('/api/process', methods=['POST'])
@login_required
def process_video():
    """Start video processing (associate job with authenticated user)"""
    data = request.json
    
    video_path = data.get('video_path')
    model_name = data.get('model', 'best.pt')
    confidence = float(data.get('confidence', 0.25))
    iou_threshold = float(data.get('iou_threshold', 0.5))
    max_age = int(data.get('max_age', 45))
    min_hits = int(data.get('min_hits', 3))
    roi_points = data.get('roi_points', None)  # ← استقبال ROI points
    roi_canvas = data.get('roi_canvas', None)
    roi_video = data.get('roi_video', None)
    
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 400
    
    model_path = os.path.join('modal', model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 400
    
    # Create unique job ID
    job_id = str(uuid.uuid4())
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], f'tracking_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
    
    # Initialize job (record owner so UI/history can show it)
    processing_jobs[job_id] = {
        'status': 'initializing',
        'progress': 0,
        'total_frames': 0,
        'processed_frames': 0,
        'output_dir': output_dir,
        'start_time': datetime.now().isoformat(),
        'roi_enabled': roi_points is not None,
        'user_id': session.get('user_id')
    }
    
    print(f"\n{'='*60}")
    print(f"[NEW JOB CREATED]")
    print(f"Job ID: {job_id}")
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"ROI Enabled: {roi_points is not None}")
    print(f"Total jobs in memory: {len(processing_jobs)}")
    print(f"{'='*60}\n")

    # Persist a minimal job-sidecar on disk so other server workers/processes can discover this job_id
    try:
        job_index_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        os.makedirs(job_index_dir, exist_ok=True)
        meta = {
            'status': processing_jobs[job_id]['status'],
            'output_dir': processing_jobs[job_id]['output_dir'],
            'start_time': processing_jobs[job_id]['start_time'],
            'video_path': video_path,
            'user_id': processing_jobs[job_id].get('user_id')
        }
        with open(os.path.join(job_index_dir, 'meta.json'), 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
    except Exception as _e:
        print(f"[WARN] could not write job index for {job_id}: {_e}")
    
    # Start processing in background thread
    # package roi metadata and processing params to pass to background task
    process_every = data.get('process_every_n_frames')
    try:
        process_every = int(process_every) if process_every is not None else None
    except Exception:
        process_every = None

    # Optional: client-provided track confidence threshold (e.g., from Vehicle clarity question)
    confidence_threshold = data.get('confidence_threshold')
    try:
        confidence_threshold = float(confidence_threshold) if confidence_threshold is not None else None
    except Exception:
        confidence_threshold = None

    roi_meta = {
        'roi_points': roi_points,
        'roi_canvas': roi_canvas,
        'roi_video': roi_video,
        'processing_params': {
            'process_every_n_frames': process_every,
            'confidence_threshold': confidence_threshold
        }
    }

    # Expose chosen value for debugging
    if process_every is not None:
        processing_jobs[job_id]['process_every_n_frames'] = process_every
        print(f"[CONFIG] process_every_n_frames set by user: {process_every}")
    if confidence_threshold is not None:
        processing_jobs[job_id]['confidence_threshold'] = confidence_threshold
        print(f"[CONFIG] confidence_threshold set by user: {confidence_threshold}")

    # Enqueue background processing via Celery (fallback to local thread if Celery unavailable)
    try:
        from celery_worker import run_process_video
        processing_jobs[job_id]['status'] = 'queued'
        run_process_video.delay(job_id, video_path, model_path, output_dir, confidence, iou_threshold, max_age, min_hits, roi_meta)
        print(f"[OK] Enqueued Celery task for job {job_id}")
    except Exception as _e:
        thread = threading.Thread(
            target=process_video_task,
            args=(job_id, video_path, model_path, output_dir, confidence, iou_threshold, max_age, min_hits, roi_meta)
        )
        thread.daemon = True
        thread.start()
        print(f"[WARN] Celery unavailable, started background thread for job {job_id}: {_e}")

    return jsonify({
        'success': True,
        'job_id': job_id,
        'analysis_id': job_id,
        'output_dir': output_dir
    })

def process_video_task(job_id, video_path, model_path, output_dir, confidence, iou_threshold, max_age, min_hits, roi_meta=None):
    """Background task for video processing using testApp1 backend"""
    try:
        print(f"\n{'='*60}")
        print(f"[TASK STARTED] PROCESS_VIDEO_TASK STARTED")
        print(f"Job ID: {job_id}")
        print(f"Video: {video_path}")
        print(f"Model: {model_path}")
        print(f"Jobs in memory: {list(processing_jobs.keys())}")
        print(f"{'='*60}\n")
        
        if job_id not in processing_jobs:
            print(f"[ERROR] Job {job_id} not found in processing_jobs!")
            print(f"Available jobs: {list(processing_jobs.keys())}")
            return

        processing_jobs[job_id]['status'] = 'processing'
        print(f"[OK] Starting processing for job {job_id} using testApp1 backend")
        
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                print(f"[GPU] CUDA detected - using CUDA for acceleration")
            else:
                print(f"[GPU] CUDA not detected - using CPU")
        except Exception as e:
            device = 'cpu'
            print(f"[WARN] Could not detect CUDA: {e}. Using CPU")
        
        config = {
            "video_path": video_path,
            "model_path": model_path,
            "out_dir": app.config['OUTPUT_FOLDER'],
            "device": device,
            "conf_threshold": confidence,
            "iou_threshold": iou_threshold,
            "max_age_frames": max_age,
            "min_hits": min_hits,
            
            "min_box_area": 200,
            "max_box_area": 80000,
            "detection_max_dim": 640,
            
            "process_every_n_frames": 3,  
            "confidence_threshold": 0.45,  # default used to split confident vs low-confidence tracks
            "bbox_smooth_alpha": 0.70,    
            "reid_window_sec": 4.0,       
            "reid_cost_threshold": 0.50, 
            "assignment_cost_threshold": 0.55,
            "center_dist_threshold": 140,  
            
            "iou_weight": 0.45,           
            "appearance_weight": 0.25,    
            "motion_weight": 0.15,        
            "distance_weight": 0.10,       
            "class_weight": 0.05,       
            
            "appearance_mode": "hsv",
            "feature_history_size": 10,
            
            "class_suspect_conf": 0.55,
            "class_lock_after_hits": 5,
            "class_change_margin": 3,
            
            "smooth_window_size": 9,
            "smooth_poly_order": 2,
            "quality_min_points": 10,
            "vel_decay": 0.90,
            
            "hide_predicted_points": False, 
            "max_predict_frames": 8,
            
            "roi_points": None,  
            
            "batch_size": 4,  
            "show_live_preview": False,
            "max_frames": 0,
            "fps_override": 0,
            "export_preview_frames": True, 
            "export_format": "all",
            "save_annotated_video": False, 
            "draw_trajectory_arrows": False,  
            "save_cfg_in_report": True,
        }
        
        # If user passed processing params (via request) override defaults
        proc_params = None
        if roi_meta and isinstance(roi_meta, dict):
            proc_params = roi_meta.get('processing_params', {})
        if proc_params:
            user_pe = proc_params.get('process_every_n_frames')
            try:
                if user_pe is not None:
                    user_pe = int(user_pe)
                    config['process_every_n_frames'] = max(1, user_pe)
                    print(f"[CONFIG] Overriding process_every_n_frames => {config['process_every_n_frames']}")
            except Exception as e:
                print(f"[WARN] invalid process_every_n_frames provided: {user_pe} -> {e}")

            # Optional: client-provided confidence threshold for splitting tracks
            user_ct = proc_params.get('confidence_threshold')
            try:
                if user_ct is not None:
                    user_ct = float(user_ct)
                    # Clamp to [0.0, 1.0]
                    user_ct = max(0.0, min(1.0, user_ct))
                    config['confidence_threshold'] = user_ct
                    print(f"[CONFIG] Overriding confidence_threshold => {config['confidence_threshold']}")
            except Exception as e:
                print(f"[WARN] invalid confidence_threshold provided: {user_ct} -> {e}")

        print(f"\n{'='*60}")
        print(f"[ROI DEBUG] ROI FILTERING DEBUG:")
        if roi_meta and isinstance(roi_meta, dict):
            incoming = roi_meta.get('roi_points')
            canvas = roi_meta.get('roi_canvas')
            video_info = roi_meta.get('roi_video')
            print(f"  roi_points from request: {incoming}")
            print(f"  roi_meta type: {type(roi_meta)}")
            print(f"  roi_points is None: {incoming is None}")
            if incoming:
                print(f"  roi_points length: {len(incoming)}")
                print(f"  roi_points content: {incoming}")
            print(f"  roi_canvas: {canvas}")
            print(f"  roi_video: {video_info}")
        else:
            print(f"  roi_meta: {roi_meta}")
        print(f"{'='*60}\n")


        # Debug: Print ROI info
        if roi_meta and isinstance(roi_meta, dict) and roi_meta.get('roi_points'):
            incoming = roi_meta.get('roi_points')
            print(f"\n[ROI] ROI FILTERING ENABLED")
            print(f"ROI Points ({len(incoming)}): {incoming}")
            print(f"{'='*60}\n")
        
        # Progress callback wrapper
        def progress_callback(progress_percent):
            if job_id in processing_jobs:
                processing_jobs[job_id]['progress'] = int(progress_percent)
                # Calculate processed frames from percentage
                total = processing_jobs[job_id].get('total_frames', 0)
                if total > 0:
                    processed = int((progress_percent / 100.0) * total)
                    processing_jobs[job_id]['processed_frames'] = processed
        
        # If ROI metadata provided, scale incoming ROI points from canvas coordinates
        scaled_roi = None
        try:
            import cv2
            cap_tmp = cv2.VideoCapture(str(video_path))
            vid_w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            vid_h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            cap_tmp.release()
        except Exception:
            vid_w = None
            vid_h = None

        if roi_meta and roi_meta.get('roi_points'):
            incoming = roi_meta.get('roi_points')
            canvas = roi_meta.get('roi_canvas') or {}
            video_info = roi_meta.get('roi_video') or {}

            # Determine source coordinate size
            src_w = canvas.get('width') or video_info.get('width') or None
            src_h = canvas.get('height') or video_info.get('height') or None

            scaled_roi = None

            # Support multiple polygons (list of polygons) or single polygon (list of points)
            if isinstance(incoming, list) and len(incoming) > 0 and isinstance(incoming[0], list):
                # incoming is list of polygons
                if vid_w and vid_h and src_w and src_h and src_w > 0 and src_h > 0:
                    sx = vid_w / float(src_w)
                    sy = vid_h / float(src_h)
                    scaled_roi = []
                    for polygon in incoming:
                        scaled_poly = [{'x': float(p['x']) * sx, 'y': float(p['y']) * sy} for p in polygon]
                        scaled_roi.append(scaled_poly)
                else:
                    scaled_roi = incoming
            else:
                # Single polygon expected (list of points/dicts)
                if vid_w and vid_h and src_w and src_h and src_w > 0 and src_h > 0:
                    sx = vid_w / float(src_w)
                    sy = vid_h / float(src_h)
                    scaled_roi = [{'x': float(p['x']) * sx, 'y': float(p['y']) * sy} for p in incoming]
                else:
                    scaled_roi = incoming

        # Inject scaled ROI into config
        if scaled_roi:
            config['roi_points'] = scaled_roi

        # Initialize and process
        processor = VideoProcessorV1(config)
        
        # Set total frames for progress tracking
        if job_id in processing_jobs:
            processing_jobs[job_id]['total_frames'] = processor.total
            processing_jobs[job_id]['processed_frames'] = 0
        
        # Process video
        result = processor.process(progress_callback=progress_callback)
        
        # Delete uploaded video after processing
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Deleted uploaded video: {video_path}")
        except Exception as delete_error:
            print(f"Warning: Could not delete uploaded video: {delete_error}")
        
        # Prepare result in compatible format
        if result and processor.root_out_dir:
            csv_path = processor.root_out_dir / "tracks.csv"
            meta_path = processor.root_out_dir / "meta.json"
            
            result_data = {
                'csv_path': str(csv_path),
                'video_path': None,  # No video output
                'meta': json.loads(meta_path.read_text()) if meta_path.exists() else {},
                'analysis_id': job_id
            }
        else:
            result_data = None
        
        # Update job status
        if job_id in processing_jobs:
            processing_jobs[job_id]['status'] = 'completed'
            processing_jobs[job_id]['progress'] = 100
            processing_jobs[job_id]['result'] = result_data
            processing_jobs[job_id]['end_time'] = datetime.now().isoformat()
            processing_jobs[job_id]['output_dir'] = str(processor.root_out_dir)
            processing_jobs[job_id]['analysis_id'] = job_id
            print(f"Job {job_id} completed successfully")

            # Persist job reference into the job's output meta.json so other processes can find it by job_id
            try:
                out_meta_path = os.path.join(str(processor.root_out_dir), 'meta.json')
                if os.path.exists(out_meta_path):
                    try:
                        m = json.load(open(out_meta_path, 'r', encoding='utf-8'))
                    except Exception:
                        m = {}
                    m['analysis_id'] = job_id
                    with open(out_meta_path, 'w', encoding='utf-8') as fh:
                        json.dump(m, fh, ensure_ascii=False, indent=2)
                    print(f"Updated output meta.json with analysis_id for job {job_id}")
            except Exception as _e:
                print(f"[WARN] Failed to update output meta.json for job {job_id}: {_e}")

            # Also write a small job-sidecar under OUTPUT_FOLDER/<job_id> pointing to the output_dir
            try:
                job_index_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
                os.makedirs(job_index_dir, exist_ok=True)
                index_meta = {
                    'status': 'completed',
                    'output_dir': str(processor.root_out_dir),
                    'start_time': processing_jobs[job_id].get('start_time'),
                    'end_time': processing_jobs[job_id].get('end_time'),
                    'video_path': video_path,
                    'analysis_id': job_id
                }
                with open(os.path.join(job_index_dir, 'meta.json'), 'w', encoding='utf-8') as fh:
                    json.dump(index_meta, fh, ensure_ascii=False, indent=2)
                print(f"Wrote job-sidecar for {job_id} -> {job_index_dir}")
            except Exception as _e:
                print(f"[WARN] could not write job index for {job_id} at completion: {_e}")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"[CRITICAL] ERROR in job {job_id}: {str(e)}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        if job_id in processing_jobs:
            processing_jobs[job_id]['status'] = 'error'
            processing_jobs[job_id]['error'] = str(e)
            processing_jobs[job_id]['end_time'] = datetime.now().isoformat()
        else:
            print(f"[WARN] Cannot update job {job_id} - already deleted from memory!")

@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status"""
    if job_id in processing_jobs:
        job_data = processing_jobs[job_id].copy()
        job_data['analysis_id'] = job_id
        return jsonify(job_data)

    # Fallback: look for a completed result folder under OUTPUT_FOLDER (direct job-index folder)
    folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    if os.path.exists(folder):
        meta_path = os.path.join(folder, 'meta.json')
        meta = {}
        try:
            if os.path.exists(meta_path):
                meta = json.load(open(meta_path, 'r', encoding='utf-8'))
        except Exception:
            meta = {}
        # return a minimal job-like structure so UI can refresh based on archive
        return jsonify({'status': 'completed', 'progress': 100, 'result': {'meta': meta, 'csv_path': os.path.join(folder, 'tracks.csv')}, 'analysis_id': job_id})

    # Second-level fallback: scan all output folders' meta.json for an analysis_id matching job_id
    try:
        for candidate in os.listdir(app.config['OUTPUT_FOLDER']):
            candidate_dir = os.path.join(app.config['OUTPUT_FOLDER'], candidate)
            if not os.path.isdir(candidate_dir):
                continue
            candidate_meta = os.path.join(candidate_dir, 'meta.json')
            if not os.path.exists(candidate_meta):
                continue
            try:
                cm = json.load(open(candidate_meta, 'r', encoding='utf-8'))
            except Exception:
                cm = {}
            if cm.get('analysis_id') == job_id or cm.get('analysis_id') == str(job_id):
                # found matching output folder
                csv_path = os.path.join(candidate_dir, 'tracks.csv')
                return jsonify({'status': 'completed', 'progress': 100, 'result': {'meta': cm, 'csv_path': csv_path}, 'analysis_id': job_id})
    except Exception as _e:
        print(f"[WARN] error while scanning OUTPUT_FOLDER for job_id {job_id}: {_e}")

    # Not found anywhere
    available_jobs = list(processing_jobs.keys())
    print(f"Job {job_id} not found. Available jobs: {available_jobs}")
    return jsonify({'error': 'Job not found', 'available_jobs': available_jobs}), 404

@app.route('/api/job/<job_id>/vehicle_classes', methods=['GET'])
def get_vehicle_classes(job_id):
    """Get vehicle class distribution from job result CSV - counts UNIQUE tracks per class

    Supports both in-memory jobs and archived result folders under OUTPUT_FOLDER.
    """
    csv_path = None

    # Prefer in-memory job data when available
    if job_id in processing_jobs:
        job_data = processing_jobs[job_id]
        result_data = job_data.get('result', {})
        csv_path = result_data.get('csv_path')
    else:
        # Fallback: look for an archived result folder under OUTPUT_FOLDER
        folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        candidate_csv = os.path.join(folder, 'tracks.csv')
        if os.path.exists(candidate_csv):
            csv_path = candidate_csv

    print(f"Getting vehicle classes for job {job_id}")
    print(f"CSV path: {csv_path}")
    
    if not csv_path or not os.path.exists(csv_path):
        print(f"CSV path doesn't exist: {csv_path}")
        return jsonify({'distribution': {}, 'count': 0}), 200
    
    # Parse CSV to get vehicle class distribution by UNIQUE tracks
    try:
        classDistribution = {}
        track_class_map = {}  # Map track_id to vehicle_type
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Total lines in CSV: {len(lines)}")
            
            if len(lines) > 0:
                # Find the required column indices from header
                header = lines[0].strip().split(',')
                print(f"CSV Header: {header}")
                
                track_id_idx = header.index('track_id') if 'track_id' in header else -1
                vehicle_type_idx = header.index('vehicle_type') if 'vehicle_type' in header else -1
                print(f"track_id column index: {track_id_idx}, vehicle_type column index: {vehicle_type_idx}")
                
                if track_id_idx >= 0 and vehicle_type_idx >= 0:
                    for line_num, line in enumerate(lines[1:], start=2):
                        if not line.strip():
                            continue
                        parts = line.strip().split(',')
                        if len(parts) > max(track_id_idx, vehicle_type_idx):
                            track_id = parts[track_id_idx].strip()
                            veh_class = parts[vehicle_type_idx].strip() if parts[vehicle_type_idx].strip() else 'unclassified'
                            # Collect all observed classes for the track (to decide a representative class)
                            if track_id:
                                track_class_map.setdefault(track_id, []).append(veh_class)
                    
                    # Decide a single class per track: pick most common non-empty / non-unknown value, otherwise 'unclassified'
                    from collections import Counter
                    for tid, class_list in track_class_map.items():
                        filtered = [c for c in class_list if c and c.lower() not in ('', 'unknown', 'none', 'nan')]
                        if filtered:
                            chosen = Counter(filtered).most_common(1)[0][0]
                        else:
                            chosen = 'unclassified'
                        classDistribution[chosen] = classDistribution.get(chosen, 0) + 1
                else:
                    print("Required columns not found in CSV header")
        
        print(f"Vehicle class distribution (UNIQUE TRACKS): {classDistribution}")
        total_tracks_from_classes = sum(int(v) for v in classDistribution.values())
        return jsonify({
            'distribution': classDistribution,
            'count': total_tracks_from_classes
        })
    except Exception as e:
        print(f"Error reading vehicle classes: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'distribution': {}, 'count': 0}), 500


@app.route('/api/job/<job_id>/low_confidence_tracks', methods=['GET'])
def get_low_confidence_tracks(job_id):
    """Return a list of tracks that are low-confidence or borderline along with a cropped image data URL for UI review.

    READ-ONLY CONTRACT: This endpoint MUST NOT re-run the classification algorithm or return authoritative quality metrics.
    It should only return `low_conf_track_ids` and `tracks` (precomputed previews and summaries). Metric recompute and persistence
    should happen as part of correction flows (e.g., `/api/job/<job_id>/correct_track`)."""
    # Support completed/archived jobs where the in-memory processing_jobs entry no longer exists
    job_data = processing_jobs.get(job_id)

    if job_data:
        result = job_data.get('result', {})
        csv_path = result.get('csv_path')
        output_dir = job_data.get('output_dir') or (os.path.dirname(csv_path) if csv_path else None)
    else:
        # Fallback to looking up a folder in the OUTPUT_FOLDER by job_id
        folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        csv_path = os.path.join(folder, 'tracks.csv') if folder else None
        output_dir = folder if os.path.exists(folder) else None
        job_data = job_data or {}



    try:
        # Prefer in-memory job result when available (READ-ONLY behaviour).
        # If the job is currently being processed, its `result` will contain `_per_track_summary` and `_corrections_applied`.
        per_track_summary = []
        corrected_ids = set()
        video_path = None
        if job_data and job_data.get('result'):
            try:
                res = job_data.get('result', {})
                        # meta may be stored under result['meta'] or flattened directly under result
                meta_from_res = res.get('meta', {}) or {}
                per_track_summary = (meta_from_res.get('_per_track_summary') or res.get('_per_track_summary') or [])
                # do not treat 'unclassify' as a suppression of low-conf listing
                corrected_ids = set(int(x['track_id']) for x in (meta_from_res.get('_corrections_applied') or res.get('_corrections_applied') or []) if ('track_id' in x and x.get('action','set_class') != 'unclassify'))
                video_path = meta_from_res.get('video_path') or res.get('video_path') or job_data.get('video_path')
                # prefer meta as the authoritative meta object if available
                meta = meta_from_res or meta
            except Exception as e:
                print(f"Warning: unable to read in-memory job result for low_conf endpoint: {e}")
                per_track_summary = []

        # If no in-memory per-track summary, prefer stored meta.json results — do NOT re-run classification algorithm here.
        meta_path = os.path.join(output_dir, 'meta.json') if output_dir else None
        meta = {}
        if not per_track_summary and meta_path and os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path, 'r', encoding='utf-8'))
                # exclude 'unclassify' actions so they still appear as low-conf for review
                corrected_ids = set(int(x['track_id']) for x in meta.get('_corrections_applied', []) if ('track_id' in x and x.get('action','set_class') != 'unclassify'))
                per_track_summary = meta.get('_per_track_summary', []) or []
                video_path = video_path or meta.get('video_path')
            except Exception as e:
                print(f"Warning: could not load meta.json for low_conf endpoint: {e}")
                per_track_summary = []

        # Prefer canonical low ids if authoritative metrics include them
        strict_low_ids_adjusted = []
        try:
            qm = meta.get('quality_metrics', {}) if isinstance(meta, dict) else {}
            if qm and isinstance(qm.get('low_conf_track_ids'), list):
                authoritative_list = [int(x) for x in qm.get('low_conf_track_ids')]
                strict_low_ids_adjusted = [tid for tid in authoritative_list if tid not in corrected_ids]
            elif per_track_summary:
                strict_low_ids = [int(t['track_id']) for t in per_track_summary if (not t.get('confident', False)) and (not t.get('borderline', False))]
                strict_low_ids_adjusted = [tid for tid in strict_low_ids if tid not in corrected_ids]
            else:
                # Fallback: read CSV and use existing 'confident'/'borderline' columns if present (NO recompute)
                try:
                    df = pd.read_csv(csv_path)
                    if 'confident' in df.columns and 'borderline' in df.columns:
                        strict_low_ids = df.loc[(~df['confident']) & (~df['borderline']), 'track_id'].astype(int).tolist()
                        strict_low_ids_adjusted = [int(x) for x in strict_low_ids if int(x) not in corrected_ids]
                    else:
                        strict_low_ids_adjusted = []
                except Exception as e:
                    print(f"Fallback CSV read error in low_conf endpoint: {e}")
                    strict_low_ids_adjusted = []
        except Exception as e:
            print(f"Error determining canonical low ids in endpoint: {e}")
            strict_low_ids_adjusted = []

        # Build tracks_out only for the requested low ids
        tracks_out = []

        # find video path from meta.json if available
        video_path = meta.get('video_path') if meta else None

        def _placeholder_dataurl(text='No image', w=360, h=240):
            svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'><rect width='100%' height='100%' fill='#0b1220'/><text x='50%' y='50%' fill='#888' font-size='16' font-family='Arial' dominant-baseline='middle' text-anchor='middle'>{text}</text></svg>"
            import urllib.parse
            return 'data:image/svg+xml;utf8,' + urllib.parse.quote(svg)

        def crop_frame_to_dataurl(video_path, frame_idx, bbox, max_dim=600):
            try:
                if not video_path or not os.path.exists(video_path):
                    print(f"Crop warning: video_path missing or not found: {video_path}")
                    return _placeholder_dataurl('No video available')

                cap = cv2.VideoCapture(str(video_path))
                attempts = [int(frame_idx)] + [int(frame_idx + i) for i in range(-5,6) if i !=0]
                found = None
                for fidx in attempts:
                    if fidx < 0:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        found = (fidx, frame)
                        break
                cap.release()

                if not found:
                    print(f"Crop warning: could not read frame around {frame_idx} in {video_path}")
                    return _placeholder_dataurl('Frame not available')

                fidx, frame = found
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = bbox
                x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(w-1, int(x2)); y2 = min(h-1, int(y2))
                if x2 <= x1 or y2 <= y1:
                    print(f"Crop warning: invalid bbox for track at frame {fidx}: {bbox}")
                    return _placeholder_dataurl('Invalid bbox')
                    return _placeholder_dataurl('Empty crop')
                scale = min(1.0, float(max_dim) / max(cw, ch))
                if scale < 1.0:
                    crop = cv2.resize(crop, (int(cw*scale), int(ch*scale)))
                # encode to PNG
                ok, buf = cv2.imencode('.png', crop)
                if not ok:
                    print(f"Crop warning: imencode failed for track at frame {fidx}")
                    return _placeholder_dataurl('Encode failed')
                import base64
                return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode('utf-8')
            except Exception as e:
                print(f"Crop error: {e}")
                import traceback
                traceback.print_exc()
                return _placeholder_dataurl('Error')

        # Read CSV once for representative detections (optional)
        df = None
        if csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Warning: could not read CSV for low_conf endpoint: {e}")
                df = None
        app.logger.debug("Low-conf ids (adjusted): %s", strict_low_ids_adjusted)

        for tid in strict_low_ids_adjusted:
            try:
                tid = int(tid)
            except Exception:
                continue

            app.logger.debug("Processing low-conf track: %s", tid)
            if df is not None:
                g = df[df['track_id'] == tid]
                if g.empty:
                    # Possibly the track isn't present in CSV - fallback to per-track summary
                    g = None
            else:
                g = None

            if g is not None:
                # prefer non-predicted and highest confidence row
                if 'is_predicted' in g.columns:
                    dets = g[~g['is_predicted']]
                else:
                    dets = g
                if dets.empty:
                    dets = g

                # pick best representative row: prefer non-predicted, maximize (confidence * area)
                try:
                    dets2 = dets.copy()
                    dets2['area'] = dets2['width'].astype(float) * dets2['height'].astype(float)
                    dets2['score'] = dets2['confidence'].astype(float).fillna(0.0) * dets2['area'].fillna(0.0)
                    rep = dets2.loc[dets2['score'].idxmax()]
                except Exception:
                    try:
                        rep = dets.loc[dets['confidence'].idxmax()] if not dets.empty else g.iloc[0]
                    except Exception:
                        rep = g.iloc[0]

                # bbox from center x,y and width/height
                cx = float(rep['x']) if 'x' in rep else None
                cy = float(rep['y']) if 'y' in rep else None
                w = float(rep['width']) if 'width' in rep else (float(rep['smoothed_x']) if 'smoothed_x' in rep else 0)
                h = float(rep['height']) if 'height' in rep else (float(rep['smoothed_y']) if 'smoothed_y' in rep else 0)
                if cx is not None and cy is not None and w > 0 and h > 0:
                    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
                    bbox = (x1, y1, x2, y2)
                else:
                    bbox = (0,0,100,100)

                frame_idx = int(rep['frame_idx']) if 'frame_idx' in rep else int(rep.name if isinstance(rep.name, (int,)) else 0)
            else:
                # No CSV data available — fallback to per-track summary values and placeholder preview
                rep = {}
                bbox = (0,0,100,100)
                frame_idx = 0

            # add padding to bbox to include context
            try:
                pad = float(job_data.get('preview_padding', 0.20))
            except Exception:
                pad = 0.20
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            x1p = bbox[0] - bw * pad
            y1p = bbox[1] - bh * pad
            x2p = bbox[2] + bw * pad
            y2p = bbox[3] + bh * pad
            bbox = (x1p, y1p, x2p, y2p)

            # Prefer pre-exported per-track preview if available
            image_data = None
            previews_dir = os.path.join(output_dir, 'previews') if output_dir else None
            if previews_dir:
                for fname in (f"track_{tid}_large.png", f"track_{tid}.png", f"track_{tid:04d}.png"):
                    ppath = os.path.join(previews_dir, fname)
                    if os.path.exists(ppath):
                        try:
                            with open(ppath, 'rb') as fh:
                                import base64
                                image_data = 'data:image/png;base64,' + base64.b64encode(fh.read()).decode('utf-8')
                            break
                        except Exception:
                            image_data = None

            if image_data is None:
                image_data = crop_frame_to_dataurl(video_path, frame_idx, bbox)

            # Gather per-track summary if available
            track_summary = None
            if per_track_summary:
                try:
                    track_summary = next((t for t in per_track_summary if int(t.get('track_id', -1)) == tid), None)
                except Exception:
                    track_summary = None

            n_points = int(track_summary.get('n_points')) if track_summary and 'n_points' in track_summary else (int(g.shape[0]) if g is not None else 0)
            n_predicted = int(track_summary.get('n_predicted', 0)) if track_summary and 'n_predicted' in track_summary else (int(g.get('n_predicted', 0).sum()) if (g is not None and 'n_predicted' in g.columns) else 0)
            # sanitize numeric fields to avoid NaN in JSON
            try:
                median_conf = float(track_summary.get('median_conf')) if track_summary and 'median_conf' in track_summary else (float(g['confidence'].median()) if (g is not None and 'confidence' in g.columns) else 0.0)
            except Exception:
                median_conf = 0.0
            try:
                mean_conf = float(track_summary.get('mean_conf')) if track_summary and 'mean_conf' in track_summary else (float(g['confidence'].mean()) if (g is not None and 'confidence' in g.columns) else 0.0)
            except Exception:
                mean_conf = 0.0
            try:
                pct_above = float(track_summary.get('pct_above')) if track_summary and 'pct_above' in track_summary else (float((g['confidence'] > job_data.get('confidence_threshold', 0.45)).mean()) if (g is not None and 'confidence' in g.columns) else 0.0)
            except Exception:
                pct_above = 0.0
            import math
            if not (isinstance(median_conf, float) and math.isfinite(median_conf)):
                median_conf = 0.0
            if not (isinstance(mean_conf, float) and math.isfinite(mean_conf)):
                mean_conf = 0.0
            if not (isinstance(pct_above, float) and math.isfinite(pct_above)):
                pct_above = 0.0
            borderline_flag = bool(track_summary.get('borderline')) if track_summary and 'borderline' in track_summary else False

            # sanitize current_class to a valid string (avoid NaN)
            current_class = None
            try:
                if 'vehicle_type' in rep:
                    current_class = rep.get('vehicle_type')
                    if isinstance(current_class, float) and (not math.isfinite(current_class)):
                        current_class = None
                    else:
                        current_class = str(current_class) if current_class is not None else None
                if not current_class:
                    current_class = 'unclassified'
            except Exception:
                current_class = 'unclassified'

            tracks_out.append({
                'track_id': tid,
                'n_points': n_points,
                'n_predicted': n_predicted,
                'median_conf': median_conf,
                'mean_conf': mean_conf,
                'pct_above': pct_above,
                'borderline': borderline_flag,
                'current_class': current_class,
                'image': image_data,
            })

        # Return only the authoritative low ids and their previews (READ-ONLY contract)
        return jsonify({'low_conf_track_ids': strict_low_ids_adjusted, 'tracks': tracks_out})
    except Exception as e:
        print(f"Error listing low confidence tracks: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'tracks': []}), 500


@app.route('/api/job/<job_id>/correct_track', methods=['POST'])
def correct_track(job_id):
    """Apply a correction for a track: set class and set confidence to 1.0 (or mark ignore).

    Supports both in-memory jobs and archived result folders under OUTPUT_FOLDER.
    """
    data = request.get_json() or {}
    track_id = data.get('track_id')
    new_class = data.get('new_class')
    action = data.get('action', 'set_class')

    # Determine CSV and output_dir: prefer in-memory job, otherwise look in OUTPUT_FOLDER
    job_data = processing_jobs.get(job_id)
    if job_data:
        result = job_data.get('result', {})
        csv_path = result.get('csv_path')
        output_dir = job_data.get('output_dir') or (os.path.dirname(csv_path) if csv_path else None)
    else:
        folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        csv_path = os.path.join(folder, 'tracks.csv') if folder else None
        output_dir = folder if os.path.exists(folder) else None
        # If the CSV is not found, job doesn't exist
        if not csv_path or not os.path.exists(csv_path):
            return jsonify({'error': 'Job not found'}), 404

    if not csv_path or not os.path.exists(csv_path):
        return jsonify({'error': 'tracks.csv not found'}), 400

    if track_id is None:
        return jsonify({'error': 'track_id missing'}), 400

    try:
        # backup csv and meta
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        csv_bak = csv_path + f'.bak.{timestamp}'
        shutil.copy2(csv_path, csv_bak)
        meta_path = os.path.join(output_dir, 'meta.json') if output_dir else None
        meta_bak = None
        if meta_path and os.path.exists(meta_path):
            meta_bak = meta_path + f'.bak.{timestamp}'
            shutil.copy2(meta_path, meta_bak)

        # update CSV
        df = pd.read_csv(csv_path)
        mask = df['track_id'] == int(track_id)
        if mask.sum() == 0:
            return jsonify({'error': 'track_id not found in CSV'}), 400

        if action == 'ignore':
            df.loc[mask, 'vehicle_type'] = 'ignore'
            df.loc[mask, 'confidence'] = 0.0
        elif action == 'unclassify':
            df.loc[mask, 'vehicle_type'] = 'unclassified'
            df.loc[mask, 'confidence'] = 0.0
        else:
            # set class and set confidence to 1 for confirmed detections
            df.loc[mask, 'vehicle_type'] = str(new_class)
            df.loc[mask, 'confidence'] = 1.0

        # write back
        df.to_csv(csv_path, index=False)

        # Load existing meta if exists (used to derive confidence rules when job is archived)
        meta = {}
        meta_path = os.path.join(output_dir, 'meta.json') if output_dir else None
        if meta_path and os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path, 'r', encoding='utf-8'))
            except Exception:
                meta = {}

        # recompute authoritative metrics and update meta.json using either job_data or stored rules
        if job_data:
            cfg = {
                'min_hits': job_data.get('min_hits', 2),
                'confidence_threshold': job_data.get('confidence_threshold', 0.45),
                # Keep default consistent with classify_tracks_from_df (max(5, min_hits))
                'confidence_min_hits': job_data.get('confidence_min_hits', max(5, job_data.get('min_hits', 2))),
                'confidence_prop_threshold': job_data.get('confidence_prop_threshold', 0.6),
                'confidence_border_delta': job_data.get('confidence_border_delta', 0.05),
                'include_predicted_in_confidence': job_data.get('include_predicted_in_confidence', False),
            }
        else:
            # derive cfg from stored meta confidence rules if available
            rules = meta.get('quality_metrics', {}).get('_confidence_rules', {})
            cfg = {
                'min_hits': int(meta.get('config', {}).get('min_hits', 2)),
                'confidence_threshold': float(rules.get('threshold', 0.45)),
                'confidence_min_hits': int(rules.get('min_hits', max(10, int(meta.get('config', {}).get('min_hits', 2))))),
                'confidence_prop_threshold': float(rules.get('prop_threshold', 0.6)),
                'confidence_border_delta': float(rules.get('border_delta', 0.05)),
                'include_predicted_in_confidence': bool(rules.get('include_predicted', False)),
            }

        authoritative_qm, per_track_df = classify_tracks_from_df(df, cfg)

        # Adjust authoritative metrics to reflect human-applied correction where appropriate
        try:
            track_row = None
            try:
                track_row = per_track_df.loc[per_track_df['track_id'] == int(track_id)]
                if not track_row.empty:
                    track_row = track_row.iloc[0]
            except Exception:
                track_row = None

            # Instead of incremental tweaks, force-calculate reporting metrics by marking corrected track CONFIDENT
            try:
                # Only treat this single correction as a CONFIRMATION for metrics if it was not an unclassify/ignore
                corrected_ids_all = set([int(track_id)]) if action not in ('unclassify', 'ignore') else set()
                adj_df = per_track_df.copy()
                if corrected_ids_all:
                    adj_df.loc[adj_df['track_id'].isin(corrected_ids_all), 'confident'] = True
                    adj_df.loc[adj_df['track_id'].isin(corrected_ids_all), 'borderline'] = False

                total2 = int(adj_df.shape[0])
                confident2 = int(adj_df['confident'].sum()) if total2 > 0 else 0
                borderline2 = int(((adj_df['borderline']) & (~adj_df['confident'])).sum()) if total2 > 0 else 0
                low2 = max(0, total2 - confident2 - borderline2)

                authoritative_qm['total_tracks'] = total2
                authoritative_qm['confident_count'] = confident2
                authoritative_qm['high_conf_count'] = confident2
                authoritative_qm['borderline_count'] = borderline2
                authoritative_qm['low_conf_count'] = low2
                authoritative_qm['confident_pct'] = float(confident2 / max(1, total2))
                authoritative_qm['low_conf_pct'] = float(low2 / max(1, total2))
            except Exception as e:
                print(f"Error recalculating authoritative metrics after single correction: {e}")

            # Write corrected metadata entry
        except Exception as e:
            print(f"Error adjusting metrics for correction: {e}")

        # Record correction and store authoritative metrics
        meta['_corrections_applied'] = meta.get('_corrections_applied', []) + [{
            'track_id': int(track_id),
            'new_class': new_class,
            'action': action,
            'timestamp': datetime.now().isoformat()
        }]

        # Deterministic: force all corrected tracks to be considered CONFIDENT in reporting
        try:
            # Exclude 'unclassify' and 'ignore' actions from the set of corrections that imply confirmed tracks
            corrected_ids_all = set(int(x['track_id']) for x in meta.get('_corrections_applied', []) if 'track_id' in x and x.get('action') not in ('unclassify', 'ignore'))
            if corrected_ids_all:
                adj_df = per_track_df.copy()
                adj_df.loc[adj_df['track_id'].isin(corrected_ids_all), 'confident'] = True
                adj_df.loc[adj_df['track_id'].isin(corrected_ids_all), 'borderline'] = False

                total2 = int(adj_df.shape[0])
                confident2 = int(adj_df['confident'].sum()) if total2 > 0 else 0
                borderline2 = int(((adj_df['borderline']) & (~adj_df['confident'])).sum()) if total2 > 0 else 0
                low2 = max(0, total2 - confident2 - borderline2)

                authoritative_qm['total_tracks'] = total2
                authoritative_qm['confident_count'] = confident2
                authoritative_qm['high_conf_count'] = confident2
                authoritative_qm['borderline_count'] = borderline2
                authoritative_qm['low_conf_count'] = low2
                authoritative_qm['confident_pct'] = float(confident2 / max(1, total2))
                authoritative_qm['low_conf_pct'] = float(low2 / max(1, total2))
        except Exception as e:
            print(f"Error recalculating authoritative metrics after batch correction: {e}")

        # Ensure the authoritative qm contains a canonical list of low ids after corrections
        try:
            # Only exclude confirmed corrections (exclude unclassify/ignore from being treated as confirmation)
            corrected_ids_all = set(int(x['track_id']) for x in meta.get('_corrections_applied', []) if 'track_id' in x and x.get('action') not in ('unclassify', 'ignore'))
            remaining_low_ids = per_track_df.loc[(~per_track_df['confident']) & (~per_track_df['borderline']) & (~per_track_df['track_id'].isin(corrected_ids_all)), 'track_id'].astype(int).tolist()
            authoritative_qm['low_conf_track_ids'] = remaining_low_ids
        except Exception:
            pass

        # Remove any stale per-track previews for this track so UI won't show outdated images
        try:
            previews_dir = os.path.join(output_dir, 'previews') if output_dir else None
            if previews_dir and os.path.exists(previews_dir):
                removed = []
                for fname in (f"track_{track_id}.png", f"track_{track_id}_large.png"):
                    p = os.path.join(previews_dir, fname)
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                            removed.append(p)
                        except Exception as e:
                            print(f"Failed to remove preview {p}: {e}")
                if removed:
                    print(f"Removed stale previews for track {track_id}: {removed}")

        # Attempt to regenerate preview for this single track (best shot logic)
                try:
                    # use global pandas and cv2 already imported at module top
                    csv_df = pd.read_csv(csv_path)
                    sub = csv_df[csv_df['track_id'] == int(track_id)]
                    if not sub.empty:
                        # ensure all rows for this track were updated by correction; if not, enforce
                        try:
                            if not all(sub['confidence'].astype(float) == 1.0) and action != 'ignore':
                                csv_df.loc[csv_df['track_id'] == int(track_id), 'confidence'] = 1.0
                                csv_df.loc[csv_df['track_id'] == int(track_id), 'vehicle_type'] = str(new_class)
                                csv_df.to_csv(csv_path, index=False)
                                sub = csv_df[csv_df['track_id'] == int(track_id)]
                        except Exception:
                            pass

                        # pick best row by score = confidence * area
                        s = sub.copy()
                        if 'is_predicted' in s.columns:
                            s = s.loc[~s['is_predicted']]
                            if s.empty:
                                s = sub.copy()
                        s['area'] = s['width'].astype(float) * s['height'].astype(float)
                        s['score'] = s['confidence'].astype(float).fillna(0.0) * s['area'].fillna(0.0)
                        rep = s.loc[s['score'].idxmax()]

                        cx, cy = float(rep['x']), float(rep['y'])
                        w, h = float(rep['width']), float(rep['height'])
                        x1 = max(0, int(cx - w / 2))
                        y1 = max(0, int(cy - h / 2))
                        x2 = int(cx + w / 2)
                        y2 = int(cy + h / 2)

                        # padding
                        pad = 0.20
                        bw = x2 - x1
                        bh = y2 - y1
                        x1p = int(max(0, x1 - bw * pad))
                        y1p = int(max(0, y1 - bh * pad))
                        x2p = int(min(int(meta.get('width', 0) or 0) - 1, x2 + bw * pad))
                        y2p = int(min(int(meta.get('height', 0) or 0) - 1, y2 + bh * pad))

                        # try to open video
                        vpath = meta.get('video_path')
                        if vpath and not os.path.isabs(vpath):
                            vpath = os.path.join(output_dir, vpath)
                        frame_img = None
                        if vpath and os.path.exists(vpath):
                            cap = cv2.VideoCapture(vpath)
                            if cap.isOpened():
                                idx = int(rep.get('frame_idx', 0))
                                # try idx +/- 5
                                for off in [0,1,-1,2,-2,3,-3,4,-4,5,-5]:
                                    fidx = idx + off
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                                    ret, frame = cap.read()
                                    if ret and frame is not None:
                                        frame_img = frame
                                        break
                                cap.release()
                        # fallback to frames directory
                        if frame_img is None:
                            frames_dir = os.path.join(output_dir, 'frames')
                            if os.path.exists(frames_dir):
                                for ff in os.listdir(frames_dir):
                                    if ff.startswith('frame_Middle') or ff.startswith('frame_First') or ff.startswith('frame_Last'):
                                        candidate = cv2.imread(os.path.join(frames_dir, ff))
                                        if candidate is not None:
                                            frame_img = candidate
                                            break

                        if frame_img is not None:
                            h0, w0 = frame_img.shape[:2]
                            x1c = max(0, min(w0 - 1, x1p)); y1c = max(0, min(h0 - 1, y1p))
                            x2c = max(0, min(w0 - 1, x2p)); y2c = max(0, min(h0 - 1, y2p))
                            if x2c > x1c and y2c > y1c:
                                crop = frame_img[y1c:y2c, x1c:x2c].copy()
                                # small
                                try:
                                    import numpy as np
                                    sh, sw = crop.shape[:2]
                                    small = cv2.resize(crop, (max(128, sw), max(128, sh)), interpolation=cv2.INTER_LINEAR)
                                except Exception:
                                    small = crop
                                try:
                                    # large
                                    lh, lw = crop.shape[:2]
                                    scale = 1.0
                                    large_max = 512
                                    if max(lw, lh) > large_max:
                                        scale = float(large_max) / float(max(lw, lh))
                                    large = cv2.resize(crop, (int(lw * scale), int(lh * scale)), interpolation=cv2.INTER_LINEAR)
                                except Exception:
                                    large = crop
                                try:
                                    out_small = os.path.join(previews_dir, f'track_{track_id}.png')
                                    out_large = os.path.join(previews_dir, f'track_{track_id}_large.png')
                                    ok1 = cv2.imwrite(out_small, small)
                                    ok2 = cv2.imwrite(out_large, large)
                                    if ok1 and ok2:
                                        print(f"Regenerated previews for track {track_id}: {out_small}, {out_large}")
                                    else:
                                        print(f"Regenerated previews failed to write for track {track_id}: ok1={ok1}, ok2={ok2}")
                                except Exception as e:
                                    print(f"Error writing regenerated previews for track {track_id}: {e}")
                except Exception as e:
                    print(f"Failed to regenerate preview for track {track_id}: {e}")
        except Exception:
            pass

        # compute remaining low ids after this correction action
        try:
            # Exclude 'unclassify' and 'ignore' actions from the set of confirmed corrections
            corrected_ids_all = set(int(x['track_id']) for x in meta.get('_corrections_applied', []) if 'track_id' in x and x.get('action') not in ('unclassify', 'ignore'))
            remaining_low_ids = per_track_df.loc[(~per_track_df['confident']) & (~per_track_df['borderline']) & (~per_track_df['track_id'].isin(corrected_ids_all)), 'track_id'].astype(int).tolist()
        except Exception:
            remaining_low_ids = []

        # Sanity check: ensure authoritative_qm low_conf_count aligns with remaining_low_ids
        try:
            low_count = int(authoritative_qm.get('low_conf_count', 0))
            if low_count != len(remaining_low_ids):
                print(f"[WARN] low_conf_count ({low_count}) != computed remaining_low_ids length ({len(remaining_low_ids)}) - reconciling")
                authoritative_qm['low_conf_count'] = len(remaining_low_ids)
                total_t = int(authoritative_qm.get('total_tracks', max(1, len(remaining_low_ids))))
                authoritative_qm['low_conf_pct'] = float(authoritative_qm.get('low_conf_count', 0) / max(1, total_t))
        except Exception:
            pass

        # Persist updated authoritative metrics and per-track summary back into meta so subsequent read-only endpoints
        # (e.g., /api/job/<id>/low_confidence_tracks) return a consistent authoritative list.
        try:
            meta['quality_metrics'] = authoritative_qm
            meta['_per_track_summary'] = per_track_df.to_dict(orient='records')
            if meta_path:
                try:
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
                    print(f"Updated meta.json after correction: {meta_path}")
                except Exception as e:
                    print(f"Warning: failed to persist meta.json after correction: {e}")
            # Also update in-memory job result meta if job is in-memory
            if job_id in processing_jobs:
                processing_jobs[job_id]['result'] = processing_jobs[job_id].get('result', {})
                processing_jobs[job_id]['result']['meta'] = meta
        except Exception as e:
            print(f"Warning: could not update meta with authoritative_qm after correction: {e}")

        return jsonify({'success': True, 'quality_metrics': authoritative_qm, 'remaining_low_ids': remaining_low_ids})
    except Exception as e:
        print(f"Error applying correction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Batch corrections endpoint
@app.route('/api/job/<job_id>/correct_tracks', methods=['POST'])
def correct_tracks(job_id):
    """Apply corrections to multiple tracks atomically. Accepts JSON:
       { tracks: [{track_id, new_class, action?}, ...] }
       or { track_ids: [...], new_class: 'truck' } for convenience.
    """
    data = request.get_json() or {}
    items = data.get('tracks')
    if not items:
        ids = data.get('track_ids') or []
        new_class = data.get('new_class')
        if ids and new_class:
            items = [{ 'track_id': int(t), 'new_class': new_class, 'action': 'set_class'} for t in ids]
    if not items:
        return jsonify({'error': 'No tracks provided'}), 400

    # Determine CSV and output_dir same as single-correction handler
    job_data = processing_jobs.get(job_id)
    if job_data:
        result = job_data.get('result', {})
        csv_path = result.get('csv_path')
        output_dir = job_data.get('output_dir') or (os.path.dirname(csv_path) if csv_path else None)
    else:
        folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        csv_path = os.path.join(folder, 'tracks.csv') if folder else None
        output_dir = folder if os.path.exists(folder) else None
        if not csv_path or not os.path.exists(csv_path):
            return jsonify({'error': 'Job not found'}), 404

    if not csv_path or not os.path.exists(csv_path):
        return jsonify({'error': 'tracks.csv not found'}), 400

    try:
        # backup csv and meta once
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        csv_bak = csv_path + f'.bak.{timestamp}'
        shutil.copy2(csv_path, csv_bak)
        meta_path = os.path.join(output_dir, 'meta.json') if output_dir else None
        meta = {}
        if meta_path and os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path, 'r', encoding='utf-8'))
            except Exception:
                meta = {}

        # update CSV in bulk
        df = pd.read_csv(csv_path)
        updated = 0
        for it in items:
            try:
                tid = int(it.get('track_id'))
            except Exception:
                continue
            action = it.get('action', 'set_class')
            new_class = it.get('new_class')
            mask = df['track_id'] == int(tid)
            if mask.sum() == 0:
                continue
            if action == 'ignore':
                df.loc[mask, 'vehicle_type'] = 'ignore'
                df.loc[mask, 'confidence'] = 0.0
            else:
                df.loc[mask, 'vehicle_type'] = str(new_class)
                df.loc[mask, 'confidence'] = 1.0
            updated += 1

        # write back (first pass)
        df.to_csv(csv_path, index=False)

        # Verify updates applied to all rows for each corrected track; enforce if necessary
        try:
            df2 = pd.read_csv(csv_path)
            modified = False
            for it in items:
                try:
                    tid = int(it.get('track_id'))
                except Exception:
                    continue
                mask = (df2['track_id'] == tid)
                if mask.sum() == 0:
                    continue
                if it.get('action', 'set_class') == 'ignore':
                    if not all(df2.loc[mask, 'vehicle_type'] == 'ignore') or not all(df2.loc[mask, 'confidence'] == 0.0):
                        df2.loc[mask, 'vehicle_type'] = 'ignore'
                        df2.loc[mask, 'confidence'] = 0.0
                        modified = True
                else:
                    new_class = str(it.get('new_class'))
                    if not all(df2.loc[mask, 'vehicle_type'] == new_class) or not all(df2.loc[mask, 'confidence'] == 1.0):
                        df2.loc[mask, 'vehicle_type'] = new_class
                        df2.loc[mask, 'confidence'] = 1.0
                        modified = True
            if modified:
                df2.to_csv(csv_path, index=False)
                df = df2
        except Exception:
            pass

        # derive cfg from job_data or meta as single-correction did
        if job_data:
            cfg = {
                'min_hits': job_data.get('min_hits', 2),
                'confidence_threshold': job_data.get('confidence_threshold', 0.45),
                # Keep default consistent with classify_tracks_from_df (max(5, min_hits))
                'confidence_min_hits': job_data.get('confidence_min_hits', max(5, job_data.get('min_hits', 2))),
                'confidence_prop_threshold': job_data.get('confidence_prop_threshold', 0.6),
                'confidence_border_delta': job_data.get('confidence_border_delta', 0.05),
                'include_predicted_in_confidence': job_data.get('include_predicted_in_confidence', False),
            }
        else:
            rules = meta.get('quality_metrics', {}).get('_confidence_rules', {})
            cfg = {
                'min_hits': int(meta.get('config', {}).get('min_hits', 2)),
                'confidence_threshold': float(rules.get('threshold', 0.45)),
                # Keep default consistent with classify_tracks_from_df (max(5, min_hits))
                'confidence_min_hits': int(rules.get('min_hits', max(5, int(meta.get('config', {}).get('min_hits', 2))))),
                'confidence_prop_threshold': float(rules.get('prop_threshold', 0.6)),
                'confidence_border_delta': float(rules.get('border_delta', 0.05)),
                'include_predicted_in_confidence': bool(rules.get('include_predicted', False)),
            }

        authoritative_qm, per_track_df = classify_tracks_from_df(df, cfg)

        # Append correction records
        nowiso = datetime.now().isoformat()
        meta['_corrections_applied'] = meta.get('_corrections_applied', []) + [
            {
                'track_id': int(it.get('track_id')),
                'new_class': it.get('new_class'),
                'action': it.get('action', 'set_class'),
                'timestamp': nowiso
            } for it in items
        ]

        # Treat corrected tracks as confirmed in authoritative metrics (same adjustment logic)
        try:
            # Only consider confirmed corrections (exclude unclassify/ignore) to adjust counts
            corrected_ids_all = set(int(x['track_id']) for x in meta.get('_corrections_applied', []) if 'track_id' in x and x.get('action') not in ('unclassify', 'ignore'))
            if corrected_ids_all:
                total_tracks = int(authoritative_qm.get('total_tracks', 0))
                adj_conf = adj_low = adj_border = 0
                for tid in corrected_ids_all:
                    row = per_track_df.loc[per_track_df['track_id'] == int(tid)]
                    if not row.empty and not bool(row.iloc[0].get('confident', False)):
                        adj_conf += 1
                        if bool(row.iloc[0].get('borderline', False)):
                            adj_border += 1
                        else:
                            adj_low += 1
                if adj_conf > 0:
                    authoritative_qm['confident_count'] = int(authoritative_qm.get('confident_count', 0)) + adj_conf
                    authoritative_qm['high_conf_count'] = authoritative_qm['confident_count']
                    authoritative_qm['borderline_count'] = max(0, int(authoritative_qm.get('borderline_count', 0)) - adj_border)
                    authoritative_qm['low_conf_count'] = max(0, int(authoritative_qm.get('low_conf_count', 0)) - adj_low)
                    if total_tracks > 0:
                        authoritative_qm['confident_pct'] = float(authoritative_qm['confident_count'] / total_tracks)
                        authoritative_qm['low_conf_pct'] = float(authoritative_qm.get('low_conf_count', 0) / total_tracks)
        except Exception:
            pass



        # Remove stale per-track previews for changed tracks; recommend using /regenerate_previews to rebuild if desired
        try:
            previews_dir = os.path.join(output_dir, 'previews') if output_dir else None
            removed = 0
            if previews_dir and os.path.exists(previews_dir):
                for it in items:
                    tid = int(it.get('track_id'))
                    for fname in (f"track_{tid}.png", f"track_{tid}_large.png"):
                        p = os.path.join(previews_dir, fname)
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                                removed += 1
                            except Exception:
                                pass
        except Exception:
            pass

        # compute remaining low ids after adjustment & corrections
        try:
            # Keep 'unclassify' and 'ignore' in the low list (they are not confirmations)
            corrected_ids_all = set(int(x['track_id']) for x in meta.get('_corrections_applied', []) if 'track_id' in x and x.get('action') not in ('unclassify', 'ignore'))
            remaining_low_ids = per_track_df.loc[(~per_track_df['confident']) & (~per_track_df['borderline']) & (~per_track_df['track_id'].isin(corrected_ids_all)), 'track_id'].astype(int).tolist()
        except Exception:
            remaining_low_ids = []

        # Record authoritative low ids explicitly on the metrics so endpoints can return them without recompute
        try:
            authoritative_qm['low_conf_track_ids'] = remaining_low_ids
        except Exception:
            pass

        # Sanity check: ensure authoritative_qm low_conf_count aligns with remaining_low_ids
        try:
            low_count = int(authoritative_qm.get('low_conf_count', 0))
            if low_count != len(remaining_low_ids):
                print(f"[WARN] low_conf_count ({low_count}) != computed remaining_low_ids length ({len(remaining_low_ids)}) - reconciling")
                authoritative_qm['low_conf_count'] = len(remaining_low_ids)
                total_t = int(authoritative_qm.get('total_tracks', max(1, len(remaining_low_ids))))
                authoritative_qm['low_conf_pct'] = float(authoritative_qm.get('low_conf_count', 0) / max(1, total_t))
        except Exception:
            pass

        # Persist authoritative metrics and per-track summary so low_conf endpoint can serve consistent IDs
        try:
            meta['quality_metrics'] = authoritative_qm
            meta['_per_track_summary'] = per_track_df.to_dict(orient='records')
            if meta_path:
                try:
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
                    print(f"Updated meta.json after batch correction: {meta_path}")
                except Exception as e:
                    print(f"Warning: failed to persist meta.json after batch correction: {e}")
            if job_id in processing_jobs:
                processing_jobs[job_id]['result'] = processing_jobs[job_id].get('result', {})
                processing_jobs[job_id]['result']['meta'] = meta
        except Exception as e:
            print(f"Warning: could not update meta after batch corrections: {e}")

        return jsonify({'success': True, 'updated': updated, 'previews_removed': removed, 'quality_metrics': authoritative_qm, 'remaining_low_ids': remaining_low_ids})
    except Exception as e:
        app.logger.exception('Error applying batch corrections')
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs', methods=['GET'])
def list_all_jobs():
    """List all active jobs (for debugging)"""
    jobs_info = {}
    for job_id, job_data in processing_jobs.items():
        jobs_info[job_id] = {
            'status': job_data.get('status', 'unknown'),
            'progress': job_data.get('progress', 0),
            'start_time': job_data.get('start_time', ''),
        }
    return jsonify({
        'total_jobs': len(processing_jobs),
        'jobs': jobs_info
    })

@app.route('/api/results')
@login_required
def list_results():
    """List processing results — users see only their own results unless admin"""
    results = []
    output_folder = app.config['OUTPUT_FOLDER']
    db = get_db()

    if os.path.exists(output_folder):
        for folder in os.listdir(output_folder):
            folder_path = os.path.join(output_folder, folder)
            if not os.path.isdir(folder_path):
                continue
            meta_file = os.path.join(folder_path, 'meta.json')
            # Only attempt to show this folder if:
            # - current user owns a request pointing to it, or
            # - an in-memory processing job for this folder is owned by current user, or
            # - current user is admin
            show = False
            if session.get('role') == 'admin':
                show = True
            else:
                # DB-backed ownership
                req = db.execute('''
                    SELECT id, user_id FROM analysis_requests
                    WHERE (results_path LIKE ? OR csv_path LIKE ?)
                    LIMIT 1
                ''', (f'%{folder}%', f'%{folder}%')).fetchone()
                if req and req['user_id'] == session.get('user_id'):
                    show = True

                # In-memory processing job ownership (for ad-hoc tracking jobs)
                if not show:
                    for job_id, job_data in processing_jobs.items():
                        out = job_data.get('output_dir')
                        if not out:
                            continue
                        try:
                            if os.path.abspath(out) == os.path.abspath(folder_path):
                                if job_data.get('user_id') == session.get('user_id'):
                                    show = True
                                    break
                        except Exception:
                            continue

            if not show:
                continue

            if os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}
                results.append({
                    'folder': folder,
                    'path': folder_path,
                    'meta': meta
                })

    db.close()
    results.sort(key=lambda x: x['folder'], reverse=True)
    return jsonify({'results': results})

@app.route('/api/download/<path:filepath>')
@login_required
def download_file(filepath):
    """Download result files (auth + ownership enforced)"""
    # Normalize path and remove 'unified_output' prefix if present
    if filepath.startswith('unified_output/') or filepath.startswith('unified_output\\'):
        filepath = filepath.replace('unified_output/', '').replace('unified_output\\', '')

    # Construct full path under OUTPUT_FOLDER
    full_path = filepath if os.path.isabs(filepath) else os.path.join(app.config['OUTPUT_FOLDER'], filepath)

    # Prevent access outside OUTPUT_FOLDER unless admin
    abs_full = os.path.abspath(full_path)
    if not abs_full.startswith(os.path.abspath(app.config['OUTPUT_FOLDER'])) and session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    if not os.path.exists(abs_full):
        print(f"Download error: File not found: {abs_full}")
        return jsonify({'error': 'File not found', 'path': filepath}), 404

    # Ownership check: map folder to analysis_requests
    rel = os.path.relpath(abs_full, app.config['OUTPUT_FOLDER']).replace('\\', '/')
    top_folder = rel.split('/')[0]
    db = get_db()
    req = db.execute('''
        SELECT * FROM analysis_requests
        WHERE (results_path LIKE ? OR csv_path LIKE ?)
        LIMIT 1
    ''', (f'%{top_folder}%', f'%{top_folder}%')).fetchone()
    db.close()
    if req:
        if req['user_id'] != session.get('user_id') and session.get('role') != 'admin':
            return jsonify({'error': 'Unauthorized'}), 403
    else:
        # No DB entry for this file; allow only admin
        if session.get('role') != 'admin':
            return jsonify({'error': 'Unauthorized'}), 403

    directory = os.path.dirname(abs_full)
    filename = os.path.basename(abs_full)
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/api/delete_result/<folder_name>', methods=['DELETE'])
@login_required
def delete_result(folder_name):
    """Delete a tracking result folder (ownership enforced)"""
    try:
        output_folder = app.config['OUTPUT_FOLDER']
        folder_path = os.path.join(output_folder, folder_name)
        
        # Security check: ensure folder is within output folder
        if not os.path.abspath(folder_path).startswith(os.path.abspath(output_folder)):
            return jsonify({'error': 'Invalid folder path'}), 400

        # Ownership check: only owner or admin may delete
        db = get_db()
        req = db.execute('''
            SELECT user_id FROM analysis_requests
            WHERE (results_path LIKE ? OR csv_path LIKE ?)
            LIMIT 1
        ''', (f'%{folder_name}%', f'%{folder_name}%')).fetchone()
        db.close()
        if req:
            if req['user_id'] != session.get('user_id') and session.get('role') != 'admin':
                return jsonify({'error': 'Unauthorized'}), 403
        else:
            # No DB entry — only admin allowed to delete orphan folders
            if session.get('role') != 'admin':
                return jsonify({'error': 'Unauthorized'}), 403
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            return jsonify({'error': 'Folder not found'}), 404
        
        # Delete the folder and all its contents
        shutil.rmtree(folder_path)
        
        return jsonify({'success': True, 'message': 'Tracking result deleted successfully'})
        
    except Exception as e:
        print(f"Error deleting result: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/regenerate_previews/<folder_name>', methods=['POST'])
@login_required
def regenerate_previews(folder_name):
    """Regenerate per-track previews for a result folder. Returns counts."""
    try:
        output_folder = app.config['OUTPUT_FOLDER']
        folder_path = os.path.join(output_folder, folder_name)

        # Ownership check: only owner or admin may regenerate previews
        db = get_db()
        req = db.execute('''
            SELECT user_id FROM analysis_requests
            WHERE (results_path LIKE ? OR csv_path LIKE ?)
            LIMIT 1
        ''', (f'%{folder_name}%', f'%{folder_name}%')).fetchone()
        db.close()
        if req:
            if req['user_id'] != session.get('user_id') and session.get('role') != 'admin':
                return jsonify({'error': 'Unauthorized'}), 403
        else:
            # No DB entry — only admin allowed to operate on orphan folders
            if session.get('role') != 'admin':
                return jsonify({'error': 'Unauthorized'}), 403

        if not os.path.exists(folder_path):
            return jsonify({'error': 'Folder not found'}), 404
        tracks_csv = os.path.join(folder_path, 'tracks.csv')
        if not os.path.exists(tracks_csv):
            return jsonify({'error': 'tracks.csv not found'}), 400

        # try to get video path from meta
        meta_path = os.path.join(folder_path, 'meta.json')
        video_path = None
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path, 'r', encoding='utf-8'))
                video_path = meta.get('video_path')
                if video_path and not os.path.isabs(video_path):
                    candidate = os.path.join(folder_path, video_path)
                    if os.path.exists(candidate):
                        video_path = candidate
                    else:
                        video_path = None
            except Exception:
                video_path = None

        cap = None
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                cap = None

        df = pd.read_csv(tracks_csv)
        frames_dir = os.path.join(folder_path, 'frames')
        previews_dir = os.path.join(folder_path, 'previews')
        os.makedirs(previews_dir, exist_ok=True)

        written = 0
        small_min_dim = 128
        large_max_dim = 512
        for tid in sorted(df['track_id'].unique()):
            sub = df[df['track_id'] == tid]
            med = int(sub['frame_idx'].median()) if 'frame_idx' in sub.columns else int(sub.index[0])
            idx = (sub['frame_idx'] - med).abs().idxmin()
            rep = sub.loc[idx]
            cx, cy = float(rep['x']), float(rep['y'])
            w, h = float(rep['width']), float(rep['height'])
            x1 = max(0, int(cx - w / 2))
            y1 = max(0, int(cy - h / 2))
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            frame_img = None
            if cap is not None:
                for offset in [0,1,-1,2,-2,3,-3,4,-4,5,-5]:
                    fidx = med + offset
                    if fidx < 0:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame_img = frame
                        break

            if frame_img is None and os.path.exists(frames_dir):
                for c in ('frame_Middle','frame_First','frame_Last'):
                    for ff in os.listdir(frames_dir):
                        if ff.startswith(c):
                            p = os.path.join(frames_dir, ff)
                            img = cv2.imread(p)
                            if img is not None:
                                frame_img = img
                                break
                    if frame_img is not None:
                        break

            if frame_img is None:
                continue

            h0,w0 = frame_img.shape[:2]
            x1c = max(0, min(w0-1, x1)); y1c = max(0, min(h0-1, y1))
            x2c = max(0, min(w0-1, x2)); y2c = max(0, min(h0-1, y2))
            if x2c <= x1c or y2c <= y1c:
                continue
            crop = frame_img[y1c:y2c, x1c:x2c].copy()

            # small preview (ensure minimum)
            ch, cw = crop.shape[:2]
            if ch < small_min_dim or cw < small_min_dim:
                new_h = max(small_min_dim, ch)
                new_w = max(small_min_dim, cw)
                try:
                    small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    small = crop
            else:
                small = crop

            # large preview
            lh, lw = crop.shape[:2]
            scale = 1.0
            if max(lw, lh) > 0:
                scale = float(large_max_dim) / float(max(lw, lh)) if max(lw, lh) > large_max_dim else 1.0
            if max(lw, lh) < 64:
                scale = max(scale, float(256) / float(max(lw, lh)))
            try:
                large = cv2.resize(crop, (int(lw * scale), int(lh * scale)), interpolation=cv2.INTER_LINEAR)
            except Exception:
                large = crop

            # overlay disabled: keep preview images free of drawn text/IDs per UI preference


            out_small = os.path.join(previews_dir, f'track_{tid}.png')
            out_large = os.path.join(previews_dir, f'track_{tid}_large.png')
            try:
                cv2.imwrite(out_small, small)
                cv2.imwrite(out_large, large)
                written += 1
            except Exception as e:
                print(f"Failed to write preview for track {tid}: {e}")

        if cap is not None:
            cap.release()

        return jsonify({'success': True, 'written': written})

        if cap is not None:
            cap.release()

        return jsonify({'success': True, 'written': written})
    except Exception as e:
        print(f"Error regenerating previews: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        print(f"Error deleting result: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_request_id_by_folder/<folder_name>')
@login_required
def get_request_id_by_folder(folder_name):
    """Get the analysis request_id for a given folder name (only if user owns it or is admin)"""
    try:
        db = get_db()
        request_data = db.execute('''
            SELECT id, user_id FROM analysis_requests 
            WHERE csv_path LIKE ? OR results_path LIKE ?
            LIMIT 1
        ''', (f'%{folder_name}%', f'%{folder_name}%')).fetchone()
        db.close()

        if not request_data:
            return jsonify({'success': False, 'message': 'No analysis request found for this folder'})

        if request_data['user_id'] != session.get('user_id') and session.get('role') != 'admin':
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403

        return jsonify({'success': True, 'request_id': request_data['id']})
        
    except Exception as e:
        print(f"Error getting request_id: {e}")
        return jsonify({'error': str(e)}), 500

# ================== Analysis Routes ==================
@app.route('/api/analyze/upload', methods=['POST'])
@login_required
def upload_for_analysis():
    """Upload tracks.csv for analysis"""
    try:
        print("\n" + "="*60)
        print("[ANALYSIS] UPLOAD REQUEST RECEIVED")
        print(f"Request files: {request.files}")
        print(f"Request form: {request.form}")
        print("="*60 + "\n")
        
        if 'file' not in request.files:
            print("[ERROR] No 'file' in request.files")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print(f"[FILE] File received: {file.filename}")
        
        if file.filename == '':
            print("[ERROR] Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            print(f"[ERROR] Invalid file type: {file.filename}")
            return jsonify({'error': 'يجب أن يكون الملف بصيغة CSV'}), 400
        
        # Create analysis folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_dir = os.path.join(app.config['OUTPUT_FOLDER'], f'analysis_{timestamp}')
        
        print(f"[DIR] Creating directory: {analysis_dir}")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save CSV file
        csv_path = os.path.join(analysis_dir, 'tracks.csv')
        print(f"[SAVE] Saving file to: {csv_path}")
        file.save(csv_path)
        
        # Verify file was saved
        if not os.path.exists(csv_path):
            print(f"[ERROR] File not saved successfully")
            return jsonify({'error': 'فشل حفظ الملف'}), 500
        
        file_size = os.path.getsize(csv_path)
        print(f"[OK] File saved successfully ({file_size} bytes)")
        
        # Create analyzed subfolder
        analyzed_dir = os.path.join(analysis_dir, 'analyzed')
        os.makedirs(analyzed_dir, exist_ok=True)
        print(f"[OK] Analysis directory created: {analyzed_dir}")
        
        result = {
            'success': True,
            'analysis_id': f'analysis_{timestamp}',
            'csv_path': csv_path,
            'analysis_dir': analysis_dir
        }
        
        print(f"[OK] Upload successful!")
        print(f"Response: {result}")
        print("="*60 + "\n")
        
        return jsonify(result)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"[CRITICAL] ERROR in upload_for_analysis:")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return jsonify({'error': f'حدث خطأ: {str(e)}'}), 500

@app.route('/api/analyze/process', methods=['POST'])
@login_required
def process_analysis():
    """Process uploaded tracks.csv and generate analysis"""
    try:
        data = request.json
        analysis_id = data.get('analysis_id')
        expected_directions = data.get('expected_directions', 'all')
        
        print(f"\n{'='*60}")
        print(f"[ANALYSIS] PROCESSING")
        print(f"Analysis ID: {analysis_id}")
        print(f"Expected Directions: {expected_directions}")
        print(f"{'='*60}\n")
        
        if not analysis_id:
            return jsonify({'error': 'Analysis ID is required'}), 400
        
        analysis_dir = os.path.join(app.config['OUTPUT_FOLDER'], analysis_id)
        csv_path = os.path.join(analysis_dir, 'tracks.csv')
        
        if not os.path.exists(csv_path):
            return jsonify({'error': 'CSV file not found'}), 404
        
        # Read and analyze CSV
        import pandas as pd
        df = pd.DataFrame(pd.read_csv(csv_path))
        
        if df.empty:
            return jsonify({'error': 'CSV file is empty'}), 400
        
        print(f"[INFO] Original data: {len(df)} records, {df['track_id'].nunique()} tracks")
        
        # Apply powerful filtering
        df_filtered, filter_stats = filter_track_data(df)
        
        if df_filtered.empty:
            return jsonify({'error': 'لا توجد مسارات صالحة بعد الفلترة'}), 400
        
        # Use filtered data for analysis
        df = df_filtered
        
        # Normalize vehicle_type: fill missing values with 'unclassified'
        if 'vehicle_type' in df.columns:
            df['vehicle_type'] = df['vehicle_type'].fillna('unclassified').astype(str)
        else:
            # If the column doesn't exist, create it and mark all as unclassified
            df['vehicle_type'] = 'unclassified'
        
        # Basic analysis
        analysis_results = {
            'total_tracks': int(df['track_id'].nunique()) if 'track_id' in df.columns else 0,
            'total_detections': len(df),
            'vehicle_types': {},
            'unclassified_tracks': 0,
            'frame_range': {},
            'track_statistics': {},
            'filter_stats': filter_stats  # إضافة إحصائيات الفلترة
        }
        
        # Vehicle type distribution (includes 'unclassified' where applicable)
        vehicle_counts = df.groupby('vehicle_type')['track_id'].nunique().to_dict()
        analysis_results['vehicle_types'] = {str(k): int(v) for k, v in vehicle_counts.items()}
        # Record unclassified tracks explicitly for clarity
        analysis_results['unclassified_tracks'] = int(analysis_results['vehicle_types'].get('unclassified', 0))
        
        # Frame range
        if 'frame_idx' in df.columns:
            analysis_results['frame_range'] = {
                'min': int(df['frame_idx'].min()),
                'max': int(df['frame_idx'].max()),
                'total_frames': int(df['frame_idx'].max() - df['frame_idx'].min() + 1)
            }
        
        # Track statistics
        if 'track_id' in df.columns:
            track_lengths = df.groupby('track_id').size()
            analysis_results['track_statistics'] = {
                'avg_length': float(track_lengths.mean()),
                'min_length': int(track_lengths.min()),
                'max_length': int(track_lengths.max()),
                'median_length': float(track_lengths.median())
            }
        
        # Save analysis results
        analyzed_dir = os.path.join(analysis_dir, 'analyzed')
        os.makedirs(analyzed_dir, exist_ok=True)
        
        # Save totals
        totals_path = os.path.join(analyzed_dir, 'totals.csv')
        totals_data = []
        for vtype, count in analysis_results['vehicle_types'].items():
            totals_data.append({'vehicle_type': vtype, 'count': count})
        pd.DataFrame(totals_data).to_csv(totals_path, index=False)
        
        # Track statistics
        track_stats = []
        track_directions = []  # For detailed direction analysis
        
        # Statistics by direction and vehicle type
        direction_vehicle_stats = {}  # {direction: {vehicle_type: count}}
        
        # Check if direction filtering is enabled
        filter_directions = expected_directions != 'all'
        matched_tracks = 0
        unmatched_tracks = 0
        
        print(f"\n{'='*60}")
        print(f"Direction Filtering Settings:")
        print(f"  Expected Directions: {expected_directions}")
        print(f"  Filter Enabled: {filter_directions}")
        print(f"{'='*60}\n")
        
        for tid in df['track_id'].unique():
            track_df = df[df['track_id'] == tid]
            
            # Basic stats
            track_stat = {
                'track_id': int(tid),
                'vehicle_type': track_df['vehicle_type'].iloc[0] if 'vehicle_type' in track_df.columns else 'unclassified',
                'length': len(track_df),
                'first_frame': int(track_df['frame_idx'].min()) if 'frame_idx' in track_df.columns else 0,
                'last_frame': int(track_df['frame_idx'].max()) if 'frame_idx' in track_df.columns else 0
            }
            
            # Direction analysis
            direction_info = analyze_track_direction(track_df)
            track_stat['overall_direction'] = direction_info['overall_direction']
            track_stat['dominant_direction'] = direction_info['dominant_direction']
            track_stat['total_distance'] = direction_info['total_distance']
            track_stat['direction_changes_count'] = len(direction_info['direction_changes'])
            
            # Advanced statistics (similar to testApp2)
            # Calculate path length in pixels
            if 'x' in track_df.columns and 'y' in track_df.columns:
                track_df_sorted = track_df.sort_values('frame_idx')
                x_coords = track_df_sorted['x'].values
                y_coords = track_df_sorted['y'].values
                
                # Path length (sum of distances between consecutive points)
                path_length_px = 0.0
                for i in range(1, len(x_coords)):
                    dx = x_coords[i] - x_coords[i-1]
                    dy = y_coords[i] - y_coords[i-1]
                    path_length_px += np.sqrt(dx**2 + dy**2)
                
                track_stat['path_length_px'] = float(path_length_px)
            else:
                track_stat['path_length_px'] = 0.0
            
            # Duration in seconds
            if 'timestamp' in track_df.columns:
                duration_sec = float(track_df['timestamp'].max() - track_df['timestamp'].min())
            elif 'frame_idx' in track_df.columns:
                # Assume 30 FPS
                duration_sec = float((track_df['frame_idx'].max() - track_df['frame_idx'].min()) / 30.0)
            else:
                duration_sec = 0.0
            
            track_stat['duration_sec'] = duration_sec
            
            # Average speed
            if 'speed' in track_df.columns:
                avg_speed = float(track_df['speed'].mean())
            elif track_stat['path_length_px'] > 0 and duration_sec > 0:
                # Calculate speed as pixels per second
                avg_speed = track_stat['path_length_px'] / duration_sec
            else:
                avg_speed = 0.0
            
            track_stat['avg_speed'] = avg_speed
            track_stat['total_distance'] = direction_info['total_distance']
            track_stat['direction_changes_count'] = len(direction_info['direction_changes'])
            
            # Count vehicles by direction and type
            dominant_dir = direction_info['dominant_direction']
            vehicle_type = track_stat['vehicle_type']
            
            if dominant_dir not in direction_vehicle_stats:
                direction_vehicle_stats[dominant_dir] = {}
            
            if vehicle_type not in direction_vehicle_stats[dominant_dir]:
                direction_vehicle_stats[dominant_dir][vehicle_type] = 0
            
            direction_vehicle_stats[dominant_dir][vehicle_type] += 1
            
            # Check if direction matches expected
            is_matched = None
            if filter_directions:
                # Check dominant direction or overall direction
                dominant = direction_info['dominant_direction']
                overall = direction_info['overall_direction']
                
                # Debug: Print first 3 tracks in detail
                if int(tid) <= 3:
                    print(f"\n[TRACK DEBUG] Track {tid} Debug:")
                    print(f"  Dominant: '{dominant}'")
                    print(f"  Overall: '{overall}'")
                    print(f"  Expected: {expected_directions}")
                    print(f"  Type of expected: {type(expected_directions)}")
                    print(f"  Dominant in expected? {dominant in expected_directions}")
                    print(f"  Overall in expected? {overall in expected_directions}")
                
                is_matched = dominant in expected_directions or overall in expected_directions
                track_stat['is_matched'] = is_matched
                
                if int(tid) <= 3:
                    print(f"  [RESULT] is_matched = {is_matched}\n")
                
                if is_matched:
                    matched_tracks += 1
                else:
                    unmatched_tracks += 1
            
            track_stats.append(track_stat)
            
            # Detailed direction changes
            if direction_info['direction_changes']:
                for change in direction_info['direction_changes']:
                    direction_entry = {
                        'track_id': int(tid),
                        'vehicle_type': track_df['vehicle_type'].iloc[0] if 'vehicle_type' in track_df.columns else 'unclassified',
                        'frame': change['frame'],
                        'direction': change['direction'],
                        'x': change['position'][0],
                        'y': change['position'][1]
                    }
                    
                    # Add matching status if filtering is enabled
                    if filter_directions:
                        direction_entry['is_matched'] = is_matched
                    
                    track_directions.append(direction_entry)
        
        # Add direction matching statistics
        if filter_directions:
            analysis_results['direction_filtering'] = {
                'enabled': True,
                'expected_directions': expected_directions if isinstance(expected_directions, list) else [expected_directions],
                'matched_tracks': matched_tracks,
                'unmatched_tracks': unmatched_tracks,
                'match_percentage': (matched_tracks / (matched_tracks + unmatched_tracks) * 100) if (matched_tracks + unmatched_tracks) > 0 else 0
            }
            print(f"[OK] Direction filtering: {matched_tracks} matched, {unmatched_tracks} unmatched")
        else:
            analysis_results['direction_filtering'] = {'enabled': False}
        
        # Add direction-vehicle statistics
        analysis_results['direction_vehicle_stats'] = direction_vehicle_stats
        print(f"[OK] Direction-Vehicle stats: {direction_vehicle_stats}")
        
        # Calculate advanced statistics (similar to testApp2)
        track_stats_df = pd.DataFrame(track_stats)
        if not track_stats_df.empty:
            advanced_stats = {
                'avg_path_length': float(track_stats_df['path_length_px'].mean()) if 'path_length_px' in track_stats_df.columns else 0.0,
                'avg_duration': float(track_stats_df['duration_sec'].mean()) if 'duration_sec' in track_stats_df.columns else 0.0,
                'avg_speed': float(track_stats_df['avg_speed'].mean()) if 'avg_speed' in track_stats_df.columns else 0.0,
                'max_path_length': float(track_stats_df['path_length_px'].max()) if 'path_length_px' in track_stats_df.columns else 0.0,
                'min_path_length': float(track_stats_df['path_length_px'].min()) if 'path_length_px' in track_stats_df.columns else 0.0,
                'max_duration': float(track_stats_df['duration_sec'].max()) if 'duration_sec' in track_stats_df.columns else 0.0,
                'min_duration': float(track_stats_df['duration_sec'].min()) if 'duration_sec' in track_stats_df.columns else 0.0,
            }
            analysis_results['advanced_statistics'] = advanced_stats
            print(f"[OK] Advanced statistics calculated")
        
        # Save track statistics
        stats_path = os.path.join(analyzed_dir, 'track_statistics.csv')
        pd.DataFrame(track_stats).to_csv(stats_path, index=False)
        
        # Save detailed direction changes
        if track_directions:
            directions_path = os.path.join(analyzed_dir, 'direction_changes.csv')
            pd.DataFrame(track_directions).to_csv(directions_path, index=False)
            
            # Add to response files
            analysis_results['directions_available'] = True
        else:
            analysis_results['directions_available'] = False
        
        # Save meta
        meta_path = os.path.join(analyzed_dir, 'analysis_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'analysis_results': analysis_results,
            'files': {
                'totals': f'{analysis_id}/analyzed/totals.csv',
                'track_statistics': f'{analysis_id}/analyzed/track_statistics.csv',
                'direction_changes': f'{analysis_id}/analyzed/direction_changes.csv' if analysis_results.get('directions_available') else None,
                'meta': f'{analysis_id}/analyzed/analysis_meta.json'
            },
            'direction_data': track_directions if track_directions else []  # Send direction data directly
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/generate_charts', methods=['POST'])
def generate_analysis_charts():
    """Generate visualization charts (heatmap, trajectory overview, etc.)"""
    try:
        data = request.json
        analysis_id = data.get('analysis_id')
        
        print(f"\n{'='*60}")
        print(f"[ANALYSIS] GENERATING ANALYSIS CHARTS")
        print(f"Analysis ID: {analysis_id}")
        print(f"{'='*60}\n")
        
        if not analysis_id:
            return jsonify({'error': 'Analysis ID is required'}), 400
        
        analysis_dir = os.path.join(app.config['OUTPUT_FOLDER'], analysis_id)
        csv_path = os.path.join(analysis_dir, 'tracks.csv')
        analyzed_dir = os.path.join(analysis_dir, 'analyzed')
        
        if not os.path.exists(csv_path):
            return jsonify({'error': 'CSV file not found'}), 404
        
        # Create charts directory
        charts_dir = os.path.join(analyzed_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # Read CSV data
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return jsonify({'error': 'CSV file is empty'}), 400
        
        # Apply filtering
        df_filtered, _ = filter_track_data(df)
        
        if df_filtered.empty:
            return jsonify({'error': 'No valid tracks after filtering'}), 400
        
        df = df_filtered
        
        # Determine frame shape from data
        if 'x' in df.columns and 'y' in df.columns:
            max_x = int(df['x'].max() + 100)
            max_y = int(df['y'].max() + 100)
            frame_shape = (max_y, max_x, 3)
        else:
            frame_shape = (1080, 1920, 3)  # Default HD
        
        print(f"[INFO] Generating charts for {len(df)} detections, {df['track_id'].nunique()} tracks")
        print(f"Frame shape: {frame_shape}")
        
        generated_charts = {}
        
        # 1. Generate Heatmap
        print("[INFO] Generating heatmap...")
        heatmap_path = os.path.join(charts_dir, 'heatmap.png')
        result = generate_heatmap(df, frame_shape, heatmap_path, bins=120)
        if result:
            generated_charts['heatmap'] = f'{analysis_id}/analyzed/charts/heatmap.png'
        
        # 2. Generate Track Visualization
        print("[INFO] Generating track visualization...")
        tracks_path = os.path.join(charts_dir, 'all_tracks.png')
        result = generate_track_visualization(df, frame_shape, tracks_path)
        if result:
            generated_charts['all_tracks'] = f'{analysis_id}/analyzed/charts/all_tracks.png'
        
        # 3. Generate Speed Histogram
        if 'speed' in df.columns:
            print("[INFO] Generating speed histogram...")
            speed_path = os.path.join(charts_dir, 'speed_histogram.png')
            result = generate_speed_histogram(df, speed_path)
            if result:
                generated_charts['speed_histogram'] = f'{analysis_id}/analyzed/charts/speed_histogram.png'
        
        # 4. Generate Direction Chart (if available)
        # Collect direction statistics
        direction_stats = {}
        for tid in df['track_id'].unique():
            track_df = df[df['track_id'] == tid]
            direction_info = analyze_track_direction(track_df)
            dominant = direction_info.get('dominant_direction', 'غير محدد')
            direction_stats[dominant] = direction_stats.get(dominant, 0) + 1
        
        if direction_stats:
            print("[INFO] Generating direction chart...")
            direction_path = os.path.join(charts_dir, 'direction_distribution.png')
            result = generate_direction_chart(direction_stats, direction_path)
            if result:
                generated_charts['direction_chart'] = f'{analysis_id}/analyzed/charts/direction_distribution.png'
        
        # 5. Generate overview with trajectories (if we have a background image)
        # Try to find a frame image in the output directory
        possible_frames = [
            os.path.join(analysis_dir, 'first_frame.jpg'),
            os.path.join(analysis_dir, 'frame.jpg'),
            os.path.join(os.path.dirname(analysis_dir), 'first_frame.jpg')
        ]
        
        background_frame = None
        for frame_path in possible_frames:
            if os.path.exists(frame_path):
                background_frame = cv2.imread(frame_path)
                if background_frame is not None:
                    print(f"[INFO] Found background frame: {frame_path}")
                    break
        
        if background_frame is not None:
            print("[INFO] Generating overview with background...")
            overview_path = os.path.join(charts_dir, 'trajectory_overview.png')
            result = generate_overview(background_frame, df, overview_path, alpha=0.7)
            if result:
                generated_charts['overview'] = f'{analysis_id}/analyzed/charts/trajectory_overview.png'
        
        print(f"\n[OK] Generated {len(generated_charts)} charts")
        print(f"Charts: {list(generated_charts.keys())}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'charts': generated_charts,
            'message': f'Successfully generated {len(generated_charts)} charts'
        })
        
    except Exception as e:
        print(f"\n[ERROR] ERROR generating charts:")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration"""
    config_file = 'user_config.json'
    
    if request.method == 'GET':
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({})
    
    elif request.method == 'POST':
        config = request.json
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        return jsonify({'success': True})

# ================== Line Drawing & Counting Routes ==================
@app.route('/api/line_drawing/upload', methods=['POST'])
def upload_for_line_drawing():
    """Upload image and CSV for line drawing"""
    try:
        if 'image' not in request.files or 'csv' not in request.files:
            return jsonify({'error': 'كل من الصورة والـ CSV مطلوبان'}), 400
        
        image_file = request.files['image']
        csv_file = request.files['csv']
        
        # Create analysis folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_dir = os.path.join(app.config['OUTPUT_FOLDER'], f'line_drawing_{timestamp}')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save files
        image_path = os.path.join(analysis_dir, 'frame.jpg')
        csv_path = os.path.join(analysis_dir, 'tracks.csv')
        
        image_file.save(image_path)
        csv_file.save(csv_path)
        
        return jsonify({
            'success': True,
            'analysis_id': f'line_drawing_{timestamp}',
            'image_path': image_path,
            'csv_path': csv_path
        })
    except Exception as e:
        print(f"Error uploading files: {e}")
        return jsonify({'error': str(e)}), 500

def _first_hit_index(track_xy, line_list, tolerance: float = None):
    """Find first intersection of track with lines. Uses unified LINE_HIT_TOLERANCE when None."""
    if tolerance is None:
        tolerance = LINE_HIT_TOLERANCE

    if track_xy is None or track_xy.shape[0] < 2:
        return None

    try:
        from shapely.geometry import LineString, Point
        SHAPELY_AVAILABLE = True
    except ImportError:
        SHAPELY_AVAILABLE = False

    if SHAPELY_AVAILABLE:
        best = None
        Ltrack = LineString(track_xy.tolist())

        for L in line_list:
            if L is None or len(L) < 2:
                continue

            # Convert to numpy array if needed
            if not isinstance(L, np.ndarray):
                L = np.array(L)

            if L.shape[0] < 2:
                continue

            try:
                line_geo = LineString(L.tolist())
                inter = Ltrack.intersection(line_geo)

                if not inter.is_empty:
                    pts = []
                    if inter.geom_type == "Point":
                        pts = [(inter.x, inter.y)]
                    elif inter.geom_type == "MultiPoint":
                        pts = [(g.x, g.y) for g in inter.geoms]
                    elif inter.geom_type == "LineString":
                        coords = list(inter.coords)
                        if coords:
                            pts = [coords[0], coords[-1]]

                    for (ix, iy) in pts:
                        d = np.hypot(track_xy[:, 0] - ix, track_xy[:, 1] - iy)
                        idx = int(np.argmin(d))
                        if best is None or idx < best:
                            best = idx
            except Exception as e:
                print(f"Error checking intersection: {e}")
                continue

        # Check tolerance zone if no direct intersection
        if best is None:
            for L in line_list:
                if L is None or len(L) < 2:
                    continue

                if not isinstance(L, np.ndarray):
                    L = np.array(L)

                line_min_x = L[:, 0].min() - tolerance
                line_max_x = L[:, 0].max() + tolerance
                line_min_y = L[:, 1].min() - tolerance
                line_max_y = L[:, 1].max() + tolerance

                try:
                    line_geo = LineString(L.tolist())
                    for idx, point in enumerate(track_xy):
                        if not (line_min_x <= point[0] <= line_max_x and
                                line_min_y <= point[1] <= line_max_y):
                            continue

                        point_geo = Point(point[0], point[1])
                        distance = point_geo.distance(line_geo)
                        if distance <= tolerance:
                            if best is None or idx < best:
                                best = idx
                except Exception:
                    continue
        return best

    # Fallback: simple distance check
    best_idx = None
    for L in line_list:
        if L is None or len(L) < 2:
            continue

        if not isinstance(L, np.ndarray):
            L = np.array(L)

        for idx, pt in enumerate(track_xy):
            for j in range(len(L) - 1):
                A, B = L[j], L[j + 1]
                AB = B - A
                AP = pt - A
                ab_len_sq = np.dot(AB, AB)
                if ab_len_sq < 1e-9:
                    dist = np.linalg.norm(AP)
                else:
                    t = np.clip(np.dot(AP, AB) / ab_len_sq, 0, 1)
                    closest = A + t * AB
                    dist = np.linalg.norm(pt - closest)

                if dist <= tolerance:
                    if best_idx is None or idx < best_idx:
                        best_idx = idx

    return best_idx

def detect_cross_ordered(track_xy, lines_in, lines_out):
    """Detect if track crosses IN then OUT lines (ordered)"""
    in_leg = out_leg = None
    in_idx = out_idx = None

    for i, Llist in enumerate(lines_in):
        if not Llist:
            continue
        idx = _first_hit_index(track_xy, Llist)
        if idx is not None and (in_idx is None or idx < in_idx):
            in_idx, in_leg = idx, i + 1

    for j, Llist in enumerate(lines_out):
        if not Llist:
            continue
        idx = _first_hit_index(track_xy, Llist)
        if idx is None:
            continue
        if in_idx is None:
            if out_idx is None or idx < out_idx:
                out_idx, out_leg = idx, j + 1
        else:
            if idx > in_idx and (out_idx is None or idx < out_idx):
                out_idx, out_leg = idx, j + 1

    if in_leg is not None and out_leg is not None and (out_idx is None or in_idx is None or out_idx <= in_idx):
        return None, None
    return in_leg, out_leg


try:
    # prefer importing the implementation from a lightweight helper module
    from lib.adaptive_line import adapt_line_to_tracks
except Exception:
    # fallback stub (no-op) if module isn't available
    def adapt_line_to_tracks(*args, **kwargs):
        return None


@app.route('/api/line_drawing/calculate', methods=['POST'])
def calculate_crossings():
    """Calculate vehicle crossings based on drawn lines"""
    try:
        data = request.json
        analysis_id = data.get('analysis_id')
        request_id = data.get('request_id')  # NEW: Get request_id
        lines_in_raw = data.get('lines_in', [])
        lines_out_raw = data.get('lines_out', [])
        leg_names_in = data.get('leg_names_in', [])
        leg_names_out = data.get('leg_names_out', [])
        canvas_width = data.get('canvas_width')
        canvas_height = data.get('canvas_height')
        image_width = data.get('image_width')
        image_height = data.get('image_height')
        
        # Scale factor from canvas to original image
        scale_x = image_width / canvas_width
        scale_y = image_height / canvas_height
        
        # Convert lines to numpy arrays and scale
        lines_in = []
        lines_out = []
        
        for leg_lines in lines_in_raw:
            leg_converted = []
            for line in leg_lines:
                pts = np.array([[p['x'] * scale_x, p['y'] * scale_y] for p in line])
                leg_converted.append(pts)
            lines_in.append(leg_converted)
        
        for leg_lines in lines_out_raw:
            leg_converted = []
            for line in leg_lines:
                pts = np.array([[p['x'] * scale_x, p['y'] * scale_y] for p in line])
                leg_converted.append(pts)
            lines_out.append(leg_converted)

        # ----------------------
        # Smart Adaptive Lines
        # Convert simple straight (2-point) lines into adaptive polylines
        # that try to touch nearby tracks while preserving endpoints and
        # keeping the overall length approximately the same.
        # ----------------------
        def _apply_adapt(lines_list):
            for li, leg in enumerate(lines_list):
                new_leg = []
                for line in leg:
                    try:
                        if line is None or len(line) < 2:
                            new_leg.append(line)
                            continue
                        # Apply only to straight lines of 2 points
                        if len(line) == 2:
                            adapted = adapt_line_to_tracks(line, df)
                            new_leg.append(adapted if adapted is not None else line)
                        else:
                            new_leg.append(line)
                    except Exception:
                        new_leg.append(line)
                lines_list[li] = new_leg

        _apply_adapt(lines_in)
        _apply_adapt(lines_out)
        
        # Load tracks CSV
        # Determine CSV path based on request_id or analysis_id
        if request_id:
            # Get CSV path from request
            db = get_db()
            request_data = db.execute('SELECT csv_path FROM analysis_requests WHERE id = ?', (request_id,)).fetchone()
            db.close()
            
            if not request_data or not request_data['csv_path']:
                return jsonify({'error': 'Request CSV not found'}), 404
            
            csv_path = request_data['csv_path']
            analysis_dir = os.path.dirname(csv_path)
        elif analysis_id:
            analysis_dir = os.path.join(app.config['OUTPUT_FOLDER'], analysis_id)
            csv_path = os.path.join(analysis_dir, 'tracks.csv')
        else:
            return jsonify({'error': 'Either request_id or analysis_id is required'}), 400
        
        if not os.path.exists(csv_path):
            return jsonify({'error': f'CSV file not found: {csv_path}'}), 404
        
        df = pd.read_csv(csv_path)
        df = ensure_xy_columns(df)
        
        # Calculate crossings
        crossed = []
        for tid, g in df.groupby("track_id"):
            if len(g) < 2:
                continue
            
            xy = g[["px", "py"]].to_numpy(float)
            in_leg, out_leg = detect_cross_ordered(xy, lines_in, lines_out)
            
            if in_leg and out_leg:
                vehicle_type = g["vehicle_type"].iloc[0] if "vehicle_type" in g.columns else "unknown"
                crossed.append([int(tid), vehicle_type, in_leg, out_leg])
        
        # Create results
        crossed_df = pd.DataFrame(crossed, columns=["track_id", "vehicle_type", "in_leg", "out_leg"])
        
        # Determine results directory
        if request_id:
            # Use request folder
            results_dir = os.path.join(analysis_dir, 'results', 'line_drawing')
        else:
            # Use analysis folder
            results_dir = os.path.join(analysis_dir, 'results')
        
        os.makedirs(results_dir, exist_ok=True)
        
        crossed_path = os.path.join(results_dir, 'crossed_pairs.csv')
        crossed_df.to_csv(crossed_path, index=False)
        
        # Calculate totals
        totals = crossed_df["vehicle_type"].value_counts().reset_index()
        totals.columns = ["vehicle_type", "count"]
        totals_path = os.path.join(results_dir, 'totals.csv')
        totals.to_csv(totals_path, index=False)
        
        # Calculate OD matrix
        vehicle_types = sorted(crossed_df["vehicle_type"].unique())
        od_rows = []
        od_summary = {}
        
        for i in range(1, len(leg_names_in) + 1):
            din = crossed_df[crossed_df["in_leg"] == i]
            for j in range(1, len(leg_names_out) + 1):
                pair = din[din["out_leg"] == j]
                row = {
                    "in_leg": i, "in_name": leg_names_in[i - 1],
                    "out_leg": j, "out_name": leg_names_out[j - 1],
                }
                total_ij = 0
                for vt in vehicle_types:
                    c = int((pair["vehicle_type"] == vt).sum())
                    row[vt] = c
                    total_ij += c
                row["total"] = total_ij
                od_rows.append(row)
                
                if total_ij > 0:
                    od_summary[f"{leg_names_in[i-1]} → {leg_names_out[j-1]}"] = total_ij
        
        od_df = pd.DataFrame(od_rows)

        # od_matrix.csv: write flat table (rows per IN→OUT pair, with per-type counts)
        od_path = os.path.join(results_dir, 'od_matrix.csv')
        od_df.to_csv(od_path, index=False)

        # od_matrix_by_type.csv: write LONG format suitable for per-type tabs
        # Columns: in_leg,in_name,out_leg,out_name,vehicle_type,count
        od_by_type_path = os.path.join(results_dir, 'od_matrix_by_type.csv')
        long_rows = []
        for r in od_rows:
            for vt in vehicle_types:
                long_rows.append({
                    'in_leg': r['in_leg'],
                    'in_name': r['in_name'],
                    'out_leg': r['out_leg'],
                    'out_name': r['out_name'],
                    'vehicle_type': vt,
                    'count': int(r.get(vt, 0))
                })
        pd.DataFrame(long_rows).to_csv(od_by_type_path, index=False)
        
        # Create Excel with separate sheets
        excel_path = os.path.join(results_dir, 'od_matrices_by_vehicle.xlsx')
        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                for vt in vehicle_types:
                    mat = pd.DataFrame(0, index=leg_names_in, columns=leg_names_out, dtype=int)
                    sub = crossed_df[crossed_df["vehicle_type"] == vt]
                    for i, in_name in enumerate(leg_names_in, start=1):
                        for j, out_name in enumerate(leg_names_out, start=1):
                            cnt = int(((sub["in_leg"] == i) & (sub["out_leg"] == j)).sum())
                            mat.loc[in_name, out_name] = cnt
                    mat.index.name = "IN_leg"
                    mat.columns.name = "OUT_leg"
                    sheet_name = vt.replace(" ", "_")[:31] or "type"
                    mat.to_excel(writer, sheet_name=sheet_name)
        except Exception as e:
            print(f"Warning: Could not create Excel file: {e}")
            excel_path = None
        
        # Generate debug overlay
        try:
            frame_path = os.path.join(analysis_dir, 'frame.jpg')
            frame = cv2.imread(frame_path)
            
            if frame is not None:
                overlay = frame.copy()
                
                # Draw lines
                for i, leg_lines in enumerate(lines_in):
                    for line in leg_lines:
                        pts = line.astype(int)
                        for j in range(len(pts) - 1):
                            cv2.line(overlay, tuple(pts[j]), tuple(pts[j + 1]), (0, 255, 0), 3)
                        cv2.putText(overlay, f"{leg_names_in[i]}", tuple(pts[0]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                for i, leg_lines in enumerate(lines_out):
                    for line in leg_lines:
                        pts = line.astype(int)
                        for j in range(len(pts) - 1):
                            cv2.line(overlay, tuple(pts[j]), tuple(pts[j + 1]), (0, 0, 255), 3)
                        cv2.putText(overlay, f"{leg_names_out[i]}", tuple(pts[0]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Draw crossed tracks
                for _, row in crossed_df.iterrows():
                    tid = int(row["track_id"])
                    g = df[df["track_id"] == tid]
                    pts = g[["px", "py"]].to_numpy(int)
                    for j in range(len(pts) - 1):
                        cv2.line(overlay, tuple(pts[j]), tuple(pts[j + 1]), (0, 255, 255), 2)
                
                debug_path = os.path.join(results_dir, 'debug_overlay.jpg')
                cv2.imwrite(debug_path, overlay)
            else:
                debug_path = None
        except Exception as e:
            print(f"Warning: Could not create debug overlay: {e}")
            debug_path = None
        
        # Expose download file paths relative to app.config['OUTPUT_FOLDER'] so
        # frontend `/api/download/<path>` resolves correctly. Use the actual
        # `results_dir` location (handles request-backed folders which write to
        # `results/line_drawing`).
        try:
            results_subpath = os.path.relpath(results_dir, app.config['OUTPUT_FOLDER']).replace('\\', '/')
            if results_subpath.startswith('..'):
                # Fallback: if relpath would escape, use the analysis_id-based path
                results_subpath = f"{analysis_id}/results" if analysis_id else os.path.basename(results_dir)
        except Exception:
            results_subpath = f"{analysis_id}/results" if analysis_id else os.path.basename(results_dir)

        # If this calculation was for a saved request, mark the request as completed
        request_completed = False
        if request_id:
            try:
                db = get_db()
                results_base = os.path.join(analysis_dir, 'results')
                db.execute('''
                    UPDATE analysis_requests
                    SET status = ?, results_path = ?, completed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', ('completed', results_base, request_id))
                db.commit()
                db.close()
                request_completed = True
                print(f"[OK] Marked request #{request_id} as completed (line_drawing)")
            except Exception as e:
                print(f"Warning: could not update request status for {request_id}: {e}")

        return jsonify({
            'success': True,
            'request_completed': request_completed,
            'results': {
                'total_crossed': len(crossed_df),
                'totals': totals.to_dict('records'),
                'od_summary': od_summary
            },
            'files': {
                'crossed': f'{results_subpath}/crossed_pairs.csv',
                'totals': f'{results_subpath}/totals.csv',
                'od_matrix': f'{results_subpath}/od_matrix.csv',
                'od_matrix_by_type': f'{results_subpath}/od_matrix_by_type.csv',
                'od_excel': f'{results_subpath}/od_matrices_by_vehicle.xlsx' if excel_path else None,
                'debug_overlay': f'{results_subpath}/debug_overlay.jpg' if debug_path else None
            }
        })
        
    except Exception as e:
        print(f"Error calculating crossings: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/line_drawing/adapt_line', methods=['POST'])
def adapt_line_endpoint():
    """API endpoint to adapt a single straight line into an adaptive polyline.

    Expects JSON payload with:
    - line: [{x, y}, ...] in canvas coordinates (typically 2 points)
    - canvas_width, canvas_height, image_width, image_height (to scale)
    - analysis_id or request_id to locate the CSV

    Returns: {success: True, adapted: True/False, line: [{x,y}, ...]} where returned points are in canvas coords.
    """
    try:
        data = request.json
        line = data.get('line')
        analysis_id = data.get('analysis_id')
        request_id = data.get('request_id')
        canvas_width = float(data.get('canvas_width') or 1)
        canvas_height = float(data.get('canvas_height') or 1)
        image_width = float(data.get('image_width') or canvas_width)
        image_height = float(data.get('image_height') or canvas_height)

        if not line or len(line) < 2:
            return jsonify({'error': 'Line must contain at least two points'}), 400

        # find csv path like in calculate_crossings
        if request_id:
            db = get_db()
            request_data = db.execute('SELECT csv_path FROM analysis_requests WHERE id = ?', (request_id,)).fetchone()
            db.close()
            if not request_data or not request_data['csv_path']:
                return jsonify({'error': 'Request CSV not found'}), 404
            csv_path = request_data['csv_path']
        elif analysis_id:
            analysis_dir = os.path.join(app.config['OUTPUT_FOLDER'], analysis_id)
            csv_path = os.path.join(analysis_dir, 'tracks.csv')
        else:
            return jsonify({'error': 'Either request_id or analysis_id is required'}), 400

        if not os.path.exists(csv_path):
            return jsonify({'error': f'CSV file not found: {csv_path}'}), 404

        df = pd.read_csv(csv_path)
        df = ensure_xy_columns(df)

        # scale to image coords
        scale_x = image_width / canvas_width
        scale_y = image_height / canvas_height
        pts_img = np.array([[p['x'] * scale_x, p['y'] * scale_y] for p in line])

        adapted = adapt_line_to_tracks(pts_img, df)
        if adapted is None:
            return jsonify({'success': True, 'adapted': False})

        # scale back to canvas coords
        adapted_canvas = [{'x': float(p[0] / scale_x), 'y': float(p[1] / scale_y)} for p in adapted]
        return jsonify({'success': True, 'adapted': True, 'line': adapted_canvas})

    except Exception as e:
        print('Error in adapt_line_endpoint:', e)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/speed_calculation', methods=['GET'])
def speed_calculation():
    """Display speed calculation page"""
    leg_names = request.args.get('leg_names', '[]')
    analysis_id = request.args.get('analysis_id', '')
    request_id = request.args.get('request_id', '')
    
    try:
        leg_names = json.loads(leg_names)
    except:
        leg_names = []
    
    return render_template('speed_calculation.html', 
                         leg_names=leg_names,
                         analysis_id=analysis_id,
                         request_id=request_id)

# ===================== SPEED CALCULATION HELPERS =====================

def get_crossing_timestamp(track_df: pd.DataFrame, line_list) -> float:
    """Get timestamp when track crosses specific lines."""
    try:
        track_xy = track_df[["px", "py"]].to_numpy(float)
        idx = _first_hit_index(track_xy, line_list)
        if idx is not None and idx < len(track_df):
            # Try different timestamp column names
            if "timestamp" in track_df.columns:
                return float(track_df.iloc[idx]["timestamp"])
            elif "frame_idx" in track_df.columns:
                # If no timestamp, use frame index
                return float(track_df.iloc[idx]["frame_idx"]) / 30.0  # Assume 30 FPS
        return None
    except Exception as e:
        print(f"Error getting crossing timestamp: {e}")
        return None

def _first_hit_index(xy: np.ndarray, line_list) -> int:
    """Find first index where xy hits any line in line_list."""
    for line in line_list:
        for i, point in enumerate(xy):
            if _point_on_line(point, line):
                return i
    return None

def _point_on_line(pt: np.ndarray, line: np.ndarray, tolerance: float = None) -> bool:
    """Check if point is on/near line. Uses unified LINE_HIT_TOLERANCE when tolerance is None."""
    if tolerance is None:
        tolerance = LINE_HIT_TOLERANCE

    if line is None or len(line) < 2:
        return False

    min_dist = float('inf')
    for j in range(len(line) - 1):
        p1 = line[j]
        p2 = line[j + 1]
        dist = _point_to_segment_distance(pt, p1, p2)
        min_dist = min(min_dist, dist)
        if min_dist < tolerance:
            return True

    return min_dist < tolerance

def _point_to_segment_distance(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate distance from point to line segment."""
    x, y = pt
    x1, y1 = p1
    x2, y2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return np.sqrt((x - x1)**2 + (y - y1)**2)
    
    t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return np.sqrt((x - proj_x)**2 + (y - proj_y)**2)

@app.route('/api/line_drawing/speed_analysis', methods=['POST'])
def speed_analysis():
    """Calculate speed for vehicles based on distance matrix"""
    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id')
        request_id = data.get('request_id')
        distance_matrix_raw = data.get('distance_matrix')  # {"{i},{j}": distance_m}
        
        if not analysis_id and not request_id:
            return jsonify({'error': 'Missing analysis_id or request_id'}), 400
        
        if not distance_matrix_raw:
            return jsonify({'error': 'Missing distance matrix'}), 400
        
        # Convert string keys to tuples
        distance_matrix = {}
        for key, value in distance_matrix_raw.items():
            try:
                i, j = key.split(',')
                distance_matrix[(int(i), int(j))] = float(value)
            except:
                continue
        
        # Load the CSV file
        if request_id:
            csv_path = None
            db = get_db()
            cursor = db.cursor()
            cursor.execute("SELECT csv_path FROM analysis_requests WHERE id = ?", (request_id,))
            row = cursor.fetchone()
            if row:
                csv_path = row['csv_path']
            db.close()
        else:
            csv_path = os.path.join(app.config['OUTPUT_FOLDER'], analysis_id, 'tracks.csv')
        
        if not csv_path or not os.path.exists(csv_path):
            return jsonify({'error': 'CSV file not found: ' + str(csv_path)}), 404
        
        # Load crossed pairs
        crossed_path = os.path.join(app.config['OUTPUT_FOLDER'], analysis_id, 'results', 'crossed_pairs.csv')
        if not os.path.exists(crossed_path):
            return jsonify({'error': 'No crossing data found at ' + crossed_path}), 404
        
        crossed_df = pd.read_csv(crossed_path)
        df = pd.read_csv(csv_path)
        
        # Initialize speed column if it doesn't exist
        if 'speed_kmh' not in crossed_df.columns:
            crossed_df['speed_kmh'] = None
        
        # Calculate speed for each vehicle
        for idx, row in crossed_df.iterrows():
            tid = int(row['track_id'])
            in_leg = int(row['in_leg'])
            out_leg = int(row['out_leg'])
            
            # Get distance between legs (including same leg)
            key = (in_leg, out_leg)
            if key not in distance_matrix:
                continue
            
            path_distance = distance_matrix[key]
            
            # Skip if distance is 0 (makes no sense for speed calculation)
            if path_distance == 0:
                continue
            
            # Get track data
            track_data = df[df['track_id'] == tid].sort_values('frame_idx') if 'track_id' in df.columns else df[df['track_id'] == tid].sort_values('frame')
            
            if track_data.empty or len(track_data) < 2:
                continue
            
            # Estimate time using frame difference
            frame_col = 'frame_idx' if 'frame_idx' in track_data.columns else 'frame'
            frame_diff = track_data[frame_col].iloc[-1] - track_data[frame_col].iloc[0]
            
            if frame_diff > 0:
                # Assume 30 FPS
                time_diff = frame_diff / 30.0
                
                if time_diff > 0.1:
                    speed_ms = path_distance / time_diff
                    speed_kmh = speed_ms * 3.6
                    crossed_df.at[idx, 'speed_kmh'] = round(speed_kmh, 2)
        
        # Save updated data
        results_dir = os.path.join(app.config['OUTPUT_FOLDER'], analysis_id, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        crossed_df.to_csv(crossed_path, index=False)
        
        # Save distance matrix
        matrix_data = []
        for (from_leg, to_leg), distance in distance_matrix.items():
            matrix_data.append({
                'from_leg': from_leg,
                'to_leg': to_leg,
                'distance_m': distance
            })
        pd.DataFrame(matrix_data).to_csv(os.path.join(results_dir, 'distance_matrix.csv'), index=False)
        
        # Calculate speed statistics
        speed_df = crossed_df[crossed_df['speed_kmh'].notna()].copy()
        speed_stats = {
            'total_vehicles_with_speed': 0,
            'avg_speed_kmh': 0,
            'max_speed_kmh': 0,
            'min_speed_kmh': 0
        }
        
        if not speed_df.empty and len(speed_df) > 0:
            speed_stats = {
                'total_vehicles_with_speed': int(len(speed_df)),
                'avg_speed_kmh': float(speed_df['speed_kmh'].mean()),
                'max_speed_kmh': float(speed_df['speed_kmh'].max()),
                'min_speed_kmh': float(speed_df['speed_kmh'].min()),
            }
            
            speed_df.to_csv(os.path.join(results_dir, 'vehicle_speeds.csv'), index=False)
        
        with open(os.path.join(results_dir, 'speed_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(speed_stats, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'speed_stats': speed_stats,
            'speed_file': f'{analysis_id}/results/vehicle_speeds.csv'
        })
        
    except Exception as e:
        print(f"Error in speed analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Advanced Vehicle Tracking System")
    print("High-precision tracking with stable IDs")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
