from datetime import datetime
import sqlite3
from flask import request, make_response
import os
from pathlib import Path
import uuid

class VisitorTracker:
    def __init__(self):
        self.db_path = Path("visitors.db")
        self._init_db()
        
    def _init_db(self):
        """Initialize the database and create tables if they don't exist"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create visitors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visitors (
                visitor_id TEXT PRIMARY KEY,
                ip TEXT,
                user_agent TEXT,
                last_visit TIMESTAMP,
                visit_count INTEGER DEFAULT 1
            )
        ''')
        
        # Create visits table for detailed visit history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visitor_id TEXT,
                path TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (visitor_id) REFERENCES visitors(visitor_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _get_visitor_id(self):
        """Get or create a unique visitor ID"""
        # Try to get visitor_id from cookie
        visitor_id = request.cookies.get('visitor_id')
        
        # If no cookie exists, create a new visitor_id
        if not visitor_id:
            visitor_id = str(uuid.uuid4())
            
        return visitor_id
        
    def track_visit(self):
        """Track a visitor and store their information"""
        visitor_id = self._get_visitor_id()
        ip = request.remote_addr
        user_agent = request.user_agent.string
        path = request.path
        timestamp = datetime.utcnow()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Update or insert visitor
        cursor.execute('''
            INSERT INTO visitors (visitor_id, ip, user_agent, last_visit, visit_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(visitor_id) DO UPDATE SET
                ip = ?,
                user_agent = ?,
                last_visit = ?,
                visit_count = visit_count + 1
        ''', (visitor_id, ip, user_agent, timestamp, ip, user_agent, timestamp))
        
        # Record the visit
        cursor.execute('''
            INSERT INTO visits (visitor_id, path, timestamp)
            VALUES (?, ?, ?)
        ''', (visitor_id, path, timestamp))
        
        conn.commit()
        conn.close()
        
        # Set cookie if it doesn't exist
        if not request.cookies.get('visitor_id'):
            response = make_response()
            response.set_cookie('visitor_id', visitor_id, max_age=365*24*60*60)  # 1 year
            return response
            
    def get_stats(self):
        """Get visitor statistics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get total unique visitors
        cursor.execute('SELECT COUNT(*) FROM visitors')
        total_visitors = cursor.fetchone()[0]
        
        # Get total visits
        cursor.execute('SELECT SUM(visit_count) FROM visitors')
        total_visits = cursor.fetchone()[0] or 0
        
        # Get recent visitors with more details
        cursor.execute('''
            SELECT visitor_id, ip, user_agent, last_visit, visit_count
            FROM visitors
            ORDER BY last_visit DESC
            LIMIT 10
        ''')
        recent_visitors = [
            {
                'visitor_id': row[0],
                'ip': row[1],
                'user_agent': row[2],
                'last_visit': row[3],
                'visit_count': row[4]
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return {
            'total_visitors': total_visitors,
            'total_visits': total_visits,
            'recent_visitors': recent_visitors
        } 