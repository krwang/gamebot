from visitor_tracker import VisitorTracker
from flask import request

def track_visitor(app):
    """Middleware to track visitors"""
    tracker = VisitorTracker()
    
    @app.before_request
    def before_request():
        # Only track visits to the homepage
        if request.path == '/':
            tracker.track_visit()
    
    return app 