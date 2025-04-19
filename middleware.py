from visitor_tracker import VisitorTracker
from flask import request

def track_visitor(app):
    """Middleware to track visitors"""
    tracker = VisitorTracker()
    
    @app.before_request
    def before_request():
        # Don't track visits to the stats page itself
        if not request.path.startswith('/app-stats'):
            tracker.track_visit()
    
    return app 