from flask import Blueprint, render_template
from visitor_tracker import VisitorTracker

stats_bp = Blueprint('stats', __name__)

@stats_bp.route('/app-stats')
def show_stats():
    tracker = VisitorTracker()
    stats = tracker.get_stats()
    return render_template('stats.html', stats=stats) 