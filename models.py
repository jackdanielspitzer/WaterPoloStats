
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    stats_private = db.Column(db.Boolean, default=False)
    profile_image = db.Column(db.String(200), default='waterpoloball.jpg')
    followed_teams = db.Column(db.Text, default='[]')  # Stores JSON string of team slugs
    password = db.Column(db.String(200), nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    high_school = db.Column(db.String(200), nullable=False)
    account_type = db.Column(db.String(20), nullable=False)
    role = db.Column(db.String(20))
    phone = db.Column(db.String(20))
    managed_team = db.Column(db.String(100))
    email_confirmed = db.Column(db.Boolean, default=False)
    confirmation_token = db.Column(db.String(100), unique=True, nullable=True)
    reset_token = db.Column(db.String(100), unique=True, nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
