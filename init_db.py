
from main import app, db
from models import User
import os

with app.app_context():
    # Create database folder if it doesn't exist
    if not os.path.exists('instance'):
        os.makedirs('instance')
        
    # Create or recreate the database
    db.drop_all()  # Drop all existing tables
    db.create_all()  # Create all tables
    print("Database initialized successfully!")
