
from main import app, db
from models import User
import os
from werkzeug.security import generate_password_hash
from datetime import datetime

with app.app_context():
    # Create database folder if it doesn't exist
    if not os.path.exists('instance'):
        os.makedirs('instance')
        
    # Create or recreate the database
    db.drop_all()  # Drop all existing tables
    db.create_all()  # Create all tables
    
    # Create admin users
    admin_users = [
        {
            'email': 'jackdanielspitzer@gmail.com',
            'password': generate_password_hash('admin123'),
            'first_name': 'Jack',
            'last_name': 'Spitzer',
            'date_of_birth': datetime.strptime('2000-01-01', '%Y-%m-%d'),
            'high_school': 'Admin',
            'account_type': 'team_manager',
            'is_admin': True,
            'email_confirmed': True
        },
        {
            'email': 'mokshejain@gmail.com',
            'password': generate_password_hash('admin123'),
            'first_name': 'Mokshe',
            'last_name': 'Jain',
            'date_of_birth': datetime.strptime('2000-01-01', '%Y-%m-%d'),
            'high_school': 'Admin',
            'account_type': 'team_manager',
            'is_admin': True,
            'email_confirmed': True
        }
    ]
    
    for admin_data in admin_users:
        admin = User(**admin_data)
        db.session.add(admin)
    
    db.session.commit()
    print("Database initialized successfully with admin users!")
