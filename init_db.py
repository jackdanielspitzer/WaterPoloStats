
from main import app, db
import os

# Remove existing database
if os.path.exists('instance/users.db'):
    os.remove('instance/users.db')

with app.app_context():
    db.create_all()
    print("Database recreated successfully!")
