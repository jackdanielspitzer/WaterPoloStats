import os
import json
import spacy
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from models import db, User
from word2number import w2n

def check_email_config():
    required = ['MAIL_USERNAME', 'MAIL_PASSWORD', 'MAIL_DEFAULT_SENDER']
    missing = [key for key in required if not app.config.get(key)]
    if missing:
        print(f"Warning: Missing email configuration: {', '.join(missing)}")
        return False
    return True

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', os.environ.get('MAIL_USERNAME'))
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_MAX_EMAILS'] = None
app.config['MAIL_ASCII_ATTACHMENTS'] = False

if not check_email_config():
    print("Warning: Email configuration is incomplete. Please set MAIL_USERNAME, MAIL_PASSWORD, and MAIL_DEFAULT_SENDER in Replit Secrets.")

db.init_app(app)
mail = Mail(app)
login_manager = LoginManager()

def load_team_permissions():
    try:
        with open('team_permissions.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_team_permissions(permissions):
    with open('team_permissions.json', 'w') as file:
        json.dump(permissions, file, indent=4)

def set_team_privacy(school_slug, is_private):
    permissions = load_team_permissions()
    if school_slug in permissions:
        permissions[school_slug]['stats_private'] = is_private
        save_team_permissions(permissions)
        return True
    return False

def is_team_private(school_slug):
    permissions = load_team_permissions()
    return permissions.get(school_slug, {}).get('stats_private', False)

login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Please sign in to view or score games and player statistics."

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
from word2number import w2n
import spacy
import json
import os
from datetime import datetime


# Helper function to get team file path
def get_team_file_path(team_name, league='SCVAL'):
    league_dir = os.path.join('teams', 'CCS', league)
    if not os.path.exists(league_dir):
        os.makedirs(league_dir)
    return os.path.join(league_dir, f"team_{team_name.replace(' ', '_')}.json")

def check_email_config():
    required = ['MAIL_USERNAME', 'MAIL_PASSWORD', 'MAIL_DEFAULT_SENDER']
    missing = [key for key in required if not app.config.get(key)]
    if missing:
        print(f"Warning: Missing email configuration: {', '.join(missing)}")
        return False
    return True
    return True

def get_team_file_path(team_name, league='SCVAL'):
    league_dir = os.path.join('teams', 'CCS', league)
    if not os.path.exists(league_dir):
        os.makedirs(league_dir)
    return os.path.join(league_dir, f"team_{team_name.replace(' ', '_')}.json")

FILE_PATH = 'team_data.json'
import json

# Define file path for storage
FILE_PATH = 'team_data.json'

# Load data from the file
def load_data():
    try:
        with open(FILE_PATH, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # Return an empty dictionary if the file doesn't exist
        return {}

# Save data to the file
def save_data_to_file(data, filename='data.json'):
    """Save the given data to a JSON file."""
    print(f"Saving data to {filename}...")  # Debugging print

    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

    print("Data saved successfully!")

# Add player to the team's roster
def add_player_to_roster(school, cap_number, player_name, grade, position):
    # Load the existing data from the file
    data = load_data()

    # If the school doesn't exist in the data, initialize it
    if school['slug'] not in data:
        data[school['slug']] = {'players': []}

    # Create a player dictionary
    new_player = {
        'cap_number': cap_number,
        'name': player_name,
        'grade': int(grade),
        'position': position
    }

    # Add the new player to the roster
    data[school['slug']]['players'].append(new_player)

    # Save the updated data back to the file
    save_data(data)


# Load the data from a file


# Assuming you have helper functions to load team data and game data

@app.route('/team/<school_slug>/score/<int:game_index>', methods=['GET'])
def start_scoring(school_slug, game_index):
    school = get_school_by_slug(school_slug)
    if not school:
        return "School not found", 404

    team_name = school['name']
    game = open_game(team_name, game_index)
    if not game:
        return "Game not found", 404

    home_team = team_name if game['home_away'] == 'Home' else game['opponent']
    away_team = game['opponent'] if game['home_away'] == 'Home' else team_name

    home_school = next((school for school in schools.values() if school['name'] == home_team), None)
    away_school = next((school for school in schools.values() if school['name'] == away_team), None)

    return render_template("score_game.html",
                         home_team=home_team,
                         away_team=away_team,
                         game_index=game_index,
                         school_slug=school_slug,
                         home_team_color=home_school['bg_color'],
                         home_team_text_color=home_school['text_color'],
                         away_team_color=away_school['bg_color'],
                         away_team_text_color=away_school['text_color'],
                         home_team_logo=home_school['logo'],
                         away_team_logo=away_school['logo'])

def initialize_team_file(team_name):
    # Ensure the 'teams' directory exists
    if not os.path.exists('teams'):
        os.makedirs('teams')  # Create the directory if it doesn't existzz

    team_file_path = get_team_file_path(team_name)
    if not os.path.exists(team_file_path):
        # Create the file with an empty "games" list
        with open(team_file_path, 'w') as file:
            json.dump({"games": []}, file, indent=4)


# Helper function to load a team's JSON file
def load_team_data(team_name):
    team_file_path = get_team_file_path(team_name)
    if os.path.exists(team_file_path):
        try:
            with open(team_file_path, 'r') as file:
                # Check if file is empty
                if os.path.getsize(team_file_path) == 0:
                    return {"games": []}  # Return an empty "games" list if file is empty
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Error: {team_file_path} contains invalid JSON.")
            return {"games": []}  # Return empty "games" list if JSON is invalid
    else:
        return {"games": []}

# Function to open a game based on team_name and game_index
def open_game(team_name, game_index):
    # Get the file path for the team's data
    team_file_path = get_team_file_path(team_name)
    print(f"Looking for game {game_index} in {team_file_path}")

    # Check if the team file exists
    if not os.path.exists(team_file_path):
        print(f"Team file not found: {team_file_path}")
        initialize_team_file(team_name)
        return None

    try:
        with open(team_file_path, 'r') as file:
            team_data = json.load(file)
            if "games" not in team_data:
                print("No games array found in team data")
                return None

            if not isinstance(game_index, int):
                game_index = int(game_index)

            if game_index < 0 or game_index >= len(team_data["games"]):
                print(f"Game index {game_index} out of bounds (0-{len(team_data['games'])-1})")
                return None

            return team_data["games"][game_index]

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading game data: {str(e)}")
        return None


# Helper function to save a team's JSON data
def save_team_data(team_name, data):
    team_file_path = get_team_file_path(team_name)
    
    # Ensure game_log exists for each game
    if "games" in data:
        for game in data["games"]:
            if "game_log" not in game:
                game["game_log"] = []
            
            # Get the game index
            game_id = str(data["games"].index(game))
            
            # If there's corresponding game data in memory, use its game log
            if game_id in game_data and "game_log" in game_data[game_id]:
                # Make a deep copy of the game log to preserve all entries exactly
                game["game_log"] = list(game_data[game_id]["game_log"])
    
    with open(team_file_path, 'w') as file:
        json.dump(data, file, indent=4)

from flask import request, jsonify













def reset_team_stats():
    global dataWhite, dataBlack
    # Reset stats for the next game
    dataWhite = {
        'Player': ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5', 'Player 6', 'Player 7', 'Player 8', 'Player 9', 'Player 10'],
        'Shot': [0] * 10,
        'Shot Attempt': [0] * 10,
        'Assists': [0] * 10,
        'Blocks': [0] * 10,
        'Steals': [0] * 10,
        'Exclusions': [0] * 10,
        'Exclusions Drawn': [0] * 10,
        'Penalties': [0] * 10,
        'Turnovers': [0] * 10,
        'Sprint Won': [0] * 10,
        'Sprint Attempt': [0] * 10
    }

    dataBlack = {
        'Player': ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5', 'Player 6', 'Player 7', 'Player 8', 'Player 9', 'Player 10'],
        'Shot': [0] * 10,
        'Shot Attempt': [0] * 10,
        'Assists': [0] * 10,
        'Blocks': [0] * 10,
        'Steals': [0] * 10,
        'Exclusions': [0] * 10,
        'Exclusions Drawn': [0] * 10,
        'Penalties': [0] * 10,
        'Turnovers': [0] * 10,
        'Sprint Won': [0] * 10,
        'Sprint Attempt': [0] * 10
    }

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    reset_team_stats()
    return jsonify({'status': 'success'})


#training model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from flask_sqlalchemy import SQLAlchemy

# Define your models after initializing db
class School(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    slug = db.Column(db.String(100), unique=True, nullable=False)
    logo = db.Column(db.String(100))
    # Add other fields as necessary

    def __repr__(self):
        return f'<School {self.name}>'

if __name__ == '__main__':
    app.run(debug=True)


# Sample dataset of sentences with labels
data = {'Sentence': [
        'Player 6 excluded from the light team',
        'Player 3 drew an exclusion for the dark team',
        'Player 3 drew a penalty for the light team',
        'Penalty on player 6 on the dark team',
        'Penalty drawn on player 4 on the light team',
        "Player 3 from light drew a penalty.",
        "Penalty for player 7 from dark.",
        "Exclusion drawn by player 5 from light.",
        "Player 6 from dark team caused an exclusion.",
        "Light team’s player 4 drew a penalty.",
        "Penalty for dark team, player 8.",
        "Player 2 from light caused an exclusion.",
        "Exclusion for dark team, player 1.",
        "Penalty on player 3 from light team.",
        "Player 5 from dark excluded from the game.",
        "Exclusion on player 6 from dark.",
        "Player 4 from light team received a penalty.",
        "Penalty on player 7 from dark.",
        "Exclusion against player 2 from light team.",
        "Player 8 from dark team was excluded.",
        "Player 1 from light excluded from the match.",
        "Player 5 had an exclusion drawn against them for the dark team.",
        "Exclusion was drawn against player 3 from the light team.",
        "The dark team’s player 7 had an exclusion drawn against them.",
        "Player 2 from the light team got an exclusion drawn against them.",
        "An exclusion against player 4 from the dark team was drawn.",
        "For the light team, an exclusion was drawn against player 6.",
        "The dark team’s exclusion was drawn against player 8.",
        "An exclusion drawn against player 1 from the light team.",
        "Player 5 from the dark team drew an exclusion for their side.",
        "Exclusion drawn for player 3 from the light team.",
        "For the dark team, player 7 drew an exclusion.",
        "Player 2 from the light team had an exclusion drawn for them.",
        "An exclusion drawn for player 4 from the dark team.",
        "Player 6 from the dark team drew an exclusion for their team.",
        "Exclusion for player 8 from the light team was drawn.",
        "For the light team, exclusion drawn for player 1.",
        "Player 5 drew a kick out for the light team.",
        "Player 2 from the dark team had a kick out drawn in their favor.",
        "The light team’s player 7 drew a kick out.",
        "A kick out was drawn for player 3 from the dark team.",
        "Player 6 from the light team successfully drew a kick out.",
        "For the dark team, a kick out was drawn by player 8.",
        "Player 4 from the light team had a kick out called in their favor.",
        "The dark team’s player 9 drew a kick out.",
        "A kick out was awarded to the light team for player 10.",
        "For the dark team, player 5 drew a kick out.",
        "Player 6 from the dark team was kicked out.",
        "A kick out was called against player 4 from the light team.",
        "Player 3 was kicked out for a foul on the dark team.",
        "The light team’s player 7 got kicked out of the game.",
        "A kick out was issued against player 2 from the light team.",
        "Player 8 from the dark team got kicked out for misconduct.",
        "The referee called a kick out against player 9 from the light team.",
        "Player 1 from the dark team was kicked out of the match.",
        "A kick out was enforced against player 5 from the light team.",
        "Player 7 on the dark team received a kick out."],

        'Label': [0,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]}  # 0 = Negative, 1 = Positive

df = pd.DataFrame(data)

# Step 1: Preprocessing - Split into train and test
X_train, X_test, y_train, y_test = train_test_split(df['Sentence'], df['Label'], test_size=0.2, random_state=42)

# Step 2: Feature extraction using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 3: Model training - Logistic Regression
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 4: Prediction and evaluation
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Step 5: Integrate the model into your workflow
def predict_connotation(sentence):
    vec_sentence = vectorizer.transform([sentence])
    prediction = model.predict(vec_sentence)
    return 'Positive' if prediction == 1 else 'Negative'

# Load spacy model

# Load the pre-trained Spacy model
nlp = spacy.load('en_core_web_sm')

# Path to the previous games JSON file
GAMES_FILE = 'previous_games.json'

previous_games = []  # List to store the data of previous games

# Load previous games from JSON file
def load_previous_games():
    if os.path.exists(GAMES_FILE):
        with open(GAMES_FILE, 'r') as file:
            return json.load(file)
    else:
        return []

# Store previous games data in memory
previous_games = load_previous_games()

# Save previous games to JSON file
def save_previous_games():
    with open("previous_games.json", "w") as file:
        json.dump(previous_games, file, indent=4)



# Route to get the previous games data
@app.route('/get_previous_games', methods=['GET'])
def get_previous_games():
    return jsonify({'games': previous_games})

@app.route('/previous_games')
def previous_games_page():
    return render_template('previous_games.html')

# Define team data with zeros
dataWhite = {
    'Player': ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5', 'Player 6', 'Player 7', 'Player 8', 'Player 9', 'Player 10'],
    'Shot': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Shot Attempt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Blocks': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Steals': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Exclusions': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Exclusions Drawn': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Penalties': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Turnovers': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Assists': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Sprint Won': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Sprint Attempt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

dataBlack = {
    'Player': ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5', 'Player 6', 'Player 7', 'Player 8', 'Player 9', 'Player 10'],
    'Shot': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Shot Attempt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Blocks': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Steals': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Exclusions': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Exclusions Drawn': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Penalties': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Turnovers': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Assists': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Sprint Won': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Sprint Attempt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}


def extract_key_phrases(text):
    # Convert 'goalie' to '1' and standardize penalty terms
    text = text.lower().replace('goalie', '1')
    
    # Replace various 5 meter/5 m variations with 'penalty'
    import re
    text = re.sub(r'5[\s-]meter', 'penalty', text)
    text = re.sub(r'5[\s-]metres', 'penalty', text)
    text = re.sub(r'5[\s-]m\s', 'penalty ', text)  # Space after m to avoid matching other words
    text = re.sub(r'five[\s-]meter', 'penalty', text)
    text = re.sub(r'five[\s-]metres', 'penalty', text)
    text = re.sub(r'five[\s-]m\s', 'penalty ', text)
    
    doc = nlp(text)
    doc_text = doc.text
    events = []
    dark_keywords = ['dark','black','blue']
    light_keywords = ['light','white']
    shot_keywords = ['goal', 'shot', 'score', 'point','scored','scores']
    block_keywords = ['block', 'blocked','blocks']
    steal_keywords = ['steal','stole','took','steals']
    exclusion_keywords = ['exclusion', 'kickout','excluded', 'kicked out', 'kick out', 'kicked']
    turnover_keywords = ['turnover', 'foul', 'lost', 'loses', 'offensive foul', 'lost the ball', 'turned the ball over', 'offensive foul', 'committed offensive foul', 'committed foul', 'committed a foul', 'under water', 'underwater', 'put the ball under', 'fouled']
    penalty_keywords = ['penalty', 'five meter', '5 meter', '5-meter', '5m', '5 m', 'five m', '5meter', 'five-meter']
    sprint_keywords = ['sprint', 'won sprint', 'won the sprint']

    # Extract all player numbers first
    all_numbers = []
    for i, token in enumerate(doc):
        try:
            # Skip penalty phrase variations like "5 m", "five meter", etc.
            next_token = doc[i + 1].text.lower() if i + 1 < len(doc) else ''
            is_penalty_phrase = (
                (token.text == '5' and next_token in ['m', 'meter', 'meters', '-meter', '-meters']) or
                (token.text.lower() == 'five' and next_token in ['m', 'meter', 'meters']) or
                token.text.lower() in ["5m", "5meter", "5-meter", "5meters", "5-meters"]
            )
            if is_penalty_phrase:
                continue
            # Convert token to number if valid player number
            num = w2n.word_to_num(token.text)
            if 1 <= num <= 13:
                all_numbers.append(str(num))
        except (ValueError, IndexError):
            continue

    # Initialize event tracking variables
    first_event = {'team': None, 'player': None, 'event': None}
    second_event = {'team': None, 'player': None, 'event': None}
    events = []

    # Extract all numbers first
    all_numbers = []
    for token in doc:
        try:
            if token.text != "five":
                num = w2n.word_to_num(token.text)
                if 1 <= num <= 13:
                    all_numbers.append(str(num))
        except ValueError:
            continue

    tokens = [token.text for token in doc]

    # Find team first
    current_team = None
    for i, token in enumerate(tokens):
        if token in dark_keywords:
            current_team = 'dark'
            break
        elif token in light_keywords:
            current_team = 'light'
            break

    if current_team is None:
        # Look for team mentions in larger context
        doc_text = doc.text.lower()
        if any(word in doc_text for word in dark_keywords):
            current_team = 'dark'
        elif any(word in doc_text for word in light_keywords):
            current_team = 'light'

    import inspect

    def debug_event(event_dict, location):
        frame = inspect.currentframe()
        line_num = frame.f_lineno
        print(f"[Line {line_num}] EVENT ASSIGNMENT: Team={event_dict['team']}, Player={event_dict['player']}, Event={event_dict['event']} at {location}")
        del frame  # Clean up frame reference properly

    # Initialize first and second events
    first_event = {'team': current_team, 'player': None, 'event': None}
    second_event = {'team': None, 'player': None, 'event': None}

    # Assign first player number found
    if all_numbers:
        first_event['player'] = all_numbers[0]
        debug_event(first_event, "initial player assignment")

    # Second pass - find player and event
    for i, token in enumerate(tokens):
        # Try to extract player number
        try:
            if token != "point" and token != "five":
                num = w2n.word_to_num(token)
                if 1 <= num <= 13 and first_event['player'] is None:
                    first_event['player'] = str(num)
                    first_event['team'] = current_team

                    # Check for single event keywords right after the number
                    if i + 1 < len(tokens):
                        if tokens[i+1] in exclusion_keywords or 'excluded' in doc_text:
                            first_event['event'] = 'Exclusions'
                            break
                        elif tokens[i+1] in penalty_keywords or 'penalty' in doc_text:
                            first_event['event'] = 'Penalties'
                            break
        except ValueError:
            pass

        # Check for goalie mentions and team assignments
        if token == 'goalie' or (token == '1' and 'goalie' in doc_text):
            if not first_event['player']:
                first_event['player'] = '1'
                # Look for team mentions before goalie
                for t_idx in range(max(0, i-5), i+1):
                    if t_idx < len(tokens):
                        if tokens[t_idx] in light_keywords:
                            first_event['team'] = 'light'
                            break
                        elif tokens[t_idx] in dark_keywords:
                            first_event['team'] = 'dark'
                            break

                # Check for event type in surrounding context
                if 'exclusion' in doc_text or 'excluded' in doc_text or 'kicked out' in doc_text:
                    first_event['event'] = 'Exclusions'
                elif tokens[i+1] in block_keywords or 'block' in doc_text:
                    first_event['event'] = 'Blocks'
                elif tokens[i+1] in penalty_keywords or 'penalty' in doc_text:
                    first_event['event'] = 'Penalties'

        # Skip penalty detection if this is a missed shot event
        if ('missed' in doc_text.lower() or 'miss' in doc_text.lower()) and not ('penalty' in doc_text.lower() or '5 meter' in doc_text.lower() or '5-meter' in doc_text.lower()):
            # This is just a missed shot, not a penalty
            if first_event['player'] is None:
                first_event['player'] = all_numbers[0] if all_numbers else None
            first_event['event'] = 'Shot Attempt'
            first_event['team'] = current_team
            break
        # Extract event type for penalties and exclusions
        elif any(phrase in doc_text for phrase in penalty_keywords):
            if '5 m by' in doc_text or 'five meter by' in doc_text:
                # Player who drew gets penalty, other player gets exclusion
                if len(all_numbers) >= 2:
                    first_event['player'] = all_numbers[0]
                    first_event['event'] = 'Penalties'
                    first_event['team'] = current_team

                    second_event['player'] = all_numbers[1]
                    second_event['event'] = 'Exclusions'
                    second_event['team'] = 'light' if current_team == 'dark' else 'dark'
            elif 'drew' in doc_text or 'drawn' in doc_text:
                # First event is who drew the penalty
                first_event['player'] = all_numbers[0]
                first_event['event'] = 'Penalties Drawn'
                first_event['team'] = current_team

                if len(all_numbers) >= 2:
                    # Second event is who got the penalty/exclusion
                    second_event['player'] = all_numbers[1]
                    second_event['event'] = 'Exclusions'
                    second_event['team'] = 'light' if current_team == 'dark' else 'dark'
            elif 'on' in doc_text or 'against' in doc_text or 'by' in doc_text:
                # Handle exclusion vs penalty
                if first_event['player'] is None:
                    if 'goalie' in doc_text:
                        first_event['player'] = '1'
                    else:
                        first_event['player'] = all_numbers[0] if all_numbers else None

                if len(all_numbers) >= 2:
                    # First player gets exclusion
                    first_event['event'] = 'Exclusions'
                    first_event['team'] = current_team

                    # Second player gets penalties drawn
                    second_event['player'] = all_numbers[1]
                    second_event['event'] = 'Penalties Drawn'
                    second_event['team'] = 'light' if current_team == 'dark' else 'dark'
                else:
                    first_event['event'] = 'Exclusions'
                    first_event['team'] = current_team
            else:
                # Default penalty event
                if first_event['player'] is None:
                    if 'goalie' in doc_text:
                        first_event['player'] = '1'
                    else:
                        first_event['player'] = all_numbers[0] if all_numbers else None
                first_event['event'] = 'Penalties'
                first_event['team'] = current_team
            break
        elif token in exclusion_keywords:
            drew_keywords = ['drew', 'draws', 'draw', 'drawn', 'by']
            if 'drawn exclusion' in doc_text or 'exclusion drawn' in doc_text:
                # Player gets exclusion drawn
                first_event['player'] = all_numbers[0]
                first_event['event'] = 'Exclusions Drawn'
                first_event['team'] = current_team

                if len(all_numbers) >= 2:
                    # Second player gets exclusion
                    second_event['player'] = all_numbers[1]
                    second_event['event'] = 'Exclusions'
                    second_event['team'] = 'light' if current_team == 'dark' else 'dark'
            elif any(word in doc_text for word in drew_keywords):
                # First event is who drew the exclusion
                first_event['player'] = all_numbers[0]
                first_event['event'] = 'Exclusions Drawn'
                first_event['team'] = current_team

                if len(all_numbers) >= 2:
                    # Second player gets exclusion
                    second_event['player'] = all_numbers[1]
                    second_event['event'] = 'Exclusions'
                    second_event['team'] = 'light' if current_team == 'dark' else 'dark'
            elif 'for' in doc_text:
                # Handle "exclusion for player X" case
                first_event['player'] = all_numbers[0] if all_numbers else None
                first_event['event'] = 'Exclusions'
                first_event['team'] = current_team
                break
            elif 'by' in doc_text and len(all_numbers) >= 2:
                # For all exclusion events with "by", first player gets negative event
                first_event['player'] = all_numbers[0]
                first_event['event'] = 'Exclusions'  # Always negative for first player
                first_event['team'] = current_team

                second_event['player'] = all_numbers[1]
                second_event['event'] = 'Exclusions Drawn'  # Always positive for second player
                second_event['team'] = 'light' if current_team == 'dark' else 'dark'
                break
            elif len(all_numbers) >= 2 and ('excluded' in doc_text or 'kicked out' in doc_text or 'kicked' in doc_text):
                if 'was excluded by' in doc_text.lower():
                    # For "X was excluded by Y", X gets excluded, Y drew it
                    first_player = all_numbers[0]  # Player who was excluded
                    second_player = all_numbers[1]  # Player who drew it
                    return [
                        (first_player, 'Exclusions', current_team),  # First player gets excluded
                        (second_player, 'Exclusions Drawn', 'light' if current_team == 'dark' else 'dark')  # Second player drew it
                    ]
                elif 'excluded' in doc_text:
                    # Handle "X excluded Y" pattern - first player drew it, second player got excluded
                    first_event['player'] = all_numbers[0]
                    first_event['event'] = 'Exclusions Drawn'
                    first_event['team'] = current_team

                    second_event['player'] = all_numbers[1]
                    second_event['event'] = 'Exclusions'
                    second_event['team'] = 'light' if current_team == 'dark' else 'dark'
                break
            elif len(all_numbers) >= 1 and ('excluded' in doc_text or 'kicked out' in doc_text or 'kicked' in doc_text):
                # Single player exclusion
                first_event['player'] = all_numbers[0]
                first_event['event'] = 'Exclusions'
                first_event['team'] = current_team
                break
            elif 'for' in doc_text:
                # Handle format: "exclusion on [team1] [player1] for [team2] [player2]"
                words = doc_text.split()
                try:
                    # Find indices of key words
                    on_idx = words.index('on')
                    for_idx = words.index('for')

                    # Parse first team (before 'for')
                    if any(word in words[on_idx:for_idx] for word in dark_keywords):
                        first_event['team'] = 'dark'
                    elif any(word in words[on_idx:for_idx] for word in light_keywords):
                        first_event['team'] = 'light'

                    # Parse second team (after 'for')
                    if any(word in words[for_idx:] for word in dark_keywords):
                        second_event['team'] = 'dark'
                    elif any(word in words[for_idx:] for word in light_keywords):
                        second_event['team'] = 'light'

                    # First event is the exclusion
                    first_event['player'] = all_numbers[0]
                    first_event['event'] = 'Exclusions'

                    # Second event is who drew it
                    second_event['player'] = all_numbers[1]
                    second_event['event'] = 'Exclusions Drawn'
                except (ValueError, IndexError):
                    pass
                break
            elif ('on' in doc_text or 'against' in doc_text or 'kicked' in doc_text or 
                  any(word in doc_text for word in ['excluded', 'kicked out'])):
                # Handle receiving an exclusion
                first_event['player'] = all_numbers[0]
                first_event['event'] = 'Exclusions'
                first_event['team'] = current_team
                break

        if 'block' in doc_text.lower() or 'blocked' in doc_text.lower() or 'save' in doc_text.lower():
            was_blocked = any(word in doc_text.lower() for word in ['was blocked', 'blocked by'])
            if was_blocked:
                # Player was blocked by someone - first player attempted shot, second player blocked
                first_event['event'] = 'Shot Attempt'
                first_event['player'] = all_numbers[0] if all_numbers else None
                first_event['team'] = current_team

                if len(all_numbers) >= 2:
                    second_event['event'] = 'Blocks'
                    second_event['player'] = all_numbers[1]
                    second_event['team'] = 'light' if current_team == 'dark' else 'dark'
            else:
                # Standard block - first player blocks, second player attempted
                first_event['event'] = 'Blocks'
                first_event['player'] = all_numbers[0] if all_numbers else None
                first_event['team'] = current_team

                if len(all_numbers) >= 2:
                    second_event['event'] = 'Shot Attempt'
                    second_event['player'] = all_numbers[1]
                    second_event['team'] = 'light' if current_team == 'dark' else 'dark'
        elif token in shot_keywords or 'goal' in doc_text.lower() or any(word in doc_text.lower() for word in ['hit the bar', 'hit the crossbar', 'bar', 'crossbar', 'missed', 'miss']):
            # Handle goalie scoring specifically
            if 'goalie' in doc_text or (all_numbers and all_numbers[0] == '1'):
                first_event['player'] = '1'  # Goalie is always number 1
                first_event['team'] = current_team

                # Check if it's a successful shot
                if 'scored' in doc_text or 'score' in doc_text or 'goal' in doc_text.lower():
                    # Only count as a goal if not missed
                    if not any(word in doc_text.lower() for word in ['missed', 'miss']):
                        first_event['event'] = 'Shot'
                        # Add shot attempt for goalie
                        second_event['event'] = 'Shot Attempt'
                        second_event['player'] = '1'
                        second_event['team'] = current_team
                    else:
                        # If missed, only count as attempt
                        first_event['event'] = 'Shot Attempt'
                else:
                    first_event['event'] = 'Shot Attempt'
                    first_event['team'] = current_team
            else:
                first_event['player'] = all_numbers[0] if all_numbers else None
                first_event['team'] = current_team

                # Check specifically for missed shots - make sure this is prioritized over penalty interpretation
                if 'missed' in doc_text.lower() or 'miss' in doc_text.lower():
                    first_event['event'] = 'Shot Attempt'
                # If it's a goal (scored) and not missed
                elif 'scored' in doc_text or 'score' in doc_text:
                    first_event['event'] = 'Shot'
                    # Add second event for shot attempt
                    second_event['event'] = 'Shot Attempt'
                    second_event['player'] = first_event['player']
                    second_event['team'] = first_event['team']
                else:
                    first_event['event'] = 'Shot Attempt'
        elif token in block_keywords:
            # Check for block scenarios
            if 'goalie' in doc_text.lower():
                # First event is the block by goalie
                first_event['event'] = 'Blocks'
                first_event['player'] = '1'  # Goalie is always number 1
                first_event['team'] = current_team

                # Second event is shot attempt by the opposing team
                if len(all_numbers) >= 1:
                    second_event['event'] = 'Shot Attempt'
                    second_event['player'] = all_numbers[0]
                    second_event['team'] = 'light' if current_team == 'dark' else 'dark'
            elif len(all_numbers) >= 2:
                # First event is the block
                first_event['event'] = 'Blocks'
                first_event['player'] = all_numbers[0]
                first_event['team'] = current_team

                # Second event is the shot attempt
                second_event['event'] = 'Shot Attempt'
                second_event['player'] = all_numbers[1]
                second_event['team'] = 'light' if current_team == 'dark' else 'dark'
            elif len(all_numbers) == 1:
                # If only one number, assume it's the blocker and the shot was from opposing team
                first_event['event'] = 'Blocks'
                first_event['player'] = all_numbers[0]
                first_event['team'] = current_team

        elif 'turnover' in doc_text or 'turned over' in doc_text or token in turnover_keywords or any(phrase in doc_text for phrase in [
            'lost the ball', 'turned the ball over', 'offensive foul', 'offensive',
            'foul on offense', 'lost possession', 'dropped the ball'
        ]) or 'foul' in doc_text:
            # Handle single player turnover events
            if 'goalie' in doc_text:
                first_event['event'] = 'Turnovers'
                first_event['player'] = '1'  # Goalie is always number 1
                first_event['team'] = current_team
            elif all_numbers:
                first_event['event'] = 'Turnovers'
                first_event['player'] = all_numbers[0]
                first_event['team'] = current_team
            break
        elif ('assist' in doc_text or 'assisted' in doc_text):
            if len(all_numbers) >= 2:
                # For input patterns like "dark 1 assisted by dark 2" or "dark 1 assisted dark 2"
                if "assisted by" in doc_text:
                    scorer = all_numbers[0]
                    assister = all_numbers[1]
                else:
                    assister = all_numbers[0]
                    scorer = all_numbers[1]
                
                # Add all relevant events
                events.append((scorer, 'Shot', current_team))  # Goal for scorer
                events.append((scorer, 'Shot Attempt', current_team))  # Shot attempt for scorer
                events.append((assister, 'Assists', current_team))  # Assist for assister
                return events
            elif len(all_numbers) >= 1:
                # If it's just an assist
                events.append((all_numbers[0], 'Assists', current_team))
                return events
        # Handle negative events with "by"
        if 'by' in doc_text and len(all_numbers) >= 2:
            if any(word in doc_text for word in ['excluded', 'kicked out']):
                # First player gets exclusion (negative)
                first_event['player'] = all_numbers[0]
                first_event['event'] = 'Exclusions'
                first_event['team'] = current_team
                
                # Second player (after "by") gets drawn (positive)
                second_event['player'] = all_numbers[1]
                second_event['event'] = 'Exclusions Drawn'
                second_event['team'] = 'light' if current_team == 'dark' else 'dark'
                break
            elif 'stolen' in doc_text or 'stole' in doc_text:
                # First player loses ball (negative)
                first_event['player'] = all_numbers[0]
                first_event['event'] = 'Turnovers'
                first_event['team'] = current_team
                
                # Second player gets steal (positive)
                second_event['player'] = all_numbers[1]
                second_event['event'] = 'Steals'
                second_event['team'] = 'light' if current_team == 'dark' else 'dark'
                break
        elif token in steal_keywords or 'stole' in doc_text or 'stolen' in doc_text:
            # Handle steals without "by"
            if all_numbers:
                first_event['event'] = 'Steals'
                first_event['player'] = all_numbers[0]
                first_event['team'] = current_team
                if len(all_numbers) >= 2:
                    second_event['event'] = 'Turnovers'
                    second_event['player'] = all_numbers[1]
                    second_event['team'] = 'light' if current_team == 'dark' else 'dark'
                break
        elif token in steal_keywords or 'stole from' in doc_text or 'steal from' in doc_text or 'under water' in doc_text or 'underwater' in doc_text or 'drew under' in doc_text or 'forced under' in doc_text or 'committed a ball under' in doc_text or 'put a ball under' in doc_text or 'forced a ball under' in doc_text or 'forced the ball under' in doc_text or 'ball under on' in doc_text or 'ball under by' in doc_text:
            # Check if it's a ball under scenario with specified players
            if 'ball under by' in doc_text:
                # Player gets steal if they forced the ball under
                first_event = {'event': 'Steals', 'player': all_numbers[0], 'team': current_team}
                if len(all_numbers) >= 2:
                    second_event = {'event': 'Turnovers', 'player': all_numbers[1], 'team': 'light' if current_team == 'dark' else 'dark'}
                break
            elif 'ball under forced against' in doc_text or 'ball under on' in doc_text:
                # Player mentioned gets a turnover when ball under is forced against them
                first_event = {'event': 'Turnovers', 'player': all_numbers[0], 'team': current_team}
                if len(all_numbers) >= 2:
                    second_event = {'event': 'Steals', 'player': all_numbers[1], 'team': 'light' if current_team == 'dark' else 'dark'}
                break
            elif 'turnover' in doc_text or 'turned over' in doc_text:
                # Handle explicit turnover mention
                first_event = {'event': 'Turnovers', 'player': all_numbers[0], 'team': current_team}
                break
            elif 'ball under by' in doc_text:
                # First player mentioned gets the steal
                first_event['event'] = 'Steals'
                if all_numbers:
                    first_event['player'] = all_numbers[0]
                    # Second player gets the turnover
                    if len(all_numbers) >= 2:
                        second_event['player'] = all_numbers[1]
                        second_event['event'] = 'Turnovers'
                        second_event['team'] = 'light' if current_team == 'dark' else 'dark'
            elif ('put' in doc_text and 'ball under' in doc_text) or ('forced' in doc_text and 'ball under' in doc_text):
                first_event['event'] = 'Steals'
                if all_numbers:
                    first_event['player'] = all_numbers[0]
                    # Check if there's a target player specified
                    if len(all_numbers) >= 2:
                        second_event['player'] = all_numbers[1]
                        second_event['event'] = 'Turnovers'
                        second_event['team'] = 'light' if first_event['team'] == 'dark' else 'dark'
                break

            first_event['event'] = 'Steals'

            # Handle underwater ball scenarios
            if any(phrase in doc_text for phrase in ['under water', 'underwater', 'drew under', 'forced under', 'put under']):
                if numbers:
                    first_event['player'] = numbers[0]  # First player mentioned gets the steal
                    first_event['event'] = 'Steals'
                    if len(numbers) >= 2:
                        # Always create turnover for the opposing player
                        second_event['player'] = numbers[1]
                        second_event['event'] = 'Turnovers'
                        second_event['team'] = 'light' if first_event['team'] == 'dark' else 'dark'
                break

            # Check if steal was from goalie
            if 'goalie' in doc_text.lower():
                second_event['player'] = '1'  # Goalie is always number 1
                second_event['event'] = 'Turnovers'
                second_event['team'] = 'light' if first_event['team'] == 'dark' else 'dark'
            else:
                # Find the second player number for turnover
                try:
                    remaining_text = ' '.join(tokens[i+1:])
                    numbers = []
                    for t in tokens[i+1:]:
                        if t != "point" and t != "five":
                            try:
                                num = w2n.word_to_num(t)
                                if 1 <= num <= 13:
                                    numbers.append(str(num))
                            except ValueError:
                                continue

                    if numbers and len(numbers) > 0:
                        if len(numbers) >= 2:
                            # For underwater ball scenarios, first number is the stealer
                            first_event['player'] = numbers[0]
                            second_event['player'] = numbers[1]
                        else:
                            second_event['player'] = numbers[-1]  # Take the last number found
                        second_event['event'] = 'Turnovers'
                        second_event['team'] = 'light' if first_event['team'] == 'dark' else 'dark'
                except ValueError:
                    pass

    # Handle sprint events
    if any(word in doc_text for word in sprint_keywords):
        if 'won' in doc_text:
            if all_numbers:
                first_event['player'] = all_numbers[0]
                first_event['event'] = 'Sprint Won'
                first_event['team'] = current_team
                second_event['event'] = 'Sprint Attempt'
                second_event['player'] = all_numbers[0]
                second_event['team'] = current_team

                if len(all_numbers) >= 2:
                    # Add attempt for opposing player
                    events.append((all_numbers[1], 'Sprint Attempt', 'light' if current_team == 'dark' else 'dark'))
        elif 'lost' in doc_text:
            if all_numbers:
                # Record Sprint Attempt for the player who lost
                first_event['player'] = all_numbers[0]
                first_event['event'] = 'Sprint Attempt'
                first_event['team'] = current_team

                # Record Sprint Won and Sprint Attempt for the opposing player
                if len(all_numbers) >= 2:
                    opposing_team = 'light' if current_team == 'dark' else 'dark'
                    events.append((all_numbers[1], 'Sprint Won', opposing_team))
                    events.append((all_numbers[1], 'Sprint Attempt', opposing_team))

    # Add complete events to the list
    if first_event['team'] and first_event['player'] and first_event['event']:
        debug_event(first_event, "final first event")
        events.append((first_event['player'], first_event['event'], first_event['team']))

    if second_event['team'] and second_event['player'] and second_event['event']:
        debug_event(second_event, "final second event")
        events.append((second_event['player'], second_event['event'], second_event['team']))

    return events if events else [(None, None, None)]




# Update team data
def sort_data(player, event, team, home_team_name, away_team_name, game_id):
    print("\n=== Debug Info ===")
    print(f"Looking for player: {player}")
    print(f"Team color: {team}")
    print(f"Home team name: '{home_team_name}'")
    print(f"Away team name: '{away_team_name}'")
    if not home_team_name or not away_team_name:
        print("ERROR: Team names are missing!")

    try:
        # Load team rosters
        with open('team_rosters.json', 'r') as file:
            team_rosters = json.load(file)
            print(f"Available teams in roster: {list(team_rosters.keys())}")

        if home_team_name not in team_rosters:
            print(f"Error: '{home_team_name}' not found in rosters")
            return False
        if away_team_name not in team_rosters:
            print(f"Error: {away_team_name} not found in rosters")
            return False

        # Get team rosters based on the current game teams
        home_roster = team_rosters[home_team_name]
        away_roster = team_rosters[away_team_name]

        print(f"Home roster: {home_roster}")
        print(f"Away roster: {away_roster}")

        print(f"Home roster cap numbers: {[str(p['cap_number']).strip() for p in home_roster]}")
        print(f"Away roster cap numbers: {[str(p['cap_number']).strip() for p in away_roster]}")
    except Exception as e:
        print(f"Error loading rosters: {str(e)}")
        return False

    if team == 'light':
        # Find the player in the away roster by cap number (light team)
        try:
            player_str = str(player)
            print(f"Searching away roster for cap number: '{player_str.strip()}'")
            player_index = next(i for i, p in enumerate(away_roster) if str(p['cap_number']).strip() == player_str.strip())
            print(f"Found player at index: {player_index}")

            # Initialize the event array if it doesn't exist or is not a list
            if event not in game_data[game_id]['dataWhite'] or not isinstance(game_data[game_id]['dataWhite'][event], list):
                game_data[game_id]['dataWhite'][event] = [0] * len(away_roster)

            game_data[game_id]['dataWhite'][event][player_index] += 1  # Light/away team uses dataWhite
            return True
        except (StopIteration, ValueError, IndexError) as e:
            print(f"ERROR: Player {player} not found in away roster: {str(e)}")
            return False
    elif team == 'dark':
        # Find the player in the home roster by cap number (dark team)
        try:
            player_str = str(player)
            print(f"Searching home roster for cap number: '{player_str.strip()}'")
            player_index = next(i for i, p in enumerate(home_roster) if str(p['cap_number']).strip() == player_str.strip())
            print(f"Found player at index: {player_index}")

            # Initialize the event array if it doesn't exist or is not a list
            if event not in game_data[game_id]['dataBlack'] or not isinstance(game_data[game_id]['dataBlack'][event], list):
                game_data[game_id]['dataBlack'][event] = [0] * len(home_roster)

            game_data[game_id]['dataBlack'][event][player_index] += 1  # Dark/home team uses dataBlack
            return True
        except (StopIteration, ValueError, IndexError) as e:
            print(f"ERROR: Player {player} not found in home roster: {str(e)}")
            return False
    else:
        print("ERROR: Invalid team color")
    return True


# Load spacy model
nlp = spacy.load('en_core_web_sm')

def extract_numbers(text):
    """Extract all player numbers from text."""
    doc = nlp(text.lower())
    numbers = []

    for token in doc:
        try:
            if token.text != "five":  # Avoid "five meter" being counted
                num = w2n.word_to_num(token.text)
                if 1 <= num <= 13:  # Valid water polo numbers
                    numbers.append(str(num))
        except ValueError:
            continue

    return numbers

def identify_team(token, next_tokens):
    """Identify team from tokens."""
    dark_keywords = ['dark', 'black', 'blue']
    light_keywords = ['light', 'white']

    for i in range(3):  # Look ahead up to 3 tokens
        if i < len(next_tokens):
            if next_tokens[i] in dark_keywords:
                return 'dark'
            if next_tokens[i] in light_keywords:
                return 'light'
    return None

def extract_events(text):
    """Extract multiple events from a single sentence."""
    doc = nlp(text.lower())
    tokens = [token.text for token in doc]
    events = []

    # Event keywords
    shot_keywords = ['shot', 'shoot', 'shooting', 'scored', 'attempt']
    block_keywords = ['block', 'blocked', 'blocks', 'save', 'saved']
    steal_keywords = ['steal', 'stole', 'took', 'steals']
    assist_keywords = ['assist', 'assisted', 'helps', 'helped']
    exclusion_keywords = ['exclusion', 'kickout', 'excluded', 'kicked out', 'kick out', 'kicked']
    penalty_keywords = ['penalty', 'five meter', '5 meter', '5-meter', '5m', '5 m']

    numbers = extract_numbers(text)

    # Process tokens
    for i, token in enumerate(tokens):
        next_tokens = tokens[i+1:] if i < len(tokens) else []
        team = identify_team(token, next_tokens)

        # Handle shots and blocks
        if token in shot_keywords:
            if numbers:
                shooter = numbers[0]
                # Check for team mentions before the shot
                team = identify_team(token, next_tokens)
                if not team:
                    for prev_token in tokens[:i]:
                        if prev_token in ['dark', 'light']:
                            team = prev_token
                            break

                # Determine if shot was unsuccessful
                was_blocked = any(word in tokens for word in block_keywords)
                missed_keywords = ['missed', 'miss', 'wide', 'over', 'post', 'attempted', 'attempt']
                was_missed = any(word in tokens for word in missed_keywords)

                # Get connotation to determine if shot was successful
                shot_connotation = predict_connotation(text)
                is_shot_attempt = was_blocked or was_missed or shot_connotation == 'Negative' or 'attempt' in tokens

                events.append({
                    'player': shooter,
                    'event': 'Shot Attempt' if is_shot_attempt else 'Shot',
                    'team': team
                })

                if was_blocked:
                    # If 'goalie' is mentioned or no blocker number is given, assign to goalie (1)
                    blocker = '1' if 'goalie' in tokens or len(numbers) == 1 else numbers[1]
                    # Use explicitly mentioned team for goalie/blocker if available
                    blocker_team = None
                    for i, t in enumerate(tokens):
                        if t == 'goalie' or t == 'blocks' or t == 'blocked':
                            # Look for team mention before and after the block keyword
                            for check_token in tokens[max(0, i-3):i+3]:
                                if check_token in ['dark', 'black', 'blue']:
                                    blocker_team = 'dark'
                                    break
                                elif check_token in ['light', 'white']:
                                    blocker_team = 'light'
                                    break
                    # Only use opposite team if no explicit team was mentioned
                    if not blocker_team:
                        blocker_team = 'dark' if team == 'light' else 'light'
                    events.append({
                        'player': blocker,
                        'event': 'Blocks',
                        'team': blocker_team
                    })

        # Handle steals and turnovers
        if token in steal_keywords and len(numbers) >= 2:
            # Check for team mentions if not already set
            if not team:
                for t in tokens:
                    if t in ['dark', 'light']:
                        team = t
                        break

            events.append({
                'player': numbers[0],
                'event': 'Steals',
                'team': team
            })
            events.append({
                'player': numbers[1],
                'event': 'Turnovers',
                'team': 'dark' if team == 'light' else 'light'
            })

        # Handle assists and goals
        if token in assist_keywords and len(numbers) >= 2:
            # Check for team before processing assist
            if not team:
                for t in tokens:
                    if t in ['dark', 'light']:
                        team = t
                        break

            # Add shot attempt and goal for the scorer
            events.append({
                'player': numbers[0],
                'event': 'Shot Attempt',
                'team': team
            })
            events.append({
                'player': numbers[0],
                'event': 'Shot',
                'team': team
            })
            # Add assist for the second player
            events.append({
                'player': numbers[1],
                'event': 'Assists',
                'team': team  # Use the same team as the shooter
            })

        # Handle exclusions and penalties
        if token in exclusion_keywords or 'kicked out' in doc_text or 'five meter' in doc_text:
            connotation = predict_connotation(text)
            if len(numbers) >= 2 and ('drew' in doc_text or 'drawn' in doc_text):
                # Handle double event (exclusion drawn)
                if connotation == 'Positive' or 'drew' in doc_text:
                    events.append({
                        'player': numbers[0],
                        'event': 'Exclusions Drawn',
                        'team': team
                    })
                    events.append({
                        'player': numbers[1],
                        'event': 'Exclusions',
                        'team': 'dark' if team == 'light' else 'light'
                    })
                else:
                    events.append({
                        'player': numbers[0],
                        'event': 'Exclusions',
                        'team': team
                    })
                    events.append({
                        'player': numbers[1],
                        'event': 'Exclusions Drawn',
                        'team': 'dark' if team == 'light' else 'light'
                    })
            elif len(numbers) >= 1:
                # Handle single exclusion event
                events.append({
                    'player': numbers[0],
                    'event': 'Exclusions',
                    'team': team
                })

        if any(word in tokens[i:i+2] for word in penalty_keywords):
            if len(numbers) >= 2:
                events.append({
                    'player': numbers[0],
                    'event': 'Penalties',
                    'team': team
                })

    return events

def process_input(text):
    """Process input text and return all events."""
    return extract_events(text)

# Load training data from sentence.txt
sentences = []
labels = []

with open('sentence.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line and not line.startswith('#'):
            sentence = line[:-4]  # Remove the (+) or (-) indicator
            label = 1 if line.endswith('(+)') else 0
            sentences.append(sentence)
            labels.append(label)

df = pd.DataFrame({'Sentence': sentences, 'Label': labels})
X_train, X_test, y_train, y_test = train_test_split(df['Sentence'], df['Label'], test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_vec, y_train)

def predict_connotation(sentence):
    vec_sentence = vectorizer.transform([sentence])
    prediction = model.predict(vec_sentence)
    return 'Positive' if prediction == 1 else 'Negative'


def phrase(number, action, team):
    player_text = "goalie" if str(number) == "1" else f"#{number}"
    if action == 'Shot':
        return f"The {team} team {player_text} scored a goal"
    elif action == 'Shot Attempt':
        return f"The {team} team {player_text} attempted a shot"
    elif action == 'Blocks':
        return f"The {team} team {number} blocked the shot"
    elif action == 'Steals':
        return f"Steal by {team} {number}"
    elif action == 'Exclusions Drawn':
        return f"The {team} team {number} drew an exclusion"
    elif action == 'Exclusions':
        if action == 'Exclusions':
            return f"The {team} team {number} was excluded"
        elif action == 'Exclusions Drawn':
            return f"The {team} team {number} drew an exclusion"
    elif action == 'Turnovers':
        return f"Turnover on {team} {number}"
    elif action == 'Penalties':
        return f"The {team} team {number} got a 5-meter penalty"
    else:
        return f"The {team} team {number} performed {action}"

def run(text):
    events = extract_key_phrases(text)
    responses = []
    home_team_name = request.form.get('home_team')
    away_team_name = request.form.get('away_team')
    result_text = request.form.get('result')

    valid_events = False
    for player, event, team in events:
        if player and event and team:
            if sort_data(player, event, team, home_team_name, away_team_name):
                valid_events = True
            else:
                responses.append(f"Error: Player #{player} not found in {team} team ({home_team_name if team == 'dark' else away_team_name}) roster.")

    if not valid_events:
        return ""  # Return empty string instead of None to avoid error message
    return result_text if result_text else ""

@app.route('/')
def home():
    upcoming_games = []
    seen_games = set()
    today = datetime.now().date()

    # Get user's followed teams if logged in
    followed_teams = []
    if current_user.is_authenticated:
        followed_teams = json.loads(current_user.followed_teams)

    for school in schools.values():
        # Skip if user is logged in and this school's slug is not in followed_teams
        if current_user.is_authenticated:
            school_slug = next((slug for slug, s in schools.items() if s['name'] == school['name']), None)
            if school_slug not in followed_teams:
                continue

        team_data = load_team_data(school['name'])
        for game in team_data.get('games', []):
            # Skip if game is scored or in the past
            if game.get('is_scored') or datetime.strptime(game['date'], '%Y-%m-%d').date() < today:
                continue
            
            game_key = f"{game['date']}-{sorted([school['name'], game['opponent']])[0]}-{sorted([school['name'], game['opponent']])[1]}"
            if game_key not in seen_games:
                game['school_name'] = school['name']
                game['school_logo'] = school['logo']
                upcoming_games.append(game)
                seen_games.add(game_key)

    # Sort games by date and get the 6 most recent
    upcoming_games.sort(key=lambda x: x['date'])
    upcoming_games = upcoming_games[:6]

    return render_template('home.html', upcoming_games=upcoming_games, schools=schools)

# Render HTML page with two tables (initial zeros)


@app.route('/team/<school_slug>/score/<int:game_index>', methods=['GET'])
@login_required
def score_game(school_slug, game_index):
    school = get_school_by_slug(school_slug)
    if not school:
        return "School not found", 404

    team_name = school['name']
    game = open_game(team_name, game_index)
    if not game:
        return "Game not found", 404

    # Load rosters for both teams
    home_team = team_name if game['home_away'] == 'Home' else game['opponent']
    away_team = game['opponent'] if game['home_away'] == 'Home' else team_name

    home_roster = get_team_roster(home_team)
    away_roster = get_team_roster(away_team)

    # Extract player cap numbers
    home_players = [player['cap_number'] for player in home_roster]
    away_players = [player['cap_number'] for player in away_roster]

    # Initialize box scores for both teams
    home_box = {
        'Player': home_players,
        'Shot': [0] * len(home_players),
        'Blocks': [0] * len(home_players),
        'Steals': [0] * len(home_players),
        'Exclusions': [0] * len(home_players),
        'Exclusions Drawn': [0] * len(home_players),
        'Penalties': [0] * len(home_players),
        'Turnovers': [0] * len(home_players),
        'Sprint Won': [0] * len(home_players),
        'Sprint Attempt': [0] * len(home_players)
    }

    away_box = {
        'Player': away_players,
        'Shot': [0] * len(away_players),
        'Blocks': [0] * len(away_players),
        'Steals': [0] * len(away_players),
        'Exclusions': [0] * len(away_players),
        'Exclusions Drawn': [0] * len(away_players),
        'Penalties': [0] * len(away_players),
        'Turnovers': [0] * len(away_players),
        'Sprint Won': [0] * len(away_players),
        'Sprint Attempt': [0] * len(away_players)
    }

    # Pass box scores and other game data to the template
    # Get team colors and logos from schools dictionary
    home_school = next((school for school in schools.values() if school['name'] == home_team), None)
    away_school = next((school for school in schools.values() if school['name'] == away_team), None)

    return render_template(
        "score_game.html",
        home_team=home_team,
        away_team=away_team,
        game_index=game_index,
        home_box=home_box,
        away_box=away_box,
        home_team_color=home_school['bg_color'],
        home_team_text_color=home_school['text_color'],
        away_team_color=away_school['bg_color'],
        away_team_text_color=away_school['text_color'],
        home_team_logo=home_school['logo'],
        away_team_logo=away_school['logo']
    )


@app.route('/teams')
def teams():
    return render_template('teams.html')

@app.route('/search')
def search():
    return render_template('search.html')


# Process user input and return updated stats
@app.route('/process', methods=['POST'])
def process_text():
    try:
        text = request.form.get('text')
        game_id = request.form.get('game_id')
        home_team = request.form.get('home_team')
        away_team = request.form.get('away_team')
        game_time = request.form.get('game_time', 'Q1 N/A')  # Get game time or use default

        if not text:
            return jsonify({'error': 'No text provided'}), 400
        if not game_id:
            return jsonify({'error': 'No game ID provided'}), 400
        if not home_team or not away_team:
            return jsonify({'error': 'Team information missing'}), 400

        if game_id not in game_data:
            # Initialize game data if it doesn't exist
            game_data[game_id] = {
                'dataWhite': {'Player': [], 'Shot': [], 'Blocks': [], 'Steals': [], 'Exclusions': [], 
                             'Exclusions Drawn': [], 'Penalties': [], 'Turnovers': [], 'Sprint Won': [], 'Sprint Attempt': []},
                'dataBlack': {'Player': [], 'Shot': [], 'Blocks': [], 'Steals': [], 'Exclusions': [], 
                             'Exclusions Drawn': [], 'Penalties': [], 'Turnovers': [], 'Sprint Won': [], 'Sprint Attempt': []},
                'game_log': []
            }

        response = run(text, game_id)
        return jsonify({'response': response})
    except Exception as e:
        app.logger.error(f"Error processing text: {str(e)}")
        return jsonify({'error': f'Error processing text: {str(e)}'}), 500

def run(text, game_id):
    # Check if there's a time format at the beginning of the text (like "3:45 dark 1 scored")
    import re
    time_pattern = re.compile(r'^(\d+:\d+)\s+(.*)')
    time_match = time_pattern.match(text)
    
    custom_time = None
    if time_match:
        custom_time = time_match.group(1)
        text = time_match.group(2)  # Remove time from text for processing
    
    events = extract_key_phrases(text)
    responses = []
    home_team_name = request.form.get('home_team')
    away_team_name = request.form.get('away_team')
    game_time = request.form.get('game_time', 'Q1 N/A')  # Get game time from request

    # Get the raw result text from the request for game log
    result_text = request.form.get('result')
    
    # Check if we're in shootout mode
    is_shootout = game_time.startswith('SO') or 'shootout' in text.lower()
    
    # Override time if provided in input text
    if custom_time:
        # Keep the quarter part and replace the time part
        quarter_part = game_time.split(' ')[0]
        game_time = f"{quarter_part} {custom_time}"
    elif ' ' not in game_time:
        # Make sure we have a default N/A if no time part exists
        game_time = f"{game_time} N/A"

    for player, event, team in events:
        if player and event and team:
            # Only update stats if not in shootout mode
            if not is_shootout:
                stat_updated = sort_data(player, event, team, home_team_name, away_team_name, game_id)
            else:
                # For shootout, we'll skip updating stats but pretend it worked for the log entry
                stat_updated = True
                
            if stat_updated:
                log_entry = phrase(player, event, team)
                if log_entry not in responses:  # Prevent duplicate entries
                    responses.append(log_entry)

                # Initialize game logs if needed
                if game_id not in game_data:
                    game_data[game_id] = {'game_log': []}
                if 'game_log' not in game_data[game_id]:
                    game_data[game_id]['game_log'] = []

                # Format quarter display correctly (OT instead of QOT)
                quarter_part = game_time.split(' ')[0]
                if quarter_part.startswith('Q') and 'OT' in quarter_part:
                    # Extract OT number if present
                    ot_num = quarter_part[3:] if len(quarter_part) > 3 else ''
                    quarter_part = f"OT{ot_num}"
                time_part = game_time.split(' ')[1]
                formatted_game_time = f"{quarter_part} {time_part}"

                # Use the raw result text for the game log
                memory_entry = result_text if result_text else ' and '.join(responses)
                
                # Add goal type tags if needed
                if is_shootout and 'scored' in memory_entry.lower():
                    memory_entry += " [SHOOTOUT GOAL]"
                elif 'scored' in memory_entry.lower() and not is_shootout:
                    found_advantage = False
                    found_penalty = False
                    
                    if time_part != 'N/A':
                        try:
                            event_time = datetime.strptime(time_part, '%M:%S')
                            
                            # Look back in game log for recent exclusions or penalties
                            recent_events = game_data[game_id].get('game_log', [])[-4:]  # Look at last 4 events
                            scoring_team = 'dark' if 'dark' in log_entry.lower() else 'light'
                            
                            # Check for exclusion advantage first
                            for prev_event in recent_events:
                                if ' - ' in prev_event:
                                    try:
                                        prev_time_str = prev_event.split(' - ')[0].split(' ')[1]
                                        if prev_time_str != 'N/A':
                                            prev_time = datetime.strptime(prev_time_str, '%M:%S')
                                            time_diff = (event_time - prev_time).total_seconds()
                                            
                                            if 'excluded' in prev_event.lower() and time_diff <= 20:
                                                found_advantage = True
                                                break
                                    except (ValueError, IndexError):
                                        continue

                            # Check for correct penalty sequence if no advantage found
                            if not found_advantage and len(recent_events) >= 4:
                                penalty_team = None
                                exclusion_team = None
                                attempt_team = None
                                sequence_found = True

                                # Check penalty drawn event
                                if 'drew a penalty' in recent_events[0].lower() or 'drew penalty' in recent_events[0].lower():
                                    penalty_team = 'dark' if 'dark' in recent_events[0].lower() else 'light'
                                else:
                                    sequence_found = False

                                # Check exclusion event
                                if sequence_found and 'excluded' in recent_events[1].lower():
                                    exclusion_team = 'dark' if 'dark' in recent_events[1].lower() else 'light'
                                    if exclusion_team == penalty_team:
                                        sequence_found = False
                                else:
                                    sequence_found = False

                                # Check attempt event
                                if sequence_found and 'attempt' in recent_events[3].lower():
                                    attempt_team = 'dark' if 'dark' in recent_events[3].lower() else 'light'
                                    if attempt_team != penalty_team:
                                        sequence_found = False
                                else:
                                    sequence_found = False

                                found_penalty = sequence_found and penalty_team == scoring_team
                        except (ValueError, TypeError):
                            # If there's an error parsing the time, just continue
                            pass

                    if found_advantage:
                        log_entry += " [ADVANTAGE GOAL]"
                    elif found_penalty:
                        log_entry += " [PENALTY GOAL]"
                    else:
                        log_entry += " [NATURAL GOAL]"
                
                game_data[game_id]['game_log'].append(f"{formatted_game_time} - {memory_entry}")
            else:
                responses.append(f"Player {player} not found in roster.")

    return " and ".join(responses) if responses else "Could not parse the input."

# Helper function to load team rosters
def load_team_rosters():
    with open('team_rosters.json', 'r') as file:
        return json.load(file)

# Store game-specific data
game_data = {}

@app.route('/get_data', methods=['GET'])
@login_required
def get_data():
    try:
        # Load team rosters
        team_rosters = load_team_rosters()

        # Get team names and game ID from query parameters
        home_team_name = request.args.get('home_team')
        away_team_name = request.args.get('away_team')
        game_id = request.args.get('game_id')

        if not all([home_team_name, away_team_name, game_id]):
            return jsonify({'error': 'Missing required parameters'}), 400

        # Get rosters for both teams
        home_roster = team_rosters.get(home_team_name, [])
        away_roster = team_rosters.get(away_team_name, [])

        if not home_roster or not away_roster:
            return jsonify({'error': 'Team rosters not found'}), 404

        if game_id not in game_data:
            game_data[game_id] = {
                'dataWhite': {
                    'Player': [str(player['cap_number']) for player in away_roster],
                    'Shot': [0] * len(away_roster),
                    'Shot Attempt': [0] * len(away_roster),
                    'Assists': [0] * len(away_roster),
                    'Blocks': [0] * len(away_roster),
                    'Steals': [0] * len(away_roster),
                    'Exclusions': [0] * len(away_roster),
                    'Exclusions Drawn': [0] * len(away_roster),
                    'Penalties': [0] * len(away_roster),
                    'Turnovers': [0] * len(away_roster),
                    'Sprint Won': [0] * len(away_roster),
                    'Sprint Attempt': [0] * len(away_roster)
                },
                'dataBlack': {
                    'Player': [str(player['cap_number']) for player in home_roster],
                    'Shot': [0] * len(home_roster),
                    'Shot Attempt': [0] * len(home_roster),
                    'Assists': [0] * len(home_roster),
                    'Blocks': [0] * len(home_roster),
                    'Steals': [0] * len(home_roster),
                    'Exclusions': [0] * len(home_roster),
                    'Exclusions Drawn': [0] * len(home_roster),
                    'Penalties': [0] * len(home_roster),
                    'Turnovers': [0] * len(home_roster),
                    'Sprint Won': [0] * len(home_roster),
                    'Sprint Attempt': [0] * len(home_roster)
                }
            }

        if not home_team_name or not away_team_name:
            return jsonify({'error': 'Missing team names'}), 400

        # Get rosters for both teams
        home_roster = team_rosters.get(home_team_name, [])
        away_roster = team_rosters.get(away_team_name, [])

        # Get game-specific data
        if game_id in game_data:
            home_box = game_data[game_id]['dataBlack']  # Home team uses dataBlack
            away_box = game_data[game_id]['dataWhite']  # Away team uses dataWhite
        else:
            # Initialize new game data if not exists
            game_data[game_id] = {
                'dataWhite': {
                    'Player': [player['cap_number'] for player in away_roster],
                    'Shot': [0] * len(away_roster),
                    'Shot Attempt': [0] * len(away_roster),
                    'Assists': [0] * len(away_roster),
                    'Blocks': [0] * len(away_roster),
                    'Steals': [0] * len(away_roster),
                    'Exclusions': [0] * len(away_roster),
                    'Exclusions Drawn': [0] * len(away_roster),
                    'Penalties': [0] * len(away_roster),
                    'Turnovers': [0] * len(away_roster),
                    'Sprint Won': [0] * len(away_roster),
                    'Sprint Attempt': [0] * len(away_roster)
                },
                'dataBlack': {
                    'Player': [player['cap_number'] for player in home_roster],
                    'Shot': [0] * len(home_roster),
                    'Shot Attempt': [0] * len(home_roster),
                    'Assists': [0] * len(home_roster),
                    'Blocks': [0] * len(home_roster),
                    'Steals': [0] * len(home_roster),
                    'Exclusions': [0] * len(home_roster),
                    'Exclusions Drawn': [0] * len(home_roster),
                    'Penalties': [0] * len(home_roster),
                    'Turnovers': [0] * len(home_roster),
                    'Sprint Won': [0] * len(home_roster),
                    'Sprint Attempt': [0] * len(home_roster)
                }
            }
            home_box = game_data[game_id]['dataBlack']
            away_box = game_data[game_id]['dataWhite']

        # Return the current game data with roster cap numbers
        return jsonify({
            'home_box': home_box,
            'away_box': away_box
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/game_details/<int:game_id>', methods=['GET'])
def game_details(game_id):
    if game_id >= len(previous_games):
        return "Game not found", 404  # Handle the case when the game ID is invalid

    game = previous_games[game_id]

    # Check if the team names exist, else use default names
    white_team_name = game.get('white_team_name', 'White Team')
    black_team_name = game.get('black_team_name', 'Black Team')

    # Pass the team names to the template
    return render_template('game_details.html', game=game, game_id=game_id, 
                           white_team_name=white_team_name, 
                           black_team_name=black_team_name)





@app.route('/player_stats/<string:player_name>', methods=['GET'])
@login_required 
def player_stats(player_name):
    school_slug = request.args.get('school_slug')
    # Check if stats are private
    manager = User.query.filter_by(managed_team=school_slug, account_type='team_manager').first()
    if manager and manager.stats_private and (not current_user.is_authenticated or manager.id != current_user.id):
        flash('These statistics are private')
        return redirect(url_for('team_page', school_slug=school_slug))

    if not school_slug:
        # Search all schools for the player
        for slug, school_data in schools.items():
            roster = get_team_roster(school_data['name'])
            if any(player['name'] == player_name for player in roster):
                school_slug = slug
                break

    if not school_slug:
        return "Player not found in any school", 404

    school = schools.get(school_slug)
    if not school:
        return "School not found", 404

    # Initialize combined stats
    combined_stats = {
        'Shot': 0,
        'Shot Attempt': 0,
        'Blocks': 0,
        'Steals': 0,
        'Exclusions': 0,
        'Exclusions Drawn': 0,
        'Penalties': 0,
        'Turnovers': 0,
        'Sprint Won': 0,
        'Sprint Attempt': 0
    }

    # Get the player's info from roster
    team_name = school['name']
    roster = get_team_roster(team_name)
    player_info = next((player for player in roster if player['name'] == player_name), None)
    if not player_info:
        return "Player not found", 404

    cap_number = str(player_info['cap_number'])
    team_file = f'teams/CCS/SCVAL/team_{team_name.replace(" ", "_")}.json'

    try:
        with open(team_file, 'r') as file:
            team_data = json.load(file)

            for game in team_data.get('games', []):
                if not game.get('is_scored'):
                    continue

                # Determine which box to check based on whether team was home or away
                box_key = 'home_box' if game['home_away'] == 'Home' else 'away_box'
                box = game.get(box_key, {})

                if not box or 'Player' not in box:
                    continue

                try:
                    # Find player's index in this box score
                    player_index = box['Player'].index(cap_number)

                    # Add up all stats from this game
                    for stat_key in combined_stats:
                        if stat_key in box and isinstance(box[stat_key], list):
                            combined_stats[stat_key] += box[stat_key][player_index]
                except (ValueError, IndexError):
                    # Player not found in this box score, skip it
                    continue

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading team file {team_file}: {str(e)}")
        return f"Error loading team statistics: {str(e)}", 500

    return render_template('player_stats.html', player_name=player_name, stats=combined_stats, school_slug=school_slug)


# Simulate a dictionary of school data (name, logo, and colors)

# Simulate a dictionary of school data (name, logo, colors, and game schedule)
schools = {
    "sf-polo": {
        "name": "SF Polo",
        "logo": "images/sf.png",
        "bg_color": "#000080",  # Navy blue
        "text_color": "#ffffff",
        "link_color": "#000080",
        "league": "Club",
        "games": [],
        "players":[]
    },
    "sj-express": {
        "name": "SJ Express",
        "logo": "images/sj.jpeg",
        "bg_color": "#000000",
        "text_color": "#ffffff",
        "link_color": "#000000",
        "league": "Club",
        "games": [],
        "players":[]
    },
    "palo-alto": {
        "name": "Palo Alto High School",
        "logo": "images/Palo Alto High School.png",
        "bg_color": "#004b23",
        "text_color": "#ffffff",
        "link_color": "#004b23",
        "league": "SCVAL",
        "games": [],
        "players":[]
    },
    "gunn": {
        "name": "Henry M. Gunn High School",
        "logo": "images/Henry M. Gunn High School.png",
        "bg_color": "#ff0000",
        "text_color": "#000000",
        "link_color": "#ff0000",
        "games": [],
        "players":[]
    },
    "harker": {
        "name": "Harker High School",
        "logo": "images/Harker High School.png",
        "bg_color": "#004b23",
        "text_color": "#ffffff",
        "link_color": "#004b23",
        "games": [],
        "players":[]
    },
    "los-gatos": {
        "name": "Los Gatos High School",
        "logo": "images/Los Gatos High School.png",
        "bg_color": "#ffa500",
        "text_color": "#ffffff",
        "link_color": "#ffa500",
        "games": [],
        "players":[]
    },
    "los-altos": {
        "name": "Los Altos High School",
        "logo": "images/Los Altos High School.png",
        "bg_color": "#000080",
        "text_color": "#ffffff",
        "link_color": "#000080",
        "games": [],
        "players":[]
    },
    "fremont": {
        "name": "Fremont High School",
        "logo": "images/Fremont High School.png",
        "bg_color": "#8B0000",
        "text_color": "#ffffff",
        "link_color": "#8B0000",
        "games": [],
        "players":[]
    },
    "mountain-view": {
        "name": "Mountain View High School",
        "logo": "images/Mountain View High School.png",
        "bg_color": "#ffdb58",
        "text_color": "#000000",
        "link_color": "#ffdb58",
        "games": [],
        "players":[]
    }
}
def get_school_by_slug(school_slug):
    return schools.get(school_slug)
def get_used_cap_numbers(school):
    # Ensure 'players' exists and is a list
    if "players" in school and isinstance(school["players"], list):
        # Extract cap numbers, converting to strings
        used_numbers = [str(player["cap_number"]) for player in school["players"] if "cap_number" in player]
    else:
        used_numbers = []  # Default to an empty list if no players
    return used_numbers
def add_player_to_roster(school, cap_number, player_name,grade, position):
    new_player = {
        'cap_number': cap_number,
        'name': player_name,
        'grade': grade,
        'position': position
    }
    school["players"].append(new_player)








@app.route('/team/<school_slug>', methods=['GET', 'POST'])
def team_page(school_slug):
    school = schools.get(school_slug)
    if not school:
        return "Team not found", 404

    # Ensure the main team file exists
    initialize_team_file(school['name'])
    team_data = load_team_data(school['name'])

    if request.method == 'POST':
        # Retrieve form data for adding a new game
        home_away = request.form.get('home_away')
        opponent_name = request.form.get('opponent')
        time = request.form.get('time')
        date = request.form.get('date')
        game_type = request.form.get('game_type')

        # Find the opponent's slug and logo
        opponent_slug = next((slug for slug, data in schools.items() if data['name'] == opponent_name), None)
        opponent_logo = schools[opponent_slug]['logo'] if opponent_slug else ''

        # Create a new game entry
        new_game = {
            'home_away': home_away,
            'opponent': opponent_name,
            'opponent_logo': opponent_logo,
            'time': time,
            'date': date,
            'game_type': game_type,
            'is_scored': False
        }

        # Format the date for display
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        day_suffix = 'th' if 11 <= date_obj.day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(date_obj.day % 10, 'th')
        new_game['formatted_date'] = date_obj.strftime(f"%B {date_obj.day}{day_suffix}, %Y")

        # Add the game to the main team’s schedule and sort by date
        team_data["games"].append(new_game)
        team_data["games"].sort(key=lambda game: datetime.strptime(game["date"], '%Y-%m-%d'))
        save_team_data(school["name"], team_data)

        # Also add it to the opponent's schedule, with home/away reversed and sort
        if opponent_slug:
            opponent_data = load_team_data(opponent_name)
            opponent_game = {
                'home_away': 'Home' if home_away == 'Away' else 'Away',
                'opponent': school['name'],
                'opponent_logo': school['logo'],
                'time': time,
                'date': date,
                'game_type': game_type,
                'is_scored': False,
                'formatted_date': new_game['formatted_date']
            }
            opponent_data["games"].append(opponent_game)
            opponent_data["games"].sort(key=lambda game: datetime.strptime(game["date"], '%Y-%m-%d'))
            save_team_data(opponent_name, opponent_data)

        return redirect(url_for('team_page', school_slug=school_slug))

    # Ensure games are sorted by date in the backend data file
    if team_data["games"]:
        team_data["games"].sort(key=lambda game: datetime.strptime(game["date"], '%Y-%m-%d'))
        save_team_data(school["name"], team_data)  # Save sorted data back to the file

    # Render the template with sorted team data
    return render_template('teams.html', 
                         school=school, 
                         schools=schools, 
                         team_data=team_data, 
                         school_slug=school_slug,
                         get_team_roster=get_team_roster,
                         sort_cap_number=sort_cap_number)






@app.route('/team/<school_slug>/delete/<int:game_id>', methods=['POST'])
def delete_game(school_slug, game_id):
    school = schools.get(school_slug)
    if not school:
        return "Team not found", 404

    # Load team data from the file
    team_data = load_team_data(school["name"])

    # Check if the game_id is valid
    if 0 <= game_id < len(team_data["games"]):
        # Remove the game from the list
        deleted_game = team_data["games"].pop(game_id)
        save_team_data(school["name"], team_data)

        # Also remove the game from the opponent's schedule if the opponent exists
        opponent_name = deleted_game["opponent"]
        opponent_data = load_team_data(opponent_name)
        opponent_data["games"] = [
            game for game in opponent_data["games"]
            if not (game["opponent"] == school["name"] and game["date"] == deleted_game["date"])
        ]
        save_team_data(opponent_name, opponent_data)

    return redirect(url_for('team_page', school_slug=school_slug))



# @app.route('/score/<school_slug>', methods=['GET'])
# def scoring_page(school_slug):
#     # Pass the school_slug to get_school() to fetch the relevant school data
#     school = get_school(school_slug)

#     if not school:
#         return "School not found", 404

#     # Fetch game data or other relevant information
#     game = get_game(school_slug)  # Example function to get a game

#     return render_template('scoring_page.html', school=school, game=game)

@app.route('/team/<school_slug>/quick_score/<int:game_index>', methods=['GET', 'POST'])
def quick_score(school_slug, game_index):
    school = schools.get(school_slug)
    if not school:
        return "School not found", 404

    team_name = school['name']
    game = open_game(team_name, game_index)
    if not game:
        return "Game not found", 404

    home_team = team_name if game['home_away'] == 'Home' else game['opponent']
    away_team = game['opponent'] if game['home_away'] == 'Home' else team_name

    # Get team colors and logos
    home_school = next((school for school in schools.values() if school['name'] == home_team), None)
    away_school = next((school for school in schools.values() if school['name'] == away_team), None)

    if not home_school or not away_school:
        return "Team configuration not found", 404

    # Load rosters
    home_roster = get_team_roster(home_team)
    away_roster = get_team_roster(away_team)

    # Process form submission
    if request.method == 'POST':
        try:
            # Initialize game data
            home_box = {
                'Player': [str(p['cap_number']) for p in home_roster],
                'Shot': [0] * len(home_roster),
                'Shot Attempt': [0] * len(home_roster),
                'Assists': [0] * len(home_roster),
                'Blocks': [0] * len(home_roster),
                'Steals': [0] * len(home_roster),
                'Exclusions': [0] * len(home_roster),
                'Exclusions Drawn': [0] * len(home_roster),
                'Penalties': [0] * len(home_roster),
                'Turnovers': [0] * len(home_roster),
                'Sprint Won': [0] * len(home_roster),
                'Sprint Attempt': [0] * len(home_roster)
            }

            away_box = {
                'Player': [str(p['cap_number']) for p in away_roster],
                'Shot': [0] * len(away_roster),
                'Shot Attempt': [0] * len(away_roster),
                'Assists': [0] * len(away_roster),
                'Blocks': [0] * len(away_roster),
                'Steals': [0] * len(away_roster),
                'Exclusions': [0] * len(away_roster),
                'Exclusions Drawn': [0] * len(away_roster),
                'Penalties': [0] * len(away_roster),
                'Turnovers': [0] * len(away_roster),
                'Sprint Won': [0] * len(away_roster),
                'Sprint Attempt': [0] * len(away_roster)
            }

            # Update stats from form data
            for stat in ['Shot', 'Shot Attempt', 'Assists', 'Blocks', 'Steals', 'Exclusions', 'Exclusions Drawn', 'Turnovers', 'Sprint Won', 'Sprint Attempt']:
                for i in range(len(home_roster)):
                    form_value = request.form.get(f'home_{stat.replace(" ", "")}_{i}', 0)
                    home_box[stat][i] = int(form_value)
                for i in range(len(away_roster)):
                    form_value = request.form.get(f'away_{stat.replace(" ", "")}_{i}', 0)
                    away_box[stat][i] = int(form_value)

            # Load and update both teams' data
            initialize_team_file(home_team)
            initialize_team_file(away_team)
            home_team_data = load_team_data(home_team)
            away_team_data = load_team_data(away_team)

            # Update home team's game
            if game_index < len(home_team_data["games"]):
                home_game = home_team_data["games"][game_index]
                home_game["is_scored"] = True
                home_game["home_box" if game['home_away'] == 'Home' else "away_box"] = home_box
                home_game["away_box" if game['home_away'] == 'Home' else "home_box"] = away_box

                # Calculate and add score
                home_score = sum(home_box['Shot'])
                away_score = sum(away_box['Shot'])
                home_game["score"] = {
                    "home_team_score": home_score,
                    "away_team_score": away_score
                }

            # Find and update away team's corresponding game
            for idx, away_game in enumerate(away_team_data["games"]):
                if (away_game["opponent"] == home_team and 
                    away_game["date"] == home_game["date"]):
                    away_game["is_scored"] = True
                    away_game["home_box" if game['home_away'] != 'Home' else "away_box"] = home_box
                    away_game["away_box" if game['home_away'] != 'Home' else "home_box"] = away_box
                    away_game["score"] = {
                        "home_team_score": away_score if game['home_away'] != 'Home' else home_score,
                        "away_team_score": home_score if game['home_away'] != 'Home' else away_score
                    }
                    break

            # Save both teams' data
            save_team_data(home_team, home_team_data)
            save_team_data(away_team, away_team_data)

            return redirect(url_for('team_page', school_slug=school_slug))
        except Exception as e:
            print(f"Error saving game data: {str(e)}")
            return f"Error saving game data: {str(e)}", 500

    # Initialize box scores
    home_box = {
        'Player': [player['cap_number'] for player in home_roster],
        'Shot': [0] * len(home_roster),
        'Assists': [0] * len(home_roster),
        'Blocks': [0] * len(home_roster),
        'Steals': [0] * len(home_roster),
        'Exclusions': [0] * len(home_roster),
        'Exclusions Drawn': [0] * len(home_roster),
        'Penalties': [0] * len(home_roster),
        'Turnovers': [0] * len(home_roster),
        'Sprint Won': [0] * len(home_roster),
        'Sprint Attempt': [0] * len(home_roster)
    }

    away_box = {
        'Player': [player['cap_number'] for player in away_roster],
        'Shot': [0] * len(away_roster),
        'Assists': [0] * len(away_roster),
        'Blocks': [0] * len(away_roster),
        'Steals': [0] * len(away_roster),
        'Exclusions': [0] * len(away_roster),
        'Exclusions Drawn': [0] * len(away_roster),
        'Penalties': [0] * len(away_roster),
        'Turnovers': [0] * len(away_roster),
        'Sprint Won': [0] * len(away_roster),
        'Sprint Attempt': [0] * len(away_roster)
    }

    return render_template('quick_score.html',
                         home_team=home_team,
                         away_team=away_team,
                         game_index=game_index,
                         school_slug=school_slug,
                         home_team_color=home_school['bg_color'],
                         home_team_text_color=home_school['text_color'],
                         away_team_color=away_school['bg_color'],
                         away_team_text_color=away_school['text_color'],
                         home_box=home_box,
                         away_box=away_box)

@app.route('/team/<school_slug>/score/<int:game_index>', methods=['GET'])
def scoring_page(school_slug, game_index):
    school = schools.get(school_slug)
    if not school:
        return "School not found", 404

    team_name = school['name']
    team_data = load_team_data(team_name)

    if not team_data or "games" not in team_data or game_index >= len(team_data["games"]):
        return "Game not found", 404

    game = team_data["games"][game_index]
    home_team = team_name if game['home_away'] == 'Home' else game['opponent']
    away_team =game['opponent'] if game['home_away'] == 'Home' else team_name

    # Get team colors and logos
    home_school = next((school for school in schools.values() if school['name'] == home_team), None)
    away_school = next((school for school in schools.values() if school['name'] == away_team), None)

    if not home_school or not away_school:
        return "Team configuration not found", 404

    # Load rosters
    home_roster = get_team_roster(home_team)
    away_roster = get_team_roster(away_team)

    # Initialize box scores
    home_box = {
        'Player': [player['cap_number'] for player in home_roster],
        'Shot': [0] * len(home_roster),
        'Shot Attempt': [0] * len(home_roster),
        'Assists': [0] * len(home_roster),
        'Blocks': [0] * len(home_roster),
        'Steals': [0] * len(home_roster),
        'Exclusions': [0] * len(home_roster),
        'Exclusions Drawn': [0] * len(home_roster),
        'Penalties': [0] * len(home_roster),
        'Turnovers': [0] * len(home_roster),
        'Sprint Won': [0] * len(home_roster),
        'Sprint Attempt': [0] * len(home_roster)
    }

    away_box = {
        'Player': [player['cap_number'] for player in away_roster],
        'Shot': [0] * len(away_roster),
        'Shot Attempt': [0] * len(away_roster),
        'Assists': [0] * len(away_roster),
        'Blocks': [0] * len(away_roster),
        'Steals': [0] * len(away_roster),
        'Exclusions': [0] * len(away_roster),
        'Exclusions Drawn': [0] * len(away_roster),
        'Penalties': [0] * len(away_roster),
        'Turnovers': [0] * len(away_roster),
        'Sprint Won': [0] * len(away_roster),
        'Sprint Attempt': [0] * len(away_roster)
    }

    return render_template("score_game.html",
                         home_team=home_team,
                         away_team=away_team,
                         game_index=game_index,
                         school_slug=school_slug,
                         home_team_color=home_school['bg_color'],
                         home_team_text_color=home_school['text_color'],
                         away_team_color=away_school['bg_color'],
                         away_team_text_color=away_school['text_color'],
                         home_team_logo=home_school['logo'],
                         away_team_logo=away_school['logo'],
                         home_box=home_box,
                         away_box=away_box)

from flask import render_template

@app.route('/admin/users', methods=['GET'])
def view_users():
    users = User.query.all()
    print("\nAll registered users:")
    for user in users:
        print(f"Email: {user.email}")
        print(f"Account type: {user.account_type}")
        print(f"Created: {user.created_at}")
        print("-" * 40)
    return render_template('admin_users.html', users=users)

@app.route('/team/<school_slug>/view/<int:game_index>', methods=['GET'])
@login_required
def view_scoring(school_slug, game_index):
    # Get manager info but don't block access
    manager = User.query.filter_by(managed_team=school_slug, account_type='team_manager').first()
    try:
        # Get school data
        school = schools.get(school_slug)
        if not school:
            return "School not found", 404
            
        team_name = school['name']
        
        # Load team rosters from team_rosters.json
        with open('team_rosters.json', 'r') as file:
            team_rosters = json.load(file)

        # Fetch school and game data
        school_name = school['name']
        team_data = load_team_data(school_name)
        
        if game_index >= len(team_data.get("games", [])):
            return "Game not found", 404
            
        game = team_data["games"][game_index]
        if not game:
            return "Game not found", 404

        # Determine home and away teams
        home_team_name = school_name if game['home_away'] == 'Home' else game['opponent']
        away_team_name = game['opponent'] if game['home_away'] == 'Home' else school_name

        # Load rosters for home and away teams
        home_roster = team_rosters.get(home_team_name, [])
        away_roster = team_rosters.get(away_team_name, [])

        # Extract cap numbers and player names for both teams
        home_players = [{'cap_number': player['cap_number'], 'name': player['name']} for player in home_roster]
        away_players = [{'cap_number': player['cap_number'], 'name': player['name']} for player in away_roster]

        # Box scores are stored based on the colors, not the team names
        # home_box is always the black team (dark), away_box is always the white team (light)
        black_team_stats = game.get('home_box', {})
        white_team_stats = game.get('away_box', {})
        
        # We don't need to flip the box based on home/away - they're already stored correctly
        # The view assignment is based on the position in the template (home is always black/dark)
        print(f"Viewing game: {home_team_name} (home/black) vs {away_team_name} (away/white)")
        
        print("Black team stats:", black_team_stats)
        print("White team stats:", white_team_stats)

        # Ensure the required stats fields exist in both team stats
        required_fields = ['Player', 'Shot', 'Shot Attempt', 'Assists', 'Blocks', 'Steals', 
                          'Exclusions', 'Exclusions Drawn', 'Penalties', 'Turnovers', 
                          'Sprint Won', 'Sprint Attempt']
        
        # Initialize missing fields in white team stats
        if not white_team_stats or not isinstance(white_team_stats, dict):
            white_team_stats = {}
        for field in required_fields:
            if field not in white_team_stats:
                if field == 'Player':
                    white_team_stats[field] = [p['cap_number'] for p in away_players]
                else:
                    white_team_stats[field] = [0] * len(away_players)
        
        # Initialize missing fields in black team stats
        if not black_team_stats or not isinstance(black_team_stats, dict):
            black_team_stats = {}
        for field in required_fields:
            if field not in black_team_stats:
                if field == 'Player':
                    black_team_stats[field] = [p['cap_number'] for p in home_players]
                else:
                    black_team_stats[field] = [0] * len(home_players)
                    
        # Ensure game log exists
        if 'game_log' not in game:
            game['game_log'] = []
            
        # Ensure score exists or calculate it
        if 'score' not in game:
            game['score'] = {
                'white_team_score': sum(white_team_stats.get('Shot', [])),
                'black_team_score': sum(black_team_stats.get('Shot', [])),
                'game_type': "(SO)" if game.get('is_shootout') else ""
            }
        
        # Print for debugging
        print("Game score:", game.get('score'))
        print("Game log entries:", len(game.get('game_log', [])))

        # Check privacy settings for both teams
        home_team_slug = next((slug for slug, s in schools.items() if s['name'] == home_team_name), None)
        away_team_slug = next((slug for slug, s in schools.items() if s['name'] == away_team_name), None)
        
        home_manager = User.query.filter_by(managed_team=home_team_slug, account_type='team_manager').first()
        away_manager = User.query.filter_by(managed_team=away_team_slug, account_type='team_manager').first()

        home_stats_private = home_manager.stats_private if home_manager else False
        away_stats_private = away_manager.stats_private if away_manager else False

        return render_template(
            "view_game.html",
            game=game,
            home_players=home_players,
            away_players=away_players,
            white_team_stats=white_team_stats,
            black_team_stats=black_team_stats,
            home_team=home_team_name,
            away_team=away_team_name,
            school_slug=school_slug,
            game_index=game_index,
            home_stats_private=home_stats_private,
            away_stats_private=away_stats_private,
            current_user_id=current_user.id if current_user.is_authenticated else None,
            home_manager_id=home_manager.id if home_manager else None,
            away_manager_id=away_manager.id if away_manager else None
        )
    except Exception as e:
        print(f"Error in view_scoring: {str(e)}")
        return f"Error: {str(e)}", 500







@app.route('/end_game', methods=['POST'])
def end_game():
    try:
        # Get required form data with validation
        print("End game form data:", request.form)
        
        school_slug = request.form.get('school_slug')
        game_id = request.form.get('game_index')
        white_team_name = request.form.get('white_team_name') 
        black_team_name = request.form.get('black_team_name')
        redirect_url = request.args.get('redirect') or f'/team/{school_slug}'
        
        print(f"Game ID: {game_id}, White Team: {white_team_name}, Black Team: {black_team_name}")
        
        if not all([game_id, white_team_name, black_team_name]):
            # Redirect to team page if we have the redirect URL
            if redirect_url:
                return redirect(redirect_url)
            elif school_slug:
                return redirect(url_for('team_page', school_slug=school_slug))
            return jsonify({'error': 'Missing required game data'}), 400
            
        game_index = int(game_id)
        current_quarter = request.form.get('current_quarter', '')
        
        # Get scores from the form
        white_score = float(request.form.get('away_score', '0'))
        black_score = float(request.form.get('home_score', '0'))
        
        # Initialize both team files
        initialize_team_file(white_team_name)
        initialize_team_file(black_team_name)

        # Load team data for both teams
        white_team_data = load_team_data(white_team_name)
        black_team_data = load_team_data(black_team_name)
        
        # Find the corresponding games in both teams' data
        white_game_index = None
        black_game_index = None
        game_date = None
        
        # Try to find the original game that was being scored
        original_game = None
        original_team_data = None
        
        # First try to find game in the black team's (home team) data if game_index is valid
        if black_team_data and "games" in black_team_data and game_index < len(black_team_data.get("games", [])):
            original_game = black_team_data["games"][game_index]
            original_team_data = black_team_data
            black_game_index = game_index
            game_date = original_game.get("date")
            
        # If not found in black team, try white team (away team)
        if original_game is None and white_team_data and "games" in white_team_data and game_index < len(white_team_data.get("games", [])):
            original_game = white_team_data["games"][game_index]  
            original_team_data = white_team_data
            white_game_index = game_index
            game_date = original_game.get("date")
        
        # Exit if we couldn't find the original game
        if original_game is None or game_date is None:
            print(f"Original game not found for index {game_index}")
            return jsonify({'error': 'Original game not found'}), 404
            
        # Now find the corresponding games in both teams' data based on date and opponent
        # If white_game_index is still None, find it in white_team_data
        if white_game_index is None:
            for i, game in enumerate(white_team_data.get("games", [])):
                if (game.get("date") == game_date and 
                    game.get("opponent") == black_team_name):
                    white_game_index = i
                    break
                    
        # If black_game_index is still None, find it in black_team_data
        if black_game_index is None:
            for i, game in enumerate(black_team_data.get("games", [])):
                if (game.get("date") == game_date and 
                    game.get("opponent") == white_team_name):
                    black_game_index = i
                    break
        
        # Exit if we still couldn't find both games
        if white_game_index is None or black_game_index is None:
            print(f"Couldn't find corresponding games: white_index={white_game_index}, black_index={black_game_index}")
            print(f"White team: {white_team_name}, Black team: {black_team_name}, Date: {game_date}")
            return jsonify({'error': 'Corresponding games not found'}), 404
            
        print(f"Found games - White team index: {white_game_index}, Black team index: {black_game_index}")
        
        # Get the current memory-based game data
        current_game_data = game_data.get(game_id, {})
        
        # Determine if this was a shootout
        is_shootout = current_quarter == 'SO'
        
        # Get the complete game log from memory
        current_game_log = current_game_data.get('game_log', [])
        
        # Save the game log to text file for backup
        game_log_dir = 'game_logs'
        os.makedirs(game_log_dir, exist_ok=True)
        log_filename = f"{game_log_dir}/{white_team_name}_{black_team_name}_{game_date}.txt"
        
        with open(log_filename, 'w') as f:
            f.write('\n'.join(current_game_log))
            
        # Set up box scores
        white_box = current_game_data.get('dataWhite', {})
        black_box = current_game_data.get('dataBlack', {})
        
        # Prepare score information
        if is_shootout:
            # Keep decimal places for shootout
            white_score = round(white_score, 1)
            black_score = round(black_score, 1)
        else:
            # Convert to integer for regular/OT games
            white_score = int(white_score)
            black_score = int(black_score)
        
        # Score info for white team's perspective (away team)
        white_score_info = {
            "white_team_score": white_score,
            "black_team_score": black_score,
            "game_type": "(SO)" if is_shootout else f"({current_quarter})" if "OT" in str(current_quarter) else ""
        }
        
        # Score info for black team's perspective (home team)
        black_score_info = {
            "white_team_score": white_score,
            "black_team_score": black_score,
            "game_type": white_score_info["game_type"]
        }
        
        # Ensure all required fields exist in box scores
        required_fields = ['Player', 'Shot', 'Shot Attempt', 'Assists', 'Blocks', 'Steals', 
                        'Exclusions', 'Exclusions Drawn', 'Penalties', 'Turnovers', 
                        'Sprint Won', 'Sprint Attempt']
        
        for field in required_fields:
            if field not in white_box:
                white_box[field] = [0] * len(white_box.get('Player', []))
            if field not in black_box:
                black_box[field] = [0] * len(black_box.get('Player', []))
        
        # Update white team's game (away team)
        white_team_data["games"][white_game_index].update({
            "is_scored": True,
            "is_shootout": is_shootout,
            "score": white_score_info,
            "game_log": current_game_log.copy(),  # Make a copy to ensure independence
            "game_log_file": log_filename,
            "away_box": white_box,
            "home_box": black_box
        })
        
        # Update black team's game (home team) 
        black_team_data["games"][black_game_index].update({
            "is_scored": True,
            "is_shootout": is_shootout,
            "score": black_score_info,
            "game_log": current_game_log.copy(),  # Make a copy to ensure independence
            "game_log_file": log_filename,
            "home_box": black_box,
            "away_box": white_box
        })
        
        # Save updated data for both teams
        save_team_data(white_team_name, white_team_data)
        save_team_data(black_team_name, black_team_data)
        
        # Clear game data from memory
        if game_id in game_data:
            del game_data[game_id]
        
        print(f"Game successfully marked as scored for both teams. Redirecting to: {redirect_url}")
        
        # Redirect to the specified URL or team page
        return redirect(redirect_url)
        
    except Exception as e:
        print(f"Error in end_game: {str(e)}")
        
        # Try to extract school_slug even in case of error
        school_slug = request.form.get('school_slug')
        if school_slug:
            print(f"Redirecting to team page for {school_slug} despite error")
            return redirect(url_for('team_page', school_slug=school_slug))
        
        return jsonify({'error': str(e)}), 500

# File path for storing rosters
ROSTER_FILE = 'team_rosters.json'


# Load rosters from file
def load_rosters():
    if os.path.exists(ROSTER_FILE):
        with open(ROSTER_FILE, 'r') as file:
            return json.load(file)
    else:
        return {}

# Save rosters to file
def save_rosters(rosters):
    with open(ROSTER_FILE, 'w') as file:
        json.dump(rosters, file, indent=4)

# Get the roster for a specific team
def get_team_roster(team_name):
    rosters = load_rosters()
    return rosters.get(team_name, [])


def save_roster(team_name, updated_roster):
    rosters = load_rosters()
    rosters[team_name] = updated_roster
    save_rosters(rosters)


@app.route('/team/<school_slug>/edit_roster', methods=['GET', 'POST'])
@login_required
def edit_roster(school_slug):
    school = get_school_by_slug(school_slug)
    if not school:
        return "School not found", 404

    if not current_user.is_admin and not current_user.account_type == 'team_manager':
        flash('Only team managers can edit rosters')
        return redirect(url_for('team_page', school_slug=school_slug))

    if not current_user.is_admin:
        if not current_user.managed_team:
            flash('You are not assigned to manage any team')
            return redirect(url_for('team_page', school_slug=school_slug))
        if current_user.managed_team != school_slug:
            flash(f'Not permitted - you can only edit the roster for {schools[current_user.managed_team]["name"]}')
            return redirect(url_for('team_page', school_slug=school_slug))

    team_name = school['name']  # Fetch team name
    roster = get_team_roster(team_name)  # Load the team's roster
    used_numbers = [player['cap_number'] for player in roster]  # Get used cap numbers

    if request.method == 'POST':
        if 'delete_cap_number' in request.form:  # Handle delete request
            delete_cap_number = request.form.get('delete_cap_number')

            # Update game data to preserve stats when removing player
            team_data = load_team_data(team_name)
            for game in team_data.get('games', []):
                if game.get('is_scored'):
                    box_key = 'home_box' if game['home_away'] == 'Home' else 'away_box'
                    if box_key in game:
                        box = game[box_key]
                        if 'Player' in box:
                            try:
                                # Find index of player to delete
                                del_idx = box['Player'].index(delete_cap_number)

                                # Remove player's stats while preserving other players' stats
                                for stat in ['Shot', 'Shot Attempt', 'Assists', 'Blocks', 'Steals',
                                           'Exclusions', 'Exclusions Drawn', 'Penalties', 'Turnovers', 'Sprint Won', 'Sprint Attempt']:
                                    if stat in box and isinstance(box[stat], list):
                                        box[stat].pop(del_idx)

                                # Remove player from Player list
                                box['Player'].pop(del_idx)
                            except ValueError:
                                pass  # Player not found in this game's box score

            # Update roster and save
            roster = [player for player in roster if player['cap_number'] != delete_cap_number]
            save_roster(team_name, sorted(roster, key=lambda x: sort_cap_number(x['cap_number'])))
            save_team_data(team_name, team_data)

            return redirect(url_for('edit_roster', school_slug=school_slug))

        else:  # Handle adding a player
            cap_number = request.form.get('cap_number')
            player_name = request.form.get('player_name')
            grade = request.form.get('grade')
            position = request.form.get('position')

            # Check if the cap number is already taken
            if cap_number in used_numbers:
                error_message = f"Cap number {cap_number} is already taken. Please choose a different number."
                return render_template('edit_roster.html', school_slug=school_slug, school=school, roster=roster, error=error_message)

            # Add the player to the roster
            new_player = {
                'cap_number': cap_number,
                'name': player_name,
                'grade': grade,
                'position': position
            }
            roster.append(new_player)  # Add to the local roster

            # Sort the roster and save
            roster = sorted(roster, key=lambda x: sort_cap_number(x['cap_number']))
            save_roster(team_name, roster)  # Save the updated and sorted roster

            # Update game data
            team_data = load_team_data(team_name)
            for game in team_data.get('games', []):
                if game.get('is_scored'):
                    box_key = 'home_box' if game['home_away'] == 'Home' else 'away_box'
                    if box_key in game:
                        old_box = game[box_key].copy()
                        old_players = old_box.get('Player', [])

                        # Initialize new box with updated roster
                        new_box = {
                            'Player': [p['cap_number'] for p in roster],
                            'Shot': [0] * len(roster),
                            'Shot Attempt': [0] * len(roster),
                            'Assists': [0] * len(roster),
                            'Blocks': [0] * len(roster),
                            'Steals': [0] * len(roster),
                            'Exclusions': [0] * len(roster),
                            'Exclusions Drawn': [0] * len(roster),
                            'Penalties': [0] * len(roster),
                            'Turnovers': [0] * len(roster),
                            'Sprint Won': [0] * len(roster),
                            'Sprint Attempt': [0] * len(roster)
                        }

                        # Map existing stats to new roster positions
                        for new_idx, cap_number in enumerate(new_box['Player']):
                            if cap_number in old_players:
                                old_idx = old_players.index(cap_number)
                                for stat in ['Shot', 'Shot Attempt', 'Assists', 'Blocks', 'Steals',
                                           'Exclusions', 'Exclusions Drawn', 'Penalties', 'Turnovers', 'Sprint Won', 'Sprint Attempt']:
                                    if stat in old_box and old_idx < len(old_box[stat]):
                                        new_box[stat][new_idx] = old_box[stat][old_idx]

                        game[box_key] = new_box

            save_team_data(team_name, team_data)

            # Redirect to the same page to refresh the roster
            return redirect(url_for('edit_roster', school_slug=school_slug))

    # Sort the roster before rendering
    roster = sorted(roster, key=lambda x: sort_cap_number(x['cap_number']))
    return render_template('edit_roster.html', school_slug=school_slug, school=school, roster=roster)

# Helper function to sort cap numbers
def sort_cap_number(cap_number):
    """
    Sort cap numbers by treating numeric and alphanumeric values appropriately.
    Goalies (1, 1A, 1B, etc.) come first, then others in ascending order.
    """
    import re
    match = re.match(r"(\d+)([A-Z]?)", cap_number)  # Match numbers followed by optional letters
    if match:
        number = int(match.group(1))  # Extract the numeric part
        letter = match.group(2)  # Extract the optional letter
        # Assign priority to goalies (1) and sort by letter (A, B, etc.)
        return (number, letter or "")
    return (float('inf'), "")  # Handle unexpected cap numbers by placing them at the end








def format_time(time_str):
    if not time_str:
        return "TBD"
    try:
        # Parse the time string
        time_obj = datetime.strptime(time_str, '%H:%M')
        # Format in 12-hour format with AM/PM
        return time_obj.strftime('%I:%M %p')
    except ValueError:
        return time_str

# Register the filter with the Jinja environment
app.jinja_env.filters['format_time'] = format_time

@app.template_filter('from_json')
def from_json(value):
    return json.loads(value) if value else []

if __name__ == '__main__':
    print("Starting Flask app...")
    app.debug = True
    app.logger.setLevel('DEBUG')
    app.run(host='0.0.0.0', port=5000)


#blocks a shot and scores a point doesn't work properly
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))

        followed_teams = request.form.getlist('followed_teams')
        user = User(
            email=email,
            password=generate_password_hash(request.form['password']),
            first_name=request.form['first_name'],
            last_name=request.form['last_name'],
            date_of_birth=datetime.strptime(request.form['date_of_birth'], '%Y-%m-%d'),
            high_school=request.form['high_school'],
            account_type=request.form['account_type'],
            followed_teams=json.dumps(followed_teams)
        )

        if user.account_type == 'team_manager':
            user.role = request.form['role']
            user.phone = request.form['phone']
            user.managed_team = request.form['managed_team']

        user.confirmation_token = secrets.token_urlsafe(32)
        db.session.add(user)
        db.session.commit()

        # Send confirmation email
        confirm_url = url_for('confirm_email', token=user.confirmation_token, _external=True)
        msg = Message('Confirm Your Account',
                     sender='noreply@waterpolostats.com',
                     recipients=[user.email])
        msg.html = f'''
        <div style="font-family: Arial, sans-serif;">
            <p>Hello {user.first_name},</p>
            <p>Please confirm your account by clicking on the link below:</p>
            <p><a href="{confirm_url}">Confirm Account</a></p>
            <p>Sincerely,<br>The Water Polo Stats Team</p>
            <p style="color: #666; font-size: 12px;">This is an automated message. Please do not reply to this email.</p>
        </div>
        '''
        mail.send(msg)

        flash('Registration successful. Please check your email to confirm your account.')
        return redirect(url_for('login'))

    return render_template('register.html', schools=schools)

@app.route('/confirm/<token>')
def confirm_email(token):
    user = User.query.filter_by(confirmation_token=token).first_or_404()
    user.email_confirmed = True
    user.confirmation_token = None
    db.session.commit()
    flash('Your account has been confirmed. Please login.')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and check_password_hash(user.password, request.form['password']):
            if not user.email_confirmed:
                flash('Please confirm your email before logging in.')
                return redirect(url_for('login'))
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/delete-account', methods=['POST'])
@login_required
def delete_account():
    user = User.query.get(current_user.id)
    db.session.delete(user)
    db.session.commit()
    logout_user()
    return '', 200

def get_team_manager(school_slug):
    return User.query.filter_by(managed_team=school_slug, account_type='team_manager').first()

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        if current_user.is_admin:
            # Handle privacy settings for all teams
            for slug in schools.keys():
                privacy_key = f'team_privacy_{slug}'
                manager = get_team_manager(slug)
                if manager:
                    new_setting = request.form.get(privacy_key) == 'on'
                    if manager.stats_private != new_setting:
                        # Update manager's privacy setting in database
                        manager.stats_private = new_setting
                        db.session.add(manager)
                        db.session.commit()

                        # Update team_permissions.json
                        permissions = load_team_permissions()
                        if slug in permissions:
                            permissions[slug]['stats_private'] = new_setting
                            save_team_permissions(permissions)
        elif current_user.account_type == 'team_manager':
            current_user.stats_private = 'stats_private' in request.form
            db.session.add(current_user)
            db.session.commit()
        if 'profile_image' in request.files:
            file = request.files['profile_image']
            if file and file.filename and file.filename != '':
                try:
                    filename = secure_filename(file.filename)
                    upload_path = os.path.join('static', 'images', 'profiles')
                    os.makedirs(upload_path, exist_ok=True)
                    file_path = os.path.join(upload_path, filename)
                    file.save(file_path)
                    current_user.profile_image = os.path.join('images', 'profiles', filename)
                except Exception as e:
                    flash(f'Error uploading file: {str(e)}')
                    return redirect(url_for('profile'))
                db.session.commit()
        elif 'selected_logo' in request.form and request.form['selected_logo']:
            current_user.profile_image = request.form['selected_logo']
            db.session.commit()

        followed_teams = request.form.getlist('followed_teams')
        current_user.followed_teams = json.dumps(followed_teams)
        db.session.commit()
        flash('Profile updated successfully')
        return redirect(url_for('home'))

    return render_template('profile.html', 
                         schools=schools,
                         followed_teams=json.loads(current_user.followed_teams),
                         get_team_manager=get_team_manager)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if not all([app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD']]):
        flash('Email service is not configured. Please contact administrator.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            email = request.form['email']
            user = User.query.filter_by(email=email).first()
            if user:
                # Generate reset token
                reset_token = secrets.token_urlsafe(32)
                user.reset_token = reset_token
                user.reset_token_expiry = datetime.utcnow() + timedelta(hours=1)
                db.session.commit()

                try:
                    # Send reset email
                    reset_url = url_for('reset_password', token=reset_token, _external=True)
                    msg = Message('Reset Your Password',
                                sender='noreply@waterpolostats.com',
                                recipients=[user.email])
                    msg.html = f'''
                    <div style="font-family: Arial, sans-serif;">
                        <p>Hello {user.first_name},</p>
                        <p>To reset your password, click the link below:</p>
                        <p><a href="{reset_url}">Reset Password</a></p>
                        <p>This link will expire in 1 hour.</p>
                        <p>Sincerely,<br>The Water Polo Stats Team</p>
                        <p style="color: #666; font-size: 12px;">This is an automated message. Please do not reply to this email.</p>
                    </div>
                    '''
                    mail.send(msg)
                    flash('Password reset instructions have been sent to your email.')
                    return redirect(url_for('login'))
                except Exception as e:
                    app.logger.error(f"Failed to send email: {str(e)}")
                    flash('Failed to send reset email. Please try again later.')
                    return redirect(url_for('forgot_password'))
        except Exception as e:
            app.logger.error(f"Error in forgot password: {str(e)}")
            flash('An error occurred. Please try again.')
            return redirect(url_for('forgot_password'))
        flash('Email address not found.')
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    if not user or not user.reset_token_expiry or user.reset_token_expiry < datetime.utcnow():
        flash('Invalid or expired reset link.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.')
            return render_template('reset_password.html')

        user.password = generate_password_hash(password)
        user.reset_token = None
        user.reset_token_expiry = None
        db.session.commit()

        flash('Your password has been reset. Please login with your new password.')
        return redirect(url_for('login'))

    return render_template('reset_password.html')