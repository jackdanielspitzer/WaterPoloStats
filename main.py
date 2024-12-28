
# Imports
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from word2number import w2n
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import spacy
import json
import os
import pandas as pd

# Flask app initialization
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Constants
FILE_PATH = 'team_data.json'
ROSTER_FILE = 'team_rosters.json'
GAMES_FILE = 'previous_games.json'

# Database Models
class School(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    slug = db.Column(db.String(100), unique=True, nullable=False)
    logo = db.Column(db.String(100))

    def __repr__(self):
        return f'<School {self.name}>'

# Game Stats Data Structures
dataWhite = {
    'Player': ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5', 'Player 6', 'Player 7', 'Player 8', 'Player 9', 'Player 10'],
    'Shot': [0] * 10,
    'Blocks': [0] * 10,
    'Steals': [0] * 10,
    'Exclusions': [0] * 10,
    'Exclusions Drawn': [0] * 10,
    'Penalties': [0] * 10,
    'Turnovers': [0] * 10
}

dataBlack = {
    'Player': ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5', 'Player 6', 'Player 7', 'Player 8', 'Player 9', 'Player 10'],
    'Shot': [0] * 10,
    'Blocks': [0] * 10,
    'Steals': [0] * 10,
    'Exclusions': [0] * 10,
    'Exclusions Drawn': [0] * 10,
    'Penalties': [0] * 10,
    'Turnovers': [0] * 10
}

# File Operations Functions
def get_team_file_path(team_name):
    return os.path.join('teams', f"team_{team_name.replace(' ', '_')}.json")

def load_data():
    try:
        with open(FILE_PATH, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_data_to_file(data, filename='data.json'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def initialize_team_file(team_name):
    if not os.path.exists('teams'):
        os.makedirs('teams')
    team_file_path = get_team_file_path(team_name)
    if not os.path.exists(team_file_path):
        with open(team_file_path, 'w') as file:
            json.dump({"games": []}, file, indent=4)

def load_team_data(team_name):
    team_file_path = get_team_file_path(team_name)
    if os.path.exists(team_file_path):
        try:
            with open(team_file_path, 'r') as file:
                if os.path.getsize(team_file_path) == 0:
                    return {"games": []}
                return json.load(file)
        except json.JSONDecodeError:
            return {"games": []}
    return {"games": []}

def save_team_data(team_name, data):
    team_file_path = get_team_file_path(team_name)
    with open(team_file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Roster Management Functions
def load_rosters():
    if os.path.exists(ROSTER_FILE):
        with open(ROSTER_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_rosters(rosters):
    with open(ROSTER_FILE, 'w') as file:
        json.dump(rosters, file, indent=4)

def get_team_roster(team_name):
    rosters = load_rosters()
    return rosters.get(team_name, [])

def save_roster(team_name, updated_roster):
    rosters = load_rosters()
    rosters[team_name] = updated_roster
    save_rosters(rosters)

def sort_cap_number(cap_number):
    import re
    match = re.match(r"(\d+)([A-Z]?)", cap_number)
    if match:
        number = int(match.group(1))
        letter = match.group(2)
        return (number, letter or "")
    return (float('inf'), "")

# Game Processing Functions
def reset_team_stats():
    global dataWhite, dataBlack
    dataWhite = {k: [0] * 10 if k != 'Player' else dataWhite['Player'] for k in dataWhite}
    dataBlack = {k: [0] * 10 if k != 'Player' else dataBlack['Player'] for k in dataBlack}

def extract_key_phrases(text):
    doc = nlp(text.lower())
    team, player, event = None, None, None
    
    dark_keywords = ['dark','black','blue']
    light_keywords = ['light','white']
    shot_keywords = ['goal', 'shot', 'score', 'point','scored','scores']
    block_keywords = ['block', 'blocked','blocks']
    steal_keywords = ['steal','stole','took','steals']
    exclusion_keywords = ['exclusion', 'kickout','excluded']
    turnover_keywords = ['turnover', 'foul','lost','loses']
    penalty_keywords = ['penalty', 'five meter']

    tokens = [token.text for token in doc]

    i = 0
    while i < len(tokens):
        token = tokens[i]
        try:
            if token == "five" and i + 1 < len(tokens) and tokens[i + 1] == "meter":
                event = "Penalties"
                i += 2
                continue
            elif token != "point":
                player = str(w2n.word_to_num(token))
        except Exception:
            pass

        if (token == "kick" or token == "kicked") and tokens[i+1] == "out":
            event = "Exclusions"

        if (token == "turn" or token == "turned") and tokens[i+1] == "over":
            event = "Turnovers"

        if token in dark_keywords:
            team = 'dark'
        elif token in light_keywords:
            team = 'light'

        if token == 'goalie':
            player = 1

        if (token in shot_keywords and event != "Blocks"):
            event = "Shot"
        elif token in block_keywords:
            event = "Blocks"
        elif token in steal_keywords:
            event = "Steals"
        elif token in exclusion_keywords:
            event = "Exclusions"
        elif token in turnover_keywords:
            event = "Turnovers"
        elif token in penalty_keywords:
            event = "Penalties"

        i += 1
    
    if event == "Exclusions" or event == "Penalties":
        if predict_connotation(text) == "Negative":
            event = "Exclusions"
        elif predict_connotation(text) == "Positive" and event == "Penalties":
            event = "Penalties"
        elif predict_connotation(text) == "Positive" and event == "Exclusions":
            event = "Exclusions Drawn"

    return player, event, team

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/team/<school_slug>', methods=['GET', 'POST'])
def team_page(school_slug):
    school = schools.get(school_slug)
    if not school:
        return "Team not found", 404

    initialize_team_file(school['name'])
    team_data = load_team_data(school['name'])

    if request.method == 'POST':
        home_away = request.form.get('home_away')
        opponent_name = request.form.get('opponent')
        time = request.form.get('time')
        date = request.form.get('date')
        game_type = request.form.get('game_type')

        opponent_slug = next((slug for slug, data in schools.items() if data['name'] == opponent_name), None)
        opponent_logo = schools[opponent_slug]['logo'] if opponent_slug else ''

        new_game = {
            'home_away': home_away,
            'opponent': opponent_name,
            'opponent_logo': opponent_logo,
            'time': time,
            'date': date,
            'game_type': game_type,
            'is_scored': False
        }

        date_obj = datetime.strptime(date, '%Y-%m-%d')
        day_suffix = 'th' if 11 <= date_obj.day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(date_obj.day % 10, 'th')
        new_game['formatted_date'] = date_obj.strftime(f"%B {date_obj.day}{day_suffix}, %Y")

        team_data["games"].append(new_game)
        team_data["games"].sort(key=lambda game: datetime.strptime(game["date"], '%Y-%m-%d'))
        save_team_data(school["name"], team_data)

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

    if team_data["games"]:
        team_data["games"].sort(key=lambda game: datetime.strptime(game["date"], '%Y-%m-%d'))
        save_team_data(school["name"], team_data)

    return render_template('teams.html', school=school, schools=schools, team_data=team_data, school_slug=school_slug)

@app.route('/team/<school_slug>/edit_roster', methods=['GET', 'POST'])
def edit_roster(school_slug):
    school = get_school_by_slug(school_slug)
    if not school:
        return "School not found", 404

    team_name = school['name']
    roster = get_team_roster(team_name)
    used_numbers = [player['cap_number'] for player in roster]

    if request.method == 'POST':
        if 'delete_cap_number' in request.form:
            delete_cap_number = request.form.get('delete_cap_number')
            roster = [player for player in roster if player['cap_number'] != delete_cap_number]
            save_roster(team_name, sorted(roster, key=lambda x: sort_cap_number(x['cap_number'])))
            return redirect(url_for('edit_roster', school_slug=school_slug))
        else:
            cap_number = request.form.get('cap_number')
            player_name = request.form.get('player_name')
            grade = request.form.get('grade')
            position = request.form.get('position')

            if cap_number in used_numbers:
                error_message = f"Cap number {cap_number} is already taken. Please choose a different number."
                return render_template('edit_roster.html', school_slug=school_slug, school=school, roster=roster, error=error_message)

            new_player = {
                'cap_number': cap_number,
                'name': player_name,
                'grade': grade,
                'position': position
            }
            roster.append(new_player)

            roster = sorted(roster, key=lambda x: sort_cap_number(x['cap_number']))
            save_roster(team_name, roster)

            return redirect(url_for('edit_roster', school_slug=school_slug))

    roster = sorted(roster, key=lambda x: sort_cap_number(x['cap_number']))
    return render_template('edit_roster.html', school_slug=school_slug, school=school, roster=roster)

@app.route('/team/<school_slug>/score/<int:game_index>', methods=['GET'])
def score_game(school_slug, game_index):
    school = schools.get(school_slug)
    if not school:
        return "School not found", 404

    team_name = school['name']
    game = open_game(team_name, game_index)
    if not game:
        return "Game not found", 404

    home_team = team_name if game['home_away'] == 'Home' else game['opponent']
    away_team = game['opponent'] if game['home_away'] == 'Home' else team_name

    home_roster = get_team_roster(home_team)
    away_roster = get_team_roster(away_team)

    home_players = [player['cap_number'] for player in home_roster]
    away_players = [player['cap_number'] for player in away_roster]

    home_box = {
        'Player': home_players,
        'Shot': [0] * len(home_players),
        'Blocks': [0] * len(home_players),
        'Steals': [0] * len(home_players),
        'Exclusions': [0] * len(home_players),
        'Exclusions Drawn': [0] * len(home_players),
        'Penalties': [0] * len(home_players),
        'Turnovers': [0] * len(home_players)
    }

    away_box = {
        'Player': away_players,
        'Shot': [0] * len(away_players),
        'Blocks': [0] * len(away_players),
        'Steals': [0] * len(away_players),
        'Exclusions': [0] * len(away_players),
        'Exclusions Drawn': [0] * len(away_players),
        'Penalties': [0] * len(away_players),
        'Turnovers': [0] * len(away_players)
    }

    return render_template(
        "score_game.html",
        home_team=home_team,
        away_team=away_team,
        game_index=game_index,
        home_box=home_box,
        away_box=away_box,
        school_slug=school_slug
    )

@app.route('/process', methods=['POST'])
def process_text():
    text = request.form['text']
    response = run(text)
    return jsonify({'response': response})

@app.route('/end_game', methods=['POST'])
def end_game():
    try:
        white_team_name = request.form.get('white_team_name')
        black_team_name = request.form.get('black_team_name')
        game_index = int(request.form.get('game_index'))

        if not white_team_name or not black_team_name:
            raise ValueError("Team names must be provided!")

        initialize_team_file(white_team_name)
        initialize_team_file(black_team_name)

        white_team_data = load_team_data(white_team_name)
        black_team_data = load_team_data(black_team_name)

        white_team_score = sum(dataWhite.get('Shot', []))
        black_team_score = sum(dataBlack.get('Shot', []))

        if game_index < len(white_team_data["games"]):
            white_game = white_team_data["games"][game_index]
            white_game_date = white_game["date"]

            black_game_index = next((i for i, game in enumerate(black_team_data["games"]) 
                                   if game["opponent"] == white_team_name and game["date"] == white_game_date), None)

            if black_game_index is not None:
                black_game = black_team_data["games"][black_game_index]

                white_game["is_scored"] = True
                black_game["is_scored"] = True

                white_game["away_box"] = dataWhite
                white_game["home_box"] = dataBlack
                black_game["home_box"] = dataBlack
                black_game["away_box"] = dataWhite

                score_data = {
                    "white_team_score": white_team_score,
                    "black_team_score": black_team_score
                }
                white_game["score"] = score_data
                black_game["score"] = score_data

                save_team_data(white_team_name, white_team_data)
                save_team_data(black_team_name, black_team_data)

        reset_team_stats()

        school_slug = request.form.get('school_slug')
        if not school_slug:
            for slug, school in schools.items():
                if school['name'] == white_team_name:
                    school_slug = slug
                    break

        return redirect(url_for('team_page', school_slug=school_slug))

    except Exception as e:
        return jsonify({'response': f'Error occurred during processing: {str(e)}'}), 500

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
