from flask import Flask, render_template, request, jsonify, redirect, url_for
app = Flask(__name__)
from word2number import w2n
import spacy
import json
import os
from datetime import datetime

# Helper function to get team file path
def get_team_file_path(team_name):
    return os.path.join('teams', f"team_{team_name.replace(' ', '_')}.json")

# Assuming you have helper functions to load team data and game data

@app.route('/team/<school_slug>/score/<int:game_index>', methods=['GET'])
def start_scoring(school_slug, game_index):
    # Load team data for the specified school
    team_data = load_team_data(school_slug)
    if not team_data or game_index >= len(team_data["games"]):
        return "Game not found", 404

    game = team_data["games"][game_index]

    # Pass necessary game data to scoring.html
    return render_template("scoring.html", game=game, school_slug=school_slug, game_index=game_index)

@app.route('/team/<school_slug>/view/<int:game_index>', methods=['GET'])
def view_scoring(school_slug, game_index):
    # Load team data for the specified school
    team_data = load_team_data(school_slug)
    if not team_data or game_index >= len(team_data["games"]):
        return "Game not found", 404

    game = team_data["games"][game_index]

    # Pass necessary game data to view_game.html for viewing purposes
    return render_template("view_game.html", game=game, school_slug=school_slug, game_index=game_index)



def initialize_team_file(team_name):
    # Ensure the 'teams' directory exists
    if not os.path.exists('teams'):
        os.makedirs('teams')  # Create the directory if it doesn't exist

    team_file_path = get_team_file_path(team_name)
    if not os.path.exists(team_file_path):
        # Create the file with an empty "games" list
        with open(team_file_path, 'w') as file:
            json.dump({"games": []}, file, indent=4)


# Helper function to load a team's JSON file
def load_team_data(team_name):
    team_file_path = get_team_file_path(team_name)
    if os.path.exists(team_file_path):
        with open(team_file_path, 'r') as file:
            return json.load(file)
    else:
        return {"games": []}

# Helper function to save a team's JSON data
def save_team_data(team_name, data):
    team_file_path = get_team_file_path(team_name)
    with open(team_file_path, 'w') as file:
        json.dump(data, file, indent=4)

@app.route('/end_game', methods=['POST'])
def end_game():
    global dataWhite, dataBlack

    # Retrieve team names from the POST request
    white_team_name = request.form.get('white_team_name', 'White Team')
    black_team_name = request.form.get('black_team_name', 'Black Team')

    initialize_team_file(white_team_name)
    initialize_team_file(black_team_name)
    
    # Calculate the scores for each team
    white_team_score = sum(dataWhite['Shot'])
    black_team_score = sum(dataBlack['Shot'])

    # Load team data from each team's file
    white_team_data = load_team_data(white_team_name)
    black_team_data = load_team_data(black_team_name)

    # Define the game entry with detailed stats
    game_entry = {
        "date": request.form.get('date'),  # You may need to pass date/time from the form
        "time": request.form.get('time'),
        "opponent": black_team_name,
        "home_away": "Home",
        "game_type": "League Game",
        "is_scored": True,
        "score": {
            "white_team_score": white_team_score,
            "black_team_score": black_team_score
        },
        "white_team_stats": dataWhite,
        "black_team_stats": dataBlack
    }

    # Add the game entry to the white team's JSON file
    white_team_data["games"].append(game_entry)
    save_team_data(white_team_name, white_team_data)

    # For the black team, we reverse the home/away and opponent fields
    game_entry_opponent = {
        "date": request.form.get('date'),
        "time": request.form.get('time'),
        "opponent": white_team_name,
        "home_away": "Away",
        "game_type": "League Game",
        "is_scored": True,
        "score": {
            "white_team_score": white_team_score,
            "black_team_score": black_team_score
        },
        "white_team_stats": dataWhite,
        "black_team_stats": dataBlack
    }

    # Add the game entry to the black team's JSON file
    black_team_data["games"].append(game_entry_opponent)
    save_team_data(black_team_name, black_team_data)

    # Reset stats for the next game
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

    return jsonify({'response': 'Game ended and data saved successfully!'})



#training model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Ensure the app's configuration is set before initializing SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'  # Replace with your actual URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with the app
db = SQLAlchemy(app)

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

#next

app = Flask(__name__)

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



# # Route to end the game, store the current game data and reset the current game
# @app.route('/end_game', methods=['POST'])
# def end_game():
#     global dataWhite, dataBlack


#     # Retrieve team names from the POST request
#     white_team_name = request.form.get('white_team_name', 'White Team')  # Default name if not provided
#     black_team_name = request.form.get('black_team_name', 'Black Team')  # Default name if not provided


#     white_team_score = sum(dataWhite['Shot'])
#     black_team_score = sum(dataBlack['Shot'])

#     previous_games.append({
#         'white_team_name': white_team_name,
#         'black_team_name': black_team_name,
#         'white_score': white_team_score,
#         'black_score': black_team_score,
#         'white_team': dataWhite,
#         'black_team': dataBlack
#     })

#     # Save the games to a JSON file
#     save_previous_games()

#     dataWhite = {
#         'Player': ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5', 'Player 6', 'Player 7', 'Player 8', 'Player 9', 'Player 10'],
#         'Shot': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         'Blocks': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         'Steals': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         'Exclusions': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         'Exclusions Drawn': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         'Penalties': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         'Turnovers': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     }

#     dataBlack = {
#         'Player': ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5', 'Player 6', 'Player 7', 'Player 8', 'Player 9', 'Player 10'],
#         'Shot': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         'Blocks': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         'Steals': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         'Exclusions': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         'Exclusions Drawn': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         'Penalties': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         'Turnovers': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     }

#     return jsonify({'response': 'Game ended and data saved successfully!'})

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
    'Blocks': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Steals': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Exclusions': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Exclusions Drawn': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Penalties': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Turnovers': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

dataBlack = {
    'Player': ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5', 'Player 6', 'Player 7', 'Player 8', 'Player 9', 'Player 10'],
    'Shot': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Blocks': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Steals': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Exclusions': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Exclusions Drawn': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Penalties': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Turnovers': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}


def extract_key_phrases(text):
    doc = nlp(text.lower())
    team, player, event = None, None, None
    #must figure out different kinds of wording
    dark_keywords = ['dark','black','blue']
    light_keywords = ['light','white']
    shot_keywords = ['goal', 'shot', 'score', 'point','scored','scores']
    block_keywords = ['block', 'blocked','blocks']
    steal_keywords = ['steal','stole','took','steals'] #stole not working?
    exclusion_keywords = ['exclusion', 'kickout','excluded']
    turnover_keywords = ['turnover', 'foul','lost','loses']
    penalty_keywords = ['penalty', 'five meter']

    tokens = [token.text for token in doc]  # Tokenize text

    i = 0
    while i < len(tokens):
        token = tokens[i]
        try:
            # Check if the word is a number unless it's part of "five meter"
            if token == "five" and i + 1 < len(tokens) and tokens[i + 1] == "meter":
                event = "Penalties"
                i += 2  # Skip the next word since it's part of "five meter"
                continue
            else:
                player = str(w2n.word_to_num(token))
        except Exception:
            pass

            if (token == "kick" or token == "kicked") and tokens[i+1] == "out":
                event = "Exclusions"

            if (token == "turn" or token == "turned") and tokens[i+1] == "over":
                event = "Turnovers"
                

        # Detect teams
        if token in dark_keywords:
            team = 'dark'
        elif token in light_keywords:
            team = 'light'

        # Special case for goalies
        if token == 'goalie':
            player = 1

        # Detect events
        if token in shot_keywords:
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
        predict_connotation(text)
        if predict_connotation(text) == "Negative":
            event = "Exclusions"
        elif predict_connotation(text) == "Positive" and event == "Penalties":
            event = "Penalties"
        elif predict_connotation(text) == "Positive" and event == "Exclusions":
            event = "Exclusions Drawn"

    return player, event, team




# Update team data
def sort_data(player, event, team):
    player_index = int(player) - 1
    if team == 'light':
        dataWhite[event][player_index] += 1
    else:
        dataBlack[event][player_index] += 1

def phrase(number, action, team):
    if number == 1:
        number = "goalie"
    if action == 'Shot':
        return f"The {team} team {number} scored a goal"
    elif action == 'Blocks':
        return f"The {team} team {number} blocked the shot"
    elif action == 'Steals':
        return f"The {team} team {number} stole the ball"
    elif action == 'Exclusions Drawn':
        return f"The {team} team {number} drew an exclusion"
    elif action == 'Exclusions':
        return f"The {team} team {number} was excluded from the game"
    elif action == 'Turnovers':
        return f"The {team} team {number} was turned over"
    elif action == 'Penalties':
        return f"The {team} team {number} got a 5-meter penalty"

def run(text):
    player, event, team = extract_key_phrases(text)
    if player and event and team:
        sort_data(player, event, team)
        return phrase(player, event, team)
    return "Could not parse the input."

@app.route('/')
def home():
    return render_template('home.html')

# Render HTML page with two tables (initial zeros)
@app.route('/scoring')
def index():
    return render_template('scoring.html')

@app.route('/teams')
def teams():
    return render_template('teams.html')

@app.route('/search')
def search():
    return render_template('search.html')


# Process user input and return updated stats
@app.route('/process', methods=['POST'])
def process_text():
    text = request.form['text']
    response = run(text)
    return jsonify({'response': response})

# Return the current team data as JSON (for updating tables)
@app.route('/get_data', methods=['GET'])
def get_data():
    return jsonify({'dataWhite': dataWhite, 'dataBlack': dataBlack})

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





@app.route('/player_stats/<string:player_name>/<string:team_color>', methods=['GET'])
def player_stats(player_name, team_color):
    # Initialize combined stats
    combined_stats = {
        'Shot': 0,
        'Blocks': 0,
        'Steals': 0,
        'Exclusions': 0,
        'Exclusions Drawn': 0,
        'Penalties': 0,
        'Turnovers': 0
    }

    # Loop through previous games and accumulate stats based on team color
    for game in previous_games:
        if team_color == 'white' and player_name in game['white_team']['Player']:
            idx = game['white_team']['Player'].index(player_name)
            for key in combined_stats:
                combined_stats[key] += game['white_team'][key][idx]
        elif team_color == 'black' and player_name in game['black_team']['Player']:
            idx = game['black_team']['Player'].index(player_name)
            for key in combined_stats:
                combined_stats[key] += game['black_team'][key][idx]

    # Render the player stats page with the combined stats
    return render_template('player_stats.html',player_name=player_name, team_color=team_color, stats=combined_stats)


# Simulate a dictionary of school data (name, logo, and colors)

# Simulate a dictionary of school data (name, logo, colors, and game schedule)
schools = {
    "palo-alto": {
        "name": "Palo Alto High School",
        "logo": "images/Palo Alto High School.png",
        "bg_color": "#004b23",
        "text_color": "#ffffff",
        "link_color": "#004b23",
        "games": []
    },
    "gunn": {
        "name": "Henry M. Gunn High School",
        "logo": "images/Henry M. Gunn High School.png",
        "bg_color": "#ff0000",
        "text_color": "#000000",
        "link_color": "#ff0000",
        "games": []
    },
    "harker": {
        "name": "Harker High School",
        "logo": "images/Harker High School.png",
        "bg_color": "#004b23",
        "text_color": "#ffffff",
        "link_color": "#004b23",
        "games": []
    },
    "los-gatos": {
        "name": "Los Gatos High School",
        "logo": "images/Los Gatos High School.png",
        "bg_color": "#ffa500",
        "text_color": "#ffffff",
        "link_color": "#ffa500",
        "games": []
    },
    "los-altos": {
        "name": "Los Altos High School",
        "logo": "images/Los Altos High School.png",
        "bg_color": "#000080",
        "text_color": "#ffffff",
        "link_color": "#000080",
        "games": []
    },
    "fremont": {
        "name": "Fremont High School",
        "logo": "images/Fremont High School.png",
        "bg_color": "#8B0000",
        "text_color": "#ffffff",
        "link_color": "#8B0000",
        "games": []
    },
    "mountain-view": {
        "name": "Mountain View High School",
        "logo": "images/Mountain View High School.png",
        "bg_color": "#ffdb58",
        "text_color": "#000000",
        "link_color": "#ffdb58",
        "games": []
    }
}

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
        am_pm = request.form.get('am_pm')  # Retrieve AM or PM

        # Ensure the opponent team file exists
        initialize_team_file(opponent_name)

        # Load opponent data
        opponent_data = load_team_data(opponent_name)

        # Find the opponent's slug and logo
        opponent_slug = next(
            (slug for slug, data in schools.items() if data['name'] == opponent_name), None
        )
        opponent_logo = schools[opponent_slug]['logo'] if opponent_slug else ''

        # Create a new game entry
        new_game = {
            'home_away': home_away,
            'opponent': opponent_name,
            'opponent_logo': opponent_logo,
            'time': f"{time} {am_pm}",
            'date': date,
            'game_type': game_type,
            'is_scored': False  # Game is initially not scored
        }

        # Format the date for display
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        day_suffix = 'th' if 11 <= date_obj.day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(date_obj.day % 10, 'th')
        new_game['formatted_date'] = date_obj.strftime(f"%B {date_obj.day}{day_suffix}, %Y")

        # Add the game to the main team’s schedule
        team_data["games"].append(new_game)
        save_team_data(school["name"], team_data)

        # Also add it to the opponent's schedule, with home/away reversed
        if opponent_slug:
            opponent_game = {
                'home_away': 'Home' if home_away == 'Away' else 'Away',
                'opponent': school['name'],
                'opponent_logo': school['logo'],
                'time': f"{time} {am_pm}",
                'date': date,
                'game_type': game_type,
                'is_scored': False,
                'formatted_date': new_game['formatted_date']
            }
            opponent_data["games"].append(opponent_game)
            save_team_data(opponent_name, opponent_data)

        return redirect(url_for('team_page', school_slug=school_slug))  # Redirect after POST

    # Render the template with team data
    return render_template('teams.html', school=school, schools=schools, team_data=team_data, school_slug=school_slug)



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



@app.route('/score/<school_slug>', methods=['GET'])
def scoring_page(school_slug):
    # Pass the school_slug to get_school() to fetch the relevant school data
    school = get_school(school_slug)

    if not school:
        return "School not found", 404

    # Fetch game data or other relevant information
    game = get_game(school_slug)  # Example function to get a game

    return render_template('scoring_page.html', school=school, game=game)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)