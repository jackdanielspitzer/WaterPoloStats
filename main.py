from flask import Flask, render_template, request, jsonify, redirect, url_for
app = Flask(__name__)
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
    print(1)
    # Load team data for the specified school
    team_data = load_team_data(school_slug)
    print(team_data,game_index)
    if not team_data or game_index >= len(team_data["games"]):
        return "Game not found", 404

    print(game_index)
    game = team_data["games"][game_index]

    # Pass necessary game data to scoring.html
    return render_template("scoring.html", game=game, school_slug=school_slug, game_index=game_index)

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
    print(team_file_path)
    # Check if the team file exists
    if os.path.exists(team_file_path):
        try:
            with open(team_file_path, 'r') as file:
                # Load the team data from the JSON file
                team_data = json.load(file)
                # Validate the game_index
                if game_index < 0 or game_index >= len(team_data["games"]):
                    return "out of bounds"  # Return None if the game_index is out of bounds
                # Return the requested game
                return team_data["games"][game_index]
        except json.JSONDecodeError:
            print(f"Error: {team_file_path} contains invalid JSON.")
            return "is invalid"  # Return None if JSON is invalid
    # Return None if the file does not exist
    return "doesn't exist"


# Helper function to save a team's JSON data
def save_team_data(team_name, data):
    team_file_path = get_team_file_path(team_name)
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
        'Turnovers': [0] * 10
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
        'Turnovers': [0] * 10
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
    'Assists': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
    'Assists': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}


def extract_key_phrases(text):
    doc = nlp(text.lower())
    doc_text = doc.text
    events = []
    dark_keywords = ['dark','black','blue']
    light_keywords = ['light','white']
    shot_keywords = ['goal', 'shot', 'score', 'point','scored','scores']
    block_keywords = ['block', 'blocked','blocks']
    steal_keywords = ['steal','stole','took','steals']
    exclusion_keywords = ['exclusion', 'kickout','excluded', 'kicked out', 'kick out']
    turnover_keywords = ['turnover', 'foul','lost','loses']
    penalty_keywords = ['penalty', 'five meter']
    
    # Extract all player numbers first
    all_numbers = []
    for token in doc:
        try:
            if token.text != "five":
                num = w2n.word_to_num(token.text)
                if 1 <= num <= 13:
                    all_numbers.append(str(num))
        except ValueError:
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

    # Initialize first and second events
    first_event = {'team': current_team, 'player': None, 'event': None}
    second_event = {'team': None, 'player': None, 'event': None}
    
    # Assign first player number found
    if all_numbers:
        first_event['player'] = all_numbers[0]

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
            
        # Extract event type
        if 'penalty' in doc_text:
            if 'got a penalty' in doc_text or 'received a penalty' in doc_text:
                # Handle when player gets a penalty/exclusion, including goalie
                if first_event['player'] is None and 'goalie' in doc_text:
                    first_event['player'] = '1'
                else:
                    first_event['player'] = all_numbers[0] if all_numbers else None
                first_event['event'] = 'Exclusions'
                first_event['team'] = current_team
            elif len(all_numbers) >= 2:
                first_event['player'] = all_numbers[0]
                first_event['event'] = 'Penalties'
                first_event['team'] = current_team
                second_event['player'] = all_numbers[1]
                second_event['event'] = 'Exclusions'
                second_event['team'] = 'dark' if first_event['team'] == 'light' else 'light'
            break
        elif token in exclusion_keywords:
            if len(all_numbers) >= 2 and ('excluded by' in doc_text or 'by player' in doc_text):
                # First event is the exclusion
                first_event['player'] = all_numbers[0]
                first_event['event'] = 'Exclusions'
                first_event['team'] = current_team
                
                # Second event is who drew it
                second_event['player'] = all_numbers[1]
                second_event['event'] = 'Exclusions Drawn'
                second_event['team'] = 'dark' if current_team == 'light' else 'light'
                break
            elif len(all_numbers) == 1:
                # Single exclusion event
                first_event['player'] = all_numbers[0]
                first_event['event'] = 'Exclusions'
                first_event['team'] = current_team
                break
                
        if token in shot_keywords and 'block' not in doc_text:
            first_event['event'] = 'Shot'
        elif token in block_keywords:
            # Check for "got block on player X" pattern
            if ('got block' in doc_text or 'got a block' in doc_text or 'makes a save' in doc_text):
                if len(all_numbers) >= 1:
                    # First event is the block
                    first_event['event'] = 'Blocks'
                    first_event['player'] = '1' if 'goalie' in doc_text else all_numbers[0]
                    first_event['team'] = current_team
                    
                    # Second event is the shot attempt
                    second_event['event'] = 'Shot Attempt'  
                    second_event['player'] = all_numbers[1] if len(all_numbers) > 1 else '5'
                    second_event['team'] = 'dark' if current_team == 'light' else 'light'
            # Check other block scenarios
            elif ('gets a block' in doc_text or 'got a block' in doc_text or 
                'makes a save' in doc_text or 'with a block' in doc_text):
                first_event['event'] = 'Blocks'
            else:
                first_event['event'] = 'Shot Attempt'
                second_event['event'] = 'Blocks'
        elif 'assist' in doc_text:
            first_event['event'] = 'Shot'
            # Find the assisting player's number
            try:
                assist_number = None
                for t in tokens[i+1:]:
                    if t != "point" and t != "five":
                        num = w2n.word_to_num(t)
                        if 1 <= num <= 13:
                            assist_number = str(num)
                            second_event['player'] = assist_number
                            second_event['event'] = 'Assists'
                            second_event['team'] = first_event['team']
                            break
            except ValueError:
                pass
        elif token in steal_keywords or 'stole from' in doc_text or 'steal from' in doc_text:
            first_event['event'] = 'Steals'
            # Find the second player number for turnover
            try:
                # Look for numbers after the steal keyword
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
                    second_event['player'] = numbers[-1]  # Take the last number found
                    second_event['event'] = 'Turnovers'
                    second_event['team'] = 'light' if first_event['team'] == 'dark' else 'dark'
            except ValueError:
                pass
            
    # Add events if complete
    if first_event['team'] and first_event['player'] and first_event['event']:
        events.append((first_event['player'], first_event['event'], first_event['team']))
        
    if second_event['team'] and second_event['player'] and second_event['event']:
        events.append((second_event['player'], second_event['event'], second_event['team']))
        
    return events if events else [(None, None, None)]


    if current_event['player'] and current_event['event'] and current_event['team']:
        events.append((current_event['player'], current_event['event'], current_event['team']))
    
    # Look for second event in the same sentence
    # Common patterns like "but was" indicate a second event
    second_half_markers = ["but", "and", "then", "while"]
    for marker in second_half_markers:
        if marker in tokens:
            idx = tokens.index(marker)
            # Process second half of sentence separately
            second_text = " ".join(tokens[idx:])
            doc2 = nlp(second_text)
            tokens2 = [token.text for token in doc2]
            current_event = {'team': None, 'player': None, 'event': None}
            
            # Process second event similar to first
            for token in tokens2:
                try:
                    if token != "point":
                        player = str(w2n.word_to_num(token))
                        current_event['player'] = player
                except:
                    pass
                
                if token in dark_keywords:
                    current_event['team'] = 'dark'
                elif token in light_keywords:
                    current_event['team'] = 'light'
                    
                if token == 'goalie':
                    current_event['player'] = '1'
                    
                if token in shot_keywords and current_event['event'] != "Blocks":
                    current_event['event'] = "Shot"
                elif token in block_keywords:
                    current_event['event'] = "Blocks"
                elif token in steal_keywords:
                    current_event['event'] = "Steals"
                elif token in exclusion_keywords:
                    current_event['event'] = "Exclusions"
                elif token in turnover_keywords:
                    current_event['event'] = "Turnovers"
                elif token in penalty_keywords:
                    current_event['event'] = "Penalties"
                    
            if current_event['player'] and current_event['event'] and current_event['team']:
                events.append((current_event['player'], current_event['event'], current_event['team']))
            
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
            # Update the player's stats directly using the found index
            game_data[game_id]['dataWhite'][event][player_index] += 1  # Light/away team uses dataWhite
        except (StopIteration, ValueError):
            print(f"ERROR: Player {player} not found in away roster")
            return False
    elif team == 'dark':
        # Find the player in the home roster by cap number (dark team)
        try:
            player_str = str(player)
            print(f"Searching home roster for cap number: '{player_str.strip()}'")
            player_index = next(i for i, p in enumerate(home_roster) if str(p['cap_number']).strip() == player_str.strip())
            print(f"Found player at index: {player_index}")
            # Update the player's stats directly using the found index
            game_data[game_id]['dataBlack'][event][player_index] += 1  # Dark/home team uses dataBlack
        except (StopIteration, ValueError):
            print(f"ERROR: Player {player} not found in home roster")
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
    exclusion_keywords = ['exclusion', 'kickout', 'excluded']
    penalty_keywords = ['penalty', 'five meter']

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

            # Only append Shot if it hasn't been recorded yet
            if not any(e['player'] == numbers[0] and e['event'] == 'Shot' for e in events):
                events.append({
                    'player': numbers[0],
                    'event': 'Shot',
                    'team': team
                })
            # Both players are from the same team in an assist
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
    if number == 1:
        number = "goalie"
    if action == 'Shot':
        return f"The {team} team {number} scored a goal"
    elif action == 'Shot Attempt':
        return f"The {team} team {number} attempted a shot"
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
    else:
        return f"The {team} team {number} performed {action}"

def run(text):
    events = extract_key_phrases(text)
    responses = []
    home_team_name = request.form.get('home_team')
    away_team_name = request.form.get('away_team')
    
    for player, event, team in events:
        if player and event and team:
            if sort_data(player, event, team, home_team_name, away_team_name):
                responses.append(phrase(player, event, team))
            else:
                responses.append(f"Player {player} not found in roster.")
    
    return " and ".join(responses) if responses else "Could not parse the input."

@app.route('/')
def home():
    # Get all upcoming games from all teams
    upcoming_games = []
    seen_games = set()  # Track unique games
    today = datetime.now().date()
    
    for school in schools.values():
        team_data = load_team_data(school['name'])
        for game in team_data.get('games', []):
            game_date = datetime.strptime(game['date'], '%Y-%m-%d').date()
            if game_date >= today and not game.get('is_scored', False):
                # Create a unique key for each game using date and team names
                game_key = f"{game['date']}-{sorted([school['name'], game['opponent']])[0]}-{sorted([school['name'], game['opponent']])[1]}"
                if game_key not in seen_games:
                    game['school_name'] = school['name']
                    game['school_logo'] = school['logo']
                    upcoming_games.append(game)
                    seen_games.add(game_key)
    
    # Sort games by date and get the 6 most recent
    upcoming_games.sort(key=lambda x: x['date'])
    upcoming_games = upcoming_games[:6]
    
    return render_template('home.html', upcoming_games=upcoming_games)

# Render HTML page with two tables (initial zeros)
@app.route('/scoring')
def index():
    return render_template('scoring.html')

@app.route('/team/<school_slug>/score/<int:game_index>', methods=['GET'])
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
    text = request.form['text']
    game_id = request.form.get('game_id')
    if game_id not in game_data:
        return jsonify({'error': 'Invalid game ID'}), 400
    response = run(text, game_id)
    return jsonify({'response': response})

def run(text, game_id):
    events = extract_key_phrases(text)
    responses = []
    home_team_name = request.form.get('home_team')
    away_team_name = request.form.get('away_team')
    
    for player, event, team in events:
        if player and event and team:
            if sort_data(player, event, team, home_team_name, away_team_name, game_id):
                responses.append(phrase(player, event, team))
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
def get_data():
    try:
        # Load team rosters
        team_rosters = load_team_rosters()
        
        # Get team names and game ID from query parameters
        home_team_name = request.args.get('home_team')
        away_team_name = request.args.get('away_team')
        game_id = request.args.get('game_id')
        
        # Get rosters for both teams
        home_roster = team_rosters.get(home_team_name, [])
        away_roster = team_rosters.get(away_team_name, [])

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
                    'Turnovers': [0] * len(away_roster)
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
                    'Turnovers': [0] * len(home_roster)
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
                    'Turnovers': [0] * len(away_roster)
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
                    'Turnovers': [0] * len(home_roster)
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
    school_slug = request.args.get('school_slug', 'palo-alto')
    return render_template('player_stats.html', player_name=player_name, team_color=team_color, stats=combined_stats, school_slug=school_slug)


# Simulate a dictionary of school data (name, logo, and colors)

# Simulate a dictionary of school data (name, logo, colors, and game schedule)
schools = {
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

@app.route('/team/<school_slug>/score/<int:game_index>', methods=['GET'])
def scoring_page(school_slug, game_index):
    school = schools.get(school_slug)
    if not school:
        return "School not found", 404

    team_name = school['name']
    roster = get_team_roster(team_name)  # Fetch the roster for this team

    # Load game data as before
    game = open_game(team_name, game_index)
    if not game:
        return "Game not found", 404

    # Pass roster and game data to the scoring template
    return render_template('score_game.html', game=game, roster=roster, school_slug=school_slug)


    # game = team_data["games"][game_index]

    # home_away = team_data["home_away"]
    # print(home_away)
    # opponent_name = team_data["opponent"]
    # print(opponent_name)

    home_team = school['name'] if team_data['home_away'] == 'Home' else team_data['opponent']
    away_team = team_data['opponent'] if team_data['home_away'] == 'Home' else school['name']
    # Render the score_game template
    return render_template("score_game.html", home_team=home_team, away_team=away_team, game_index=game_index)

from flask import render_template

@app.route('/team/<school_slug>/view/<int:game_index>', methods=['GET'])
def view_scoring(school_slug, game_index):
    try:
        # Load team rosters from team_rosters.json
        with open('team_rosters.json', 'r') as file:
            team_rosters = json.load(file)

        # Fetch school and game data
        school = schools.get(school_slug)
        if not school:
            return "School not found", 404

        school_name = school['name']
        game = open_game(school_name, game_index)
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

        # Determine which box scores to use based on home/away status
        if game['home_away'] == 'Home':
            # If home team viewing, use home_box for black team stats
            black_team_stats = game.get('home_box', {})
            white_team_stats = game.get('away_box', {})
            print("viewed from home")
            print(black_team_stats)
            print(white_team_stats)
        else:
            # If away team viewing, use away_box for white team stats
            white_team_stats = game.get('away_box', {})
            black_team_stats = game.get('home_box', {})
            print("view from away")
            print(black_team_stats)
            print(white_team_stats)

        # Fill missing stats with empty lists
        if not white_team_stats or 'Player' not in white_team_stats:
            white_team_stats = {'Player': [], 'Shot': [], 'Blocks': [], 'Steals': [], 'Exclusions': [], 'Exclusions Drawn': [], 'Penalties': [], 'Turnovers': []}
        if not black_team_stats or 'Player' not in black_team_stats:
            black_team_stats = {'Player': [], 'Shot': [], 'Blocks': [], 'Steals': [], 'Exclusions': [], 'Exclusions Drawn': [], 'Penalties': [], 'Turnovers': []}

        # Calculate scores based on correct home/away assignment
        game['score'] = {
            'white_team_score': sum(white_team_stats.get('Shot', [])),
            'black_team_score': sum(black_team_stats.get('Shot', []))
        }

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
            game_index=game_index
        )
    except Exception as e:
        return f"Error: {str(e)}", 500







@app.route('/end_game', methods=['POST'])
def end_game():
    print("Received request to /end_game")

    try:
        white_team_name = request.form.get('white_team_name')
        black_team_name = request.form.get('black_team_name')
        game_index = int(request.form.get('game_index'))

        print(f"White Team Name: {white_team_name}")
        print(f"Black Team Name: {black_team_name}")

        if not white_team_name or not black_team_name:
            raise ValueError("Team names must be provided!")

        # Initialize team files if they don't exist
        initialize_team_file(white_team_name)
        initialize_team_file(black_team_name)

        # Load both teams' data
        white_team_data = load_team_data(white_team_name)
        black_team_data = load_team_data(black_team_name)

        # Calculate scores
        white_team_score = sum(dataWhite.get('Shot', []))
        black_team_score = sum(dataBlack.get('Shot', []))
        print(f"Calculated scores -> White: {white_team_score}, Black: {black_team_score}")

        # Function to find corresponding game in other team's data
        def find_matching_game(games, opponent, date):
            for idx, game in enumerate(games):
                if game["opponent"] == opponent and game["date"] == date:
                    return idx
            return None

        # Update white team's game
        if game_index < len(white_team_data["games"]):
            white_game = white_team_data["games"][game_index]
            white_game_date = white_game["date"]

            # Find corresponding game in black team's data
            black_game_index = find_matching_game(black_team_data["games"], white_team_name, white_game_date)

            if black_game_index is not None:
                # Get both games
                black_game = black_team_data["games"][black_game_index]

                # Mark both games as scored
                white_game["is_scored"] = True
                black_game["is_scored"] = True

                # Always put dataWhite in the away_box for white team's game
                # and dataBlack in the home_box for black team's game
                white_game["away_box"] = dataWhite
                white_game["home_box"] = dataBlack
                black_game["home_box"] = dataBlack
                black_game["away_box"] = dataWhite

                # Add scores to both games
                white_game["score"] = {
                    "white_team_score": white_team_score,
                    "black_team_score": black_team_score
                }
                black_game["score"] = {
                    "white_team_score": white_team_score,
                    "black_team_score": black_team_score
                }

                # Save both teams' data
                save_team_data(white_team_name, white_team_data)
                save_team_data(black_team_name, black_team_data)

                print(f"Successfully updated game data for both {white_team_name} and {black_team_name}")
            else:
                print(f"Could not find matching game for {black_team_name}")

        # Reset stats for next game
        reset_team_stats()

        # Redirect to /home after processing
        school_slug = request.form.get('school_slug')

        if not school_slug:
            for slug, school in schools.items():
                if school['name'] == white_team_name:
                    school_slug = slug
                    break

        # Redirect to the original team's page
        return redirect(url_for('team_page', school_slug=school_slug))

    except Exception as e:
        print(f"Exception during end_game processing: {str(e)}")
        return jsonify({'response': f'Error occurred during processing: {str(e)}'}), 500

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
def edit_roster(school_slug):
    school = get_school_by_slug(school_slug)
    if not school:
        return "School not found", 404

    team_name = school['name']  # Fetch team name
    roster = get_team_roster(team_name)  # Load the team's roster
    used_numbers = [player['cap_number'] for player in roster]  # Get used cap numbers

    if request.method == 'POST':
        if 'delete_cap_number' in request.form:  # Handle delete request
            delete_cap_number = request.form.get('delete_cap_number')
            roster = [player for player in roster if player['cap_number'] != delete_cap_number]
            save_roster(team_name, sorted(roster, key=lambda x: sort_cap_number(x['cap_number'])))  # Save updated and sorted roster
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

if __name__ == '__main__':
    print("calling app.run")
    app.run(host='0.0.0.0', port=5000, debug=True)


#blocks a shot and scores a point doesn't work properly