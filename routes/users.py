import io
from bson import ObjectId
from flask import Blueprint, request, jsonify
import numpy as np
from db import users_collection
import bcrypt

from lib.serialize_mongo_document import serialize_mongo_document

users_bp = Blueprint('users', __name__)

# User Endpoints

@users_bp.route('/create', methods=['POST'])
def create_user():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    age = data.get('age')
    gender = data.get('gender')
    
    if not all([name, email, password, age, gender]):
        return jsonify({'error': 'Missing data'}), 400
    if users_collection.find_one({'email': email}):
        return jsonify({'error': 'Email already exists'}), 409
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user = {
        'name': name,
        'email': email,
        'password': hashed_pw,
        'age': age,
        'gender': gender,
        'muscles_max_emg': {},
        'history': []
    }
    users_collection.insert_one(user)
    return jsonify({'message': 'User created successfully'}), 201

@users_bp.route('/', methods=['GET'])
def get_users():
    users = []
    for user in users_collection.find({}, {'password': 0}):
        user['_id'] = str(user['_id'])
        users.append(user)
    return jsonify(users), 200

@users_bp.route('/<user_id>', methods=['GET'])
def get_user_data(user_id):
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400

    user = users_collection.find_one({'_id': ObjectId(user_id)}, {'password': 0})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    user['_id'] = str(user['_id'])
    return jsonify(user), 200

@users_bp.route('/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        result = users_collection.delete_one({'_id': ObjectId(user_id)})
        if result.deleted_count == 1:
            return jsonify({'message': 'User deleted successfully'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Invalid user_id or error: {str(e)}'}), 400

@users_bp.route('/<user_id>', methods=['PATCH'])
def update_user(user_id):
    try:
        update_data = request.get_json()
        if not update_data:
            return jsonify({'error': 'No update data provided'}), 400

        # Update user document with the fields provided in update_data
        result = users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': update_data}
        )
        if result.matched_count == 1:
            updated_user = users_collection.find_one({'_id': ObjectId(user_id)})
            updated_user = serialize_mongo_document(updated_user)
            updated_user.pop('password', None)
            return jsonify({'message': 'User updated successfully', 'updated_user': updated_user}), 200
        else:
            return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Invalid user_id or error: {str(e)}'}), 400
    
# Login
@users_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({'error': 'Missing email or password'}), 400

    user = users_collection.find_one({'email': email})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    # Password is stored as a bcrypt hash (binary)
    if not bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return jsonify({'error': 'Incorrect password'}), 401

    # Remove sensitive info before sending
    user['_id'] = str(user['_id'])
    user.pop('password', None)
    return jsonify({'message': 'Login successful', 'user': user}), 200

# Update user muscle max emg
@users_bp.route('/<user_id>/muscle_max_emg', methods=['POST'])
def add_muscle_max_emg(user_id):
    muscle_name = request.form.get('muscle_name')
    muscle_emg_file = request.files.get('muscle_emg_file')  # <--- the file input name in your form

    if not all([user_id, muscle_name, muscle_emg_file]):
        return jsonify({"error": "Missing data"}), 400

    try:
        # Read file contents as string and convert to array (assumes CSV/txt)
        emg_contents = muscle_emg_file.read().decode('utf-8')
        max_emg = np.loadtxt(io.StringIO(emg_contents)).flatten().tolist()
    except Exception as e:
        return jsonify({"error": "Could not parse EMG file", "details": str(e)}), 400
    
    try:
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {f"muscles_max_emg.{muscle_name}": max_emg}}
        )

        if result.modified_count == 1:
            return jsonify({"message": f"Max EMG for {muscle_name} updated!", "max_emg": max_emg}), 200
        else:
            return jsonify({"error": "User not found or value not changed."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Get user muscle max emg recorded
@users_bp.route('/<user_id>/muscle_max_emg', methods=['GET'])
def get_muscle_max_emg(user_id):
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404

        muscles_max_emg = user.get("muscles_max_emg", {})
        muscle_names = list(muscles_max_emg.keys())

        return jsonify({"muscle_names": muscle_names}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@users_bp.route('/<user_id>/history', methods=['DELETE'])
def remove_history_request(user_id):
    data = request.get_json()
    date = data.get('date')
    history_category = data.get('historyCategory')

    if not date or not history_category:
        return jsonify({'error': 'date and historyCategory are required'}), 400

    try:
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Check if the category exists and is a list
        history = user.get('history', {})
        history_list = history.get(history_category, [])
        if not isinstance(history_list, list):
            return jsonify({'error': 'History category not found or not a list'}), 400

        # Filter out the entry with the matching date
        new_history_list = [entry for entry in history_list if entry.get('date') != date]
        history[history_category] = new_history_list

        # Update the user's history in the database
        users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {'history': history}}
        )

        return jsonify({'message': 'History entry removed successfully'}), 200

    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 400

