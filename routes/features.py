from bson import ObjectId
from flask import Blueprint, request, jsonify
import numpy as np
import io

from lib.muscles import save_to_user_history
from model.model import load_optimization_check
from model.model import detect_and_resolve_imbalance_across_bilateral_muscle_pair
from db import users_collection

features_bp = Blueprint('features', __name__)

@features_bp.route('/<user_id>/load_optimization_check', methods=['POST'])
def load_optimization_check_request(user_id):
    try:
        emg_file = request.files.get('emg_file')
        muscle_name = request.form.get('muscle_name')

        if emg_file is None or muscle_name is None or user_id is None:
            return jsonify({"error": "Missing data"}), 400

        # Fetch user by ObjectId
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Get the muscle's max_emg
        muscles_max_emg = user.get('muscles_max_emg', {})
        max_emg = muscles_max_emg.get(muscle_name)

        if max_emg is None:
            return jsonify({"error": "This muscle doesn't have a recorded max voluntary contraction emg!"}), 500
        
        emg_str = emg_file.read().decode('utf-8')
        emg_array = np.loadtxt(io.StringIO(emg_str))
        response = load_optimization_check(max_emg, emg_array)
        response['muscle_name'] = muscle_name
        
        save_to_user_history(response, user_id, load_optimization=True)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@features_bp.route('/<user_id>/detect_and_resolve_imbalance_across_bilateral_muscle_pair', methods=['POST'])
def detect_and_resolve_imbalance_across_bilateral_muscle_pair_request(user_id):
    try:
        emg_left_file = request.files.get('emg_left')
        emg_right_file = request.files.get('emg_right')
        muscle_name = request.form.get('muscle_name')
        emg_left_str = emg_left_file.read().decode('utf-8')
        emg_right_str = emg_right_file.read().decode('utf-8')
        emg_left_array = np.loadtxt(io.StringIO(emg_left_str))
        emg_right_array = np.loadtxt(io.StringIO(emg_right_str))
        prediction = detect_and_resolve_imbalance_across_bilateral_muscle_pair(
            emg_left_array, emg_right_array, muscle_name)

        save_to_user_history(prediction, user_id, bilateral_imbalance=True)

        return jsonify(prediction)
    except Exception as e:
        return jsonify({"Error, something went wrong!"}), 500