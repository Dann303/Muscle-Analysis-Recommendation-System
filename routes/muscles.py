from flask import Blueprint, jsonify
from model.model import all_unique_muscles

muscles_bp = Blueprint('muscles', __name__)

# You'll need to define 'all_unique_muscles' elsewhere and import it here.
@muscles_bp.route('/get_muscle_names_bilateral_muscle_pair', methods=['GET'])
def get_muscle_names():
    try:
        return jsonify({"muscle_names": all_unique_muscles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500