from bson import ObjectId
from db import users_collection
from datetime import datetime

def save_to_user_history(result, user_id, bilateral_imbalance=False, load_optimization=False):
    if bilateral_imbalance:
        feature_name = "bilateral_imbalance"
    elif load_optimization:
        feature_name = "load_optimization"
    else:
        raise ValueError("Must specify which feature to save")

    # Step 1: Fix the type if history is not a dict
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if user:
        if not isinstance(user.get("history", {}), dict):
            users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"history": {}}}
            )

    # Step 2: Add timestamp to result (UTC ISO format)
    result_with_date = dict(result)  # don't mutate the original
    result_with_date["date"] = datetime.utcnow().isoformat() + "Z"

    # Step 3: Now it's safe to $push to the correct history.<feature_name> array
    users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$push": {f"history.{feature_name}": result_with_date}}
    )