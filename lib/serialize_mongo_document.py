from bson import ObjectId

def serialize_mongo_document(doc):
    # Convert ObjectId and bytes to string
    for key, value in doc.items():
        if isinstance(value, bytes):
            doc[key] = value.decode('utf-8', errors='ignore')
        elif isinstance(value, ObjectId):
            doc[key] = str(value)
    return doc