from pymongo import MongoClient
import os

uri = 'mongodb+srv://danielekdawi:51uQxsl7IQpV3ESc@recommendationsystem.vprovtj.mongodb.net/?retryWrites=true&w=majority&appName=RecommendationSystem'
client = MongoClient(os.environ.get('MONGO_URI', uri))
db = client['recommendation_system']
users_collection = db['users']