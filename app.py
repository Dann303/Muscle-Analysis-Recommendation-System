from flask import Flask
from routes.users import users_bp
from routes.muscles import muscles_bp
from routes.features import features_bp

app = Flask(__name__)

@app.route('/')
def home():
    return "Server is alive"

app.register_blueprint(users_bp, url_prefix='/users')
app.register_blueprint(muscles_bp, url_prefix='/')
app.register_blueprint(features_bp, url_prefix='/users')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
