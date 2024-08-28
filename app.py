from flask import Flask
import os

def create_app():
    app = Flask(__name__)

    from routes.ml import ml
    app.register_blueprint(blueprint=ml, url_prefix="/ml")

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))