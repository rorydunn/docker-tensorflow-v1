from flask.ext.api import FlaskAPI
from flask import request, current_app, abort
from functools import wraps

app = FlaskAPI(__name__)
from werkzeug.debug import DebuggedApplication
app.wsgi_app = DebuggedApplication(app.wsgi_app, True)
app.debug = True
app.config.from_object('settings')

@app.route('/')
def hello_world():
    return 'ML Root Endpoint'

def token_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-TOKEN', None) != current_app.config['API_TOKEN']:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

@app.route('/tensorflowpredict', methods=['POST'])
@token_auth
def tensorflowpredict():
    from models import model
    description = request.data.get('description')
    return model.tensorflowPredict(description)

if __name__ == '__main__':
    app.debug = True
    app.run()
