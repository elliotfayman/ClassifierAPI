import markdown
import os
import jwt
import functools
import datetime
import ML

# Import the framework
from flask import Flask, g, jsonify, request, make_response
from flask_restful import Resource, Api, reqparse


# Create an instance of Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'password'

def token_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        token = request.args.get('token')
        if not token:
            return jsonify({'message' : 'Token is missing!'}), 401
        try:
            data =jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except:
            return jsonify({'message' : 'Token is invalid!'}), 401
        return f(*args, **kwargs)
    return decorated



@app.route('/preidct')
@token_required
def preidct(*args, **kwargs):
    image = request.args.get('image')

    if not image:
        return jsonify({'message' : 'Image URL is missing!'})
    else:
        return jsonify({'message' : f'{ML.main(image)}'})


@app.route('/generate_key')
def generate_key():
    auth = request.authorization


    if auth and auth.password == 'password':
        t = jwt.encode({'user': auth.username, 'exp' : datetime.datetime.utcnow() + datetime.timedelta(minutes=200)}, app.config['SECRET_KEY'], algorithm='HS256')
        print(type(t))
        return jsonify({'token': t})

    return make_response('Could not verify!', 401, {'WWW-Authenticate' : 'Basic realm="Login Required"'})


if __name__ == '__main__':
    app.run(debug=True)