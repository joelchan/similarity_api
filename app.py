from flask import Flask, jsonify, request
from lsaSim import rank_paths

app = Flask(__name__)


# use this to test if the API is live
@app.route('/')
def hello():
    return "Hello World!"

# given a set of ideas and a knowledge test, return the 
@app.route('/LSArank', methods=['GET', 'POST'])
def get_sim_ranks():
    data = request.get_json()
    ideas = data['ideas']
    ideaBag = " ".join(ideas)
    return jsonify(rankings = rank_paths(ideaBag))

if __name__ == '__main__':
    app.run(debug=True)