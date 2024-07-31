import os
from flask import Flask, request, jsonify, render_template
from model.freshprompt import llm
from dotenv import load_dotenv
from search.bing import *

app = Flask(__name__)
load_dotenv()

def configure():
    load_dotenv()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    question = data['question']
    model_name = "azureai"
    check_premise = True

    print('chatbot answer loading....')
    answer, _, _ = llm.call_freshprompt(model_name, question, check_premise=check_premise)
    
    # Return only the answer
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
