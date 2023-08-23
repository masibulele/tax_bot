from flask import Flask, render_template, request, jsonify
from tax_agent import agent

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    response= agent.run(text)
    return response


if __name__ == '__main__':
    app.run()
