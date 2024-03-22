from flask import Flask, render_template, request
from sentiment_analyser import sentiment_predictor

app = Flask(__name__)

@app.route("/", methods = ["POST", "GET"])
def index():
    review = request.form.get("review")
    if review:
        sent = sentiment_predictor(review)
        score = sent.sentiment_analyser()
    else:
        score = None
    return render_template("index.html", score = score)

if __name__ == '__main__':
    app.run(debug = True, port = 8003)