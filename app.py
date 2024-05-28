# app.py
from flask import Flask, request, render_template
from bilstm import predict_emotion

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = ""
    if request.method == 'POST' and request.form['text'] != "":
        text = request.form['text']
        emotion = predict_emotion(text)
    return render_template('index.html', emotion=emotion)


if __name__ == '__main__':
    app.run(debug=True)
