# app.py
from flask import Flask, request, render_template
from bilstm import predict_emotion
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = ""
    if request.method == 'POST' and request.form['text'] != "":
        text = request.form['text']
        emotion = predict_emotion(text)
    return render_template('index.html', emotion=emotion)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))

    is_prod = os.environ.get('RAILWAY_ENVIRONMENT_NAME') is not None

    app.run(host='0.0.0.0', port=port, debug=not is_prod)
