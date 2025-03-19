#IMPORTS FLASK
from flask import Flask, render_template, request
#IMPORTS ML
import joblib

app = Flask(__name__)
#Necessário para o funcionamento da ML
model = joblib.load(open("models/twitter.joblib", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("entrada.html")
@app.route("/resposta", methods=["POST"])
def resposta():
    resposta = request.form["comentário"]

    # Fazendo previsões nos novos dados
    prediction = model.predict([resposta])

    return render_template("saída.html", resposta=resposta,
                                                           prediction=prediction[0])
if __name__ == "__main__":
    app.run(debug=True)