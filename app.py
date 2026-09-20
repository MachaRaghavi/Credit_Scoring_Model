from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("models/credit_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    duration = float(request.form["duration"])
    credit_amount = float(request.form["credit_amount"])
    installment_commitment = float(
        request.form["installment_commitment"]
    )
    residence_since = float(request.form["residence_since"])
    age = float(request.form["age"])
    existing_credits = float(request.form["existing_credits"])
    num_dependents = float(request.form["num_dependents"])

    input_data = [[
        duration,
        credit_amount,
        installment_commitment,
        residence_since,
        age,
        existing_credits,
        num_dependents
    ]]

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    risk_percentage = round((1 - probability) * 100, 2)

    if prediction == 1:
        result = "Good Credit Risk"
        message = "The model indicates a relatively lower credit risk."
    else:
        result = "Higher Credit Risk"
        message = "The model indicates a relatively higher credit risk."

    return render_template(
        "index.html",
        result=result,
        message=message,
        risk=risk_percentage
    )


if __name__ == "__main__":
    app.run(debug=True)
