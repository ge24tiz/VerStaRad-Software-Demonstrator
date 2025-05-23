from flask import Flask, render_template, request
import numpy as np
from joblib import load

app = Flask(__name__)

# 1. load trained prediction models and scaler
svr_y1_model = load("models/svr_y1_model.joblib")
svr_y2_model = load("models/svr_y2_model.joblib")
xgb_y3_model = load("models/xgb_y3_model.joblib")
scaler = load("models/scaler.joblib")

# 2. define input parameters
factors = [
    "MastHeight", "MastLength", "MastWallThickness1", "MastWidth", "MastWallThickness2",
    "MastAddMass", "MastCGX", "TraverseMass", "TraverseCGX", "WheelDistance", "WheelDiameter",
    "Hardness", "LiftMass", "WheelA", "WheelV", "WheelSlipTime", "LiftA"
]

# 3. value range of each input parameter [min, max, unit]
factor_ranges = {
    "MastHeight":         {"min": 8,       "max": 45,    "unit": "m"},
    "MastLength":         {"min": 0.8,     "max": 1.2,   "unit": "m"},
    "MastWallThickness1": {"min": 0.001,   "max": 0.02,  "unit": "m"},
    "MastWidth":          {"min": 0.2,     "max": 0.5,   "unit": "m"},
    "MastWallThickness2": {"min": 0.005,   "max": 0.01,  "unit": "m"},
    "MastAddMass":        {"min": 200,     "max": 2000,  "unit": "kg"},
    "MastCGX":            {"min": 1.5,     "max": 4,     "unit": "m"},
    "TraverseMass":       {"min": 1000,    "max": 5000,  "unit": "kg"},
    "TraverseCGX":        {"min": 2,       "max": 4,     "unit": "m"},
    "WheelDistance":      {"min": 0.3,     "max": 0.8,   "unit": "m"},
    "WheelDiameter":      {"min": 0.3,     "max": 0.8,   "unit": "m"},
    "Hardness":           {"min": 500,     "max": 1500,  "unit": "MPa"},
    "LiftMass":           {"min": 1000,    "max": 4500,  "unit": "kg"},
    "WheelA":             {"min": 0.5,     "max": 2,     "unit": "m/s²"},
    "WheelV":             {"min": 1,       "max": 5,     "unit": "m/s"},
    "WheelSlipTime":      {"min": 0.25,    "max": 2,     "unit": "s"},
    "LiftA":              {"min": 0.5,     "max": 2,     "unit": "m/s²"}
}

@app.route('/', methods=['GET', 'POST'])
def home():
    """main page route includes prediction of cycles+Real+Norm"""
    user_inputs = {factor: {"real": "", "norm": ""} for factor in factors}
    cycles_input = "100"
    y1 = y2 = y3 = None
    error = None

    if request.method == 'POST':
        try:
            # read cycles
            cycles_str = request.form.get("cycles", "100").strip()
            if cycles_str == "":
                cycles_str = "100"
            cycles_val = float(cycles_str)

            # collect norm
            inputs_norm = []
            for factor in factors:
                norm_str = request.form.get(f"{factor}_norm", "").strip()
                if norm_str == "":
                    norm_str = "0"  # empty => 0
                val_norm = float(norm_str)
                inputs_norm.append(val_norm)

                # echo real/norm
                real_str = request.form.get(f"{factor}_real", "")
                user_inputs[factor]["real"] = real_str
                user_inputs[factor]["norm"] = norm_str

            # predict
            input_array = np.array(inputs_norm).reshape(1, -1)
            y1 = svr_y1_model.predict(input_array)[0]
            y2 = svr_y2_model.predict(input_array)[0]
            y3 = xgb_y3_model.predict(input_array)[0]

            # if "cycles" is a multiple of 100 => amplify wear result
            if cycles_val % 100 == 0 and cycles_val >= 100:
                multiple = cycles_val / 100
                y1 *= multiple
                y2 *= multiple
                y3 *= multiple

            y1 = round(y1, 4)
            y2 = round(y2, 4)
            y3 = round(y3, 4)

            cycles_input = cycles_str

        except Exception as e:
            error = str(e)

    return render_template(
        'index.html',
        factors=factors,
        factor_ranges=factor_ranges,
        user_inputs=user_inputs,
        cycles=cycles_input,
        y1=y1,
        y2=y2,
        y3=y3,
        error=error
    )

@app.route('/about')
def about():
    """About page route includes overview of this demonstrator, workflow and parameter description usw."""
    return render_template('about.html')

@app.route('/params')
def params():
    """Parameter Description describes how input parameters are defined"""
    return render_template(
        'params.html')

if __name__ == '__main__':
    app.run(debug=True)











