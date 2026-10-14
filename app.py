import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained Naive Bayes model from naive.pkl
with open('naive.pkl', 'rb') as f:
    model = pickle.load(f)

# Embedded HTML & CSS Template with an attractive, modern layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            --card-bg: #ffffff;
            --text-main: #1f2937;
            --text-muted: #6b7280;
            --border: #d1d5db;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: var(--card-bg);
            width: 100%;
            max-width: 650px;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        }

        header {
            text-align: center;
            margin-bottom: 30px;
        }

        header h1 {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 8px;
        }

        header p {
            font-size: 14px;
            color: var(--text-muted);
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            color: #374151;
        }

        input, select {
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
            background-color: #f9fafb;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
            background-color: #ffffff;
        }

        button {
            grid-column: span 2;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            margin-top: 10px;
        }

        button:hover {
            background-color: var(--primary-hover);
        }

        button:active {
            transform: scale(0.98);
        }

        .result-card {
            margin-top: 25px;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
            font-size: 15px;
            font-weight: 600;
            animation: fadeIn 0.3s ease-in-out;
        }

        .result-success {
            background-color: #ecfdf5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }

        .result-danger {
            background-color: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-5px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            .form-group.full-width {
                grid-column: span 1;
            }
            button {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Employee Prediction System</h1>
            <p>Enter employee parameters below to run Naive Bayes classification</p>
        </header>

        <form method="POST" action="/predict">
            <div class="form-grid">
                <!-- Education (Assuming encoded/mapped categories: e.g., 0 for Bachelors, 1 for Masters, 2 for PhD or similar numeric mapping based on your training data) -->
                <div class="form-group">
                    <label for="education">Education</label>
                    <select name="education" id="education" required>
                        <option value="0">Bachelors</option>
                        <option value="1">Masters</option>
                        <option value="2">PHD</option>
                    </select>
                </div>

                <!-- JoiningYear -->
                <div class="form-group">
                    <label for="joining_year">Joining Year</label>
                    <input type="number" name="joining_year" id="joining_year" placeholder="e.g., 2018" value="2018" required>
                </div>

                <!-- City -->
                <div class="form-group">
                    <label for="city">City</label>
                    <select name="city" id="city" required>
                        <option value="0">Bangalore</option>
                        <option value="1">Pune</option>
                        <option value="2">New Delhi</option>
                    </select>
                </div>

                <!-- PaymentTier -->
                <div class="form-group">
                    <label for="payment_tier">Payment Tier</label>
                    <select name="payment_tier" id="payment_tier" required>
                        <option value="1">Tier 1</option>
                        <option value="2" selected>Tier 2</option>
                        <option value="3">Tier 3</option>
                    </select>
                </div>

                <!-- Age -->
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" name="age" id="age" placeholder="e.g., 28" value="28" required>
                </div>

                <!-- Gender -->
                <div class="form-group">
                    <label for="gender">Gender</label>
                    <select name="gender" id="gender" required>
                        <option value="0">Male</option>
                        <option value="1">Female</option>
                    </select>
                </div>

                <!-- EverBenched -->
                <div class="form-group">
                    <label for="ever_benched">Ever Benched</label>
                    <select name="ever_benched" id="ever_benched" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <!-- ExperienceInCurrentDomain -->
                <div class="form-group">
                    <label for="experience">Experience In Domain (Years)</label>
                    <input type="number" name="experience" id="experience" placeholder="e.g., 3" value="3" required>
                </div>

                <button type="submit">Run Prediction</button>
            </div>
        </form>

        {% if prediction_text is not none %}
            <div class="result-card {% if prediction == 1 %}result-danger{% else %}result-success{% endif %}">
                {{ prediction_text }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features corresponding to model input features:
        # Education, JoiningYear, City, PaymentTier, Age, Gender, EverBenched, ExperienceInCurrentDomain[cite: 1]
        features = [
            float(request.form['education']),
            float(request.form['joining_year']),
            float(request.form['city']),
            float(request.form['payment_tier']),
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['ever_benched']),
            float(request.form['experience'])
        ]
        
        # Reshape for single prediction
        final_features = np.array([features])
        prediction = model.predict(final_features)[0]
        
        # Customize output message based on your target classes
        if prediction == 1:
            prediction_text = "Prediction Result: High likelihood of turnover / positive condition (Class: 1)"
        else:
            prediction_text = "Prediction Result: Low likelihood / negative condition (Class: 0)"
            
        return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text, prediction=prediction)
    
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        return render_template_string(HTML_TEMPLATE, prediction_text=error_msg, prediction=1)

if __name__ == '__main__':
    app.run(debug=True)
