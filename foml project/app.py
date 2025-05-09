from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))

model = None
try:
    model = joblib.load('model.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print("Model not found or failed to load:", e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not trained yet'})
    
    try:
        data = {
            'satisfaction_level': float(request.form['satisfaction_level']),
            'last_evaluation': float(request.form['last_evaluation']),
            'number_project': int(request.form['number_project']),
            'average_montly_hours': int(request.form['average_montly_hours']),
            'time_spend_company': int(request.form['time_spend_company']),
            'work_accident': int(request.form['work_accident']),
            'promotion_last_5years': int(request.form['promotion_last_5years']),
            'salary': request.form['salary']
        }
        salary_map = {'low': 0, 'medium': 1, 'high': 2}
        data['salary'] = salary_map[data['salary']]
        features = np.array([[
            data['satisfaction_level'],
            data['last_evaluation'],
            data['number_project'],
            data['average_montly_hours'],
            data['time_spend_company'],
            data['work_accident'],
            data['promotion_last_5years'],
            data['salary']
        ]])
        prediction = model.predict_proba(features)[0]
        probability_leave = prediction[1] * 100
        probability_stay = 100 - probability_leave
        return jsonify({
            'probability_stay': round(probability_stay, 2),
            'probability_leave': round(probability_leave, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 