#This file only to be used for referencing purposes

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the SVR model
model = pickle.load(open('svrmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('templates.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define feature names
        cut_options = ['Fair', 'Good', 'Ideal', 'Premium', 'Very Good']
        color_options = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
        clarity_options = ['I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2']
        numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
        
        # Collect categorical data (one-hot encoding)
        input_data = [0] * (len(cut_options) + len(color_options) + len(clarity_options))
        
        cut_value = request.form.get('cut')
        if cut_value in cut_options:
            input_data[cut_options.index(cut_value)] = 1
        
        color_value = request.form.get('color')
        if color_value in color_options:
            input_data[len(cut_options) + color_options.index(color_value)] = 1
        
        clarity_value = request.form.get('clarity')
        if clarity_value in clarity_options:
            input_data[len(cut_options) + len(color_options) + clarity_options.index(clarity_value)] = 1
        
        # Collect numerical data
        for feature in numerical_features:
            input_data.append(float(request.form.get(feature, 0)))
        
        features_array = np.array([input_data])
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        return render_template('templates.html', prediction_text=f'Predicted Price: ${prediction:.2f}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

#Note: Please make sure to check your model's file path and add it accordingly to your main.py file and also you can chnage the port at the end accordignly as per need