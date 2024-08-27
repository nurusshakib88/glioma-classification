from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '../svm_classifier_model.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Ensure svm_classifier_model.pkl is in the correct location.")

best_ann_classifier = joblib.load(model_path)

# Define a function to predict cancer grade
def predict_cancer_grade(gene_expression_values):
    # Ensure gene_expression_values is in the correct format (numpy array or pandas DataFrame)
    input_data = np.array([gene_expression_values])  # Convert input to numpy array
    
    # Predict cancer grade
    predicted_grade = best_ann_classifier.predict(input_data)
    
    return predicted_grade[0]  # Assuming single prediction

# Define route for home page
@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        if request.method == 'POST':
            # Get form data from POST request
            gene1 = float(request.form['gene1'])
            gene2 = float(request.form['gene2'])
            gene3 = float(request.form['gene3'])
            gene4 = float(request.form['gene4'])
            gene5 = float(request.form['gene5'])
            
            # Make prediction
            gene_values = [gene1, gene2, gene3, gene4, gene5]
            predicted_grade = predict_cancer_grade(gene_values)
            
            # Render result template with prediction
            return render_template('index.html', predicted_grade=predicted_grade)
        
        # Render home page template for GET request
        return render_template('index.html')
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
