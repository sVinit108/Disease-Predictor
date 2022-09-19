# Importing essential libraries
import sys
from flask import Flask, render_template, request
import pickle
import numpy as np

import pandas as pd
train_df = pd.read_csv("training.csv")
train_df.drop(['Unnamed: 133'],axis='columns',inplace=True)

filename = '1-disease-prediction-rfc-model.pkl'
Model1 = pickle.load(open(filename, 'rb'))
filename = '2-disease-prediction-rfc-model.pkl'
Model2 = pickle.load(open(filename, 'rb'))
filename = '3-disease-prediction-rfc-model.pkl'
Model3 = pickle.load(open(filename, 'rb'))
filename = '4-disease-prediction-rfc-model.pkl'
Model4 = pickle.load(open(filename, 'rb'))


trin_x = train_df.iloc[:,:-1]
symptoms = trin_x.columns.values

symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index
 
data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne',
       'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma',
       'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
       'Common Cold', 'Dengue', 'Diabetes ',
       'Dimorphic hemmorhoids(piles)', 'Drug Reaction',
       'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia',
       'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine',
       'Osteoarthristis', 'Paralysis (brain hemorrhage)',
       'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis',
       'Typhoid', 'Urinary tract infection', 'Varicose veins',
       'hepatitis A']
}

def predictDisease(symptoms):
    symptoms = symptoms.split(",")
     
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1

    input_data = np.array(input_data).reshape(1,-1)

    model1_prediction = data_dict["predictions_classes"][Model1.predict(input_data)[0]]
    model2_prediction = data_dict["predictions_classes"][Model2.predict(input_data)[0]]
    #model3_prediction = data_dict["predictions_classes"][Model3.predict(input_data)[0]]
    model4_prediction = data_dict["predictions_classes"][Model4.predict(input_data)[0]]
    #final_prediction = f"{np.mod(model1_prediction,model2_prediction,model4_prediction)}"

    # predictions = f''' SVM: {model1_prediction} \n 
    # Decision Tree: {model2_prediction} \n 
    # Random Forest: {model4_prediction} \n '''
    predictions = f"{model2_prediction}"

    
#"Naives Bayes": {model3_prediction},
    return predictions

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
	#return render_template('second.html')
    return render_template('index2.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        syms = request.form.get('Symptoms')
        return render_template('result.html', prediction=predictDisease(symptoms=syms)) 

if __name__ == '__main__':
	app.run(debug=True)

