# Symptom-Based Disease Prediction using Random Forest and Support Vector Machine

This project predicts diseases from user-reported symptoms using machine learning. It implements two complementary classifiers — Random Forest and Support Vector Classifier (SVC) — and combines their outputs into a hybrid decision pipeline.


User symptom inputs are encoded into symptom–severity feature vectors. Both models produce probability distributions over disease classes; the system extracts the top-3 disease candidates from each model and computes a hybrid confidence score that blends model probability with a severity-based score. The pipeline then selects the model (RF or SVM) whose top prediction has the higher final confidence, producing a ranked list of likely diagnoses.  

---

1. Dependencies

- Python 3.8 or above  
- numpy  
- pandas  
- scikit-learn  
- joblib  
- matplotlib (for visualizations)  
- seaborn (optional)  
- notebook or jupyterlab (to run the `.ipynb` file)

---

2. Requirements

All dependencies can be installed using:

pip install -r requirements.txt

---

3. Description of Each File
|
Main Notebook:

-mlp.ipynb — The main notebook that handles data preprocessing, model training (SVM and Random Forest), and prediction pipeline creation. It also saves the trained models and scaler objects into the saved_models folder.

Data Files:

-Training_like_AI_Doctor.csv : Primary dataset used for training the models.
-Symptom-severity.csv : Maps each symptom to its severity weight, used for feature encoding and confidence scoring.
-symtoms_df.csv : Symptom name reference file.
-description.csv : Contains disease descriptions for output display.
-precautions_df.csv : Lists recommended precautions for each disease.
-medications.csv : Maps diseases to suggested medications.
-diets.csv : Contains dietary recommendations for diseases.
-workout_df.csv : Contains suggested exercises and activities for each condition.

Saved Models

saved_models/ai_doctor_pipeline.pkl : Full trained model pipeline integrating preprocessing and prediction.

saved_models/svm_model.pkl : Standalone SVM model used for top-3 disease prediction.

saved_models/scaler.pkl : StandardScaler object used for feature normalization.

saved_models/model_metadata.pkl : Metadata storing training configurations, metrics, and mappings.

Instructions for Training and Inference
Training:

To train the models from scratch:

Open the mlp.ipynb notebook.
Run all cells sequentially.

The notebook will:
1. Load and preprocess the dataset (Training_like_AI_Doctor.csv)
2. Encode symptoms using Symptom-severity.csv
3. Train both Random Forest and SVM models
4. Save trained models to the saved_models/ folder

After execution, the models and scalers will be available for inference.

Inference (Prediction):

1. You can use the same notebook (mlp.ipynb) to test predictions.

Provide user symptom inputs in the required format (e.g., a list of symptoms).

The pipeline will:
1. Generate predictions from both models (RF and SVM)
2. Extract the top-3 disease predictions from each
3. Compare their probabilities and select the higher-confidence result

The final output includes:
1. Top predicted disease(s)
2. Description, precautions, medications, diet, and workout suggestions from the corresponding CSV files


Alternatively, you can run the `main.py` file to interact with the project through a web interface.

1. Open a terminal in the project folder and run:
   streamlit run main.py


Contributors

Member 1: Data collection and preprocessing

Member 2: Model training and evaluation

Member 3: Results analysis and documentation

