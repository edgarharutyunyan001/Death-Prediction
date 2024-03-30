# Classification_survival_dataset

This project is designed to classify whether patient will survive or not based on the results of the tests. 

Before runing the run_pipline.py, user should provide filename of the dataset and set mode: "training" or "testing" (default is "tranining").
run_pipeline.py connects preprocessor.py and model.py files.

preprocessor.py provides Preprocessor class which reads data, deletes unnecessary columns and fills NaN values using KNN Imputer.
KNN Imputer takes 5 neighbors and weighted distance, as hyperparametres.
In the end Preprocessor returns X, y if the mode is set "training" and returns only X if the mode is set "testing".

model.py Model class which has fit and predict methods.
"fit" method trains model and saves it as Model.joblib file.
"predict" method reads already saved trained model and returns predicted probabilites for the second class.
In the end this probablities are saved in predictions.json file with threshold.

For model evaluating I took into attention F1_score and AUC, which were highest for RandomForestClassifier. 
Thus as a final model for Survival_dataset classificaton has been chosen RandomForestClassifier. 
It gets weights for corresponding classes, and minimum sapmles of a split is set 5. 

Results of all observed models are represented below:
![table](https://github.com/AvMariam/Classification_survival_dataset/assets/125482296/e0537a1d-355b-4c93-bd18-6c1319a95327)
