from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

def create_preprocessing_pipeline():
    categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'NumOfProducts']
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    return preprocessor

def train_models(X_processed, y):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    trained_models = {}
    cv_results = {}
    best_model = None
    best_f1 = 0

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()

    for name, model in models.items():
        # Create a pipeline that first transforms the data and then fits the model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', model)])
        
        # Perform 5-fold cross-validation
        cv_f1_scores = cross_val_score(pipeline, X_processed, y, cv=5, scoring='f1')
        cv_accuracy_scores = cross_val_score(pipeline, X_processed, y, cv=5, scoring='accuracy')
        cv_precision_scores = cross_val_score(pipeline, X_processed, y, cv=5, scoring='precision')
        cv_recall_scores = cross_val_score(pipeline, X_processed, y, cv=5, scoring='recall')

        # Train the model on the full dataset
        pipeline.fit(X_processed, y)
        trained_models[name] = pipeline  # Store the trained model
        
        # Calculate mean scores
        cv_results[name] = {
            'cv_f1_mean': cv_f1_scores.mean(),
            'cv_accuracy_mean': cv_accuracy_scores.mean(),
            'cv_precision_mean': cv_precision_scores.mean(),
            'cv_recall_mean': cv_recall_scores.mean()
        }

        # Save the model if it's the best one based on F1 Score
        if cv_results[name]['cv_f1_mean'] > best_f1:
            best_f1 = cv_results[name]['cv_f1_mean']
            best_model = pipeline

    # Create models directory outside the notebooks folder
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Save the best model
    if best_model:
        joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))  # Save the best model

    return trained_models, cv_results

    