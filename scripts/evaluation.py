from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from model_training import LogisticRegression, RandomForestClassifier, KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder





def evaluate_model_comparison(X_processed, y_processed, X_original, y_original):
    # Split the original data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X_original, y_original, test_size=0.2, random_state=42)

    # Define numeric and categorical features
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'NumOfProducts']

    # Create a ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )

    # Define your models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }
    
    comparison_results = {}
    
    for name, model in models.items():
        # Create a pipeline with preprocessor and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train on processed data
        pipeline.fit(X_processed, y_processed)
        y_pred_processed = pipeline.predict(X_test)

        # Train on original data
        pipeline.fit(X_train, y_train)
        y_pred_original = pipeline.predict(X_test)

        # Calculate metrics
        cv_f1 = cross_val_score(pipeline, X_processed, y_processed, cv=5, scoring='f1').mean()
        cv_accuracy = cross_val_score(pipeline, X_processed, y_processed, cv=5, scoring='accuracy').mean()
        cv_precision = cross_val_score(pipeline, X_processed, y_processed, cv=5, scoring='precision').mean()
        cv_recall = cross_val_score(pipeline, X_processed, y_processed, cv=5, scoring='recall').mean()
        
        tt_f1 = f1_score(y_test, y_pred_original)
        tt_accuracy = accuracy_score(y_test, y_pred_original)
        tt_precision = precision_score(y_test, y_pred_original)
        tt_recall = recall_score(y_test, y_pred_original)

        comparison_results[name] = {
            'cv_f1_mean': cv_f1,
            'cv_accuracy_mean': cv_accuracy,
            'cv_precision_mean': cv_precision,
            'cv_recall_mean': cv_recall,
            'tt_f1': tt_f1,
            'tt_accuracy': tt_accuracy,
            'tt_precision': tt_precision,
            'tt_recall': tt_recall
        }

    return comparison_results


    