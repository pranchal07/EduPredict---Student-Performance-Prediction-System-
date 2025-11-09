A machine learning web application to predict student math scores based on demographic and academic performance data. Built with Flask, scikit-learn, and deployed on AWS EC2, this project demonstrates a complete end-to-end ML workflow, from data exploration to deployment.

## 📋 Table of Contents

1.  [Project Overview](#project-overview)
3.  [Features](#features)
4.  [Technology Stack](#technology-stack)
5.  [Project Architecture & Workflow](#project-architecture-&-workflow)
6.  [Getting Started](#getting-started)
7.  [Running the Application](#running-the-application)
8.  [Jupyter Notebooks](#jupyter-notebooks)
9.  [Development Workflow](#development-workflow)
10. [Model Performance](#model-performance)
11. [Configuration](#configuration)
12. [Troubleshooting](#troubleshooting)
13. [API Endpoints](#api-endpoints)
14. [Contributing](#contributing)
15. [License](#license)
16. [Future Enhancements](#future-enhancements)


## <a id="project-overview"></a>📖 Project Overview
<!-- ## 📖 Project Overview -->

This project aims to understand and predict student performance in mathematics. By analyzing features such as gender, ethnicity, parental education level, and test preparation, we can build a model that provides an accurate estimate of a student's math score.

The application serves as a practical example of building and deploying a production-ready machine learning system, complete with a web interface for real-time predictions.

## <a id="features"></a>✨ Features
<!-- ## ✨ Features -->

-   **Predictive Modeling**: Utilizes regression models to predict student math scores.
-   **Comprehensive EDA**: Detailed exploratory data analysis to uncover insights and relationships in the data.
-   **Multi-Model Evaluation**: Trains and evaluates several models (Random Forest, Decision Tree, Gradient Boosting, Linear Regression, CatBoost, AdaBoost, and K-Neighbors) to select the best performer.
-   **Hyperparameter Tuning**: Employs `GridSearchCV` to find the optimal parameters for each model.
-   **Modular Pipeline**: A structured, reusable pipeline for data ingestion, transformation, and model training.
- **Model Persistence**: Saved trained models and preprocessors into pickle files for production use.
-   **Web Interface**: A user-friendly web form built with Flask to input student data and receive instant predictions.
-   **Robust Engineering**: Features custom logging, exception handling, and a modular project structure for maintainability.

## <a id="technology-stack"></a>🛠️ Technology Stack
<!-- ## 🛠️ Technology Stack -->

-   **Backend**: Flask
-   **ML & Data Science**: Scikit-learn, CatBoost, Pandas, NumPy
-   **Data Visualization**: Matplotlib, Seaborn
-   **Development Environment**: Jupyter Notebook, **uv** (or venv/pip)
-   **Deployment**: AWS EC2 with Elastic Beanstalk

## <a id="project-architecture-&-workflow"></a>🏗️ Project Architecture & Workflow
<!-- ## 🏗️ Project Architecture & Workflow -->

The project is organized into a modular structure that separates concerns and makes the system easy to maintain and scale.

### Directory Structure

```
├── artifacts/                          # Stores output files like models and preprocessors
│   ├── model.pkl                       # Trained model object
│   └── preprocessor.pkl                # Preprocessing pipeline object
├── notebooks/                          # Jupyter notebooks for EDA and initial modeling
├── src/                                # Source code for the application
│   ├── components/                     # Core ML pipeline components
│   │   ├── data_ingestion.py           # Data loading and splitting
│   │   ├── data_transformation.py      # Feature engineering and preprocessing
│   │   └── model_trainer.py            # Model training and evaluation
│   ├── pipeline/                       # Manages training and prediction workflows
│   │   ├── prediction_pipeline.py
│   │   └── training_pipeline.py   
│   ├── exception.py                    # Custom exception handling
│   ├── logger.py                       # Logging configuration
│   └── utils.py                        # Utility functions
├── application.py                      # Main Flask application entry point
├── requirements.txt                    # Project dependencies
└── README.md                           # This file
```

### ML Pipeline Workflow

1.  **Data Ingestion (`data_ingestion.py`)**:
    -   Reads the raw data from `notebooks/data/stud.csv`.
    -   Splits the data into training and testing sets.
    -   Saves the raw, train, and test CSVs into the `artifacts/` directory.
    -   Triggers the data transformation and model training steps.

2.  **Data Transformation (`data_transformation.py`)**:
    -   Creates a preprocessing pipeline using `ColumnTransformer`.
    -   Applies `StandardScaler` to numerical features and `OneHotEncoder` to categorical features.
    -   Saves the fitted preprocessor object as `preprocessor.pkl` for later use.

3.  **Model Training (`model_trainer.py`)**:
    -   Receives the transformed data.
    -   Runs a suite of regression models through `GridSearchCV` to find the best model and hyperparameters.
    -   Selects the model with the highest R² score (minimum threshold of 0.6).
    -   Saves the best-performing model as `model.pkl`.

4.  **Prediction (`prediction_pipeline.py` & `application.py`)**:
    -   The Flask app captures user input from the web form.
    -   The `PredictPipeline` loads the saved `preprocessor.pkl` and `model.pkl`.
    -   It transforms the new input data and feeds it to the model to generate a prediction, which is then displayed to the user.

## <a id="getting-started"></a>🚀 Getting Started
<!-- ## 🚀 Getting Started -->

### Step 1: Clone the Repository

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/GoJo-Rika/Student-Performance-Prediction-System.git
cd Student-Performance-Prediction-System
```

### Step 2: Set Up The Environment and Install Dependencies
We recommend using `uv`, a fast, next-generation Python package manager, for setup.



<!-- ## 🔄 Development Workflow -->

This project follows a modular, pipeline-based architecture:

1. **Experimentation**: Initial development in Jupyter notebooks
2. **Modularization**: Successful experiments converted to reusable components
3. **Pipeline Integration**: Components connected in training and prediction pipelines
4. **Error Handling**: Custom exceptions and logging for debugging
5. **Testing**: Iterative testing and refinement
6. **Deployment**: AWS EC2 deployment with Elastic Beanstalk configuration

## <a id="model-performance"></a>📈 Model Performance
<!-- ## 📈 Model Performance -->

The system evaluates multiple algorithms and selects the best performer:
- Minimum R² score threshold: 0.6
- Grid search hyperparameter optimization
- Cross-validation for robust evaluation

## <a id="configuration"></a>🔧 Configuration
<!-- ## 🔧 Configuration -->

### AWS Deployment
- **EC2 Instance**: Configured via `.ebextensions/python.config`
- **WSGI**: Flask application served through `application:application`
- **Environment**: Production-ready with proper logging

### File Structure
```
artifacts/
├── model.pkl               # Trained model
├── preprocessor.pkl        # Feature transformation pipeline
├── train.csv               # Training data
├── test.csv                # Test data
└── data.csv                # Raw data

logs/
└── [timestamp].log         # Application logs
```

## <a id="troubleshooting"></a>🐛 Troubleshooting
<!-- ## 🐛 Troubleshooting -->

**Common Issues:**
1. **Import errors**: Ensure all dependencies are installed
2. **Data not found**: Check `notebooks/data/stud.csv` exists
3. **Model not found**: Run training pipeline first
4. **Prediction errors**: Check input data format

**Debugging:**
- Check logs in `logs/` directory
- Custom exceptions provide detailed error context
- Use logging output for pipeline debugging

## <a id="api-endpoints"></a>📝 API Endpoints
<!-- ## 📝 API Endpoints -->

- `GET /`: Home page
- `GET /predictdata`: Prediction form
- `POST /predictdata`: Submit prediction request

## <a id="contributing"></a>🤝 Contributing
<!-- ## 🤝 Contributing -->

Contributions are welcome! If you have suggestions or want to improve the project, please follow these steps:

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

## <a id="license"></a>📄 License
<!-- ## 📄 License -->

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## <a id="future-enhancements"></a>🎯 Future Enhancements
<!-- ## 🎯 Future Enhancements -->

- REST API for programmatic access
- Model training & retraining pipeline
- Performance monitoring dashboard
- Additional ML algorithms
- A/B testing framework
