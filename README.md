## Project Title - Employee Performance Analytics
End-to-End Employee Performance Analytics project using Machine Learning, Flask, Docker, and CI/CD deployment workflows.

## Business Problem
INX Future Inc. has experienced a decline in employee performance despite being recognized for attracting top talent. The organization’s leadership seeks data-driven insights to identify the factors contributing to reduced workforce performance while maintaining employee morale and protecting the company’s employer brand reputation.

This project applies data analytics and machine learning techniques to uncover key performance drivers and support strategic HR decision-making.

## Objectives
This project aims to analyze employee data to:

- Identify the key factors influencing employee performance.
- Analyze performance trends across different departments.
- Build a machine learning model to predict employee performance levels.
- Generate actionable insights to support workforce productivity and strategic HR decision-making while maintaining employee morale and brand reputation.

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn.

## Workflow Pipeline
- Data Collection
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Model Training
- Evaluation
- Deployment

## Key Insights
A positive relationship was observed between employee work-life balance and performance ratings.
Employee environment satisfaction emerged as one of the strongest factors influencing employee performance.
Salary hike percentage showed a noticeable positive impact on employee performance outcomes.

## Model Performance Summary

After training and evaluating multiple classification models on the employee performance dataset, the following results were obtained:

| Model | Accuracy | F1 Score (Macro) | Hyperparameter Tuning | Cross Validation |
|---|---|---|---|---|
| Random Forest | 0.93 | 0.88 | Yes | Yes |
| SVC | 0.82 | 0.72 | Yes | Yes |
| XGBoost | 0.93 | 0.88 | No | Yes |
| ANN (MLPClassifier) | 0.84 | 0.76 | No | Yes |

### Best Performing Models
- Random Forest and XGBoost achieved the highest predictive performance with an accuracy score of 93% and a macro F1-score of 0.88.
- Random Forest was selected as the preferred model due to its strong predictive performance, interpretability, and robustness.
- Cross-validation was applied across all models to improve model reliability and reduce overfitting risk.


## Software and Tools requirements

1. [Github Account](https://github.com)
2. [HerokuAccount](https://heroku.com)
3. [VSCodeIDE](https://code.visualstudio.com/)
4. [GitCLI](https://git-scm.com/install/)

## Future Improvements
Future Enhancements:
- Deploy Model Using FASTAPI.

