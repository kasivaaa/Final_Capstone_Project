### Final_Capstone_Project

Predicting Child Personality Traits Using Household Structure and parental behavior
### 1. Project Overview

This project investigates whether household structure, parental behaviour, socioeconomic conditions, and other developmental factors can be used to predict personality traits in young adulthood using machine learning.

The project will use longitudinal data from the National Longitudinal Study of Adolescent to Adult Health (Add Health). Information collected during adolescence, such as family structure and parent-child relationships, will be used to predict Big Five personality traits measured later in young adulthood.

The five personality traits are:

Openness
Conscientiousness
Extraversion
Agreeableness
Neuroticism

The purpose is not to determine whether being raised in a single-parent or two-parent household causes a particular personality. Instead, the project will investigate whether household structure and parental behaviour provide meaningful predictive information about later personality.

### 2. Problem Statement

Children grow up in different family environments, including single-parent, two-parent, and blended households. They also experience different levels of parental support, communication, involvement, and interaction.

However, household structure alone may not explain differences in personality. Other factors such as socioeconomic circumstances, parental behaviour, and individual characteristics may also contribute.

Therefore, this project asks:

How effectively can household structure, parental behaviour, and other developmental factors predict Big Five personality traits in young adulthood?

### 3. Main Objective

To develop and evaluate machine learning models that predict Big Five personality traits using household structure, parental behaviour, socioeconomic characteristics, and other relevant developmental factors.

### 4. Research Questions
Is household structure associated with differences in Big Five personality traits?
Which parental behaviours provide the strongest predictive information about personality?
Does parental behaviour provide more predictive information than household structure alone?
Does household structure still contribute useful predictive information after other factors are considered?
Which machine learning model provides the best predictions of personality traits?

### 5. Methodology

The project will follow these main stages:

Data Understanding → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Statistical Analysis → Machine Learning → Model Evaluation → Explainable AI → Streamlit Deployment

Several regression models will be compared, including:

Linear Regression
Ridge Regression
Random Forest
Potentially XGBoost

Dimensionality-reduction techniques such as PCA and Isomap will also be investigated where appropriate.

Model performance will be evaluated using:

MAE
RMSE
R²

Explainable AI techniques such as feature importance and SHAP will be used to understand which factors contribute most to the predictions.

### 6. Proposed Modelling Approach

The project will compare models using progressively richer information:

Model 1: Household structure only

↓

Model 2: Household structure + demographics

↓

Model 3: Household structure + demographics + parental behaviour

↓

Model 4: Household structure + parental behaviour + socioeconomic and other relevant factors

This will allow the project to investigate whether household structure provides additional predictive value beyond parenting and other environmental factors.

### 7. Expected Outcome

The final system will determine:

Whether family and parental characteristics contain useful predictive information about personality.
Which factors are most strongly associated with the predicted personality traits.
Which machine learning approach performs best.
Whether adding parental behaviour and socioeconomic factors improves predictions compared with using household structure alone.

A Streamlit application will be developed to demonstrate the final predictive model and present the findings interactively.

### 8. Ethical Consideration

The project will focus on prediction and association, not causation.

The model will not be presented as a psychological diagnostic tool or as evidence that a particular household structure causes a specific personality.

Its purpose is to demonstrate how machine learning can be responsibly applied to psychological and developmental research.
