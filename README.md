# Car Insurance Claim Risk Assessment

## Project Overview
"On the Road" car insurance requested a predictive model to determine the likelihood of a customer making a claim during their policy period. Due to limited deployment infrastructure, the company required a simplified solution: identifying the **single feature** that results in the best-performing model.

This repository contains a Python analysis that iterates through customer data features to find the optimal predictor for insurance claims based on model accuracy.

## Dataset
The analysis utilizes `car_insurance.csv`, containing **10,000 records** with **18 columns** detailing client demographics, driving history, and vehicle information.

**Key Features:**
* **Demographics:** `age`, `gender`, `education`, `income`, `married`, `children`, `postal_code`.
* **Driving History:** `driving_experience`, `speeding_violations`, `duis`, `past_accidents`.
* **Vehicle Info:** `vehicle_ownership`, `vehicle_year`, `vehicle_type`, `annual_mileage`.
* **Financial:** `credit_score`.
* **Target Variable:** `outcome` (0 = No Claim, 1 = Claim).

## Methodology

### 1. Data Preprocessing
* **Missing Values:** Identified missing data in `credit_score` and `annual_mileage` columns.
* **Imputation:** Filled missing values with the mean of their respective columns to preserve data integrity for modeling.

### 2. Modeling Strategy
To find the single best feature, the analysis employed an iterative approach:
1.  **Feature Selection:** Isolate every feature column (excluding `id` and `outcome`).
2.  **Model Loop:** Iterate through each feature and build a Logistic Regression model (`logit`) using `statsmodels`.
    * *Constraint Check:* The script includes a check to skip features with less than 2 unique values to prevent convergence errors.
3.  **Evaluation:** Calculate the accuracy for each model using a confusion matrix (True Negatives + True Positives / Total Observations).

## Results
After testing all potential variables, the analysis identified the single most predictive feature.

| Best Feature | Accuracy |
| :--- | :--- |
| **driving_experience** | **77.71%** |

The feature `driving_experience` yielded the highest accuracy of roughly **0.7771**, making it the recommended variable for the client's simple production model.

## Technologies Used
* **Python**: Core programming language.
* **pandas**: Data manipulation and I/O.
* **statsmodels**: Logistic regression modeling (`logit`).
* **numpy**: Numerical operations.

## Usage
To replicate this analysis:
1. Ensure `car_insurance.csv` is in the root directory.
2. Run the Jupyter Notebook or Python script.
3. The script will output the list of model accuracies and a dataframe containing the best feature.
