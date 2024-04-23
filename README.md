# FairLens CreditAI

Data bias poses a substantial threat to the accurate prediction of credit card default payments, influencing the efficacy, fairness, and transparency of machine learning (ML) models. Numerous existing solutions that anticipate defaulters incorporate variables such as age and gender in their feature set, introducing inherent bias to the data. This bias can compromise the precision and generalizability of ML models, potentially leading to oversight of genuine patterns and relationships in the data or excessive tailoring to specific characteristics present in the training
data.

The initiative to develop a machine learning model for forecasting credit card default risks, with a specific emphasis on integrating AI ethics—particularly focusing on fairness, transparency, and privacy—constitutes a noteworthy advancement in the responsible deployment of artificial intelligence within the financial sector. The core objectives of this project are centered on leveraging cutting-edge machine learning techniques to enhance the accuracy of credit risk predictions. Concurrently, there is a deliberate commitment to prioritizing ethical considerations, addressing potential biases, ensuring transparency in the decision-making process of the model, and upholding stringent measures to safeguard user privacy.

## Dataset Information
Utilizing the "Default of Credit Card Clients" dataset from the
UCI Machine Learning Repository, this project explores predictive modeling in financial
behavior while emphasizing ethical AI practices

The dataset is publicly available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

## Data Card
>30000*24

Data Types:

|# | Column                      | Data Type | Description |
|---|---------------------------|----------|--------------|
|1 | ID                          | Numerical | ID of each client |
|2 | LIMIT_BAL                   | Numerical | Amount of given credit in NT dollars (includes individual and family/supplementary credit) |
|3 | SEX                         | Numerical | Gender (1=male, 2=female) |
|4 | EDUCATION                   | Numerical | (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown) |
|5 | MARRIAGE                    | Numerical | Marital status (1=married, 2=single, 3=others) |
|6 | AGE                         | Numerical | Age in years |
|7 | PAY_0                       | Numerical | Repayment status in September 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above) |
|8 | PAY_2                       | Numerical | Repayment status in August 2005 (scale same as above) |
|9 | PAY_3                       | Numerical | Repayment status in July 2005 (scale same as above) |
|10| PAY_4                       | Numerical | Repayment status in June 2005 (scale same as above) |
|11| PAY_5                       | Numerical | Repayment status in May 2005 (scale same as above) |
|12| PAY_6                       | Numerical | Repayment status in April 2005 (scale same as above) |
|13| BILL_AMT1                   | Numerical | Amount of bill statement in September 2005 (NT dollar) |
|14| BILL_AMT2                   | Numerical | Amount of bill statement in August 2005 (NT dollar) |
|15| BILL_AMT3                   | Numerical | Amount of bill statement in July 2005 (NT dollar) |
|16| BILL_AMT4                   | Numerical | Amount of bill statement in June 2005 (NT dollar) |
|17| BILL_AMT5                   | Numerical | Amount of bill statement in May 2005 (NT dollar) |
|18| BILL_AMT6                   | Numerical | Amount of bill statement in April 2005 (NT dollar) |
|19| PAY_AMT1                    | Numerical | Amount of previous payment in September 2005 (NT dollar) |
|20| PAY_AMT2                    | Numerical | Amount of previous payment in August 2005 (NT dollar) |
|21| PAY_AMT3                    | Numerical | Amount of previous payment in July 2005 (NT dollar) |
|22| PAY_AMT4                    | Numerical | Amount of previous payment in June 2005 (NT dollar) |
|23| PAY_AMT5                    | Numerical | Amount of previous payment in May 2005 (NT dollar) |
|24| PAY_AMT6                    | Numerical | Amount of previous payment in April 2005 (NT dollar) |
|25| default payment next month | Numerical | Indicates whether the client defaulted on the payment next month (1=yes, 0=no) |
##	Data Rights and Privacy: 
The dataset is anonymized, with all personal identifiers removed to comply with privacy regulations. As per UCI platform regulations, this allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

## Data Planning and Splits
As part of data preprocessing, we will first check for any null values, outliers, and duplicates present in the data and correct them. We will later perform initial exploratory data analysis to get deeper insights into the dataset.
We will split the data in the ratio of 70-15-15 for training, validation, and testing and, ensure that we maintain the distribution of classes across the training, validation, and testing sets such that each class is represented proportionally in each subset.

## Diagram

![Diagram](https://drive.google.com/uc?export=download&id=1rbkIr1U8tqNm1M1AP0dg4P-sr56abreS)

## Files
The code files are present inside airflow/dags/


## Tools Used for MLOps

* GitHub
* Airflow
* DVC
* Google Cloud Platform (GCP)
* MLflow
* TensorFlow Data Validation

## Data Pipeline
We have 2 DAGs in use for our project.
1. Train data DAG
2. Test data DAG

## Train Data Pipeline Components

### 1. Pre-processing Data:
The first stage involves downloading the dataset into the `data` directory. Then the following processes are executed:
- `dataSplit.py`: Responsible for downloading the dataset from the specified source and splitting the data in a 90:10 ratio, where, the split 90% of the train data is stored in `train_val_data.xlsx`.
- `preprocess.py`: We drop the columns 'ID',' EDUCATION', 'MARRIAGE', and 'AGE' from train data as a part of this step to avoid bias based on the mentioned columns and then store the processed data in `train_processed_data.pkl`
### 2. Validate Data and Train Model:
- `train_validate.py`: This function loads processed data from a pickle file, splits it into training and validation sets, in a 75:25 ratio, and dumps them into `train_val_data` and `test_val_data` pickle files respectively<br>
Then it infers the schema from the training data, writes the inferred schema to `schema.pbtxt`, generates statistics from the validation data, validates the statistics against the inferred schema, and logs any anomalies.
- `train_model.py`: Loads training and test data from the pickle file, scales the features using StandardScaler and trains a LightGBM model, logs relevant metrics and the model to MLflow, and saves the run ID to a pickle file.

## Test Data Pipeline Components

### 1. Pre-processing Data:
We use  `test_data.xlsx` from the `dataSplit.py` run from the train DAG. Then the following processes are executed:

- `preprocess.py`: We drop the columns 'ID',' EDUCATION', 'MARRIAGE', and 'AGE' from test data as a part of this step to avoid bias based on the mentioned columns and then store the processed data in `new_processed_data.pkl`

- `new_data_validate.py`: Loads a predefined schema from `schema.pbtxt`, verifies the non-presence of the target feature in the new data,
    generates statistical summaries from the new data, validates these statistics against the predefined schema, and logs any detected anomalies.

- `predict.py`: Loads a pre-trained model and scaler from MLflow, scales the newly processed data, makes predictions using the model, logs the predictions, and saves the predicted data to a CSV file.


## Experimental tracking (Mlflow)
We track our model using mlflow and Python<br>
For different values of learning_rate, num_leaves, max_depth we can see the AUC scores achieved.

![Diagram](https://drive.google.com/file/d/1IbudxVBHbgCeSWhxvGMBN5fACHlS7Kxk/view?usp=drive_link)

## Staging, Production, and Archived models
We rely on MLflow for managing models for Archiving, Staging, and Production as it allows us to reuse the models from the artifacts registry and serve them on a predefined port on the go.

![Diagram](https://drive.google.com/file/d/1nUYTORQB7dZ7PqW2w8tFrvWjFMWnE7Qn/view?usp=sharing)

## Machine Learning Model 

### Train the Model
The model is trained using light gradient-boosting machine model after scaling the features using StandardScaler. It takes preprocessed data file as input, logs relevant metrics, and saves the run ID to a pickle file as output.

### Save the Model
The model is saved in mlflow

### Hyper Parameter Tuning
The model has three hyperparameters namely learning_rate, num_leaves, max_depth. We use MLFLOW for checking the model with different hyperparameter values.

### Model Analysis
The model is analyzed by using the TensorFlow Board. We analyze on Accuracy, AUC_Score, and Logloss.

Accuracy
![Accuracy](https://drive.google.com/uc?export=download&id=1wAqg_qIzufbwNOl18MzU7Y5-lSz8orTe)

AUC_Score
![AUCCurve](https://drive.google.com/uc?export=download&id=1XP97NrRURjgjOJFXb6NByUcM7RgLNW4p)

Logloss
![LogLoss](https://drive.google.com/uc?export=download&id=16zNrY9mLvpzhFs0_Z-oBUHlPFBhhN-ez)

### Model Efficacy Report and Visuals

We get the model efficacy by using `line_profiler`. We used the code below to get the output.
```python
!pip install line_profiler

%load_ext line_profiler
 
# Run line-by-line profiling
%lprun -f modelcomputational_report modelcomputational_report()
```
Output
![Comp_Repo](https://drive.google.com/uc?export=download&id=1syTU8nstG_pZrm4MBS3r22ObAmpD6GF-)


ROC Curve

![ROC](https://drive.google.com/uc?export=download&id=1_GSmz47qeDd-2kbzwnSm-KIAj252IZIh)


Confusion Matrix

![ConfusionMatrix](https://drive.google.com/uc?export=download&id=1bishYT8N7ZkIcMkNd1kNe0uSF47lUYBl)
