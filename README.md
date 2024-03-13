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

|1 | ID        | Numerical | ID of each client |

|2 | LIMIT_BAL | Numerical | Amount of given credit in NT dollars (includes individual and family/supplementary credit) |
|3 | SEX       | Numerical | Gender (1=male, 2=female) |
|4 | EDUCATION | Numerical | (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown) |
|5 | MARRIAGE  | Numerical | Marital status (1=married, 2=single, 3=others) |
|6 | AGE       | Numerical | Age in years |
|7 | PAY_0     | Numerical | Repayment status in September 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above) |
|8 | PAY_2     | Numerical | Repayment status in August 2005 (scale same as above) |
|9 | PAY_3     | Numerical | Repayment status in July 2005 (scale same as above) |
|10| PAY_4     | Numerical | Repayment status in June 2005 (scale same as above) |
|11| PAY_5     | Numerical | Repayment status in May 2005 (scale same as above) |
|12| PAY_6     | Numerical | Repayment status in April 2005 (scale same as above) |
|13| BILL_AMT1 | Numerical | Amount of bill statement in September 2005 (NT dollar) |
|14| BILL_AMT2 | Numerical | Amount of bill statement in August 2005 (NT dollar) |
|15| BILL_AMT3 | Numerical | Amount of bill statement in July 2005 (NT dollar) |
|16| BILL_AMT4 | Numerical | Amount of bill statement in June 2005 (NT dollar) |
|17| BILL_AMT5 | Numerical | Amount of bill statement in May 2005 (NT dollar) |
|18| BILL_AMT6 | Numerical | Amount of bill statement in April 2005 (NT dollar) |
|19| PAY_AMT1  | Numerical | Amount of previous payment in September 2005 (NT dollar) |
|20| PAY_AMT2  | Numerical | Amount of previous payment in August 2005 (NT dollar) |
|21| PAY_AMT3  | Numerical | Amount of previous payment in July 2005 (NT dollar) |
|22| PAY_AMT4  | Numerical | Amount of previous payment in June 2005 (NT dollar) |
|23| PAY_AMT5  | Numerical | Amount of previous payment in May 2005 (NT dollar) |
|24| PAY_AMT6  | Numerical | Amount of previous payment in April 2005 (NT dollar) |
|25| default payment next month | Numerical | Indicates whether the client defaulted on the payment next month (1=yes, 0=no) |

##	Data Rights and Privacy: 
The dataset is anonymized, with all personal identifiers removed to comply with privacy regulations. As per UCI platform regulations, this allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

## Data Planning and Splits
As part of data preprocessing, we will first check for any null values, outliers, and duplicates present in the data and correct them. We will later perform initial exploratory data analysis to get deeper insights into the dataset.
We will split the data in the ratio of 70-15-15 for training, validation, and testing and, ensure that we maintain the distribution of classes across the training, validation, and testing sets such that each class is represented proportionally in each subset.

