#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
loan_predict_pyspark.py
Description:
    This PySpark script implements an end-to-end machine learning pipeline
    to predict loan approval outcomes (Loan_Status) using PySpark MLlib.
    It covers:
        1. Session Initialization
        2. Dataset Download & Ingestion
        3. Data Cleaning: Handling missing values
        4. Data Cleaning: Handling Outlier
        5. Feature Discretization
        6. Feature Selection and Encoding
        7. Pipeline Construction & Data Preprocessing Execution
        8. Training and Evaluating Three Classifiers:
           - Logistic Regression
           - Decision Tree Classifier
           - Random Forest Classifier
        9. Model Comparison & Accuracy Output to 'output.txt'
"""

import os
import sys
import urllib.request
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count
from pyspark.ml import Pipeline
from pyspark.ml.feature import Bucketizer, StringIndexer, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    # ==============================================================================
    # 1. Session Initialization
    # ==============================================================================
    print("Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("Predictive Loan Modeling with PySpark") \
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # ==============================================================================
    # 2. Dataset Download & Ingestion
    # ==============================================================================
    url = "https://raw.githubusercontent.com/learningtechnologieslab/mds_cloud_computing/refs/heads/main/apache_spark/loan_data.csv"
    local_csv_path = "loan_data.csv"
    
    if not os.path.exists(local_csv_path):
        try:
            print(f"Downloading dataset from {url}...")
            urllib.request.urlretrieve(url, local_csv_path)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            sys.exit(1)
    else:
        print("file exist")

    print(f"Loading dataset from {local_csv_path}...")
    df = spark.read.csv(local_csv_path, header=True, inferSchema=True)
    print(f"Dataset successfully loaded. Number of rows: {df.count()}, Columns: {len(df.columns)}")
    df.printSchema()

    # ==============================================================================
    # 3. Data Cleaning: Handling Missing Values
    # ==============================================================================
    # JUSTIFICATION:
    # 1. For the Categorical Columns (Gender, Married, Dependents, Self_Employed):
    #    I filled the missing values with the mode of each respective column.
    #    It is becasue it's a common and robust approach for categorical data since it
    #    will not discorder the distribution and frequency structure since it will not 
    #    introducing non-existent categories.
    # 2. Numerical Columns (LoanAmount, Loan_Amount_Term):
    #    I filled the missing values using the MEDIAN since it is more robust to
    #    outliers and skewed distributions compared to the mean.
    # 3. Credit_History:
    #    for Credit_History, it is considered as categorical variable and impute  
    #    missing values with its mode.
    # ==============================================================================
    print("Preprocessing: Cleaning and replacing text nulls...")

    #  function to compute mode
    def get_mode(dataframe, col_name):
        dtype = dict(dataframe.dtypes)[col_name]
        if dtype == 'string':
            valid_filter = col(col_name).isNotNull() & (col(col_name) != "")
        else:
            valid_filter = col(col_name).isNotNull()
        mode_row = dataframe.filter(valid_filter) \
                            .groupBy(col_name).count().orderBy("count", ascending=False)
        if mode_row.count() > 0:
            return mode_row.first()[0]
        return None

    # function to compute median
    def get_median(dataframe, col_name):
        non_null_df = dataframe.filter(col(col_name).isNotNull())
        if non_null_df.count() > 0:
            return non_null_df.approxQuantile(col_name, [0.5], 0.001)[0]
        return None

    print("Missing value counts before imputation:")
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    
    print("Imputing missing values...")
    for column, dtype in df.dtypes:
        if dtype == 'string':
            df = df.withColumn(column, when(col(column).isin("", "NA", "null", "NULL"), None).otherwise(col(column)))
    
    # Categorical Columns Imputation
    categorical_to_impute = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
    for cat_col in categorical_to_impute:
        col_mode = get_mode(df, cat_col)
        if col_mode is not None:
            if cat_col == "Credit_History":
                df = df.withColumn(cat_col, col(cat_col).cast("double"))
                col_mode = float(col_mode)
            print(f"  - Imputing categorical '{cat_col}' missing values with mode: '{col_mode}'")
            df = df.fillna({cat_col: col_mode})

    # Numerical Columns Imputation
    numerical_to_impute = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    for num_col in numerical_to_impute:
        df = df.withColumn(num_col, col(num_col).cast("double"))
        col_median = get_median(df, num_col)
        if col_median is not None:
            print(f"  - Imputing numerical '{num_col}' missing values with median: {col_median}")
            df = df.fillna({num_col: col_median})

    # Verify that there are no missing values remaining
    print("Missing value counts after imputation:")
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

    # ==============================================================================
    # 4. Data Cleaning: Handling Outlier
    # ==============================================================================
    # JUSTIFICATION FOR OUTLIER DETECTION AND TREATMENT:
    # 1. Outlier Detection: Outliers in ApplicantIncome, CoapplicantIncome, LoanAmount, 
    #    Loan_Amount_Term can be detected using IQR - Interquartile Range, as it is commonly 
    #    used and easy to impliment.
    # 2. Decision: the outliers in our dataset should be kept as:
    #    1. Since we are discretizing (binning) these continuous variables later in the pipeline, 
    #    those outlier values will fall into the highest bin. It will naturally controls the high 
    #    influence of these extreme values on model parameters without losing valid data.
    #    2. High-income applicants and exceptionally large loan requests are valid cases in loan 
    #    applications. Dropping them would lead to a model that cannot generalize the high value loans.
    # ==============================================================================
    print("Outlier Treatment:")
    for col_name in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]:
        q1, q3 = df.approxQuantile(col_name, [0.25, 0.75], 0.001)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = df.filter((col(col_name) < lower_bound) | (col(col_name) > upper_bound)).count()
        print(f"  - For {col_name}: Q1={q1}, Q3={q3}, IQR={iqr}")
        print(f"    Outlier Boundaries: Lower={lower_bound}, Upper={upper_bound}")
        print(f"    Detected Outliers count: {outlier_count} (Decision: KEPT because discretization handles them natively)")

    # ==============================================================================
    # 5. Feature Discretization
    # ==============================================================================
    # JUSTIFICATION FOR DISCRETIZATION OF NUMERIC COLUMNS:
    # 1. ApplicantIncome:
    #    - Bins: [0.0, 3000.0, 6000.0, 10000.0, float('inf')]
    #    - Reasoning: Splits based on monthly socio-economic income: Low Income (<3k),
    #      Lower-Middle (3k-6k), Upper-Middle (6k-10k), and High Income (>10k).
    # 2. CoapplicantIncome:
    #    - Bins: [0.0, 3000.0, 6000.0, 10000.0, float('inf')]
    #    - Reasoning: also split cases based on monthly socio-economic income: Low Income (<3k),
    #      Lower-Middle (3k-6k), Upper-Middle (6k-10k), and High Income (>10k).
    # 3. LoanAmount:
    #    - Bins: [0.0, 100.0, 200.0, 350.0, float('inf')]
    #    - Reasoning: Based on the levels of debt requests: Small (<100k), Standard (100k-200k),
    #      Large (200k-350k), and Premium (>350k).
    # 4. Loan_Amount_Term:
    #    - Bins: [0.0, 180.0, 300.0, 360.0, float('inf')]
    #    - Reasoning: Based on the duration into Short-Term (<15 years), Medium-Term (15-25 years),
    #      Standard 30-Year term (360 months), and Long-Term (>30 years).
    # ==============================================================================
    print("Discretizing numeric columns into structured categories...")
    
    applicant_income_bucket = Bucketizer(
        splits=[0.0, 3000.0, 6000.0, 10000.0, float('inf')],
        inputCol="ApplicantIncome",
        outputCol="ApplicantIncome_Bin"
    )

    coapplicant_income_bucket = Bucketizer(
        splits=[0.0, 3000.0, 6000.0, 10000.0, float('inf')],
        inputCol="CoapplicantIncome",
        outputCol="CoapplicantIncome_Bin"
    )

    loan_amount_bucket = Bucketizer(
        splits=[0.0, 100.0, 200.0, 350.0, float('inf')],
        inputCol="LoanAmount",
        outputCol="LoanAmount_Bin"
    )

    loan_term_bucket = Bucketizer(
        splits=[0.0, 180.0, 300.0, 360.0, float('inf')],
        inputCol="Loan_Amount_Term",
        outputCol="Loan_Amount_Term_Bin"
    )

    # ==============================================================================
    # 6. Feature Selection and Encoding
    # ==============================================================================
    # JUSTIFICATION FOR FEATURE SELECTION:
    # - Included Predictors:
    #   1. Gender_Index, Married_Index, Dependents_Index, Education_Index, Self_Employed_Index,
    #      Property_Area_Index: as the above features are transformed and contain socio-demographic 
    #      information.
    #   2. Credit_History: Historically the single strongest indicator of loan repayment success.
    #   3. ApplicantIncome_Bin, CoapplicantIncome_Bin, LoanAmount_Bin, Loan_Amount_Term_Bin:
    #      Our robust discretized financial predictors.
    # - Excluded Predictors:
    #   1. Loan_ID: It is applicant primary key. It doesnt contain useful information
    #      and would bring noise.
    #   2. Raw Numerical Features (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term):
    #      Including both the raw and discretized columns would bring multi-collinearity
    #      and redundancy.
    # ==============================================================================
    print("Configuring Categorical Label Encoding and Vector Assembler...")
    
    # Columns to label encode (we handle missing values before doing this)
    categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    
    indexers = [
        StringIndexer(inputCol=col_name, outputCol=col_name + "_Index", handleInvalid="keep")
        for col_name in categorical_cols
    ]

    # Label encoding for the Target Variable 'Loan_Status' (Y -> 1.0, N -> 0.0)
    target_indexer = StringIndexer(inputCol="Loan_Status", outputCol="label", handleInvalid="keep")

    # List of engineered features to construct our feature vector
    feature_cols = [
        "Gender_Index", "Married_Index", "Dependents_Index", "Education_Index", 
        "Self_Employed_Index", "Property_Area_Index", "Credit_History",
        "ApplicantIncome_Bin", "CoapplicantIncome_Bin", "LoanAmount_Bin", "Loan_Amount_Term_Bin"
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    selector = ChiSqSelector(
        numTopFeatures=8, 
        featuresCol="raw_features", 
        outputCol="features", 
        labelCol="label"
    )

    # ==============================================================================
    # 7. Pipeline Construction & Data Preprocessing Execution
    # ==============================================================================
    print("Assembling and running the preprocessing Pipeline with ChiSqSelector...")
    preprocessing_pipeline = Pipeline(stages=[
        applicant_income_bucket,
        coapplicant_income_bucket,
        loan_amount_bucket,
        loan_term_bucket,
        *indexers,
        target_indexer,
        assembler,
        selector
    ])
    pipeline_model = preprocessing_pipeline.fit(df)
    processed_df = pipeline_model.transform(df)
    
    # Split preprocessed data into train and test sets (80% train, 20% test)
    print("Partitioning data into Train (80%) and Test (20%) sets...")
    train_data, test_data = processed_df.randomSplit([0.8, 0.2], seed=42)
    print(f"  - Training records count: {train_data.count()}")
    print(f"  - Testing records count: {test_data.count()}")
    
    # ==============================================================================
    # 8. Model Training, Evaluation & Comparison
    # ==============================================================================
    # JUSTIFICATION FOR SELECTION OF CLASSIFIERS:
    # The following three models are selected:
    # 1. Logistic Regression:
    #    - A widely used, interpretable baseline model for binary classification.
    #    - Easy to understand the feature weights and relationship directions.
    # 2. Decision Tree Classifier:
    #    - Also widely used baseline model for binary classification.
    #    - Highly interpretable and matches the decision-making logic of credit underwriters.
    # 3. Random Forest Classifier:
    #    - An ensemble method of bagging multiple decision trees.
    #    - Significantly reduces overfitting, handles high variance, and typically yields
    #      superior predictive accuracy compared to a single decision tree.
    # ================================================================================


    # Initialize Multi-class classification evaluator for accuracy
    print("Training models and evaluating accuracies...")
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    # 1. Logistic Regression Classifier
    print("Training Classifier 1: Logistic Regression...")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
    lr_model = lr.fit(train_data)
    lr_preds = lr_model.transform(test_data)
    lr_accuracy = evaluator.evaluate(lr_preds)
    print(f"  -> Logistic Regression Accuracy: {lr_accuracy:.4%}")

    # 2. Decision Tree Classifier
    print("Training Classifier 2: Decision Tree...")
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", seed=42)
    dt_model = dt.fit(train_data)
    dt_preds = dt_model.transform(test_data)
    dt_accuracy = evaluator.evaluate(dt_preds)
    print(f"  -> Decision Tree Accuracy: {dt_accuracy:.4%}")

    # 3. Random Forest Classifier
    print("Training Classifier 3: Random Forest...")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50, seed=42)
    rf_model = rf.fit(train_data)
    rf_preds = rf_model.transform(test_data)
    rf_accuracy = evaluator.evaluate(rf_prexds)
    print(f"  -> Random Forest Accuracy: {rf_accuracy:.4%}")

    # ==============================================================================
    # 9. Output to Text File (output.txt)
    # ==============================================================================
    output_filename = "output.txt"
    print(f"Saving final evaluation results to '{output_filename}'...")
    
    output_content = (
        "========================================================\n"
        "PySpark MLlib Model Evaluation Results - Loan Approval\n"
        "========================================================\n"
        f"1. Logistic Regression Accuracy : {lr_accuracy:.4%}\n"
        f"2. Decision Tree Accuracy       : {dt_accuracy:.4%}\n"
        f"3. Random Forest Accuracy        : {rf_accuracy:.4%}\n"
        "========================================================\n"
    )
    
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(output_content)
        print(f"Successfully wrote accuracy report to: '{os.path.abspath(output_filename)}'")
    except Exception as e:
        print(f"Error writing output file: {e}")

    # Clean up and stop Spark session
    print("\nStopping Spark Session...")
    spark.stop()
    print("Execution complete. All steps of the PySpark ML pipeline executed successfully.")

if __name__ == "__main__":
    main()
