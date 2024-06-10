# Instagram Data Classification with PySpark

This project focuses on classifying Instagram captions into different categories using various machine learning models in PySpark. The dataset contains captions labeled into categories such as finance, sports, art, technology, and personal. The classification models used include Decision Tree, Random Forest, Naive Bayes, and Logistic Regression.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [Unlabeled Data Analysis](#unlabeled-data-analysis)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need to have PySpark installed. You can install PySpark using pip:

```sh
pip install pyspark
```

Additionally, ensure you have the following libraries installed:
- pandas
- matplotlib

You can install these using pip as well:

```sh
pip install pandas matplotlib
```

## Dataset

The dataset is a CSV file containing Instagram captions with their corresponding labels. The labels include categories such as finance, sports, art, technology, and personal.

## Data Preprocessing

1. **Label Encoding**: Convert categorical labels into numeric codes.
2. **Text Cleaning**: Lowercase transformation, removal of digits and special characters.
3. **Tokenization and Stop Words Removal**: Tokenize the text and remove common stop words.
4. **Feature Extraction**: Use `CountVectorizer` and `IDF` to extract features from the text.

## Model Training

The models used in this project are:
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Naive Bayes**
- **Logistic Regression**

Each model is trained using a pipeline that includes tokenization, stop words removal, count vectorization, and IDF.

## Evaluation

The models are evaluated based on their accuracy on the test data. The accuracy is computed using `MulticlassClassificationEvaluator`.

## Results

The results of the models are saved in a CSV file `predictions.csv`. The CSV file includes the original captions, the true labels, and the predictions made by each model.

## Visualization

The performance of each model is visualized using a bar chart. The chart shows the accuracy of each model.

```python
# Plotting the results
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Model')
ax1.set_ylabel('Accuracy', color=color)
ax1.bar(results_df['Model'], results_df['Accuracy'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # Layout adjustments
plt.title("Model Performance: Accuracy ")
plt.show()
```

## Unlabeled Data Analysis

The project also includes the capability to analyze unlabeled data. Given a new dataset without labels, the models can predict the categories of the captions.

```python
# Analyzing Unlabeled Data
test_dosyasi = input("Enter the path of the dataset to be analyzed:")

df_unlabeled = spark.read.option('header', 'true').option('encoding', 'utf-8').csv(test_dosyasi)

# Data preparation
df_unlabeled = df_unlabeled.withColumn("only_str", lower(col('caption')))
df_unlabeled = df_unlabeled.withColumn("only_str", regexp_replace(col('only_str'), '\d+', ''))
df_unlabeled = df_unlabeled.withColumn("only_str", regexp_replace(col('only_str'), '[^\w\s]', ' '))

# Feature engineering
unlabeled_transformed = pipeline_model.transform(df_unlabeled)

# Making predictions
all_unlabeled_predictions = None

for model_name, model in models.items():
    model_fit = model.fit(countVectorizer_train)
    predictions = model_fit.transform(unlabeled_transformed)
    
    if all_unlabeled_predictions is None:
        all_unlabeled_predictions = predictions.withColumnRenamed("prediction", f"{model_name}_prediction")
    else:
        all_unlabeled_predictions = all_unlabeled_predictions.join(predictions.select("caption", col("prediction").alias(f"{model_name}_prediction")), on="caption")
        
all_unlabeled_predictions = all_unlabeled_predictions.withColumn("Decision Tree_prediction",
    when(col("Decision Tree_prediction") == 1, "finans")
    .when(col("Decision Tree_prediction") == 2, "spor")
    .when(col("Decision Tree_prediction") == 3, "sanat")
    .when(col("Decision Tree_prediction") == 4, "teknoloji")
    .when(col("Decision Tree_prediction") == 5, "kişisel")
    .otherwise("other"))
all_unlabeled_predictions = all_unlabeled_predictions.withColumn("Random Forest_prediction",
    when(col("Random Forest_prediction") == 1, "finans")
    .when(col("Random Forest_prediction") == 2, "spor")
    .when(col("Random Forest_prediction") == 3, "sanat")
    .when(col("Random Forest_prediction") == 4, "teknoloji")
    .when(col("Random Forest_prediction") == 5, "kişisel")
    .otherwise("other"))

all_unlabeled_predictions = all_unlabeled_predictions.withColumn("Naive Bayes_prediction",
    when(col("Naive Bayes_prediction") == 1, "finans")
    .when(col("Naive Bayes_prediction") == 2, "spor")
    .when(col("Naive Bayes_prediction") == 3, "sanat")
    .when(col("Naive Bayes_prediction") == 4, "teknoloji")
    .when(col("Naive Bayes_prediction") == 5, "kişisel")
    .otherwise("other"))
all_unlabeled_predictions = all_unlabeled_predictions.withColumn("Logistic Regression_prediction",
    when(col("Logistic Regression_prediction") == 1, "finans")
    .when(col("Logistic Regression_prediction") == 2, "spor")
    .when(col("Logistic Regression_prediction") == 3, "sanat")
    .when(col("Logistic Regression_prediction") == 4, "teknoloji")
    .when(col("Logistic Regression_prediction") == 5, "kişisel")
    .otherwise("other"))

# Saving the results
all_unlabeled_predictions.select("caption", "Decision Tree_prediction", "Random Forest_prediction", "Naive Bayes_prediction", "Logistic Regression_prediction") \
    .toPandas().to_csv("predictions_unlabeled.csv", index=False)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README provides a high-level overview of the Instagram data classification project using PySpark. For detailed implementation, refer to the code comments and the accompanying Jupyter notebook (if available).
