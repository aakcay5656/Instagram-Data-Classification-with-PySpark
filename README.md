# Instagram Data Classification with PySpark

This project aims to classify Instagram captions into different categories using various machine learning models in PySpark. The dataset contains captions labeled into different categories such as finance, sports, art, technology, and personal. The classification models used include Decision Tree, Random Forest, Naive Bayes, and Logistic Regression.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualization](#visualization)
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README provides a high-level overview of the Instagram data classification project using PySpark. For detailed implementation, refer to the code comments and the accompanying Jupyter notebook (if available).
