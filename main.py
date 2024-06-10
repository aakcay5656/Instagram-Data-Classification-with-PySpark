import pandas as pd

# Dosya yolunu al
dosya_yolu = input("Dosya yolunu giriniz:")

# JSON dosyasını Pandas DataFrame olarak oku
obj = pd.read_json(dosya_yolu, orient='values')
print(obj)

# Geçici olarak Pandas DataFrame'i bir CSV dosyasına kaydet
name = input("CSV dosyasının ismini giriniz:")
name = name + ".csv"
obj.to_csv(name)
print(name)

# Spark oturumunu oluşturma


#%%
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, when
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, NaiveBayes, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import pandas as pd
import matplotlib.pyplot as plt

dosya_yolu = input("Dosya yolunu giriniz:")

spark = SparkSession.builder.appName('Instagram Data').getOrCreate()

# Geçici CSV dosyasını Spark DataFrame olarak yükleme
df = spark.read.option('header', 'true').option('encoding', 'utf-8').csv(dosya_yolu)

# Sütunları etiketlemek
df = df.withColumn("label_code",
    when(col("label") == "finans", 1)
    .when(col("label") == "spor", 2)
    .when(col("label") == "sanat", 3)
    .when(col("label") == "teknoloji", 4)
    .when(col("label") == "kişisel", 5)
    .otherwise(0))

# Metin ve etiket işlemleri
df_with_numeric = df.withColumn("label_code", col("label_code").cast("float"))

# Eksik değerleri düşürme
df_cleaned = df_with_numeric.na.drop()

# Eğitim ve test verilerini ayırma
train_data, test_data = df_cleaned.randomSplit([0.8, 0.2], seed=42)

# Eğitim verilerini hazırlama
ml_df = train_data.select("caption", "label_code").dropna()
ml_df = ml_df.withColumn("only_str", lower(col('caption')))
ml_df = ml_df.withColumn("only_str", regexp_replace(col('only_str'), '\d+', ''))
ml_df = ml_df.withColumn("only_str", regexp_replace(col('only_str'), '[^\w\s]', ' '))

# Özellik mühendisliği ve metin işleme
regex_tokenizer = RegexTokenizer(inputCol="only_str", outputCol="words", pattern="\\W")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
cv = CountVectorizer(inputCol="filtered", outputCol="raw_features")
idf = IDF(inputCol="raw_features", outputCol="features")

# Pipeline oluşturma
pipeline = Pipeline(stages=[regex_tokenizer, remover, cv, idf])
pipeline_model = pipeline.fit(ml_df)
countVectorizer_train = pipeline_model.transform(ml_df)
countVectorizer_train = countVectorizer_train.withColumn("label_", col('label_code'))

# Test verilerini hazırlama
testData = test_data.select("caption", "label_code")
testData = testData.withColumn("only_str", lower(col('caption')))
testData = testData.withColumn("only_str", regexp_replace(col('only_str'), '\d+', ''))
testData = testData.withColumn("only_str", regexp_replace(col('only_str'), '[^\w\s]', ' '))
testData = pipeline_model.transform(testData)
testData = testData.withColumn("label_", col('label_code'))

# Değerlendirme fonksiyonu
def evaluate_model(predictions, model_name):
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label_", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_acc.evaluate(predictions)
    print(f"{model_name} Model Accuracy: {accuracy}")
    return accuracy

# Model eğitim ve değerlendirme
models = {
    "Decision Tree": DecisionTreeClassifier(featuresCol='features', labelCol='label_', maxDepth=10),
    "Random Forest": RandomForestClassifier(featuresCol='features', labelCol='label_', numTrees=100, maxDepth=10),
    "Naive Bayes": NaiveBayes(featuresCol='features', labelCol='label_'),
    "Logistic Regression": LogisticRegression(featuresCol='features', labelCol='label_')
}

results = []
all_predictions = None

for model_name, model in models.items():
    model_fit = model.fit(countVectorizer_train)
    predictions = model_fit.transform(testData)
   
    acc = evaluate_model(predictions, model_name)
    results.append((model_name, acc))

    if all_predictions is None:
        all_predictions = predictions.withColumnRenamed("prediction", f"{model_name}_prediction")
    else:
        all_predictions = all_predictions.join(predictions.select("caption", col("prediction").alias(f"{model_name}_prediction")), on="caption")
#Etiketleri anlamlı kategorilere dönüştürme
all_predictions = all_predictions.withColumn("label_",
    when(col("label_") == 1, "finans")
    .when(col("label_") == 2, "spor")
    .when(col("label_") == 3, "sanat")
    .when(col("label_") == 4, "teknoloji")
    .when(col("label_") == 5, "kişisel")
    .otherwise("other"))

all_predictions = all_predictions.withColumn("Decision Tree_prediction",
    when(col("Decision Tree_prediction") == 1, "finans")
    .when(col("Decision Tree_prediction") == 2, "spor")
    .when(col("Decision Tree_prediction") == 3, "sanat")
    .when(col("Decision Tree_prediction") == 4, "teknoloji")
    .when(col("Decision Tree_prediction") == 5, "kişisel")
    .otherwise("other"))
all_predictions = all_predictions.withColumn("Random Forest_prediction",
    when(col("Random Forest_prediction") == 1, "finans")
    .when(col("Random Forest_prediction") == 2, "spor")
    .when(col("Random Forest_prediction") == 3, "sanat")
    .when(col("Random Forest_prediction") == 4, "teknoloji")
    .when(col("Random Forest_prediction") == 5, "kişisel")
    .otherwise("other"))

all_predictions = all_predictions.withColumn("Naive Bayes_prediction",
    when(col("Naive Bayes_prediction") == 1, "finans")
    .when(col("Naive Bayes_prediction") == 2, "spor")
    .when(col("Naive Bayes_prediction") == 3, "sanat")
    .when(col("Naive Bayes_prediction") == 4, "teknoloji")
    .when(col("Naive Bayes_prediction") == 5, "kişisel")
    .otherwise("other"))
all_predictions = all_predictions.withColumn("Logistic Regression_prediction",
    when(col("Logistic Regression_prediction") == 1, "finans")
    .when(col("Logistic Regression_prediction") == 2, "spor")
    .when(col("Logistic Regression_prediction") == 3, "sanat")
    .when(col("Logistic Regression_prediction") == 4, "teknoloji")
    .when(col("Logistic Regression_prediction") == 5, "kişisel")
    .otherwise("other"))
# Sonuçları kaydetme
all_predictions.select("caption", "label_", "Decision Tree_prediction", "Random Forest_prediction", "Naive Bayes_prediction", "Logistic Regression_prediction") \
    .toPandas().to_csv('predictions.csv', index=False)

# Performans sonuçlarını görselleştirme
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

# Grafik oluşturma
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Model')
ax1.set_ylabel('Accuracy', color=color)
ax1.bar(results_df['Model'], results_df['Accuracy'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # Layout düzenlemeleri
plt.title("Model Performance: Accuracy ")
plt.show()

#%%
# Etiketsiz veri kümesi ile çalışmak

a = input("Analiz Edilecek Dosyanın Yolunu Giriniz:")

df_unlabeled = spark.read.option('header', 'true').option('encoding', 'utf-8').csv(a)

# Veri hazırlığı
df_unlabeled = df_unlabeled.withColumn("only_str", lower(col('caption')))
df_unlabeled = df_unlabeled.withColumn("only_str", regexp_replace(col('only_str'), '\d+', ''))
df_unlabeled = df_unlabeled.withColumn("only_str", regexp_replace(col('only_str'), '[^\w\s]', ' '))

# Özellik mühendisliği
unlabeled_transformed = pipeline_model.transform(df_unlabeled)

# Tahminlerin yapılması
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
# Sonuçları kaydetme
all_unlabeled_predictions.select("caption", "Decision Tree_prediction", "Random Forest_prediction", "Naive Bayes_prediction", "Logistic Regression_prediction") \
    .toPandas().to_csv("pre_"+name, index=False)
