# Databricks notebook source
# MAGIC %md
# MAGIC ##Name: Omkar Suresh Kadam
# MAGIC
# MAGIC ##Net-ID: oxk240000
# MAGIC
# MAGIC ##Course: CS 6350.001
# MAGIC
# MAGIC ##Assignment 2

# COMMAND ----------

# MAGIC %md
# MAGIC ###Part 1: Friend Recommendation using Mutual Friends

# COMMAND ----------

list_of_friends_txtFile = sc.textFile("/FileStore/tables/soc_LiveJournal1Adj.txt")

# COMMAND ----------

formatted_data = list_of_friends_txtFile.map(lambda row: (row.split("\t")[0], list(map(int, row.split("\t")[1].split(","))) if len(row.split("\t")) > 1 and row.split("\t")[1] else []))

# COMMAND ----------

original_list_dict = dict(formatted_data.collect())

# COMMAND ----------

def get_FoF(person, friends):
    friends_of_friends = []
    for friend in friends:
        key = str(friend)
        if key in original_list_dict and original_list_dict[key]:
            friends_of_friends.extend(original_list_dict[key])
    return (person, friends_of_friends)

# COMMAND ----------

expanded_rdd = formatted_data.map(lambda x: get_FoF(x[0], x[1]))

# COMMAND ----------

fof_count_rdd = expanded_rdd.map(lambda x: (x[0], [(friend, 1) for friend in x[1]]))

# COMMAND ----------

flattened_rdd = fof_count_rdd.flatMap(lambda x: [(x[0], friend) for friend in x[1]])

# COMMAND ----------

reduced_rdd = flattened_rdd.map(lambda x: ((x[0], x[1][0]), x[1][1]))

# COMMAND ----------

reduced_rdd_new = reduced_rdd.reduceByKey(lambda a, b: a + b)

# COMMAND ----------

reduced_rdd_new.take(5)

# COMMAND ----------

grouped_rdd = reduced_rdd_new.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().mapValues(list)

# COMMAND ----------

grouped_rdd.take(1)

# COMMAND ----------

sorted_rdd = grouped_rdd.mapValues(lambda friends: sorted(friends, key=lambda x: x[1], reverse=True))

# COMMAND ----------

sorted_rdd.take(3)

# COMMAND ----------

def filter_direct_friends(person, friends_list):
    direct_friends = original_list_dict.get(person, set())
    return [(friend, count) for friend, count in friends_list if friend != int(person) and friend not in direct_friends]

# COMMAND ----------

direct_friends_filtered_rdd = sorted_rdd.map(lambda x: (x[0], filter_direct_friends(x[0], x[1])))

# COMMAND ----------

direct_friends_filtered_rdd.take(2)

# COMMAND ----------

random_users_subset = direct_friends_filtered_rdd.takeSample(False, 10)

# COMMAND ----------

def get_top_n_recommendations(recommendations, n=10):
    return recommendations[:n]
                           
top_10_recommendations = []
for user, recommendations in random_users_subset:
    top_10_recommendations.append((user, get_top_n_recommendations(recommendations)))

# COMMAND ----------

ser = 1
for user, recommendations in top_10_recommendations:
    print(f"Top 10 Recommendations for User {ser} => {user}:")
    for friend, count in recommendations:
        print(f"  Friend ID: {friend}")
    ser+=1

# COMMAND ----------

# MAGIC %md
# MAGIC ###Part 2: Implementing Naive Bayes Classifier using Spark MapReduce

# COMMAND ----------

data = sc.textFile("/FileStore/tables/SMSSpamCollection").map(lambda line: line.split("\t")).filter(lambda x: len(x) == 2)

# COMMAND ----------

data.take(5)

# COMMAND ----------

!pip install nltk

# COMMAND ----------

import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# COMMAND ----------

nltk.download("all")

# COMMAND ----------

stop_words = set(stopwords.words("english"))

# COMMAND ----------

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return words

# COMMAND ----------

preprocessed_data = data.map(lambda x: (x[0], preprocess(x[1])))

# COMMAND ----------

preprocessed_data.take(5)

# COMMAND ----------

from pyspark.ml.feature import HashingTF, IDF

# COMMAND ----------

preprocesseddf = spark.createDataFrame(preprocessed_data.map(lambda x: (x[0], x[1])), ["label", "words"])

# COMMAND ----------

hashing_Term_Frequency = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=100000)
tf_data = hashing_Term_Frequency.transform(preprocesseddf)


inverse_doc_freq = IDF(inputCol="rawFeatures", outputCol="features")
inverse_doc_freq_model = inverse_doc_freq.fit(tf_data)
tfidf_data = inverse_doc_freq_model.transform(tf_data)

# COMMAND ----------

tfidf_rdd = tfidf_data.select("label", "features").rdd.map(tuple)
print(tfidf_rdd.take(5))

# COMMAND ----------

training_data, testing_data = tfidf_rdd.randomSplit([80,20], seed = 567)

# COMMAND ----------

class_labels = training_data.map(lambda x: x[0])
class_counts = class_labels.countByValue()
total_documents = sum(class_counts.values())
class_priors = {cls: count / total_documents for cls, count in class_counts.items()}
print("Class Priors:", class_priors)

# COMMAND ----------

training_data.take(1)[0]

# COMMAND ----------

word_counts = training_data.flatMap(lambda x: [((x[0], word_index), tfidf_value) for word_index, tfidf_value in zip(x[1].indices, x[1].values)]).reduceByKey(lambda a, b: a + b) 

# COMMAND ----------

word_counts.collect()

# COMMAND ----------

total_tfidf_per_class = word_counts.map(lambda x: (x[0][0], x[1])).reduceByKey(lambda a, b: a + b)

# COMMAND ----------

total_tfidf_per_class.collect()

# COMMAND ----------

import numpy as np
from collections import defaultdict

# COMMAND ----------

def sums_and_counts_of_features(data):
    target_label, tf_idf_features = data
    features_array = np.array(tf_idf_features.toArray())
    counts_of_feature = defaultdict(float)
    sums_of_feature = defaultdict(float)

    for counter, value in enumerate(features_array):
        if value > 0:
            counts_of_feature[counter] += 1
            sums_of_feature[counter] += value

    return (target_label, (dict(sums_of_feature), dict(counts_of_feature)))

# COMMAND ----------

sums_and_counts_features_rdd = training_data.map(sums_and_counts_of_features)

# COMMAND ----------

sums_and_counts_features_rdd_new = sums_and_counts_features_rdd.reduceByKey(
    lambda x, y: (
    {k: x[0].get(k, 0) + y[0].get(k, 0) for k in set(x[0]) | set(y[0])},
    {k: x[1].get(k, 0) + y[1].get(k, 0) for k in set(x[1]) | set(y[1])}
    )
)

# COMMAND ----------

sums_and_counts_features_rdd_new.collect()

# COMMAND ----------

feature_likelihoods = sums_and_counts_features_rdd_new.mapValues(
    lambda sums_and_counts: {
        counter: (sums_and_counts[0][counter] / sums_and_counts[1][counter]) 
        for counter in sums_and_counts[0]
    }
).collectAsMap()

# COMMAND ----------

def create_predictions(features, class_priors, feature_likelihoods):
    log_probabilities = {class_label: np.log(prior_for_class) for class_label, prior_for_class in class_priors.items()}  
    array_of_features = np.array(features.toArray())
    for class_label, feature_likelihood in feature_likelihoods.items():
        for counter, value_of_feature in enumerate(array_of_features):
            if value_of_feature > 0:
                if counter in feature_likelihood:
                    log_probabilities[class_label] += value_of_feature * np.log(feature_likelihood[counter])
                else:
                    log_probabilities[class_label] += value_of_feature * np.log(1e-9)
    prediction = max(log_probabilities, key=log_probabilities.get)
    return prediction

# COMMAND ----------

all_preds = testing_data.map(lambda sample: (sample[0], create_predictions(sample[1], class_priors, feature_likelihoods)))

# COMMAND ----------

all_preds.take(5)

# COMMAND ----------

correct_preds = all_preds.filter(lambda x: x[0] == x[1]) 

# COMMAND ----------

accuracy = correct_preds.count() / float(testing_data.count())
print(f"Accuracy: {accuracy}")

# COMMAND ----------

elements_in_cf = all_preds.map(lambda x: ((x[0], x[1]), 1)).reduceByKey(lambda x, y: x + y).collectAsMap()

# COMMAND ----------

true_positive = elements_in_cf.get(('spam', 'spam'), 0)
false_positive = elements_in_cf.get(('ham', 'spam'), 0)
true_negative = elements_in_cf.get(('ham', 'ham'), 0)
false_negative = elements_in_cf.get(('spam', 'ham'), 0)

# COMMAND ----------

recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# COMMAND ----------

print(f"Confusion Matrix: TP={true_positive}, FP={false_positive}, TN={true_negative}, FN={false_negative}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1_score}")

# COMMAND ----------

true_positive_spam = elements_in_cf.get(('spam', 'spam'), 0)
true_positive_ham = elements_in_cf.get(('ham', 'ham'), 0)

total_spam = all_preds.filter(lambda x: x[0] == 'spam').count()
total_ham = all_preds.filter(lambda x: x[0] == 'ham').count()

accuracy_spam = true_positive_spam / total_spam if total_spam > 0 else 0
accuracy_ham = true_positive_ham / total_ham if total_ham > 0 else 0

print(f"Accuracy for 'spam': {accuracy_spam:.4f}")
print(f"Accuracy for 'ham': {accuracy_ham:.4f}")