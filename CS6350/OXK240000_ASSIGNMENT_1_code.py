# Databricks notebook source
# MAGIC %md
# MAGIC +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assignment I
# MAGIC ### Name: Omkar Suresh Kadam
# MAGIC ### Net-ID: oxk240000
# MAGIC ### CS6350.001.25S - Big Data Management and Analytics

# COMMAND ----------

# MAGIC %md
# MAGIC +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# COMMAND ----------

# MAGIC %md
# MAGIC **_Part 1 - WordCount for Named Entities_**

# COMMAND ----------

wizard_rdd = sc.textFile("/FileStore/tables/the_wonderful_wizard_of_OZ.txt")

# COMMAND ----------

wizard_rdd.take(10)

# COMMAND ----------

# MAGIC %pip install nltk

# COMMAND ----------

import nltk

# COMMAND ----------

nltk.download('all')

# COMMAND ----------

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# COMMAND ----------

def extract_named_entities(line):
    sentences = sent_tokenize(line)
    named_entities = []

    for sentence in sentences:
        words = word_tokenize(sentence)  
        tagged_words = pos_tag(words)
        chunked = ne_chunk(tagged_words) 
        for chunk in chunked:
            if hasattr(chunk, "label"):
                named_entities.append(" ".join(c[0] for c in chunk))

    return named_entities

# COMMAND ----------

named_entities_rdd = wizard_rdd.flatMap(extract_named_entities)

# COMMAND ----------

named_entities_rdd.take(10)

# COMMAND ----------

mapped_rdd = named_entities_rdd.map(lambda entity: (entity, 1))

# COMMAND ----------

reduced_rdd = mapped_rdd.reduceByKey(lambda x, y: x + y)

# COMMAND ----------

sorted_rdd = reduced_rdd.sortBy(lambda x: -x[1]).collect()

# COMMAND ----------

print("Top 10 Named Entities:")
print("-" * 30)
for entity, count in sorted_rdd[:10]:
    print(f"{entity:<20} | {count:>3}")

# COMMAND ----------

# MAGIC %md
# MAGIC **_Part 2 - Search Engine for Movie Plot Summaries_**

# COMMAND ----------

from nltk.corpus import stopwords
import math
from numpy import dot
from numpy.linalg import norm
from collections import defaultdict

# COMMAND ----------

plot_summaries = sc.textFile("/FileStore/tables/plot_summaries.txt")

# COMMAND ----------

num_of_summaries = plot_summaries.count()
print(num_of_summaries)

# COMMAND ----------

stop_words = set(stopwords.words('english'))

# COMMAND ----------

def remove_stopwords(movie_id, plot_summary):
    plot_summary_lowered = plot_summary.lower()
    words = word_tokenize(plot_summary_lowered)
    words_without_stopwords = [w for w in words if w.isalnum() and w not in stop_words]
    return (movie_id, words_without_stopwords)

# COMMAND ----------

processed_lines = plot_summaries.map(lambda line: line.split("\t")).map(lambda x: remove_stopwords(x[0], x[1])).collect()

# COMMAND ----------

for movie_id, filtered_summary in processed_lines[:5]:
    print(f"Movie ID: {movie_id}")
    print(f"Filtered Summary: {' '.join(filtered_summary)}")
    print("-" * 40)

# COMMAND ----------

def map_term_frequencies(document):
    document_id, terms = document
    return [(document_id, term, 1) for term in terms]

# COMMAND ----------

mapped_terms = sc.parallelize(processed_lines).flatMap(map_term_frequencies)

# COMMAND ----------

mapped_terms_with_keys = mapped_terms.map(lambda x: ((x[0], x[1]), x[2]))

# COMMAND ----------

term_frequencies = mapped_terms_with_keys.reduceByKey(lambda x, y: x + y)

# COMMAND ----------

term_frequencies_collected = term_frequencies.collect()

# COMMAND ----------

term_frequencies_collected[:5]

# COMMAND ----------

word_document_pairs = sc.parallelize(term_frequencies_collected).map(lambda x: (x[0][1], 1))

# COMMAND ----------

document_frequency = word_document_pairs.reduceByKey(lambda x, y: x + y)

# COMMAND ----------

document_frequency_collected = document_frequency.collect()

# COMMAND ----------

document_frequency_collected[:5]

# COMMAND ----------

inverse_doc_freq = sc.parallelize(document_frequency_collected).map(lambda x: (x[0], math.log(num_of_summaries/x[1])))

# COMMAND ----------

inverse_doc_freq_collected = inverse_doc_freq.collect()

# COMMAND ----------

inverse_doc_freq_collected[:5]

# COMMAND ----------

term_frequencies_rearranged = sc.parallelize(term_frequencies_collected).map(lambda x: (x[0][1], (x[0][0], x[1])))

# COMMAND ----------

term_frequencies_rearranged_collected = term_frequencies_rearranged.collect()

# COMMAND ----------

term_frequencies_rearranged_collected[:2]

# COMMAND ----------

tf_idf = sc.parallelize(term_frequencies_rearranged_collected).join(sc.parallelize(inverse_doc_freq_collected))

# COMMAND ----------

tf_idf_values = tf_idf.map(lambda x: (x[1][0][0], (x[0], x[1][0][1] * x[1][1])))

# COMMAND ----------

tf_idf_collected = tf_idf_values.collect()

# COMMAND ----------

tf_idf_collected[:5]

# COMMAND ----------

movie_metadata = sc.textFile("/FileStore/tables/movie_metadata-1.tsv")

# COMMAND ----------

movies_id_name = movie_metadata.map(lambda x: ((x.split("\t"))[0], (x.split("\t"))[2])).collectAsMap()

# COMMAND ----------

user_search_queries = sc.textFile("/FileStore/tables/user_search_queries.txt")

# COMMAND ----------

all_queries = user_search_queries.collect()

# COMMAND ----------

single_term_queries = user_search_queries.take(5)

# COMMAND ----------

multiple_term_queries = all_queries[-5:]

# COMMAND ----------

def get_top_10_movies_single_term_query(term):
    mapped_tf_idf_tuples_for_term = sc.parallelize(tf_idf_collected).flatMap(lambda doc: [(doc[0], doc[1][1])] if doc[1][0]==term else [])
    rdd_sorted = mapped_tf_idf_tuples_for_term.sortBy(lambda x: x[1], ascending=False)
    rdd_sorted_new = rdd_sorted.collect()
    top_10_movies = [(movies_id_name.get(id_, "[unknown]"), movie_tf_idf_score) 
                     for id_, movie_tf_idf_score in rdd_sorted_new[:10]]
    return top_10_movies

# COMMAND ----------

for single_query_term in single_term_queries:
    top_10_movies_single_term = get_top_10_movies_single_term_query(single_query_term.lower())
    print(f"\n Top 10 movies for '{single_query_term}':")
    print("-" * 77)  
    print(f"| {'Movie Name':<50} | {'TF-IDF':<20} |")
    print("-" * 77)  
    for movie_and_tf_idf_score in top_10_movies_single_term:
        print(f"| {movie_and_tf_idf_score[0]:50} | {movie_and_tf_idf_score[1]:<20} |")
    print("-" * 77)

# COMMAND ----------

tf_idf_document_id_grouping = sc.parallelize(tf_idf_collected).groupByKey()
tf_idf_document_id_grouping = tf_idf_document_id_grouping.mapValues(dict)
tf_idf_final = tf_idf_document_id_grouping.collectAsMap()

# COMMAND ----------

def create_multiple_term_vectors(multiple_term_query, inverse_doc_freq_map):
    multiple_term_query_count = defaultdict(int)
    words = word_tokenize(multiple_term_query)
    multiple_term_query_without_stopwords = [w for w in words if w not in stop_words and w.isalnum()]

    for term in multiple_term_query_without_stopwords:
        multiple_term_query_count[term] = multiple_term_query_count[term] + 1

    tf_of_muti_term_query = {term: count for term, count in multiple_term_query_count.items()}
    tf_idf_of_multi_term_query = {term: tf_of_muti_term_query[term] * inverse_doc_freq_map.get(term, 0) for term in tf_of_muti_term_query}
    return tf_idf_of_multi_term_query

# COMMAND ----------

inverse_doc_freq_map = inverse_doc_freq.collectAsMap()

# COMMAND ----------

vectors_of_multiple_term_queries = {query: create_multiple_term_vectors(query, inverse_doc_freq_map) for query in multiple_term_queries}

# COMMAND ----------

def calculate_cosine_similarity(vector1, vector2):
    intersection_of_terms = set(vector1.keys()).intersection(set(vector2.keys()))
    magnitude_of_vector1 = norm(list(vector1.values()))
    magnitude_of_vector2 = norm(list(vector2.values()))
    cosine_sim = (sum([vector1[term] * vector2[term] for term in intersection_of_terms])) / (magnitude_of_vector1 * magnitude_of_vector2)
    return cosine_sim

# COMMAND ----------

for multiple_term_query in multiple_term_queries:
    multiple_term_query_vector = vectors_of_multiple_term_queries[multiple_term_query]
    similarities = [(movie_id, calculate_cosine_similarity(multiple_term_query_vector, tf_idf)) for movie_id, tf_idf in tf_idf_final.items()]
    top_10_movies = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
    print(f"\n Top 10 movies for '{multiple_term_query}':")
    print("-" * 77)
    print(f"| {'Movie Name':<50} | {'Cosine Similarity':<20} |")
    print("-" * 77)
    
    for movie_id, cosine_sim in top_10_movies:
        movie_name = movies_id_name.get(movie_id, "[unknown]")
        print(f"| {movie_name:<50} | {cosine_sim:<20.6f} |")

    print("-" * 77)
