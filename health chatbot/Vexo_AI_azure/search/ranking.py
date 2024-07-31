from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def rank_results(results, query):
  # Extract the search results from the dictionary
  search_results = results['organic_results']

  # Extract the text of the search results
  texts = [result['snippet'] for result in search_results]

  # Add the query to the texts
  texts.append(query)

  # Create a TF-IDF Vectorizer and fit it to the texts
  vectorizer = TfidfVectorizer().fit(texts)

  # Transform the texts into a matrix of TF-IDF features
  tfidf_matrix = vectorizer.transform(texts)

  # Compute the cosine similarity between the query and each of the search results
  cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

  # Create a recency score based on the order of the results
  recency_scores = np.array(range(len(search_results), 0, -1))

  # Normalize the recency scores to the same range as the cosine similarities
  recency_scores = MinMaxScaler().fit_transform(recency_scores.reshape(-1, 1)).flatten()

  # Combine the cosine similarities and recency scores to get a final score for each result
  final_scores = cosine_similarities.flatten() + recency_scores

  # Get the indices of the final scores sorted in descending order
  sorted_indices = final_scores.argsort()[::-1]

  # Use these indices to sort the search results
  ranked_results = [search_results[i] for i in sorted_indices]

  # Wrap the ranked results in a dictionary under the key 'organic_results'
  ranked_results = {'organic_results': ranked_results}

  return ranked_results

query = "IPL 2024 Latest updates"
ranked_results = rank_results(results, query)
print(ranked_results)



from serpapi import GoogleSearch
from serpapi import BingSearch
from dotenv import load_dotenv
import os
import pdb

def configure():
  load_dotenv()

def call_google_search_engine(query):
  configure()
  serpapi_api_key = os.getenv('SERPAPI_API_KEY')
  search_engines = ["google", "duckduckgo"]

  for engine in search_engines:
    params = {
      "q": query,
      "engine": engine,
      "hl": "en",
      "gl": "in",
      "google_domain": "google.com",
      "api_key": serpapi_api_key,
    }
    # Call the API with these parameters
  search = GoogleSearch(params)
  return search.get_dict()

def call_bing_search_engine(query):
  configure()
  serpapi_api_key = os.getenv('SERPAPI_API_KEY')
  params = {
    "q": query,
    "hl": "en",
    "gl": "in",
    
    "api_key": serpapi_api_key,
  }

  search = BingSearch(params)
  return search.get_dict()


def call_search_engine(query):
  google_result = call_google_search_engine(query)
  bing_result = call_bing_search_engine(query)
  
  merged_results = {**google_result, **bing_result}

  return merged_results
 




# print(results)

# results

# def rank_results(results):
#   # Extract the search results from the dictionary
#   search_results = results['organic_results']
  
#   # Sort the search results based on relevance or any other criteria
#   ranked_results = sorted(search_results, key=lambda x: x['position'])
  
#   return ranked_results

# ranked_results = rank_results(results)
# print(ranked_results)



def rank_results(results, query, threshold=0.9):
    # Extract the search results from the dictionary
    search_results = results['organic_results']

    # Extract the text of the search results
    texts = [result['snippet'] for result in search_results]

    # Add the query to the texts
    texts.append(query)

    # Create a TF-IDF Vectorizer and fit it to the texts
    vectorizer = TfidfVectorizer().fit(texts)

    # Transform the texts into a matrix of TF-IDF features
    tfidf_matrix = vectorizer.transform(texts)

    # Compute the cosine similarity between the query and each of the search results
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Create a recency score based on the order of the results
    recency_scores = np.array(range(len(search_results), 0, -1))

    # Normalize the recency scores to the same range as the cosine similarities
    recency_scores = MinMaxScaler().fit_transform(recency_scores.reshape(-1, 1)).flatten()

    # Combine the cosine similarities and recency scores to get a final score for each result
    final_scores = cosine_similarities.flatten() + recency_scores

    # Filter results based on the threshold
    filtered_indices = np.where(final_scores >= threshold)[0]

    # Use these indices to sort the search results
    ranked_results = [search_results[i] for i in filtered_indices]

    # Wrap the ranked results in a dictionary under the key 'organic_results'
    ranked_results = {'organic_results': ranked_results}

    return ranked_results



from serpapi import GoogleSearch, BingSearch
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def configure():
    load_dotenv()

def call_google_search_engine(query):
    configure()
    serpapi_api_key = os.getenv('SERPAPI_API_KEY')
    search_engines = ["google", "duckduckgo"]

    for engine in search_engines:
        params = {
            "q": query,
            "engine": engine,
            "hl": "en",
            "gl": "in",
            "google_domain": "google.com",
            "api_key": serpapi_api_key,
        }
        # Call the API with these parameters
    search = GoogleSearch(params)
    return search.get_dict()

def call_bing_search_engine(query):
    configure()
    serpapi_api_key = os.getenv('SERPAPI_API_KEY')
    params = {
        "q": query,
        "hl": "en",
        "gl": "in",
        "api_key": serpapi_api_key,
    }

    search = BingSearch(params)
    return search.get_dict()

def call_search_engine(query):
    google_result = call_google_search_engine(query)
    bing_result = call_bing_search_engine(query)
  
    merged_results = {**google_result, **bing_result}

    return rank_results(merged_results, query)  # Call the ranking function directly here

def rank_results(results, query):
    # Extract the search results from the dictionary
    search_results = results.get('organic_results', [])  # Get the search results list or an empty list if it's not present

    # Initialize an empty list to store the snippets
    texts = []

    # Extract the 'snippet' from each search result, handling cases where 'snippet' may not be present
    for result in search_results:
        try:
            snippet = result['snippet']
            texts.append(snippet)
        except KeyError:
            pass  # Handle the case where 'snippet' is not present

    # Add the query to the texts
    texts.append(query)

    # Create a TF-IDF Vectorizer and fit it to the texts
    vectorizer = TfidfVectorizer().fit(texts)

    # Transform the texts into a matrix of TF-IDF features
    tfidf_matrix = vectorizer.transform(texts)

    # Compute the cosine similarity between the query and each of the search results
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Create a recency score based on the order of the results
    recency_scores = np.array(range(len(search_results), 0, -1))

    # Normalize the recency scores to the same range as the cosine similarities
    recency_scores = MinMaxScaler().fit_transform(recency_scores.reshape(-1, 1)).flatten()

    # Combine the cosine similarities and recency scores to get a final score for each result
    final_scores = cosine_similarities.flatten() + recency_scores

    # Filter out older articles based on a threshold score
    threshold_score = 0.5
    ranked_results = [result for result, score in zip(search_results, final_scores) if score >= threshold_score]

    # Sort the ranked results based on final scores in descending order
    ranked_results.sort(key=lambda x: final_scores[search_results.index(x)], reverse=True)

    # Wrap the ranked results in a dictionary under the key 'organic_results'
    ranked_results = {'organic_results': ranked_results}

    return ranked_results

