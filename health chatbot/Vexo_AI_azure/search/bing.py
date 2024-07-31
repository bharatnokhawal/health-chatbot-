from serpapi import GoogleSearch, BingSearch
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta
def configure():
    load_dotenv()

def call_google_search_engine(query):
    configure()
    serpapi_api_key = os.getenv('serpapi_api_key')
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
    serpapi_api_key = os.getenv('serpapi_api_key')
    params = {
        "q": query,
        "hl": "en",
        "gl": "in",
        "api_key": serpapi_api_key,
    }

    search = BingSearch(params)
    return search.get_dict()





# Example usage





def parse_published_date(published_date_str):
    if "hour" in published_date_str or "hours" in published_date_str:
        try:
            hours = int(published_date_str.split()[0])
            return datetime.now() - timedelta(hours=hours)
        except ValueError:
            return datetime.now()
    elif "day" in published_date_str or "days" in published_date_str:
        try:
            days = int(published_date_str.split()[0])
            return datetime.now() - timedelta(days=days)
        except ValueError:
            return datetime.now()
    elif "week" in published_date_str or "weeks" in published_date_str:
        try:
            weeks = int(published_date_str.split()[0])
            return datetime.now() - timedelta(weeks=weeks)
        except ValueError:
            return datetime.now()
    elif "month" in published_date_str or "months" in published_date_str:
        try:
            months = int(published_date_str.split()[0])
            return datetime.now() - timedelta(days=months * 30)
        except ValueError:
            return datetime.now()
    else:
        return datetime.now()

def call_image_search_engine(query):
    configure()
    serpapi_api_key = os.getenv('serpapi_api_key')
    search_engine = "google_images"

    params = {
        "q": query,
        "engine": search_engine,
        "api_key": serpapi_api_key,
    }

    search = GoogleSearch(params)
    image_results = search.get_dict()
    results = image_results.get("images_results", [])

    # Apply parse_published_date function to each image result
    for result in results:
        result['parsed_date'] = parse_published_date(result.get('published_date', ''))

    # Sort the image results based on parsed dates (most relevant or most published)
    sorted_results = sorted(results, key=lambda x: x.get('parsed_date', datetime.now()), reverse=True)

    # Limit the output to 4-5 titles and links
    limited_results = sorted_results[:5]

    # Print the limited results
    for idx, result in enumerate(limited_results, 1):
        print(f"{idx}. Title: {result.get('title', 'Unknown')} - Link: {result.get('original', 'Unknown')}")


# Example usage
##call_image_search_engine(query)

def parse_published_date(published_date_str):
    if "hour" in published_date_str or "hours" in published_date_str:
        try:
            hours = int(published_date_str.split()[0])
            return datetime.now() - timedelta(hours=hours)
        except ValueError:
            return datetime.now()
    elif "day" in published_date_str or "days" in published_date_str:
        try:
            days = int(published_date_str.split()[0])
            return datetime.now() - timedelta(days=days)
        except ValueError:
            return datetime.now()
    elif "week" in published_date_str or "weeks" in published_date_str:
        try:
            weeks = int(published_date_str.split()[0])
            return datetime.now() - timedelta(weeks=weeks)
        except ValueError:
            return datetime.now()
    elif "month" in published_date_str or "months" in published_date_str:
        try:
            months = int(published_date_str.split()[0])
            return datetime.now() - timedelta(days=months * 30)
        except ValueError:
            return datetime.now()
    else:
        return datetime.now()

def call_youtube_search_engine(query):
    configure()
    serpapi_api_key = os.getenv('serpapi_api_key')
    search_engine = "youtube"

    params = {
        "search_query": query,
        "engine": search_engine,
        "api_key": serpapi_api_key,
        "num": 5  # Limit the number of results to 5
    }

    search = GoogleSearch(params)
    youtube_results = search.get_dict()
    results = youtube_results["video_results"]
    
    relevant_results = []
    for movie in results:
        if 'title' in movie and 'link' in movie and 'published_date' in movie:
            published_date = parse_published_date(movie['published_date'])
            relevant_results.append((movie['title'], movie['link'], published_date))

    # Sort by publish date in descending order (most recent first)
    relevant_results.sort(key=lambda x: x[2], reverse=True)

    # Print the most relevant 4-5 titles, links, and publish dates
    for title, link, publish_date in relevant_results[:5]:
        print("Title:", title)
        print("Link:", link)
        print("Published Date:", publish_date.strftime("%Y-%m-%d %H:%M:%S"))

def call_google_scholar_search(query):
    configure()
    api_key = os.getenv('serpapi_api_key')
    
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": api_key
    }

    search = GoogleSearch(params)
    return search.get_dict()






#call_youtube_search_engine('arvind kejriwal arrest')
def call_search_engine(query):
    google_result = call_google_search_engine(query)
    bing_result = call_bing_search_engine(query)
    
    merged_results = {**google_result, **bing_result}

    return rank_results(merged_results, query)  # Call the ranking function directly here

def rank_results(results, query):
    # Extract the search results from the dictionary
    threshold = 0.2
    search_results = results['organic_results']

    # Extract the text of the search results with snippets available
    texts = [result['snippet'] for result in search_results if 'snippet' in result]

    # Check if there are snippets available
    if len(texts) == 0:
        return {'organic_results': []}  # Return empty results if no snippets are available

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

    # Reshape recency_scores to match the number of search results
    recency_scores = recency_scores[:cosine_similarities.shape[1]]

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