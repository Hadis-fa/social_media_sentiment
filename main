import praw # reddit api 
import pandas as pd
import numpy as np
from transformers import pipeline
from langdetect import detect, LangDetectException  # for language 
from datetime import datetime
from googletrans import Translator
import requests
import matplotlib.pyplot as plt


# --- API Keys ---
# ALPHA_VANTAGE_API_KEY = 
CLIENT_ID = 
CLIENT_SECRET = 
USER_AGENT = 


# --- Reddit Auth ---
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# --- Sentiment Model & Language Support ---
translator = Translator()
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_labels = ["Positive", "Negative", "Neutral"]

# --- Alpha Vantage Lookup ---
# def get_company_name_from_alphavantage(ticker: str, api_key: str) -> str:
#     url = "https://www.alphavantage.co/query"
#     params = {
#         "function": "SYMBOL_SEARCH",
#         "keywords": ticker,
#         "apikey": api_key
#     }
#     response = requests.get(url, params=params)
#     if response.status_code == 200:
#         data = response.json()  # converting the response from api into a python dic
#         matches = data.get("bestMatches", [])
#         if matches:
#             name = matches[0].get("2. name", "")
#             return name.replace(" Inc", "").replace(" Co. Ltd", "").strip()
#         else: 
#             print(f"no matches found for the ticker.")

#     else:
#         print(f"Error: Alpha Vantage API request failed with status code {response.status_code}. Using ticker '{ticker}' instead.")
#     return ticker

# --- Population Stats ---
def population_stats(values):
    if not values:
        return 0, 0, 0
    return np.mean(values), np.var(values), np.std(values)

# --- Reddit Post & Comment Fetching ---
def fetch_comments_with_keyword(keywords, subreddits, posts_per_subreddit, comments_per_post):
    collected = []
    pos_scores, neg_scores, neu_scores = [], [], []

    for subreddit_name in subreddits:
        print(f"\n--- Searching subreddit: r/{subreddit_name} ---")
        subreddit = reddit.subreddit(subreddit_name)
        stream = subreddit.search(query=" OR ".join(keywords), sort="new", time_filter="year", limit=1000000)

        matched_posts = 0

        for post in stream:
            if any(kw.lower() in post.title.lower() for kw in keywords):
                print(f"\nMatched Post: {post.title[:100]}")
                matched_posts += 1
                post.comments.replace_more(limit=0)  # getting all the comments and sub comments
                count = 0
                for comment in post.comments.list():
                    if count >= comments_per_post:
                        break
                    text = comment.body.strip()
                    if not text:
                        continue
                    try:
                        if detect(text) != 'en':
                            text = translator.translate(text, dest='en').text  # getting the text out of the returned object
                    except (LangDetectException, Exception):
                        continue

                    result = classifier(text, candidate_labels=sentiment_labels)
                    sentiment = dict(zip(result["labels"], result["scores"]))
                    pos, neg, neu = sentiment["Positive"], sentiment["Negative"], sentiment["Neutral"]

                    collected.append({
                        "Subreddit": subreddit_name,
                        "Post Title": post.title,
                        "Post Date": datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d"),
                        "Comment": text,
                        "Positive Score": pos,
                        "Negative Score": neg,
                        "Neutral Score": neu
                    })
                    pos_scores.append(pos)
                    neg_scores.append(neg)
                    neu_scores.append(neu)
                    count += 1

                if matched_posts >= posts_per_subreddit:
                    break

    return collected, pos_scores, neg_scores, neu_scores


def plot_sentiment_pie_chart(df):
    total_pos = df['Positive Score'].sum()
    total_neg = df['Negative Score'].sum()
    total_neu = df['Neutral Score'].sum()
    sizes = [total_pos, total_neg, total_neu]
    labels = ['Positive', 'Negative', 'Neutral']
    colors = ['green', 'red', 'gray']
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title("Overall Sentiment Distribution (Pie Chart)")
    plt.tight_layout()
    plt.show()


def plot_monthly_sentiment_trend(df):
    df['Post Date'] = pd.to_datetime(df['Post Date'])
    df['Month'] = df['Post Date'].dt.to_period('M')
    monthly_avg = df.groupby('Month')[['Positive Score', 'Negative Score', 'Neutral Score']].mean().reset_index()
    monthly_avg['Month'] = monthly_avg['Month'].astype(str)
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_avg['Month'], monthly_avg['Positive Score'], label='Positive', color='green', linewidth=2)
    plt.plot(monthly_avg['Month'], monthly_avg['Negative Score'], label='Negative', color='red', linewidth=2)
    plt.plot(monthly_avg['Month'], monthly_avg['Neutral Score'], label='Neutral', color='gray', linewidth=2)
    plt.title("Monthly Average Reddit Sentiment")
    plt.xlabel("Month")
    plt.ylabel("Average Sentiment Score")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_sentiment_volume_by_month(df):
    # Convert 'Post Date' to datetime
    df['Post Date'] = pd.to_datetime(df['Post Date'])
    df['Post Month'] = df['Post Date'].dt.to_period('M').dt.to_timestamp()  # Group by month

    # Assign label for dominant sentiment (highest score)
    def label_dominant(row):
        scores = {
            "Positive": row["Positive Score"],
            "Negative": row["Negative Score"],
            "Neutral": row["Neutral Score"]
        }
        return max(scores, key=scores.get)

    df["Dominant Sentiment"] = df.apply(label_dominant, axis=1)

    # Count number of each sentiment per month
    monthly_counts = df.groupby(["Post Month", "Dominant Sentiment"]).size().unstack(fill_value=0)

    # Ensure all sentiment columns exist
    for sentiment in ["Positive", "Negative", "Neutral"]:
        if sentiment not in monthly_counts.columns:
            monthly_counts[sentiment] = 0
    monthly_counts = monthly_counts[["Positive", "Negative", "Neutral"]]

    # Plot
    monthly_counts.plot(
        kind="bar",
        stacked=True,
        figsize=(14, 6),
        color={"Positive": "green", "Negative": "red", "Neutral": "gold"}
    )
    plt.title("Monthly Sentiment Volume")
    plt.xlabel("Month")
    plt.ylabel("Number of Comments")
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # ticker = input("Enter stock ticker (e.g., AAPL, TSLA): ").strip().upper()
    company_name = input("Enter the company name: ").strip().upper()
    # company_name_clean = company_name.replace(" Inc", "").replace(" Co. Ltd", "").strip()
    print("Searching Reddit posts for:", company_name)
    posts_per_subreddit = int(input("How many posts per subreddit: "))
    comments_per_post = int(input("How many comments per post: "))
    finance_subreddits = [
    "investing", "stocks", "wallstreetbets", "StockMarket", "options",
    "personalfinance", "finance", "FinancialIndependence", "CryptoCurrency",
    "algotrading", "economy"]


    keywords = [company_name]
    comments, pos_scores, neg_scores, neu_scores = fetch_comments_with_keyword(
        keywords, finance_subreddits, posts_per_subreddit, comments_per_post
    )
    df = pd.DataFrame(comments)
    filename = f"reddit_sentiment_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    df.to_excel(filename, index=False)
    print(f"\nSaved {len(df)} comment(s) to {filename}")
    plot_sentiment_pie_chart(df)
    plot_monthly_sentiment_trend(df)
    plot_sentiment_volume_by_month(df)

    pos_mean, pos_var, pos_std = population_stats(pos_scores)
    neg_mean, neg_var, neg_std = population_stats(neg_scores)
    neu_mean, neu_var, neu_std = population_stats(neu_scores)
    print("\nSentiment Summary:")
    print(f"Positive - Mean: {pos_mean:.4f}, Variance: {pos_var:.4f}, Std Dev: {pos_std:.4f}")
    print(f"Negative - Mean: {neg_mean:.4f}, Variance: {neg_var:.4f}, Std Dev: {neg_std:.4f}")
    print(f"Neutral  - Mean: {neu_mean:.4f}, Variance: {neu_var:.4f}, Std Dev: {neu_std:.4f}")
