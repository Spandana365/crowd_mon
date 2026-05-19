"""
ENHANCED SENTIMENT ANALYSIS - TEXT + EASYOCR IMAGE TEXT EXTRACTION
- VADER: Text sentiment from captions
- EasyOCR: Extracts text from images (English + Hindi support)
- Combined sentiment analysis for pre-event prediction
"""

import sys
from apify_client import ApifyClient
import re
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import json
import requests
import time
import io
from PIL import Image
import os
import signal
from contextlib import contextmanager
import threading

@contextmanager
def timeout_context(seconds):
    """Context manager for timeout handling (cross-platform)"""
    def timeout_handler():
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    timer = threading.Timer(seconds, timeout_handler)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Optional EasyOCR import - handle memory issues gracefully
try:
    import easyocr
    OCR_ENABLED = True
    print("EasyOCR loaded successfully (English + Hindi support)")
except (ImportError, OSError) as e:
    print(f"EasyOCR not available: {e}")
    print("  OCR features will be disabled")
    easyocr = None
    OCR_ENABLED = False

# Initialize EasyOCR reader if available
reader = None
if OCR_ENABLED and easyocr:
    try:
        reader = easyocr.Reader(['en', 'hi'], gpu=False)
    except Exception as e:
        print(f"EasyOCR reader initialization failed: {e}")
        OCR_ENABLED = False

# Initialize VADER
try:
    sia = SentimentIntensityAnalyzer()
    print("VADER loaded successfully")
except:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    print("VADER downloaded and loaded")



# Load API token from environment variable (set in .env file)
APIFY_TOKEN = os.environ.get("APIFY_TOKEN", "")


# 🧹 CLEAN TEXT (preserves Unicode: Hindi, emojis, etc.)
def clean_text(text):
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove hashtags but keep the content
    text = re.sub(r"#", "", text)
    # Remove ONLY special punctuation, but keep Unicode letters/numbers
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text, flags=re.UNICODE)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# � QUERY BUILDER
def build_event_queries(event_name):
    if not event_name:
        event_name = "Ram Navami"

    cleaned = re.sub(r"[^A-Za-z0-9\s]", "", event_name).strip()
    words = [w for w in cleaned.split() if w]
    if not words:
        words = ["Ram", "Navami"]

    hashtag_candidates = []
    hashtag_candidates.append("".join(words).lower())
    hashtag_candidates.append(words[0].lower())
    if len(words) > 1:
        hashtag_candidates.append((words[0] + words[1]).lower())
    if len(words) > 2:
        hashtag_candidates.append((words[0] + words[1] + words[2]).lower())
    if "temple" in cleaned.lower() or "mandir" in cleaned.lower():
        hashtag_candidates.extend(["temple", "mandir"])

    hashtags = []
    for tag in hashtag_candidates:
        if tag and tag not in hashtags:
            hashtags.append(tag)
    hashtags = hashtags[:5]

    query_terms = [f'"{cleaned}"']
    if len(words) > 1:
        query_terms.append(f'"{words[0]} {words[1]}"')
    query_terms.append(f'"{words[0]}"')
    if "temple" not in cleaned.lower() and "mandir" not in cleaned.lower():
        query_terms.append('"temple"')
        query_terms.append('"mandir"')

    query = " OR ".join(query_terms)
    return hashtags, query


# �🚫 SPAM FILTER
def is_spam(text):
    spam_keywords = ["bet", "casino", "bonus", "deposit", "telegram"]
    return any(word in text.lower() for word in spam_keywords)


# 🕒 FORMAT TIME
def format_time(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", "")).strftime("%Y-%m-%d %H:%M")
    except:
        return ""


# 🤖 TEXTBLOB SENTIMENT
# 🤖 VADER SENTIMENT
def get_vader_sentiment(text):
    """VADER Sentiment Analysis - 0.0 to 1.0"""
    if not text or len(text.strip()) == 0:
        return 0.5

    try:
        scores = sia.polarity_scores(text)
        compound = scores['compound']
        return (compound + 1) / 2
    except Exception as e:
        print(f"  ❌ VADER error: {e}")
        return 0.5


# 📸 EASYOCR - IMAGE TEXT EXTRACTION
def extract_text_from_image(image_url, likes_count=0):
    """
    Uses EasyOCR to extract text from images.
    Supports English and Hindi text from event posters.
    Returns: extracted_text, confidence_score
    """

    if not OCR_ENABLED or reader is None:
        return "", 0.0

    if not image_url:
        return "", 0.0

    if likes_count < 50:
        return "", 0.0  # Skip low-engagement posts

    try:
        print(f"   📸 EasyOCR processing image...")

        # Download image
        img_response = requests.get(image_url, timeout=10)
        img_response.raise_for_status()

        # Open image with PIL
        img = Image.open(io.BytesIO(img_response.content))

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save temporarily for EasyOCR (it works with file paths)
        temp_path = "temp_image_for_ocr.jpg"
        img.save(temp_path)

        try:
            # Extract text using EasyOCR
            # detail=0 returns just the strings, detail=1 returns boxes and confidence
            results = reader.readtext(temp_path, detail=1)

            # Filter text with reasonable confidence (>0.5)
            extracted_texts = []
            confidences = []

            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Only high confidence text
                    text = text.strip()
                    if text and len(text) > 2:  # Filter out single characters
                        extracted_texts.append(text)
                        confidences.append(confidence)

            # Combine all extracted text
            full_text = ' '.join(extracted_texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Clean the extracted text
            cleaned_text = clean_text(full_text)

            # Remove temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if cleaned_text:
                print(f"   📝 Extracted: '{cleaned_text[:60]}...' (Conf: {avg_confidence:.2f})")
                return cleaned_text, avg_confidence
            else:
                return "", 0.0

        except Exception as ocr_error:
            # Clean up temp file even if OCR fails
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise ocr_error

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Image download error: {e}")
        return "", 0.0
    except Exception as e:
        print(f"   ❌ EasyOCR error: {str(e)[:60]}")
        return "", 0.0


# 📊 MAIN FUNCTION
# 📊 MAIN SCRAPING & ANALYSIS
def analyze_and_export(event_name=None, start_date_str=None, end_date_str=None, timeout_seconds=120):
    """
    Analyze social media sentiment for an event with timeout protection.
    timeout_seconds: Maximum time to wait for scraping (default 2 minutes)
    """
    try:
        with timeout_context(timeout_seconds):
            return _analyze_and_export_internal(event_name, start_date_str, end_date_str)
    except TimeoutError as e:
        print(f"⚠️ Scraping timed out: {e}")
        return {
            'total_posts': 0,
            'instagram_posts': 0,
            'reddit_posts': 0,
            'average_sentiment_overall': 0.5,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'posts': [],
            'error': f'Scraping timed out after {timeout_seconds} seconds. Try with a shorter date range or different event name.'
        }
    except Exception as e:
        print(f"⚠️ Scraping error: {e}")
        return {
            'total_posts': 0,
            'instagram_posts': 0,
            'reddit_posts': 0,
            'average_sentiment_overall': 0.5,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'posts': [],
            'error': f'Scraping failed: {str(e)}'
        }

def _analyze_and_export_internal(event_name=None, start_date_str=None, end_date_str=None):
    # Parse date range if provided
    start_date = None
    end_date = None
    if start_date_str and end_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            print(f"📅 Filtering posts from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        except ValueError as e:
            print(f"⚠️ Date parsing error: {e}. Scraping without date filter.")
            start_date = None
            end_date = None

    client = ApifyClient(APIFY_TOKEN)
    posts = []
    instagram_error = None

    # ============================================================================
    # INSTAGRAM SCRAPING
    # ============================================================================

    print("\n" + "="*70)
    print("🚀 INSTAGRAM - Scraping posts (LIMIT: 50)...")
    print("="*70)

    hashtags, reddit_query = build_event_queries(event_name)
    print(f"🔎 Using Instagram hashtags: {hashtags}")
    print(f"🔎 Using Reddit query: {reddit_query}")

    # Use apify hashtag scraper with date filtering
    run_input_insta = {
        "hashtags": hashtags[:5],  # Use top 5 hashtags
        "resultsLimit": 50,
        "shouldDownloadCovers": False,
        "shouldDownloadSlideshows": False
    }

    # Add date filtering if dates are provided
    if start_date and end_date:
        run_input_insta["searchFrom"] = start_date.strftime("%Y-%m-%d")
        run_input_insta["searchTo"] = end_date.strftime("%Y-%m-%d")
        print(f"📅 Date filter applied: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    try:
        run_insta = client.actor("apify/instagram-hashtag-scraper").call(
            run_input=run_input_insta,
            timeout_secs=60  # 60 second timeout for the API call
        )

        insta_count = 0
        for item in client.dataset(run_insta["defaultDatasetId"]).iterate_items():
            if insta_count >= 50:
                break

            caption = item.get("caption", "")
            likes = item.get("likesCount", 0)
            comments = item.get("commentsCount", 0)
            username = item.get("ownerUsername", "unknown")
            timestamp = item.get("timestamp", "")
            url = item.get("url", "")
            display_url = item.get("displayUrl", "")

            if likes == -1:
                likes = 0

            if caption and not is_spam(caption):
                cleaned_caption = clean_text(caption)

                if cleaned_caption:
                    # TEXT SENTIMENT
                    text_sentiment = get_vader_sentiment(cleaned_caption)

                    # IMAGE TEXT EXTRACTION & SENTIMENT
                    image_text = ""
                    image_text_sentiment = 0.5
                    ocr_confidence = 0.0

                    if display_url and likes >= 50 and OCR_ENABLED:
                        image_text, ocr_confidence = extract_text_from_image(display_url, likes)
                        if image_text:
                            image_text_sentiment = get_vader_sentiment(image_text)
                        else:
                            image_text_sentiment = 0.5  # Neutral if no text found

                    # COMBINED SENTIMENT
                    # Weight: 40% caption + 60% image text (if available)
                    if image_text and ocr_confidence > 70:
                        final_sentiment = (text_sentiment * 0.4) + (image_text_sentiment * 0.6)
                    else:
                        final_sentiment = text_sentiment

                    post = {
                        "platform": "instagram",
                        "author": username,
                        "caption": cleaned_caption,
                        "likes": likes,
                        "comments": comments,
                        "timestamp": format_time(timestamp),
                        "url": url,
                        "image_url": display_url,
                        "sentiment": {
                            "text_sentiment": round(text_sentiment, 3),
                            "image_text": image_text if image_text else None,
                            "image_text_sentiment": round(image_text_sentiment, 3) if image_text else None,
                            "ocr_confidence": round(ocr_confidence, 1) if ocr_confidence > 0 else None,
                            "final_sentiment": round(final_sentiment, 3)
                        }
                    }

                    posts.append(post)
                    insta_count += 1
                    print(f"  ✓ [{insta_count}] {cleaned_caption[:50]}...")
                    print(f"       Caption: {text_sentiment:.2f} | Image Text: {image_text_sentiment:.2f} | Final: {final_sentiment:.2f}")
                    if image_text:
                        print(f"       📝 OCR: '{image_text[:40]}...'")

        print(f"✅ Instagram: {insta_count} posts")
    except Exception as e:
        instagram_error = str(e)
        print(f"❌ Instagram error: {instagram_error}")

    # ============================================================================
    # REDDIT SCRAPING
    # ============================================================================

    print("\n" + "="*70)
    print("🚀 REDDIT - Scraping posts (LIMIT: 50)...")
    print("="*70)
    subreddits = ["india", "IndiaSpeaks", "news"]
    QUERY = reddit_query
    LIMIT = 50
    PER_PAGE = 25

    headers = {
        "User-Agent": "windows:ramnavami.analysis:v1.0"
    }

    reddit_posts = []

    for sub in subreddits:
        if len(reddit_posts) >= LIMIT:
            break

        after = None

        while len(reddit_posts) < LIMIT:
            url = f"https://www.reddit.com/r/{sub}/search.json"
            params = {
                "q": QUERY,
                "sort": "new",
                "limit": PER_PAGE,
                "after": after,
                "restrict_sr": "on"
            }

            response = requests.get(url, headers=headers, params=params, timeout=30)  # 30 second timeout

            if response.status_code != 200:
                print(f"  ⚠ r/{sub}: Status {response.status_code}")
                time.sleep(2)
                break

            data = response.json()
            children = data.get("data", {}).get("children", [])

            if not children:
                break

            for item in children:
                if len(reddit_posts) >= LIMIT:
                    break

                d = item["data"]

                full_text = f"{d.get('title', '')} {d.get('selftext', '')}"
                cleaned = clean_text(full_text)

                if cleaned and not is_spam(cleaned):
                    text_sentiment = get_vader_sentiment(cleaned)
                    post_date = datetime.fromtimestamp(d.get("created_utc", 0))
                    timestamp = post_date.strftime("%Y-%m-%d %H:%M")

                    post = {
                        "platform": "reddit",
                        "author": d.get("author"),
                        "caption": cleaned,
                        "likes": d.get("score", 0),
                        "comments": d.get("num_comments", 0),
                        "timestamp": timestamp,
                        "url": f"https://reddit.com{d.get('permalink')}",
                        "image_url": None,
                        "sentiment": {
                            "text_sentiment": round(text_sentiment, 3),
                            "image_text": None,
                            "image_text_sentiment": None,
                            "ocr_confidence": None,
                            "final_sentiment": round(text_sentiment, 3)
                        }
                    }

                    reddit_posts.append(post)
                    print(f"  ✓ Reddit: {cleaned[:50]}...")

            after = data.get("data", {}).get("after")
            if not after:
                break

            time.sleep(1)

    # ============================================================================
    # SUMMARY & SAVE
    # ============================================================================

    # Merge Reddit posts into main posts list
    posts.extend(reddit_posts)
    print(f"✅ Reddit: {len(reddit_posts)} posts")

    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)

    insta_posts = [p for p in posts if p["platform"] == "instagram"]
    fb_posts = []  # Facebook removed
    reddit_posts_final = [p for p in posts if p["platform"] == "reddit"]

    avg_insta = sum(p["sentiment"]["final_sentiment"] for p in insta_posts) / len(insta_posts) if insta_posts else 0
    avg_fb = 0  # Facebook removed
    avg_reddit = sum(p["sentiment"]["final_sentiment"] for p in reddit_posts_final) / len(reddit_posts_final) if reddit_posts_final else 0
    avg_overall = sum(p["sentiment"]["final_sentiment"] for p in posts) / len(posts) if posts else 0

    print(f"\nPlatform Breakdown:")
    print(f"  Instagram: {len(insta_posts)} posts (Avg: {avg_insta:.3f})")
    print(f"  Facebook:  {len(fb_posts)} posts (removed)")
    print(f"  Reddit:    {len(reddit_posts_final)} posts (Avg: {avg_reddit:.3f})")
    print(f"  TOTAL:     {len(posts)} posts (Avg: {avg_overall:.3f})")

    # Count posts with OCR text
    ocr_posts = [p for p in posts if p["sentiment"].get("image_text")]
    print(f"\n📸 EasyOCR Results: {len(ocr_posts)} posts had extractable text from images")

    try:
        output_file = "insta_facebook_reddit_easyocr_sentiment.json"

        data = {
            "posts": posts,
            "summary": {
                "total_posts": len(posts),
                "instagram_posts": len(insta_posts),
                "facebook_posts": len(fb_posts),  # Will be 0
                "reddit_posts": len(reddit_posts_final),
                "posts_with_ocr_text": len(ocr_posts),
                "average_sentiment_instagram": round(avg_insta, 3),
                "average_sentiment_facebook": round(avg_fb, 3),  # Will be 0
                "average_sentiment_reddit": round(avg_reddit, 3),
                "average_sentiment_overall": round(avg_overall, 3),
                "analysis_method": "VADER (text) + EasyOCR (image text) + Combined sentiment",
                "ocr_enabled": OCR_ENABLED,
                "facebook_removed": True,
                "instagram_error": instagram_error,
                "date_filter_applied": start_date is not None,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "timestamp": datetime.now().isoformat()
            }
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Output saved: {output_file}")
        print("="*70 + "\n")

        return data["summary"]  # Return summary data for app.py integration

    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
        return {
            "total_posts": len(posts),
            "instagram_posts": len(insta_posts),
            "facebook_posts": 0,
            "reddit_posts": len(reddit_posts_final),
            "average_sentiment_instagram": round(avg_insta, 3),
            "average_sentiment_facebook": 0,
            "average_sentiment_reddit": round(avg_reddit, 3),
            "average_sentiment_overall": round(avg_overall, 3)
        }


# 🔥 RUN
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 ENHANCED SENTIMENT ANALYSIS - TEXT + EASYOCR")
    print("="*70)
    print("📝 TEXT: VADER Sentiment (0.0-1.0)")
    print("📸 IMAGE: EasyOCR text extraction + sentiment (English + Hindi)")
    print("⚡ Posts per platform: 5 (Limited for testing)")
    print("🚫 Facebook: Removed from scraping")
    print("="*70)

    if not OCR_ENABLED:
        print("\n⚠️  EASYOCR NOT INSTALLED!")
        print("   Install with: pip install easyocr")
        print("   Also install: pip install torch torchvision torchaudio opencv-python-headless")
        print("="*70)

    result = analyze_and_export()
    print(f"✨ Analysis complete!")