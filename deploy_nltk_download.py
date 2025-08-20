import nltk
import sys

def download_nltk_data():
    """Download required NLTK data for the chatbot"""
    try:
        print("Downloading NLTK data...")
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        print("NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False

if __name__ == "__main__":
    success = download_nltk_data()
    sys.exit(0 if success else 1)