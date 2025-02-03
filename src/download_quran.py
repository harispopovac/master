import requests
import pandas as pd
import os
import json

def download_quran_dataset():
    """
    Download the Quran dataset from the Quran.com API
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    try:
        # Download Arabic text and English translation
        print("Downloading Quran dataset...")
        
        # We'll use Quran.com API v4
        base_url = "https://api.quran.com/api/v4"
        
        # Get all chapters
        chapters_response = requests.get(f"{base_url}/chapters?language=en")
        chapters = chapters_response.json()['chapters']
        
        data = []
        total_verses = sum(chapter['verses_count'] for chapter in chapters)
        
        print(f"Total chapters: {len(chapters)}")
        print(f"Total verses: {total_verses}")
        
        for chapter in chapters:
            surah_number = chapter['id']
            print(f"Processing Surah {surah_number}...")
            
            # Get verses for this chapter
            verses_response = requests.get(
                f"{base_url}/verses/by_chapter/{surah_number}",
                params={
                    "language": "en",
                    "words": "true",
                    "translations": "131",  # Sahih International translation
                    "fields": "text_uthmani",
                    "per_page": 1000
                }
            )
            
            verses = verses_response.json()['verses']
            
            for verse in verses:
                verse_data = {
                    'surah': surah_number,
                    'ayah': verse['verse_number'],
                    'text': verse['text_uthmani'],
                    'translation': verse['translations'][0]['text']
                }
                data.append(verse_data)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv('data/quran.csv', index=False)
        print(f"\nDataset saved to data/quran.csv with {len(df)} verses")
        return df
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return None

if __name__ == "__main__":
    download_quran_dataset() 