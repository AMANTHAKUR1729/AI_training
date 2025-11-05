import requests
from bs4 import BeautifulSoup
import re

class WebScraper:
    @staticmethod
    def scrape_url(url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text if text.strip() else "No readable text content found on this page."
        except Exception as e:
            return f"Error scraping URL: {str(e)}"

class TextCleaner:
    @staticmethod
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        return text.strip()