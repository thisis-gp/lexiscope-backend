from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import os
import requests
from urllib.parse import urljoin
from webdriver_manager.chrome import ChromeDriverManager

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in background
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# Set up WebDriver (update chromedriver path)
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

base_url = "http://www.liiofindia.org/in/cases/cen/INSC/2024/"

def get_all_case_links():
    driver.get(base_url)
    time.sleep(3)  # Wait for page load
    
    # Extract all case links
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    case_links = []
    
    # Find all monthly sections
    months = soup.find_all('h3', class_='make-database')
    for month in months:
        ul = month.find_next_sibling('ul')
        if ul:
            links = ul.find_all('a', class_='make-database')
            for link in links:
                href = link.get('href')
                full_url = urljoin(base_url, href)
                case_links.append(full_url)
    
    return case_links

def download_pdf(case_url):
    driver.get(case_url)
    time.sleep(2)  # Wait for PDF object to load
    
    # Find PDF URL
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    object_tag = soup.find('object', {'data': True})
    
    if object_tag:
        pdf_path = object_tag['data']
        pdf_url = urljoin(case_url, pdf_path)
        
        # Create download directory
        os.makedirs('supreme_court_pdfs', exist_ok=True)
        
        # Download PDF
        response = requests.get(pdf_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        if response.status_code == 200:
            filename = os.path.join('supreme_court_pdfs', pdf_url.split('/')[-1])
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {pdf_url}")

# Main execution
try:
    case_links = get_all_case_links()
    print(f"Found {len(case_links)} cases")
    
    for i, link in enumerate(case_links, 1):
        print(f"Processing case {i}/{len(case_links)}: {link}")
        download_pdf(link)
        time.sleep(1)  # Be polite with delays
        
finally:
    driver.quit()