import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

web_texts = []
max_pages = 10

for page in range(max_pages):
    url = 'https://www.fmkorea.com/index.php?mid=best&page=' + str(page)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    items = soup.select(".hotdeal_var8")

    for item in items[4:]:
        web_text = item.text
        web_text = web_text.strip()
        web_text = re.sub(r'\xa0\t\[\d+\]$', '', web_text)
        web_texts.append(web_text)

print(web_texts)
