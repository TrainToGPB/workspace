import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import random
import re
from itertools import chain
from datetime import datetime
import time

url_home = 'https://www.fmkorea.com/'
url_main = 'index.php?mid=best&page='
max_pages = 3
time_now = datetime.now().strftime('%y%m%d_%H%M%S')


def get_thread(page):
    url_main_total = url_home + url_main + str(page)
    response_main = requests.get(url_main_total)
    soup_main = BeautifulSoup(response_main.text, 'html.parser')

    contents_main = soup_main.select('.hotdeal_var8')

    return contents_main

def get_cml(max_pages):
    contents_main_list = []
    for page in range(1, max_pages + 1):
        time.sleep(random.uniform(5, 10))

        contents_main = get_thread(page)
        contents_main_list.append(contents_main)
        print(f"Page {page} crawled")

    return contents_main_list
    
# 포텐 터진 게시판 1페이지부터 탐방
contents_main_list = get_cml(max_pages)
thread_texts_list = []
for page, contents_main in enumerate(contents_main_list, 1):
    thread_num = 1
    start = 4 if page == 1 else 0
    for content_main in contents_main[start:]:
        time.sleep(random.uniform(30, 40))

        thread_texts = []
        url_thread = url_home + content_main.get('href')
        response_thread = requests.get(url_thread)
        soup_thread = BeautifulSoup(response_thread.text, 'html.parser')

        # 게시글 제목
        title = soup_thread.select(".np_18px")
        print(title[0].text.strip())
        thread_texts.append(title[0].text.strip())

        # 게시 일시
        date = soup_thread.select(".date.m_no")
        thread_texts.append(date[0].text.strip())

        # 게시판 종류(메인 + 서브)
        category_raw = soup_thread.select(".pop_more")
        category = category_raw[0].text.strip()
        category_main, category_sub = category.split('\t\t - ')
        thread_texts.append(category_main.strip())
        thread_texts.append(category_sub.strip())

        # 조회 수, 추천 수, 댓글
        attentions = soup_thread.select(".side.fr")
        clicks, likes, replies = map(int, re.findall(r'\d+', attentions[0].text))
        thread_texts.append(clicks)
        thread_texts.append(likes)
        thread_texts.append(replies)

        # 본문 - 텍스트만
        content_text_raw = soup_thread.select(".xe_content")
        # content_text = content_text_raw[0].text.strip()
        content_text = content_text_raw[0].get_text(strip=True, separator='\n')
        content_text = re.sub(r'\s+', ' ', content_text)
        content_text = re.sub(r'\xa0', '', content_text)
        content_text = re.sub(r'\r', '', content_text)
        content_text = re.sub(r'\n+', '\n', content_text)
        content_text = re.sub(r'https?://\S+', '', content_text)
        content_text = re.sub(r'Video 태그를 지원하지 않는 브라우저입니다.', '', content_text)
        thread_texts.append(content_text)

        # 베스트 댓글 - 텍스트, 추천, 비추천
        comments_raw = soup_thread.select(".fdb_itm.clear.comment_best.comment-2")
        comments_text = []
        for comment_raw in comments_raw:
            comment_text_raw = comment_raw.select('.comment-content')
            comment_text = []
            for ctr in comment_text_raw:
                if ctr.select_one(".findParent"):
                    ct = '[R2C] ' + ctr.text.strip().split(maxsplit=1)[1] # R2C: Reply to Comment
                else:
                    ct = ctr.text.strip()
                ct = re.sub(r'\[댓글이 수정되었습니다: .*?\]', '', ct)
                
                comment_likes = int(comment_raw.select_one('.voted_count').text.strip())
                if comment_raw.select_one('.blamed_count').text.strip():
                    comment_hates = int(comment_raw.select_one('.blamed_count').text.strip())
                else:
                    comment_hates = 0
                
                comment_text.append(ct)
                comment_text.append(comment_likes)
                comment_text.append(comment_hates)
                
            comments_text.append(comment_text)
            
        max_comments = 4
        for _ in range(max_comments - len(comments_text)):
            comments_text.append([np.nan, np.nan, np.nan])
        thread_texts.extend(chain(*comments_text))

        print(f'Page {page} Thread {thread_num} Crawled')
        thread_num += 1

        thread_texts_list.append(thread_texts)

web_data_columns = ['Title',
                    'Time',
                    'Gallery',
                    'Gallery - Tab',
                    'Clicks',
                    'Total Likes (Likes - Hates)',
                    'Comments',
                    'Content Text',
                    'Best Comment 1',
                    'Best Comment 1 - Likes',
                    'Best Comment 1 - Hates',
                    'Best Comment 2',
                    'Best Comment 2 - Likes',
                    'Best Comment 2 - Hates',
                    'Best Comment 3',
                    'Best Comment 3 - Likes',
                    'Best Comment 3 - Hates',
                    'Best Comment 4',
                    'Best Comment 4 - Likes',
                    'Best Comment 4 - Hates',
                    ]
web_data = pd.DataFrame(thread_texts_list, columns=web_data_columns)
web_data.to_csv(f"fmkorea_{time_now}_poten_crawl.csv")