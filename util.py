import json
import urllib.request
from pandas import DataFrame as df
from time import sleep
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from bs4 import Comment
import re
import os
import datetime

def get_novels(page):
    params = {
        'category': '0',
        'story_type': '2', # เรื่องยาว
        'is_end': '1', # นิยายจบแล้ว
        'nexttime': 'i_warn_u',
        'sort_by': '0', # เรียงตามอัปเดตล่าสุด
        'offset': str(
            page * # เลขหน้า (0-indexed)
            30 # 30 เรื่องต่อหน้า
        )
    }

    headers = {
        'Host': 'www.dek-d.com',
        'User-Agent': ' Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:65.0) Gecko/20100101 Firefox/65.0',
        'Accept': '*/*',
        'Accept-Language': 'en,en-US;q=0.7,th;q=0.3',
        # 'Accept-Encoding': 'gzip, deflate, br', # Not usable - not supported by Python URLLib
        'Referer': 'https://www.dek-d.com/home/writer/',
        #'X-CSRF-TOKEN': '66dc07b97aac8200e7e485037d126ab0',
        #'X-Requested-With': 'XMLHttpRequest',
        'Connection': 'keep-alive',
        #'Cookie': 'writer-hp-store-abtest=1; G_ENABLED_IDPS=google; fbm_193207127471363=base_domain=.dek-d.com; __gddads=100000358969339:undefined:0; _serc=dh3zsl20-4kcw-1dg2; PHPSESSID=bcn2l9rdjs8h4c54r3243vv41g; xcsrf-dd_login=95cc7cbdba8210d1e8afa0f563b4e533; fbsr_193207127471363=tRs2q7Cc1ImB6FmhoVK0b7hF67Zvupu9RLcyRdD_uDg.eyJhbGdvcml0aG0iOiJITUFDLVNIQTI1NiIsImNvZGUiOiJBUURrYWhsNTJnVWlNS040QzRNdnBYY2xzNWxEelFrY19ueEZMTWVfdnR5TjR6bWNBQWJxUmtvZDM0d0xxUzFWbFZNY29FelFQcHpVS1FsbjhwcXREZEJfVTFOSF90b1JQLWpWRjl1UWU3cks0Ti1Fc2o5WC1INEV6clBzQnEtQy1xUHZsQVJlVW9zUmU1NDlqTVdmSEVIdEdqSV9QdU1jU2pDRER1eHk3bzVvRERhdmpRc1VhTlRfNUFTcURnMXhNVUxmYVRDQlU3cGd5QTlNT3BQZlB3QXVrWUR0bl9RUjVFcjlHdlFjV3ItUUJmNGo4WlFSaXhZYjlMM2lSSG9DcWJRYlhZU3dlUGs1bDRWcVBMaWRsVUFXLTQyNzdjZFZGeE44YldUUzVJVWNrREpqcUJRSF9oS2hJYzlXbTRjZzdpak90UE96Umgya3h2eTU2cENnbFd3VyIsImlzc3VlZF9hdCI6MTU0OTM0ODM1OCwidXNlcl9pZCI6IjEwMDAwMDM1ODk2OTMzOSJ9',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache'
    }

    req = urllib.request.Request(
        'https://www.dek-d.com/homepage/ajax/writer_query?{params}'.format(params=urllib.parse.urlencode(params)),
        headers=headers
    )

    res = urllib.request.urlopen(req)

    res_bytes = res.read()

    res_str = res_bytes.decode('utf-8', 'backslashreplace')

    res_json = json.loads(res_str[9:])

    return res_json['o']['list']


if __name__ == '__main__':
    print(get_novels(0))


def get_chapter(i):
    headers = {
        'Host': 'www.dek-d.com',
        'User-Agent': ' Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:65.0) Gecko/20100101 Firefox/65.0',
        'Accept': '*/*',
        'Accept-Language': 'en,en-US;q=0.7,th;q=0.3',
        # 'Accept-Encoding': 'gzip, deflate, br', # Not usable - not supported by Python URLLib
        'Referer': 'https://www.dek-d.com/home/writer/',
        # 'X-CSRF-TOKEN': '66dc07b97aac8200e7e485037d126ab0',
        'X-Requested-With': 'XMLHttpRequest',
        'Connection': 'keep-alive',
        # 'Cookie': 'writer-hp-store-abtest=1; G_ENABLED_IDPS=google; fbm_193207127471363=base_domain=.dek-d.com; __gddads=100000358969339:undefined:0; _serc=dh3zsl20-4kcw-1dg2; PHPSESSID=bcn2l9rdjs8h4c54r3243vv41g; xcsrf-dd_login=95cc7cbdba8210d1e8afa0f563b4e533; fbsr_193207127471363=tRs2q7Cc1ImB6FmhoVK0b7hF67Zvupu9RLcyRdD_uDg.eyJhbGdvcml0aG0iOiJITUFDLVNIQTI1NiIsImNvZGUiOiJBUURrYWhsNTJnVWlNS040QzRNdnBYY2xzNWxEelFrY19ueEZMTWVfdnR5TjR6bWNBQWJxUmtvZDM0d0xxUzFWbFZNY29FelFQcHpVS1FsbjhwcXREZEJfVTFOSF90b1JQLWpWRjl1UWU3cks0Ti1Fc2o5WC1INEV6clBzQnEtQy1xUHZsQVJlVW9zUmU1NDlqTVdmSEVIdEdqSV9QdU1jU2pDRER1eHk3bzVvRERhdmpRc1VhTlRfNUFTcURnMXhNVUxmYVRDQlU3cGd5QTlNT3BQZlB3QXVrWUR0bl9RUjVFcjlHdlFjV3ItUUJmNGo4WlFSaXhZYjlMM2lSSG9DcWJRYlhZU3dlUGs1bDRWcVBMaWRsVUFXLTQyNzdjZFZGeE44YldUUzVJVWNrREpqcUJRSF9oS2hJYzlXbTRjZzdpak90UE96Umgya3h2eTU2cENnbFd3VyIsImlzc3VlZF9hdCI6MTU0OTM0ODM1OCwidXNlcl9pZCI6IjEwMDAwMDM1ODk2OTMzOSJ9',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache'
    }

    url_is = 'https://writer.dek-d.com/punn3kay/story/view.php?id=' + str(i)

    time_now = datetime.datetime.now().strftime('%Y%m%d-%H:%M')
    print(time_now)
    print(url_is)
    page = urllib.request.urlopen(
        urllib.request.Request(url_is, headers=headers)).read().decode("utf8", errors='ignore')
    #     print(page)
    soup = BeautifulSoup(page, features='lxml')

    # chapter_list = soup.find_all(class_='chapter-list')

    # chapter_list_items = chapter_list[0].find_all(class_='chapter-item')

    readable_chapter_list_items = [chapter_list_item for chapter_list_item in soup.find_all(class_='chapter-item')
                                   if (not 'chapter-state-hidden' in chapter_list_item.attrs['class']) and
                                   (not 'chapter-sell' in chapter_list_item.attrs['class'])]
    #     print(readable_chapter_list_items)

    # readable_chapter_links = [li.find('a', class_='txt-link').attrs['href'] for li in filter(lambda li: li.find('a', class_='txt-link') is not None, readable_chapter_list_items)]
    readable_chapter_links = list(
        map(
            lambda a: a.attrs['href'],
            filter(
                lambda a: a is not None,
                [li.find('a', class_='txt-link') for li in readable_chapter_list_items])))

    return readable_chapter_links


if __name__ == '__main__':
    print(get_chapter(5))


def get_text(reg_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}
    req = Request(url=reg_url, headers=headers)
    # open url
    html_doc = urlopen(req).read().decode("utf8", errors='ignore')
    # cut
    soup = BeautifulSoup(html_doc, 'html.parser')

    for element in soup(text=lambda text: isinstance(text, Comment)):
        element.extract()

    #     readable_chapter_list_items = [chapter_list_item for chapter_list_item in soup.find_all(class_='chapter-item')
    #                                 if (not 'chapter-state-hidden' in chapter_list_item.attrs['class']) and
    #                                 (not 'chapter-sell' in chapter_list_item.attrs['class'])]

    div = soup.find("div", id="story-content")
    # print(div)
    novel = ''
    if div is not None and div != []:

        sc = div.find_all("script")
        for element in sc:
            element.extract()
        e1 = div.find("span", class_="er1")
        for element in e1:
            element.extract()
        e2 = div.find("span", class_="er2")
        for element in e2:
            element.extract()
        e3 = div.find("span", class_="er3")
        for element in e3:
            element.extract()
        e4 = div.find("span", class_="er4")
        for element in e4:
            element.extract()

        style = div.find_all("style")
        for element in style:
            element.extract()

        novel = div.get_text()

        # div = [d for d in div.find_all("div", id=None)]
        # for i in div:
        #     # print('+',i)
        #     novel += i.get_text()

    return novel


def into_folder(folder):
    folder = str(folder)
    if folder in os.listdir():
        os.chdir(folder)
    else:
        os.mkdir(folder)
        os.chdir(folder)


