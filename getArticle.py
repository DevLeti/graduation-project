import dotenv
import os
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

# Get environment variables - Naver API
dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)


def searchArticle(keyword):
    '''
    :param keyword: search keyword with keyword generation
    :return: type:'json', Naver article API response
    '''
    query = keyword  # keyword
    display = 40  # number of result
    sort = "sim"  # sort method, sim: accurate, date: date
    URL = f"https://openapi.naver.com/v1/search/news.json?query={query}&display={display}&sort={sort}"

    res = requests.get(URL, headers={"X-Naver-Client-Id": os.environ["NAVER_API_ID"],
                                     "X-Naver-Client-Secret": os.environ["NAVER_API_SECRET"]})
    res_json = res.json()

    return res_json


def getOriginalLinks(res_json):
    '''
    :param res_json: type:'json', Naver article API response
    :return: type:list<'str'>, 네이버 포털에 나오는 기사의 원 출처 링크. 네이버 기사 포털에 나오는 매일 경제 기사면 매일 경제 사이트 링크를 알려줌.
    '''
    links = []
    for i in res_json["items"]:
        links.append(i["originallink"])
    return links


def getLinks(res_json):
    '''
    :param res_json: type:'json', Naver article API response
    :return: type:list<'str'>, article URLs
    '''
    links = []
    for i in res_json["items"]:
        links.append(i["link"])
    return links


def getOnlyNaverLinks(res_json):
    '''
    :param res_json: type:'json', Naver article API response
    :return: type:list<'str'>, naver article URLs
    '''
    links = []
    for i in res_json["items"]:
        if ("news.naver.com" in i["link"]):
            links.append(i["link"])
    return links


def getArticleDetail(URL):
    '''
    :param URL: Target article URL
    :return: type:'str', Article 본문
    '''
    # sid, 100:정치, 101:경제, 102: 사회, 103:생활/문화, 104:세계, 105:IT/과학, 106: 예능
    # '106: 예능'의 경우 연예 페이지로 파싱을 다르게 해야함
    # sports은 sid가 없으므로 다른 방식으로 크롤링 해야함
    # URL = "https://n.news.naver.com/mnews/article/003/0012106374?sid=106"
    try:
        URLparts = urlparse(URL)
        res = requests.get(URL)
        soup = BeautifulSoup(res.text, features="html.parser")
        if 'sports' in URLparts.netloc:
            print(f"Article type is sports.")
            detail = soup.find(id="newsEndContents").text
            detail = detail.replace("\n\n", "\n")
            print(f"Article: {detail}")
        else:
            query_list = parse_qs(URLparts.query)
            article_type = query_list["sid"][0]  # type: 'str'
            print(f"Article type is {article_type}")
            if (article_type == "106"):  # 연예
                detail = soup.find(id="articeBody").text
                detail = detail.replace("\n\n", "\n")
                print(f"Article: {detail}")
            else:
                detail = soup.find(id="dic_area").text
                detail = detail.replace("\n\n", "\n")
                print(f"Article: {detail}")

        return detail
    except:
        print(URLparts)
        print(f'error found.')
        if(query_list is not None):
            print(f'query_list: {query_list}')
        return ""


def getArticleDetailBulk(URLs):
    '''
    :param URLs: Target article URLs, so the param's type should be list.
    :return: type: list<'str'>, Articles
    '''
    article_details = []
    for URL in URLs:
        article_details.append(getArticleDetail(URL))
        # print("wait 1 second...")
        # time.sleep(1)
    return article_details

def getArticleDetailBulkWithStr(user_keyword):
    '''
    :param user_keyword: search article keyword by user
    :return: type: 'str', Articles
    '''
    search_result = searchArticle(user_keyword)
    article_links = getOnlyNaverLinks(search_result)
    article_list = getArticleDetailBulk(article_links)
    article_string = ""
    for i in range(0, len(article_list)):
        article_string += f"\n{i + 1}번째 기사:"
        article_string += article_list[i].strip('\n')
        article_string += "\n"
    return article_string

if __name__ == "__main__":
    keyword = input()
    # search_result = searchArticle(keyword)
    # links = getOnlyNaverLinks(search_result)
    # # print(links)
    #
    # articles = getArticleDetailBulk(links)
    # print(articles)
    print(getArticleDetailBulkWithStr(keyword))