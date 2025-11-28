import sys
import os
import requests
import time
import random
from bs4 import BeautifulSoup
from typing import List, Dict
from sqlalchemy import text

# 明确导出接口，修复 Flask 导入问题
__all__ = ['crawl_douban_movies', 'validate_cookie']

def validate_cookie(cookie: str) -> bool:
    """验证Cookie是否有效"""
    test_url = 'https://movie.douban.com/mine'
    headers = {
        'Cookie': cookie,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        resp = requests.get(test_url, headers=headers, timeout=10, allow_redirects=False)
        # 如果重定向到登录页，说明Cookie无效
        if resp.status_code == 302 and 'accounts.douban.com' in resp.headers.get('Location', ''):
            return False
        return True
    except:
        return False

def crawl_douban_movies(douban_id: str, cookie: str) -> Dict[str, List[Dict]]:
    """
    爬取豆瓣用户主页的"看过"和"想看"电影
    返回 {'watched': [...], 'wish': [...]}，每项为 dict: {'douban_id', 'title', 'poster_url'}
    """
    headers = {
        'Cookie': cookie,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Referer': 'https://movie.douban.com/',
        'Upgrade-Insecure-Requests': '1',
    }
    base_url = f'https://movie.douban.com/people/{douban_id}/'
    result = {'watched': [], 'wish': []}
    def fetch_list(list_type):
        movies = []
        page = 1
        max_pages = 10
        while page <= max_pages:
            time.sleep(random.uniform(1, 3))
            url = f'{base_url}{list_type}?start={(page-1)*15}&sort=time&rating=all&filter=all&mode=list'
            print(f'[豆瓣爬虫] 正在获取 {list_type} 第 {page} 页: {url}')
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                print(f'[豆瓣爬虫] 请求状态码: {resp.status_code}')
                if 'accounts.douban.com' in resp.url:
                    print('[豆瓣爬虫] 被重定向到登录页，可能Cookie失效或触发了反爬')
                    break
                if resp.status_code == 403:
                    print('[豆瓣爬虫] 403 Forbidden，可能IP被限制')
                    break
                if resp.status_code != 200:
                    print(f'[豆瓣爬虫] 状态码异常: {resp.status_code}')
                    break
                if page == 1:
                    try:
                        with open(f'douban_sync_debug_{list_type}.html', 'w', encoding='utf-8') as f:
                            f.write(resp.text)
                        print(f'[豆瓣爬虫] 已保存第一页HTML到 douban_sync_debug_{list_type}.html')
                    except Exception as e:
                        print(f'[豆瓣爬虫] 保存HTML失败: {e}')
                soup = BeautifulSoup(resp.text, 'html.parser')
                login_prompt = soup.find('div', class_='login')
                if login_prompt:
                    print('[豆瓣爬虫] 页面包含登录提示，需要登录')
                    break
                items = soup.select('.item') or soup.select('.grid-view .item')
                if not items:
                    print(f'[豆瓣爬虫] 第{page}页未找到电影条目')
                    no_content = soup.find('div', class_='note') or soup.find('p', class_='pl')
                    if no_content and ('没有' in no_content.text or 'No' in no_content.text):
                        print(f'[豆瓣爬虫] 用户没有{list_type}列表')
                        break
                    try:
                        with open(f'douban_error_{list_type}_page{page}.html', 'w', encoding='utf-8') as f:
                            f.write(resp.text)
                        print(f'[豆瓣爬虫] 已保存错误页面到 douban_error_{list_type}_page{page}.html')
                    except:
                        pass
                    break
                print(f'[豆瓣爬虫] 第{page}页找到 {len(items)} 个条目')
                for item in items:
                    douban_id = None
                    link = item.select_one('a[href*="/subject/"]')
                    if link and link.get('href'):
                        href = link['href']
                        try:
                            parts = href.split('/')
                            for i, part in enumerate(parts):
                                if part == 'subject' and i + 1 < len(parts):
                                    douban_id = parts[i + 1]
                                    break
                        except Exception as e:
                            print(f'[豆瓣爬虫] 提取豆瓣ID失败: {e}')
                    title = None
                    title_elem = (item.select_one('.info .title') or 
                                 item.select_one('.title') or
                                 item.select_one('li.title a') or
                                 item.select_one('em'))
                    if title_elem:
                        title = title_elem.get_text().strip()
                    poster_url = None
                    img_elem = item.select_one('img')
                    if img_elem and img_elem.get('src'):
                        poster_url = img_elem['src']
                    if douban_id and title:
                        movies.append({
                            'douban_id': douban_id,
                            'title': title,
                            'poster_url': poster_url
                        })
                        print(f'[豆瓣爬虫] 找到电影: {title} (ID: {douban_id})')
                    else:
                        print(f'[豆瓣爬虫] 条目信息不完整: ID={douban_id}, title={title}')
                next_link = soup.select_one('.next a')
                if not next_link:
                    print(f'[豆瓣爬虫] 没有下一页，停止爬取')
                    break
                page += 1
            except requests.exceptions.Timeout:
                print(f'[豆瓣爬虫] 请求超时，第{page}页')
                break
            except requests.exceptions.RequestException as e:
                print(f'[豆瓣爬虫] 网络请求异常: {e}')
                break
            except Exception as e:
                print(f'[豆瓣爬虫] 解析异常: {e}')
                break
        print(f'[豆瓣爬虫] {list_type}列表获取完成，共{len(movies)}部电影')
        return movies
    print(f'[豆瓣爬虫] 开始获取用户 {douban_id} 的豆瓣电影列表')
    result['watched'] = fetch_list('collect')
    result['wish'] = fetch_list('wish')
    print(f'[豆瓣爬虫] 获取完成: 看过{len(result["watched"])}部，想看{len(result["wish"])}部')
    return result
# 明确导出接口，修复 Flask 导入问题
__all__ = ['crawl_douban_movies', 'validate_cookie']
# douban_sync.py
"""
豆瓣个人主页同步工具
- 封装 crawl_douban_me2.py 的核心逻辑为 crawl_douban_movies(douban_id, cookie)
- 返回 {'watched': [...], 'wish': [...]}，每项为 dict: {'douban_id', 'title', 'poster_url'}
"""
import sys
import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

# 可根据 crawl_douban_me2.py 进一步完善

def crawl_douban_movies(douban_id: str, cookie: str) -> Dict[str, List[Dict]]:
    """
    爬取豆瓣用户主页的“看过”和“想看”电影
    返回 {'watched': [...], 'wish': [...]}，每项为 dict: {'douban_id', 'title', 'poster_url'}
    """
    headers = {
        'Cookie': cookie,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    }
    base_url = f'https://movie.douban.com/people/{douban_id}/'
    result = {'watched': [], 'wish': []}
    
    def fetch_list(list_type):
        # list_type: 'collect'（看过）或 'wish'（想看）
        movies = []
        page = 1
        first_html_saved = False
        while True:
            url = f'{base_url}{list_type}?start={(page-1)*15}&sort=time&rating=all&filter=all&mode=list'
            try:
                resp = requests.get(url, headers=headers, timeout=10)
            except Exception as e:
                print(f'[豆瓣爬虫] 请求失败: {e}')
                break
            print(f'[豆瓣爬虫] 请求 {url} 状态码: {resp.status_code}')
            if page == 1:
                # 保存第一页 HTML
                try:
                    with open(f'douban_sync_debug_{list_type}.html', 'w', encoding='utf-8') as f:
                        f.write(resp.text)
                    print(f'[豆瓣爬虫] 已保存第一页 HTML 到 douban_sync_debug_{list_type}.html')
                except Exception as e:
                    print(f'[豆瓣爬虫] 保存 HTML 失败: {e}')
                first_html_saved = True
            if resp.status_code != 200:
                print(f'[豆瓣爬虫] 状态码异常: {resp.status_code}')
                break
            soup = BeautifulSoup(resp.text, 'html.parser')
            items = soup.select('.item')
            if not items:
                print(f'[豆瓣爬虫] 页面无电影条目，页面片段:')
                print(resp.text[:500])
                break
            for item in items:
                # 豆瓣ID
                link = item.select_one('div.pic a')
                douban_id = None
                if link and link.get('href'):
                    # 链接如 https://movie.douban.com/subject/1292052/
                    try:
                        douban_id = link['href'].split('/')[-2]
                    except Exception:
                        douban_id = None
                # 标题
                title = item.select_one('div.info span.title')
                title = title.text.strip() if title else None
                # 海报
                img = item.select_one('div.pic img')
                poster_url = img['src'] if img and img.get('src') else None
                if douban_id:
                    movies.append({
                        'douban_id': douban_id,
                        'title': title,
                        'poster_url': poster_url
                    })
            # 翻页判断
            next_btn = soup.select_one('span.next a')
            if next_btn:
                page += 1
            else:
                break
        return movies
    
    result['watched'] = fetch_list('collect')
    result['wish'] = fetch_list('wish')
    return result

# 测试入口
if __name__ == '__main__':
    douban_id = input('豆瓣ID: ').strip()
    cookie = input('Cookie: ').strip()
    data = crawl_douban_movies(douban_id, cookie)
    print(f"看过: {len(data['watched'])}部，想看: {len(data['wish'])}部")
    print('示例:', data['watched'][:2], data['wish'][:2])
