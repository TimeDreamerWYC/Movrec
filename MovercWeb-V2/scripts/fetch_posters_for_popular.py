#!/usr/bin/env python3
"""根据推荐引擎的筛选结果抓取对应的豆瓣海报。

用法示例：
    python scripts/fetch_posters_for_popular.py --count 200 --sleep 1.0

脚本会调用 recommend_engine.engine.get_popular_movies 从数据中选取电影（按评分/投票等条件），
然后对返回的 MOVIE_ID 列表逐条调用豆瓣页面抓取海报并下载到 static/posters。
"""
import os
import time
import argparse
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

import pandas as pd

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36'}


def fetch_poster_url(douban_id):
    url = f'https://movie.douban.com/subject/{douban_id}/'
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            print(f"WARN: {douban_id} 页面返回 {r.status_code}")
            return None
        soup = BeautifulSoup(r.text, 'html.parser')
        og = soup.find('meta', property='og:image')
        if og and og.get('content'):
            return og['content']
        img = soup.find('img', {'rel': 'v:image'}) or soup.find('img', class_='poster')
        if img and img.get('src'):
            return img['src']
    except Exception as e:
        print(f"ERROR fetching {douban_id}: {e}")
    return None


def download_image(url, dest_base):
    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=15)
        if r.status_code == 200:
            parsed = urlparse(url)
            fname = os.path.basename(parsed.path)
            ext = os.path.splitext(fname)[1] if '.' in fname else '.jpg'
            dest = dest_base + ext
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            return dest
        else:
            print(f"WARN: 下载图片返回 {r.status_code} for {url}")
    except Exception as e:
        print(f"ERROR downloading {url}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=200, help='要抓取的影片数量')
    parser.add_argument('--min_score', type=float, default=6.5, help='最小豆瓣评分，默认 6.5')
    parser.add_argument('--min_votes', type=int, default=3000, help='最小投票数，默认 3000')
    parser.add_argument('--sleep', type=float, default=1.0)
    parser.add_argument('--out', default='static/posters')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 直接从 CSV 中筛选，避免导入 recommend_engine 过程中可能的依赖问题
    movies_csv = os.path.join(os.getcwd(), 'data', 'movies.csv')
    if not os.path.exists(movies_csv):
        print(f'movies.csv 不存在: {movies_csv}')
        return

    try:
        movies = pd.read_csv(movies_csv, dtype=str)
    except Exception as e:
        print('读取 movies.csv 失败:', e)
        return

    # 转换为数值以便比较
    if 'DOUBAN_SCORE' in movies.columns:
        movies['DOUBAN_SCORE'] = pd.to_numeric(movies['DOUBAN_SCORE'], errors='coerce')
    else:
        movies['DOUBAN_SCORE'] = pd.NA
    if 'DOUBAN_VOTES' in movies.columns:
        movies['DOUBAN_VOTES'] = pd.to_numeric(movies['DOUBAN_VOTES'], errors='coerce')
    else:
        movies['DOUBAN_VOTES'] = pd.NA

    popular = movies[(movies['DOUBAN_SCORE'].ge(args.min_score)) & (movies['DOUBAN_VOTES'].ge(args.min_votes))]
    if popular.empty:
        print(f'没有找到符合条件的电影 (评分>={args.min_score}, 评分人数>={args.min_votes})')
        return

    # 按评分和投票数降序排列，优先高质量高人气电影，取前 N 条
    popular_sorted = popular.sort_values(by=['DOUBAN_SCORE', 'DOUBAN_VOTES'], ascending=[False, False])
    df = popular_sorted.head(args.count).reset_index(drop=True)
    ids = [str(row.get('MOVIE_ID') or row.get('subject_id') or row.get('douban_id')) for _, row in df.iterrows() if (row.get('MOVIE_ID') or row.get('subject_id') or row.get('douban_id'))]

    print(f'将要抓取海报的 ID 数量：{len(ids)}')

    for i, mid in enumerate(ids, start=1):
        base = os.path.join(args.out, mid)
        # skip if exists
        existing = [f for f in os.listdir(args.out) if f.startswith(mid + '.') or f == mid]
        if existing:
            print(f'[{i}/{len(ids)}] {mid} 已存在，跳过')
            continue
        print(f'[{i}/{len(ids)}] 抓取 {mid} ...')
        url = fetch_poster_url(mid)
        if url:
            saved = download_image(url, base)
            if saved:
                print(f'已保存: {saved}')
            else:
                print(f'保存失败: {mid}')
        else:
            print(f'未找到海报: {mid}')
        time.sleep(args.sleep)


if __name__ == '__main__':
    main()
