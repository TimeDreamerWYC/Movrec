#!/usr/bin/env python3
"""脚本：根据 movies.csv 中的 MOVIE_ID（豆瓣 id）抓取豆瓣页面并下载海报到 static/posters/ 目录。

用法:
    python scripts/fetch_douban_posters.py --data data/movies.csv --out static/posters --start 0 --limit 100

注意：请遵守豆瓣的使用条款和 robots.txt，适当控制速率（默认 sleep=1s）。
"""
import requests
from bs4 import BeautifulSoup
import csv
import os
import time
import argparse
from urllib.parse import urljoin, urlparse


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36'
}


def fetch_poster_url(douban_id):
    url = f'https://movie.douban.com/subject/{douban_id}/'
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            print(f"WARN: {douban_id} 页面返回 {r.status_code}")
            return None
        html = r.text
        soup = BeautifulSoup(html, 'html.parser')
        # 优先 og:image
        og = soup.find('meta', property='og:image')
        if og and og.get('content'):
            return og['content']
        # 尝试查找 poster img
        img = soup.find('img', {'rel': 'v:image'}) or soup.find('img', class_='poster')
        if img and img.get('src'):
            return img['src']
        # 兜底：查找第一张大图
        imgs = soup.select('img')
        for im in imgs:
            src = im.get('src') or im.get('data-src')
            if src and ('doubanio.com' in src or 'doubani' in src):
                return src
    except Exception as e:
        print(f"ERROR fetching {douban_id}: {e}")
    return None


def download_image(url, dest_path):
    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=15)
        if r.status_code == 200:
            # try to determine extension
            parsed = urlparse(url)
            fname = os.path.basename(parsed.path)
            if '.' in fname:
                ext = os.path.splitext(fname)[1]
            else:
                ext = '.jpg'
            if not dest_path.lower().endswith(ext.lower()):
                dest_path = dest_path + ext
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            return dest_path
        else:
            print(f"WARN: 下载图片返回 {r.status_code} for {url}")
    except Exception as e:
        print(f"ERROR downloading {url}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/movies.csv')
    parser.add_argument('--out', default='static/posters')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--sleep', type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.data, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    total = len(rows)
    print(f"Total rows in CSV: {total}")
    processed = 0
    for i, row in enumerate(rows[args.start:], start=args.start):
        if args.limit and processed >= args.limit:
            break
        mid = row.get('MOVIE_ID') or row.get('subject_id') or row.get('douban_id')
        if not mid or mid.strip() == '':
            continue
        mid = mid.strip()
        out_base = os.path.join(args.out, mid)
        # 如果已存在任意文件以 mid 开头，则跳过
        exists = False
        for f in os.listdir(args.out):
            if f.startswith(mid + '.') or f.startswith(mid + '_') or f == mid:
                exists = True
                break
        if exists:
            print(f"Skip {mid} (already exists)")
            processed += 1
            continue

        print(f"[{i}/{total}] Fetching poster for {mid} ...")
        poster_url = fetch_poster_url(mid)
        if poster_url:
            saved = download_image(poster_url, out_base)
            if saved:
                print(f"Saved poster: {saved}")
            else:
                print(f"Failed to save poster for {mid}")
        else:
            print(f"No poster found for {mid}")

        processed += 1
        time.sleep(args.sleep)


if __name__ == '__main__':
    main()
