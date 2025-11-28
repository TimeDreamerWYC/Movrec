#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导入脚本：将 data/user_wish.csv 和 data/user_dislike.csv 导入到指定用户的记录
"""
import csv
import os
from app import app
from models import db, User, UserMoviePreference, UserMovieDislike

def import_csv_to_user(username, csv_path, model_class, record_type):
    """
    通用导入函数：将 CSV 的 douban_id 导入到用户的记录
    
    Args:
        username (str): 目标用户名
        csv_path (str): CSV 文件路径
        model_class: ORM 模型类（UserMoviePreference 或 UserMovieDislike）
        record_type (str): 记录类型标签（"喜欢" 或 "不喜欢"）
    """
    with app.app_context():
        # 1. 查找用户
        user = User.query.filter_by(username=username).first()
        if not user:
            print(f"❌ 用户 '{username}' 不存在！")
            return 0
        
        # 2. 检查 CSV 文件
        if not os.path.exists(csv_path):
            print(f"❌ CSV 文件不存在: {csv_path}")
            return 0
        
        print(f"📥 正在导入 [{record_type}]: {os.path.basename(csv_path)}")
        
        imported_count = 0
        skipped_count = 0
        duplicate_count = 0
        
        # 3. 读取 CSV（处理 BOM 字符）
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 处理 BOM 字符：尝试 'douban_id' 或 '\ufeffdouban_id'
                douban_id = row.get('douban_id', row.get('\ufeffdouban_id', '')).strip()
                
                # 跳过空行
                if not douban_id:
                    skipped_count += 1
                    continue
                
                # 检查是否已存在
                existing = model_class.query.filter_by(
                    user_id=user.id,
                    movie_douban_id=douban_id
                ).first()
                
                if existing:
                    duplicate_count += 1
                    continue
                
                # 插入新记录
                try:
                    record = model_class(
                        user_id=user.id,
                        movie_douban_id=douban_id
                    )
                    db.session.add(record)
                    imported_count += 1
                except Exception as e:
                    print(f"  ⚠️ 插入失败 (douban_id={douban_id}): {e}")
                    skipped_count += 1
        
        # 4. 提交
        try:
            db.session.commit()
            print(f"  ✓ 新增: {imported_count} 条")
            if duplicate_count > 0:
                print(f"  ⊘ 重复: {duplicate_count} 条（已跳过）")
            if skipped_count > 0:
                print(f"  ✗ 跳过: {skipped_count} 条（空或错误）")
            return imported_count
        except Exception as e:
            db.session.rollback()
            print(f"❌ 提交失败: {e}")
            return 0

if __name__ == '__main__':
    import sys
    
    # 从命令行参数读取用户名，默认为 'zjm'
    username = sys.argv[1] if len(sys.argv) > 1 else 'zjm'
    
    with app.app_context():
        user = User.query.filter_by(username=username).first()
        if not user:
            print(f"❌ 用户 '{username}' 不存在！")
            sys.exit(1)
        
        print(f"✅ 找到用户: {username} (id={user.id})\n")
    
    # 导入喜欢和不喜欢
    print("=" * 50)
    wish_count = import_csv_to_user(
        username,
        'data/user_wish.csv',
        UserMoviePreference,
        "喜欢"
    )
    print()
    
    dislike_count = import_csv_to_user(
        username,
        'data/user_dislike.csv',
        UserMovieDislike,
        "不喜欢"
    )
    print("=" * 50)
    
    # 显示最终统计
    with app.app_context():
        user = User.query.filter_by(username=username).first()
        total_liked = UserMoviePreference.query.filter_by(user_id=user.id).count()
        total_disliked = UserMovieDislike.query.filter_by(user_id=user.id).count()
        
        print(f"\n✅ 导入完成！")
        print(f"  📊 用户 '{username}' 的偏好统计:")
        print(f"     • 喜欢: {total_liked} 部电影")
        print(f"     • 不喜欢: {total_disliked} 部电影")
