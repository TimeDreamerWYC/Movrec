# app.py
try:
    # 尝试从旧版本 werkzeug 导入 (向后兼容)
    from werkzeug.urls import url_parse
except ImportError:
    # 如果失败，则从 urllib.parse 导入 (适用于 Werkzeug >= 3.0)
    from urllib.parse import urlparse as url_parse # 保持别名 url_parse
from forms import LoginForm, RegistrationForm
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
try:
    # 尝试从旧版本 werkzeug 导入 (向后兼容)
    from werkzeug.urls import url_parse
except ImportError:
    # 如果失败，则从 urllib.parse 导入 (适用于 Werkzeug >= 3.0)
    from urllib.parse import urlparse as url_parse
from config import Config
from models import db, User, UserMoviePreference, UserMovieDislike
from recommend_engine.engine import initialize_engine, recommand, get_movies_dataframe
import os
import pandas as pd

app = Flask(__name__)
app.config.from_object(Config)

# 初始化数据库
db.init_app(app)

# 初始化登录管理器
login = LoginManager(app)
login.login_view = 'login'

@login.user_loader
def load_user(id):
    return User.query.get(int(id))

import os # 确保文件顶部已导入 os

# --- 新增：全局标志位，用于确保引擎和数据库表只初始化一次 ---
_engine_initialized = False

# --- 新增/修改：初始化函数，包含数据库和推荐引擎 ---
def initialize_app_once():
    global _engine_initialized
    if not _engine_initialized:
        # 1. 初始化数据库表
        print("🗄️  初始化数据库表...")
        db.create_all() # 在 app_context 内部调用是安全的
        print("✅ 数据库表初始化完成!")

        # 2. 初始化推荐引擎
        print("🔧 初始化推荐引擎...")
        # 注意：这里根据你的 initialize_engine 函数签名调整参数
        # 假设你的 config.py 中定义了 DATA_FOLDER
        data_folder = app.config.get('DATA_FOLDER', os.path.join(app.root_path, 'data'))
        model_cache = os.path.join(app.root_path, 'model_cache.pkl') # 缓存文件路径

        try:
            # 调用你修改后的 initialize_engine 函数
            initialize_engine(data_folder_path=data_folder, model_cache_path=model_cache)
            _engine_initialized = True
            print("✅ 推荐引擎初始化完成!")
        except Exception as e:
            print(f"❌ 推荐引擎初始化失败: {e}")
            # 根据你的需求决定是否要阻止应用启动
            # raise e # 如果初始化失败是致命的，取消注释此行

# --- 修改：使用 before_request 替代 before_first_request ---
@app.before_request
def ensure_app_initialized():
    """确保在第一次请求时初始化应用（数据库和引擎）"""
    initialize_app_once()

# --- 移除或注释掉旧的初始化代码 ---
# with app.app_context():
#     print("🔧 初始化推荐引擎...")
#     initialize_engine(app.config['DATA_FOLDER'])
#     print("✅ 推荐引擎初始化完成!")

# --- 移除或注释掉旧的装饰器 ---
# @app.before_first_request
# def create_tables():
#     db.create_all()
@app.route('/')
@app.route('/index')
def index():
    # --- 新增调试信息 ---
    print("\n--- DEBUG INDEX ROUTE ---")
    # 1. 获取电影数据 DataFrame
    movies_df = get_movies_dataframe()
    print(f"get_movies_dataframe() returned type: {type(movies_df)}")
    if movies_df is not None:
        print(f"get_movies_dataframe() returned shape: {movies_df.shape}")
        print(f"First few rows of DataFrame:\n{movies_df.head()}")
        print(f"Column names: {list(movies_df.columns)}")
        # 检查 MOVIE_ID 列
        if 'MOVIE_ID' in movies_df.columns:
             print(f"Sample MOVIE_ID values: {movies_df['MOVIE_ID'].head().tolist()}")
             print(f"Type of first MOVIE_ID value: {type(movies_df['MOVIE_ID'].iloc[0]) if not movies_df.empty else 'N/A'}")
        else:
             print("Warning: 'MOVIE_ID' column not found in DataFrame!")
    else:
        print("get_movies_dataframe() returned None!")
    print("--- DEBUG INDEX ROUTE END ---\n")
    # --- 新增调试信息结束 ---

    # 2. 处理数据
    movies_list = []
    if movies_df is not None and not movies_df.empty:
        movies_list = movies_df.to_dict('records')
    else:
        flash("暂时无法加载电影列表。") # 可选

    # 3. 传递给模板
    return render_template('index.html', movies=movies_list)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = RegistrationForm()
    # --- 新增调试代码：打印验证状态和错误 ---
    print("--- DEBUG REGISTER FORM ---")
    print(f"Form is submitted: {form.is_submitted()}")
    print(f"Form is valid: {form.validate()}") # 这会触发验证
    if form.errors:
        print("Form errors:", form.errors)
    print("--- DEBUG REGISTER FORM END ---")
    # --- 新增调试代码结束 ---
    
    if form.validate_on_submit(): # 这里面包含了 is_submitted() 和 validate()
        username = form.username.data
        email = form.email.data
        password = form.password.data

        print(f"--- DEBUG REGISTER START ---")
        print(f"Attempting to register user: {username}, email: {email}")

        user = User(username=username, email=email)
        user.set_password(password)
        print(f"Password hash generated: {user.password_hash}")

        db.session.add(user)
        try:
            db.session.commit()
            print(f"User {username} committed to database successfully.")
            inserted_user = User.query.filter_by(username=username).first()
            print(f"Re-queried user from DB: {inserted_user}, Hash: {inserted_user.password_hash if inserted_user else 'N/A'}")
            print(f"--- DEBUG REGISTER END ---")
            
            flash('恭喜你，注册成功！')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            print(f"--- DEBUG REGISTER ERROR ---")
            print(f"Error committing user to database: {e}")
            print(f"--- DEBUG REGISTER ERROR ---")
            flash('注册失败，请重试。')
    
    # 如果验证失败或 GET 请求，渲染表单
    return render_template('register.html', title='Register', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        remember_me = form.remember_me.data

        # --- 调试信息 1: 登录尝试 ---
        print(f"--- DEBUG LOGIN ATTEMPT ---")
        print(f"Login attempt for username: '{username}'")

        user = User.query.filter_by(username=username).first()
        
        # --- 调试信息 2: 查询结果 ---
        print(f"User found in DB: {user}")
        if user:
            print(f"Stored password hash: {user.password_hash}")
            password_check_result = user.check_password(password)
            print(f"Password check result: {password_check_result}")
        else:
            print("No user found with that username.")
        print(f"--- DEBUG LOGIN ATTEMPT END ---")

        if user is None or not user.check_password(password):
            flash('无效的用户名或密码')
            return redirect(url_for('login'))
        login_user(user, remember=remember_me)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    # 获取当前用户的喜好和厌恶列表
    liked_movies_ids = [pref.movie_douban_id for pref in current_user.liked_movies.all()]
    disliked_movies_ids = [dis.movie_douban_id for dis in current_user.disliked_movies.all()]

    # 从全局 movies_new DataFrame 中查找详细信息
    movies_df = get_movies_dataframe()
    liked_movies_info = movies_df[movies_df['MOVIE_ID'].isin(liked_movies_ids)].to_dict('records')
    disliked_movies_info = movies_df[movies_df['MOVIE_ID'].isin(disliked_movies_ids)].to_dict('records')

    return render_template('profile.html', title='Profile',
                           liked_movies=liked_movies_info,
                           disliked_movies=disliked_movies_info)


# 注意：由于你的电影数据主要来自 CSV，这个路由需要能访问到该数据。
# 假设 get_movies_dataframe() 返回包含所有电影信息的 DataFrame
@app.route('/movie/<string:movie_douban_id>') # 使用 douban_id 作为 URL 参数
def movie_detail(movie_douban_id):
    # 从全局 DataFrame 获取电影信息
    movies_df = get_movies_dataframe()
    if movies_df is None or movies_df.empty:
         flash('电影数据未加载。')
         return redirect(url_for('index'))

    # 筛选特定电影
    movie_row = movies_df[movies_df['MOVIE_ID'] == movie_douban_id]
    if movie_row.empty:
        flash('未找到指定的电影。')
        return redirect(url_for('index'))

    # 将 Series 转换为字典以便在模板中使用
    movie_info = movie_row.iloc[0].to_dict()

    # 检查当前用户偏好状态 (需要在 app context 内)
    user_has_liked = False
    user_has_disliked = False
    if current_user.is_authenticated:
        # 查询关联表
        liked_entry = UserMoviePreference.query.filter_by(user_id=current_user.id, movie_douban_id=movie_douban_id).first()
        disliked_entry = UserMovieDislike.query.filter_by(user_id=current_user.id, movie_douban_id=movie_douban_id).first()
        user_has_liked = liked_entry is not None
        user_has_disliked = disliked_entry is not None

    return render_template('movie_detail.html', movie=movie_info,
                           user_has_liked=user_has_liked,
                           user_has_disliked=user_has_disliked)

# --- 新增/修改：优化后的 toggle_preference API 路由 ---
# 使用 session 批量操作以提高效率并保证原子性
@app.route('/api/toggle_preference_optimized', methods=['POST'])
@login_required
def toggle_preference_optimized():
    """
    优化版本的偏好切换API，使用数据库事务确保一致性，
    并返回更新后的按钮状态给前端。
    """
    data = request.get_json()
    movie_douban_id = data.get('movie_douban_id')
    action = data.get('action') # 'like' or 'dislike'

    if not movie_douban_id or action not in ['like', 'dislike']:
        return jsonify({'error': 'Invalid data'}), 400

    try:
        with db.session.begin(): # 开始一个数据库事务
            # 先删除相反的操作
            if action == 'like':
                UserMovieDislike.query.filter_by(
                    user_id=current_user.id, movie_douban_id=movie_douban_id
                ).delete(synchronize_session=False) # synchronize_session=False 提高性能
                # 检查是否已存在喜欢记录
                existing_like = UserMoviePreference.query.filter_by(
                    user_id=current_user.id, movie_douban_id=movie_douban_id
                ).first()
                if not existing_like:
                    new_pref = UserMoviePreference(user_id=current_user.id, movie_douban_id=movie_douban_id)
                    db.session.add(new_pref)
                    new_status = 'liked'
                else:
                    # 如果已存在，则本次操作是取消喜欢
                    db.session.delete(existing_like)
                    new_status = 'none'
            else: # action == 'dislike'
                UserMoviePreference.query.filter_by(
                    user_id=current_user.id, movie_douban_id=movie_douban_id
                ).delete(synchronize_session=False)
                existing_dislike = UserMovieDislike.query.filter_by(
                    user_id=current_user.id, movie_douban_id=movie_douban_id
                ).first()
                if not existing_dislike:
                    new_dislike = UserMovieDislike(user_id=current_user.id, movie_douban_id=movie_douban_id)
                    db.session.add(new_dislike)
                    new_status = 'disliked'
                else:
                     # 如果已存在，则本次操作是取消不喜欢
                    db.session.delete(existing_dislike)
                    new_status = 'none'

        # 成功提交事务后，返回新状态
        return jsonify({'success': True, 'new_status': new_status})

    except Exception as e:
        db.session.rollback()
        print(f"[错误] 切换偏好失败: {e}") # 记录日志
        return jsonify({'error': '服务器内部错误'}), 500

@app.route('/api/toggle_preference', methods=['POST'])
@login_required
def toggle_preference():
    data = request.get_json()
    movie_douban_id = data.get('movie_douban_id')
    action = data.get('action') # 'like' or 'dislike'

    if not movie_douban_id or action not in ['like', 'dislike']:
        return jsonify({'error': 'Invalid data'}), 400

    # 查找或创建 Movie 实体（如果数据库中没有）
    # 注意：这里为了简化，我们直接操作关联表，不强制要求 Movie 表存在
    # 如果未来 Movie 表完善，这里需要先查询/创建 Movie

    # 先删除相反的操作
    if action == 'like':
        UserMovieDislike.query.filter_by(user_id=current_user.id, movie_douban_id=movie_douban_id).delete()
        # 检查是否已存在
        existing = UserMoviePreference.query.filter_by(user_id=current_user.id, movie_douban_id=movie_douban_id).first()
        if not existing:
            new_pref = UserMoviePreference(user_id=current_user.id, movie_douban_id=movie_douban_id)
            db.session.add(new_pref)
    else: # dislike
        UserMoviePreference.query.filter_by(user_id=current_user.id, movie_douban_id=movie_douban_id).delete()
        existing = UserMovieDislike.query.filter_by(user_id=current_user.id, movie_douban_id=movie_douban_id).first()
        if not existing:
            new_dislike = UserMovieDislike(user_id=current_user.id, movie_douban_id=movie_douban_id)
            db.session.add(new_dislike)

    db.session.commit()
    return jsonify({'success': True})

@app.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    recommendations = None
    query = ""
    if request.method == 'POST':
        query = request.form.get('movie_query', '').strip()
        if query:
            recommendations = recommand(query, sample_top=10, pick_n=5)
            if recommendations is None or recommendations.empty:
                 flash(f'未找到与 "{query}" 相关的电影')
                 recommendations = pd.DataFrame() # Empty DF for template
    # GET 请求时不执行推荐，显示空白表单
    return render_template('recommendations.html', title='Recommend', query=query, recommendations=recommendations)

if __name__ == '__main__':
    # 确保 instance 文件夹存在
    os.makedirs(os.path.join(app.root_path, 'instance'), exist_ok=True)
    app.run(debug=True) # 生产环境请设置 debug=False