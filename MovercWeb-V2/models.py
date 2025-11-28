# models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True, nullable=False)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    # 关联关系：一个用户有多部喜欢和不喜欢的电影
    liked_movies = db.relationship('UserMoviePreference', backref='user', lazy='dynamic',
                                   foreign_keys='UserMoviePreference.user_id',
                                   primaryjoin='User.id==UserMoviePreference.user_id')
    
    disliked_movies = db.relationship('UserMovieDislike', backref='user', lazy='dynamic',
                                      foreign_keys='UserMovieDislike.user_id',
                                      primaryjoin='User.id==UserMovieDislike.user_id')


    # 新增：豆瓣同步字段
    douban_id = db.Column(db.String(32), nullable=True)
    douban_cookie = db.Column(db.Text, nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return '<User {}>'.format(self.username)


# 注意：这里的 Movie 表主要用于关联用户偏好，实际电影信息仍从 CSV 加载
# 如果需要完全数据库化，可以将 movies_new 的内容也存入数据库
class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    douban_id = db.Column(db.String(20), unique=True, nullable=False) # 对应 movies_new['MOVIE_ID']
    title = db.Column(db.String(255), nullable=False) # 对应 movies_new['NAME']

    def __repr__(self):
        return '<Movie {}>'.format(self.title)

# 用户喜欢的电影关联表 (多对多简化为一对多，指向 Movie ID)
class UserMoviePreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_douban_id = db.Column(db.String(20), nullable=False) # 直接存储豆瓣ID，避免频繁查Movie表
    # 可选：添加时间戳等字段

# 用户不喜欢的电影关联表
class UserMovieDislike(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_douban_id = db.Column(db.String(20), nullable=False)


# 用户混合推荐权重配置表
class UserHybridWeights(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    dvae_weight = db.Column(db.Float, default=0.6)  # DVAE 权重
    itemcf_weight = db.Column(db.Float, default=0.4)  # itemCF 权重
    # 使用 Python 的 datetime.utcnow 作为默认值，避免将 SQL 表达式作为绑定参数
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return '<UserHybridWeights user_id={} dvae={} itemcf={}>'.format(
            self.user_id, self.dvae_weight, self.itemcf_weight)


# 用户对推荐的反馈表（用于改进 itemCF 模型）
class RecommendationFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query_movie_id = db.Column(db.String(20), nullable=False)  # 查询电影ID
    recommended_movie_id = db.Column(db.String(20), nullable=False)  # 推荐电影ID
    feedback = db.Column(db.String(20), nullable=False)  # 'helpful' / 'not_helpful' / 'dislike'
    recommendation_method = db.Column(db.String(50), default='hybrid')  # 推荐方法标识
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<RecommendationFeedback user_id={} query={} rec={} feedback={}>'.format(
            self.user_id, self.query_movie_id, self.recommended_movie_id, self.feedback)
