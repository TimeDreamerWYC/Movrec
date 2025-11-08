# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess-this-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
       'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance', 'database.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 推荐引擎相关路径
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
    RECOMMEND_ENGINE_FOLDER = os.path.join(os.path.dirname(__file__), 'recommend_engine')
   