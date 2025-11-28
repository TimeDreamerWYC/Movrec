import os


class Config:
   SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess-this-secret-key'
   SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or (
      'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance', 'database.db')
   )
   SQLALCHEMY_TRACK_MODIFICATIONS = False

   # Flask-WTF CSRF 保护
   WTF_CSRF_ENABLED = True
   WTF_CSRF_TIME_LIMIT = None  # 禁用令牌过期时间限制

   # 测试模式开关（可通过环境变量启用）
   # 在测试时会关闭 CSRF 保护以便自动化测试使用 test_client 发起 POST
   TESTING = os.environ.get('FLASK_TESTING', '0') == '1'

   # 推荐引擎相关路径
   DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
   RECOMMEND_ENGINE_FOLDER = os.path.join(os.path.dirname(__file__), 'recommend_engine')
