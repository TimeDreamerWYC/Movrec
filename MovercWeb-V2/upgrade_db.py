# 升级 user 表，添加 douban_id 和 douban_cookie 字段（安全，不影响已有数据）
from app import app, db
from sqlalchemy.exc import OperationalError
from sqlalchemy import text

def add_column_if_not_exists(table, column, coltype):
    with app.app_context():
        # 检查字段是否已存在
        insp = db.inspect(db.engine)
        columns = [c['name'] for c in insp.get_columns(table)]
        if column not in columns:
            try:
                with db.engine.connect() as conn:
                    conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {column} {coltype}'))
                print(f"已添加字段: {column}")
            except OperationalError as e:
                print(f"添加字段失败: {e}")
        else:
            print(f"字段已存在: {column}")

if __name__ == '__main__':
    add_column_if_not_exists('user', 'douban_id', 'VARCHAR(32)')
    add_column_if_not_exists('user', 'douban_cookie', 'TEXT')
    print("数据库升级完成。请重启 Flask 项目。")
