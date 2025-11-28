from app import app, db
from models import User, UserHybridWeights

with app.app_context():
    user = User.query.filter_by(username='test_save_user').first()
    if not user:
        print('No test user found, nothing to delete')
    else:
        # 删除用户的 UserHybridWeights（如果有）
        wh = UserHybridWeights.query.filter_by(user_id=user.id).first()
        if wh:
            db.session.delete(wh)
            print('Deleted UserHybridWeights id=', wh.id)
        # 删除用户
        db.session.delete(user)
        db.session.commit()
        print('Deleted test user id=', user.id)
