# ItemCF 准确度改进指南

## 一、ItemCF 算法原理

ItemCF（基于物品的协同过滤）通过计算用户共同喜欢的电影来推断电影之间的相似度。

**核心公式：**
```
相似度(电影A, 电影B) = 交集用户数 / sqrt(喜欢A的用户数 × 喜欢B的用户数)
```

**算法依赖的核心数据：**
1. **用户喜欢关系**（UserMoviePreference 表）
2. **用户不喜欢关系**（UserMovieDislike 表）

---

## 二、提高 ItemCF 准确度需要的关键反馈数据

### 1. **用户偏好反馈** ⭐⭐⭐⭐⭐（最重要）

**数据来源：** `/api/toggle_preference` 接口

**作用：**
- 累积用户的喜欢/不喜欢历史
- 扩充 movie_users 映射，增加共现信息
- 提高长尾电影的相似度计算质量

**收集方式：**
```python
# 用户在电影详情页点击"喜欢"或"不喜欢"按钮
POST /api/toggle_preference
{
    "movie_douban_id": "27611447",
    "action": "like"  # or "dislike"
}
```

**为什么重要：**
- itemCF 需要用户行为数据来建立"谁喜欢哪些电影"的映射
- 用户偏好越多 → 共现交集越多 → 相似度计算越精准

**目标指标：**
- 系统内每个用户至少标记 10-20 部电影偏好
- 系统内累积至少 1000+ 用户偏好记录才能显现 itemCF 优势

---

### 2. **推荐反馈（关键）** ⭐⭐⭐⭐

**数据来源：** `/api/recommend_feedback` 接口

**三种反馈类型：**

#### A. `helpful`（推荐有帮助）
```json
{
    "query_movie_id": "123456",
    "recommended_movie_id": "654321",
    "feedback": "helpful",
    "recommendation_method": "hybrid"
}
```
**含义：** 用户认为推荐的电影与查询电影在风格/质量/主题上确实相似
**改进信号：** itemCF 对这对电影的相似度评估是正确的，应该保留或增强

#### B. `not_helpful`（推荐没帮助但可接受）
```json
{
    "query_movie_id": "123456",
    "recommended_movie_id": "654321",
    "feedback": "not_helpful",
    "recommendation_method": "hybrid"
}
```
**含义：** 推荐有一定关联但不够相关，或用户已看过类似作品
**改进信号：** itemCF 的相似度可能被高估，需要调整或降低权重

#### C. `dislike`（强烈反对）
```json
{
    "query_movie_id": "123456",
    "recommended_movie_id": "654321",
    "feedback": "dislike",
    "recommendation_method": "hybrid"
}
```
**含义：** 推荐完全不相关或质量差，两部电影没有可比性
**改进信号：** itemCF 对这对电影的共现权重太高，应该降低或排除

**为什么重要：**
- 直接评估推荐算法的质量
- 识别 itemCF 的错误关联
- 帮助调整权重和阈值

**目标指标：**
- 累积 500+ 反馈记录
- 有用率（helpful / 总数）应该 > 60%
- 强反对率 < 15%

---

### 3. **用户评分反馈** ⭐⭐⭐（可选但推荐）

**数据来源：** 用户在看过电影后的评分（可选扩展）

**改进提案：** 在 RecommendationFeedback 表中添加 rating 字段

```python
# 扩展 RecommendationFeedback 模型
class RecommendationFeedback(db.Model):
    rating = db.Column(db.Integer)  # 1-5 星评分
    watched = db.Column(db.Boolean, default=False)  # 用户是否看过推荐电影
```

**作用：**
- 用户评分高的电影 → 说明推荐质量好
- 用户评分低的电影 → 说明推荐有问题
- 帮助区分"不相似"vs"相似但低质"

---

### 4. **观看历史反馈** ⭐⭐⭐

**数据来源：** 用户实际观看的电影

**收集方式（需要扩展）：**
```python
# 新建表
class UserMovieWatchHistory(db.Model):
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    movie_douban_id = db.Column(db.String(20))
    watched_at = db.Column(db.DateTime)
    rating = db.Column(db.Integer)  # 用户给出的评分 1-5
    is_recommended = db.Column(db.Boolean)  # 是否来自推荐
```

**作用：**
- itemCF 应该推荐用户可能会看/喜欢的电影
- 观看历史是验证推荐质量的真实信号
- 用户看过推荐的电影 → 推荐有效性高

**为什么重要：**
- 最真实的用户反馈信号
- 可以计算 itemCF 的"转化率"：被推荐且被观看 / 总推荐数

**目标指标：**
- 推荐转化率 > 20%（被推荐的电影中有 20% 用户会看）
- 推荐电影的平均评分 > 7.0/10

---

## 三、反馈数据收集的完整流程

### 当前已实现的流程：

```
1. 用户搜索电影 
   ↓
2. 混合推荐（DVAE + itemCF）显示结果
   ↓
3. 用户点击推荐电影 → 进入电影详情页
   ↓
4. 用户点击"喜欢"或"不喜欢" 
   ↓ (调用 /api/toggle_preference)
   UserMoviePreference / UserMovieDislike 表更新
   ↓
5. itemCF 在下次计算时使用最新的用户偏好数据
```

### 推荐反馈流程（需要前端集成）：

```
1. 推荐结果展示
   ↓
2. 用户对每个推荐点击反馈按钮
   ↓
   [有帮助] [没帮助] [不相关]
   ↓ (调用 /api/recommend_feedback)
   RecommendationFeedback 表记录
   ↓
3. 后端定期分析反馈数据
   ↓
4. 调整 itemCF 权重或参数
```

---

## 四、后端分析和改进机制

### A. 实时反馈统计

**现有端点：** `GET /api/itemcf_feedback_stats`

```python
{
    "total_feedback": 500,
    "helpful": 300,
    "not_helpful": 150,
    "dislike": 50,
    "helpful_rate": 0.60  # 关键指标
}
```

**目标：**
- helpful_rate 越高越好（目标 > 65%）
- dislike 越低越好（目标 < 10%）

### B. 建议的后端改进方向

#### 1. **动态权重调整**

根据反馈自动调整 DVAE vs itemCF 权重：

```python
def adjust_weights_based_feedback():
    """
    如果 itemCF 反馈不好 → 降低 itemCF 权重
    如果 DVAE 反馈不好 → 降低 DVAE 权重
    """
    # 分析最近 N 条反馈
    # 如果 helpful_rate < 50% → 考虑降低当前权重方案
    # 自动建议调整为 dvae:0.7, itemcf:0.3
```

#### 2. **电影对相似度校正**

记录每对电影的反馈，自动调整它们的关联强度：

```python
class MoviePairFeedback(db.Model):
    movie_a_id = db.Column(db.String(20))
    movie_b_id = db.Column(db.String(20))
    helpful_count = db.Column(db.Integer, default=0)
    not_helpful_count = db.Column(db.Integer, default=0)
    dislike_count = db.Column(db.Integer, default=0)
    
    # 计算信心度
    def confidence_score(self):
        total = self.helpful_count + self.not_helpful_count + self.dislike_count
        if total == 0:
            return 0
        return (self.helpful_count - self.dislike_count * 2) / total
```

#### 3. **用户偏好权重**

不同类型的用户的反馈价值不同：

```python
class UserRecommendationQuality(db.Model):
    user_id = db.Column(db.Integer)
    # 用户反馈的有用性评分
    accuracy_score = 0.8  # 基于该用户的历史反馈评估
    
    # 权重高的用户反馈在改进算法时权重更大
```

---

## 五、前端需要收集的反馈

### 立即可实现的：

1. **推荐反馈按钮** - 在推荐结果下方添加：
   ```html
   <div class="recommendation-feedback">
       <span>这个推荐有帮助吗？</span>
       <button onclick="submitFeedback('helpful')">👍 有帮助</button>
       <button onclick="submitFeedback('not_helpful')">😐 没帮助</button>
       <button onclick="submitFeedback('dislike')">👎 不相关</button>
   </div>
   ```

2. **权重调整滑块** - 已通过 API 实现，需前端 UI

3. **观看标记** - 用户可标记"已观看"或给出评分

### 后续可扩展的：

4. **观看历史导入** - 用户上传观看过的电影列表
5. **详细评论反馈** - 用户给出反馈原因（太相似、太不同、质量差等）
6. **对比反馈** - 用户对比两部电影相似度的评分

---

## 六、数据收集的最小可行产品（MVP）

**快速启动改进 itemCF 的最小需求：**

```
1. ✅ 用户偏好标记（已实现）
   - 每个用户标记 10+ 部电影

2. ✅ 推荐反馈收集（已实现 API，需前端 UI）
   - 用户对推荐点击 helpful/not_helpful/dislike

3. 📊 反馈分析（已实现统计端点）
   - 定期检查 helpful_rate
   - 识别表现差的推荐组合

4. 🔧 权重调整（已实现 API）
   - 基于反馈调整 dvae_weight vs itemcf_weight
```

---

## 七、数据质量指标

| 指标 | 目标 | 含义 |
|------|------|------|
| 累积偏好记录 | 1000+ | 足够构建共现矩阵 |
| 每用户平均偏好 | 15+ | 用户参与度 |
| 反馈记录 | 500+ | 足以评估推荐质量 |
| helpful_rate | > 65% | 推荐准确率 |
| dislike_rate | < 10% | 推荐不相关率 |
| 反馈覆盖率 | 20%+ | 用户愿意提供反馈的比例 |

---

## 八、快速测试 itemCF 改进效果

### API 查询反馈统计：

```bash
curl http://127.0.0.1:5000/api/itemcf_feedback_stats
```

### 设置用户权重：

```bash
curl -X POST http://127.0.0.1:5000/api/hybrid_weights \
  -H "Content-Type: application/json" \
  -d '{"dvae_weight": 0.5, "itemcf_weight": 0.5}'
```

### 提交反馈：

```bash
curl -X POST http://127.0.0.1:5000/api/recommend_feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query_movie_id": "123456",
    "recommended_movie_id": "654321",
    "feedback": "helpful"
  }'
```

---

## 总结

**itemCF 准确度改进的数据需求优先级：**

1. **最关键** 📌：用户偏好标记（喜欢/不喜欢）
2. **非常重要** 📌：推荐反馈（有帮助/没帮助/不相关）
3. **重要** 📌：观看历史和评分
4. **可选** 📌：用户评论和详细反馈

**立即行动：**
- ✅ 添加前端推荐反馈按钮（3 个按钮）
- ✅ 鼓励用户标记电影偏好
- ✅ 定期监控 helpful_rate 和 dislike_rate
- ✅ 根据反馈调整权重配置
