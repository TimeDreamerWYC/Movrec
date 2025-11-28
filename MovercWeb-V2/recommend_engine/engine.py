# 独立演员检索与模糊匹配函数
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def _normalize_name(s: str):
    """简单归一化：去首尾空白、转小写、去掉常见标点（保留中文字符）。"""
    if s is None:
        return ''
    s = str(s).strip()
    # 小写化
    s_low = s.lower()
    # 去掉常见拉丁标点和括号中的内容（例如角色说明），保留中文与字母数字、空格、点、连字符
    s_low = re.sub(r'[\(\)\[\]\{\}<>«»"\'`，,。！？!?:;—\-–]', ' ', s_low)
    # 压缩连续空白
    s_low = re.sub(r"\s+", ' ', s_low).strip()
    return s_low


def extract_actors_and_search(movies_df, query, top_k=10):
    """
    输入演员名关键字，返回：
    - 'direct': 包含关键字的演员原名列表（优先返回，最多 top_k）
    - 'fuzzy': 如果 direct 为空，返回基于 TF-IDF+余弦相似度的最相似演员原名或 None

    说明：匹配使用归一化形式（小写、去标点）进行比较，以提高命中率。
    """
    query = (query or '').strip()
    if not query or movies_df is None or 'ACTORS' not in movies_df.columns:
        return {'direct': [], 'fuzzy': None}

    # 构建原名集合与归一化映射
    actor_set = set()
    for actors in movies_df['ACTORS'].dropna():
        for name in re.split(r'[\\/;，,、\n]+', str(actors)):
            n = name.strip()
            if n:
                actor_set.add(n)
    actor_list = sorted(actor_set)

    # 归一化映射：norm -> list(orig)
    norm_map = {}
    norm_list = []
    for a in actor_list:
        na = _normalize_name(a)
        if na in norm_map:
            norm_map[na].append(a)
        else:
            norm_map[na] = [a]
            norm_list.append(na)

    nq = _normalize_name(query)

    # 直接包含：在归一化名字中查找包含关系
    direct_norm_hits = [na for na in norm_list if nq in na]
    direct_hits = []
    for na in direct_norm_hits:
        direct_hits.extend(norm_map.get(na, []))

    # 限制结果数量
    direct_hits = direct_hits[:top_k]

    fuzzy_hit = None
    if not direct_hits and norm_list:
        try:
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,2))
            tfidf = vectorizer.fit_transform(norm_list + [nq])
            sim = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
            idx = int(np.argmax(sim))
            if sim[idx] > 0.08:  # 放宽阈值以兼容短输入
                best_norm = norm_list[idx]
                # 选择该归一化名对应的第一个原名作为展示
                fuzzy_hit = norm_map.get(best_norm, [None])[0]
        except Exception as e:
            print(f"[ERROR] fuzzy actor match failed: {e}")

    # 调试输出
    if not direct_hits and not fuzzy_hit:
        print(f"[DEBUG] actor search no hits for query='{query}' (normalized='{nq}'), actor_count={len(actor_list)}")

    return {'direct': direct_hits, 'fuzzy': fuzzy_hit}
# 演员模糊匹配函数
def fuzzy_search_by_actor(actor_query, top_n=10):
    """
    支持演员名模糊匹配，返回相关电影列表。
    按归一化文本后用 TF-IDF + 余弦相似度匹配。
    """
    global movies_new, A, actor_index
    if movies_new is None or A is None or actor_index is None:
        return pd.DataFrame()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    # 归一化输入
    norm_query = normalize_text(actor_query)
    # 构造候选演员列表
    all_actors = list(actor_index.keys())
    # TF-IDF 向量化
    tfidf = TfidfVectorizer().fit(all_actors + [norm_query])
    query_vec = tfidf.transform([norm_query])
    actor_vecs = tfidf.transform(all_actors)
    sims = cosine_similarity(query_vec, actor_vecs).flatten()
    # 找到最相近的演员
    best_idx = sims.argmax()
    best_actor = all_actors[best_idx]
    # 找到所有包含该演员的电影
    import numpy as np
    actor_mask = A[:, best_idx] > 0
    matched_movies = movies_new[actor_mask]
    # 返回前 top_n 部电影
    return matched_movies.head(top_n).reset_index(drop=True)
def normalize_text(s):
    """统一文本格式：去除空格、标点、分隔符，转为小写。"""
    if not isinstance(s, str):
        s = str(s)
    # 去除所有空格、标点、分隔符
    s = re.sub(r'[\s·/|,;:._\-]', '', s)
    s = re.sub(r'[\u3000-\u303F\u2000-\u206F\uFF00-\uFFEF]', '', s)  # 中文符号
    s = s.lower()
    return s
# recommend_engine/engine.py
import pandas as pd
import numpy as np
import jieba
import re
import os
import pickle # 用于缓存预处理数据和模型
import logging
import threading

# NOTE: Heavy dependencies (tensorflow, sklearn) are imported lazily inside
# functions to avoid long import times on module import. This allows lightweight
# operations (e.g. checking get_movies_dataframe) without pulling in TF/Scipy.

# --- 全局变量存储模型状态 ---
# 这些将在 initialize_engine 中被填充
movies_new = None
cv = None
encoder = None
feature = None
similarity = None
G = None
D = None
A = None  # 演员特征矩阵
genre_index = None
director_index = None
actor_index = None

# logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 初始化进度状态（供外部监控）
init_progress_percent = 0
init_progress_messages = []
init_progress_lock = threading.Lock()

# --- 混合推荐权重配置（可动态调整） ---
hybrid_weights = {'dvae': 0.6, 'itemcf': 0.4}  # 默认权重（内部仍保存 itemcf，外部只需设置 dvae）
weights_lock = threading.Lock()

# --- 初始化状态标志（供外部查询） ---
cv_loaded = False
encoder_available = False
encoder_weights_loaded = False
feature_loaded = False
similarity_loaded = False

def get_hybrid_weights():
    """获取当前混合推荐权重（外部只需关心 dvae，itemcf=1-dvae）"""
    with weights_lock:
        dvae = float(hybrid_weights.get('dvae', 0.6))
        itemcf = 1.0 - dvae
        # 保证数值边界
        dvae = max(0.0, min(1.0, dvae))
        itemcf = max(0.0, min(1.0, itemcf))
        return {'dvae': dvae, 'itemcf': itemcf}

def set_hybrid_weights(dvae_weight=None):
    """设置混合推荐权重（仅接受 dvae 权重），itemcf 由 1-dvae 计算。

    参数:
        dvae_weight: DVAE 权重 (0-1)

    返回: 归一化后的权重字典
    """
    global hybrid_weights
    with weights_lock:
        if dvae_weight is not None:
            dvae = max(0.0, min(1.0, float(dvae_weight)))
        else:
            dvae = hybrid_weights.get('dvae', 0.6)

        # 计算 itemcf 为补余
        itemcf = 1.0 - dvae
        hybrid_weights = {'dvae': dvae, 'itemcf': itemcf}
        return {'dvae': dvae, 'itemcf': itemcf}


def get_engine_initialization_status():
    """返回引擎初始化的详细状态，供外部（例如 Flask）展示给用户。"""
    global init_progress_percent, init_progress_messages
    status = {
        'progress_percent': int(init_progress_percent),
        'messages': list(init_progress_messages[-20:]) if init_progress_messages else [],
        'cv_loaded': bool(cv is not None),
        'encoder_available': bool(encoder is not None),
        'encoder_weights_loaded': bool(encoder_weights_loaded),
        'feature_loaded': bool(feature is not None),
        'similarity_loaded': bool(similarity is not None),
        'hybrid_weights': get_hybrid_weights(),
    }
    return status

def _set_progress(percent, message=None):
    """线程安全地设置进度和附加消息（保留最近若干条消息）。"""
    global init_progress_percent, init_progress_messages
    with init_progress_lock:
        try:
            init_progress_percent = int(max(0, min(100, int(percent))))
        except Exception:
            init_progress_percent = 0
        if message:
            init_progress_messages.append(message)
            # 限制消息长度以免无限增长
            if len(init_progress_messages) > 200:
                init_progress_messages = init_progress_messages[-200:]

# 权重文件名统一
WEIGHTS_FILENAME = 'encoder.weights.h5'

# Note: heavy imports (sklearn, tensorflow) are performed inside functions when needed.
import pandas as pd


# --- 辅助函数：创建 CountVectorizer ---
def _get_stopwords():
    """返回中文停用词列表"""
    return [
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
        "为", "之", "对", "与", "而", "并", "等", "被", "及", "或",
        "但", "所以", "如果", "因为", "然后", "而且", "那么", "他们",
        "我们", "你们", "它们", "什么", "哪个", "哪些", "哪里", "时候",
        "他", "她", "它", "咱们", "大家", "谁", "怎样", "怎么", "多少", "为什么",
        "这里", "那里", "这样", "那样", "这个", "那个", "这些", "那些",
        "地", "得", "所", "过", "吗", "呢", "吧", "啊", "呀", "嘛", "哇", "啦",
        "从", "自", "以", "向", "关于", "对于", "根据", "按照", "通过", "由于",
        "并且", "或者", "虽然", "即使", "尽管", "不管", "只要", "只有", "除非",
        "最", "太", "更", "非常", "十分", "特别", "极其", "比较", "稍微", "有点",
        "刚", "才", "正在", "已经", "曾经", "马上", "立刻", "永远", "一直", "总是",
        "常常", "经常", "往往", "不断", "偶尔", "又", "再", "还", "仅", "光",
        "能", "能够", "可以", "可能", "应该", "应当", "想", "愿意", "肯", "敢",
        "来", "去", "进", "出", "回", "起", "开",
        "些", "一些", "所有", "每个", "某个", "各种", "多个", "几个", "第一", "第二",
        "就是", "只是", "可是", "真是", "也是", "不是", "正是",
        "一样", "一般", "一点", "一起", "一直", "一下", "一种", "一次"
    ]


def _create_count_vectorizer():
    """创建并返回配置好的 CountVectorizer 实例"""
    stopwords = _get_stopwords()
    # 使用顶层 tokenizer，避免 lambda 导致不可序列化的问题
    def jieba_tokenize(text):
        return jieba.lcut(str(text))

    # local import to avoid heavy imports at module import time
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(
        max_features=10000,
        tokenizer=jieba_tokenize,
        stop_words=stopwords,
        token_pattern=None
    )
    return cv


# ---------------------------
# Genre / Director feature helpers
# ---------------------------
def _extract_tokens_from_series(series, sep_chars=r'[|,;/]'):
    """把字符串列拆分成 token 列表"""
    return series.fillna('').astype(str).apply(lambda s: [t.strip() for t in re.split(sep_chars, s) if t.strip()])


def compute_genre_director_actor_features(movies_df, top_k_directors=500, top_k_genres=200, top_k_actors=5000):
    """计算并缓存全量电影的流派（G）、导演（D）、演员（A）稀疏矩阵（行归一化）。

    全局赋值：G, D, A, genre_index, director_index, actor_index
    """
    global G, D, A, genre_index, director_index, actor_index, movies_new
    # 安全拷贝 DataFrame 指针
    df = movies_df

    # 1) genres/tags 处理
    if 'GENRES' in df.columns:
        genres_lists = _extract_tokens_from_series(df['GENRES'], sep_chars=r'[|,;/]')
    elif 'TAGS' in df.columns:
        genres_lists = _extract_tokens_from_series(df['TAGS'], sep_chars=r'[|,;/]')
    else:
        genres_lists = df['INFO'].fillna('').astype(str).apply(lambda s: [])

    from collections import Counter
    all_genres = Counter([g for lst in genres_lists for g in lst])
    top_genres = [g for g, _ in all_genres.most_common(top_k_genres)]
    genre_index = {g: i for i, g in enumerate(top_genres)}

    import numpy as _np
    from sklearn.preprocessing import normalize as _normalize

    G_mat = _np.zeros((len(df), len(top_genres)), dtype=_np.float32)
    for i, lst in enumerate(genres_lists):
        for g in lst:
            if g in genre_index:
                G_mat[i, genre_index[g]] = 1.0
    if G_mat.shape[1] > 0:
        G_mat = _normalize(G_mat, norm='l2', axis=1)
    G = G_mat

    # 2) directors 处理
    directors_lists = _extract_tokens_from_series(df['DIRECTORS'].fillna(''), sep_chars=r'[/,;]')
    directors_lists = directors_lists.apply(lambda lst: [normalize_text(d) for d in lst])
    all_dirs = Counter([d for lst in directors_lists for d in lst])
    top_dirs = [d for d, _ in all_dirs.most_common(top_k_directors)]
    director_index = {d: i for i, d in enumerate(top_dirs)}

    D_mat = _np.zeros((len(df), len(top_dirs)), dtype=_np.float32)
    for i, lst in enumerate(directors_lists):
        for d in lst:
            if d in director_index:
                D_mat[i, director_index[d]] = 1.0
    if D_mat.shape[1] > 0:
        D_mat = _normalize(D_mat, norm='l2', axis=1)
    D = D_mat

    # 3) actors 处理
    actors_lists = _extract_tokens_from_series(df['ACTORS'].fillna(''), sep_chars=r'[/,;]')
    actors_lists = actors_lists.apply(lambda lst: [normalize_text(a) for a in lst])
    all_acts = Counter([a for lst in actors_lists for a in lst])
    top_acts = [a for a, _ in all_acts.most_common(top_k_actors)]
    actor_index = {a: i for i, a in enumerate(top_acts)}

    A_mat = _np.zeros((len(df), len(top_acts)), dtype=_np.float32)
    for i, lst in enumerate(actors_lists):
        for a in lst:
            if a in actor_index:
                A_mat[i, actor_index[a]] = 1.0
    if A_mat.shape[1] > 0:
        A_mat = _normalize(A_mat, norm='l2', axis=1)
    A = A_mat

    # 同步到全局 movies_new（如果尚未指向的话）
    movies_new = df
    return G, D, A, genre_index, director_index, actor_index


def build_user_pref_vectors_from_ids(movie_ids):
    """基于用户喜欢的 movie_ids（MOVIE_ID 列）构建 U_c/U_g/U_d 向量。

    返回 dict {'U_c','U_g','U_d'} 或 None（当无法构建时）。
    """
    global feature, G, D, movies_new
    if movie_ids is None:
        return None
    # 构造 id->idx 映射
    id_series = movies_new['MOVIE_ID'].astype(str)
    id_to_idx = {v: i for i, v in enumerate(id_series)}
    idxs = [id_to_idx.get(str(mid)) for mid in movie_ids if str(mid) in id_to_idx]
    idxs = [i for i in idxs if i is not None]
    if not idxs:
        return None

    import numpy as _np

    result = {}
    if feature is not None and getattr(feature, 'shape', None):
        try:
            U_c = _np.mean(feature[idxs], axis=0)
            U_c = U_c / (_np.linalg.norm(U_c) + 1e-8)
            result['U_c'] = U_c
        except Exception:
            result['U_c'] = None
    else:
        result['U_c'] = None

    if G is not None and getattr(G, 'shape', None) and G.shape[1] > 0:
        try:
            U_g = _np.mean(G[idxs], axis=0)
            U_g = U_g / (_np.linalg.norm(U_g) + 1e-8)
            result['U_g'] = U_g
        except Exception:
            result['U_g'] = None
    else:
        result['U_g'] = None

    if D is not None and getattr(D, 'shape', None) and D.shape[1] > 0:
        try:
            U_d = _np.mean(D[idxs], axis=0)
            U_d = U_d / (_np.linalg.norm(U_d) + 1e-8)
            result['U_d'] = U_d
        except Exception:
            result['U_d'] = None
    else:
        result['U_d'] = None

    return result


def enhanced_recommend_for_user(movie_name, user_pref_vectors=None, weights=None, sample_top=50, pick_n=15):
    """结合内容/流派/导演与用户偏好向量的增强推荐。

    参数:
        movie_name: 查询电影标题
        user_pref_vectors: dict from build_user_pref_vectors_from_ids
        weights: 权重字典（见函数内默认值）
    返回: pd.DataFrame 推荐列表
    """
    global movies_new, feature, G, D
    # 默认权重
    if weights is None:
        weights = {
            'content': 0.5,
            'genre': 0.15,
            'director': 0.15,
            'user_content': 0.1,
            'user_genre': 0.05,
            'user_director': 0.05,
        }

    # 查找电影索引
    # 统一格式化电影名
    norm_movie_name = normalize_text(movie_name)
    norm_names = movies_new['NAME'].fillna('').apply(normalize_text)
    matches = movies_new[norm_names == norm_movie_name]
    if matches.empty:
        # 模糊匹配第一项
        similar = movies_new[norm_names.str.contains(norm_movie_name, na=False)]
        if similar.empty:
            return pd.DataFrame()
        q = similar.index[0]
    else:
        q = matches.index[0]

    # 计算相似度分量
    n = len(movies_new)
    import numpy as _np
    try:
        from sklearn.metrics.pairwise import cosine_similarity as _cos
    except Exception:
        # sklearn 不可用时退化为空向量
        def _cos(a, b):
            return _np.zeros((a.shape[0], b.shape[0]))

    # content
    if feature is not None and getattr(feature, 'shape', None):
        sims_content = _cos(feature[q:q+1], feature).flatten()
    else:
        sims_content = _np.zeros(n, dtype=_np.float32)

    # genre & director
    sims_genre = _np.zeros(n, dtype=_np.float32)
    sims_director = _np.zeros(n, dtype=_np.float32)
    if G is not None and getattr(G, 'shape', None) and G.shape[1] > 0:
        sims_genre = _cos(G[q:q+1], G).flatten()
    if D is not None and getattr(D, 'shape', None) and D.shape[1] > 0:
        sims_director = _cos(D[q:q+1], D).flatten()

    combined = (_np.zeros(n, dtype=_np.float32)
                + weights.get('content', 0.0) * sims_content
                + weights.get('genre', 0.0) * sims_genre
                + weights.get('director', 0.0) * sims_director)

    # user preference contributions
    if user_pref_vectors is not None:
        U_c = user_pref_vectors.get('U_c')
        U_g = user_pref_vectors.get('U_g')
        U_d = user_pref_vectors.get('U_d')
        if U_c is not None and feature is not None:
            user_content_sim = _cos(feature, U_c.reshape(1, -1)).flatten()
            combined += weights.get('user_content', 0.0) * user_content_sim
        if U_g is not None and G is not None and G.shape[1] > 0:
            user_genre_sim = _cos(G, U_g.reshape(1, -1)).flatten()
            combined += weights.get('user_genre', 0.0) * user_genre_sim
        if U_d is not None and D is not None and D.shape[1] > 0:
            user_dir_sim = _cos(D, U_d.reshape(1, -1)).flatten()
            combined += weights.get('user_director', 0.0) * user_dir_sim

    # 排除自身
    combined[q] = -1

    top_idxs = _np.argsort(-combined)[:sample_top]
    pick_n = min(pick_n, len(top_idxs))
    if pick_n <= 0:
        return pd.DataFrame()
    picks = _np.random.choice(top_idxs, pick_n, replace=False)

    recs = []
    for idx in picks:
        r = movies_new.iloc[idx]
        recs.append({
            # 保持与模板兼容的字段名
            'MOVIE_ID': r.get('MOVIE_ID'),
            '电影名': r.get('NAME'),
            '豆瓣评分': r.get('DOUBAN_SCORE'),
            '流派': r.get('LABEL') if 'LABEL' in r.index else None,
            '导演': r.get('DIRECTORS'),
            '相似度': float(combined[idx])
        })
    return pd.DataFrame(recs)


def build_user_pref_vectors_from_user(user_id):
    """从数据库加载用户喜欢的 movie_douban_id，并构建用户偏好向量。

    该函数在内部延迟导入 `models` 以避免循环依赖。
    返回与 build_user_pref_vectors_from_ids 相同格式的字典，或 None。
    """
    try:
        # 延迟导入以避免循环引用
        from models import UserMoviePreference
    except Exception as e:
        # 无法导入 models（在某些测试场景），直接返回 None
        print(f"⚠️ 无法导入 models: {e}")
        return None

    try:
        prefs = UserMoviePreference.query.filter_by(user_id=user_id).all()
        movie_ids = [p.movie_douban_id for p in prefs]
        return build_user_pref_vectors_from_ids(movie_ids)
    except Exception as e:
        print(f"⚠️ 从数据库构建用户偏好向量失败: {e}")
        return None


# --- 主初始化函数 ---
# 假设 movies_new, encoder, feature, similarity, _build_encoder_structure 是在模块级别定义的全局变量或函数
# from somewhere import movies_new, encoder, feature, similarity, _build_encoder_structure

def initialize_engine(data_folder_path, model_cache_path="model_cache.pkl"):
    # 调试：输出演员索引数量和前10个演员名
    import sys
    def _debug_actor_index():
        global actor_index
        if actor_index:
            print(f"[DEBUG] 演员索引数量: {len(actor_index)}", file=sys.stderr)
            print(f"[DEBUG] 演员样例: {list(actor_index.keys())[:10]}", file=sys.stderr)
        else:
            print("[DEBUG] 演员索引为空！", file=sys.stderr)

    """
    初始化推荐引擎：加载数据、预处理、训练DVAE模型（如果缓存不存在）。
    """
    # 声明需要修改的全局变量
    global movies_new, cv, encoder, feature, similarity
    # 注意：'encoder' 只需声明一次，如果之前已声明过，请删除重复的 global encoder

    # local imports of heavy libs to avoid module-level import cost
    # Use tolerant imports so cached-only startup can succeed without TF
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as _e:
        tf = None
        keras = None
        print(f"⚠️ tensorflow import failed or unavailable: {_e}")
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as _e:
        cosine_similarity = None
        print(f"⚠️ sklearn.metrics.pairwise.cosine_similarity import failed: {_e}")

    cache_exists = os.path.exists(model_cache_path)
    _set_progress(1, "开始初始化：检查缓存")
    if cache_exists:
        _set_progress(5, "检测到 model_cache，尝试从缓存加载...")
        print("🔍 尝试从缓存加载预处理模型和特征...")
        try:
            with open(model_cache_path, 'rb') as f:
                cache = pickle.load(f)
                # 尽量从缓存恢复不依赖 heavy lib 的数据（movies_new/feature/similarity）
                movies_new = cache.get('movies_new', None)
                feature = cache.get('feature', None)
                similarity = cache.get('similarity', None)

                # 尝试恢复流派/导演特征（如果缓存中存在）
                try:
                    G = cache.get('G', None)
                    D = cache.get('D', None)
                    genre_index = cache.get('genre_index', None)
                    director_index = cache.get('director_index', None)
                except Exception:
                    G = D = genre_index = director_index = None

                # 尝试恢复 director_to_label
                try:
                    director_to_label = cache.get('director_to_label', None)
                except Exception:
                    director_to_label = None

                # 重新创建 CountVectorizer 并恢复词表（如果存在）
                try:
                    cv = _create_count_vectorizer()
                    vocab = cache.get('cv_vocab', None)
                    if vocab is not None:
                        cv.vocabulary_ = vocab
                except Exception:
                    cv = _create_count_vectorizer()

                # 尝试构建 encoder 并加载权重；若系统缺少 tensorflow，则跳过但不阻塞初始化
                try:
                    _build_encoder_structure(cache.get('inp_dim'), cache.get('code_dim'))
                    try:
                        encoder.load_weights(os.path.join(os.path.dirname(model_cache_path), WEIGHTS_FILENAME))
                    except Exception as e:
                        # 权重加载失败（可能缺少文件或不兼容），记录但继续
                        print(f"⚠️ 载入 encoder 权重失败: {e}")
                except Exception as e:
                    # 如果 tensorflow 不可用或构建失败，记录并继续（不阻塞）
                    print(f"⚠️ 无法重建 encoder（可能缺少 tensorflow）：{e}")

                _set_progress(100, "成功从缓存加载完成（部分功能可能被降级）")
                print("✅ 成功从缓存加载（部分功能可能被降级）!")
                return
        except Exception as e:
            print(f"⚠️ 缓存加载失败: {e}，将重新计算...")

    _set_progress(10, "缓存不可用或加载失败，开始预处理数据和训练模型")
    print("🔄 开始预处理数据和训练模型...")

    # 1. 读入原始数据
    movies_path = os.path.join(data_folder_path, "movies.csv")
    movies_db_path = os.path.join(data_folder_path, "movies_db.csv")
    director_label_path = os.path.join(data_folder_path, "director_label.csv")

    _set_progress(12, "读取 CSV 数据")
    movies = pd.read_csv(movies_path)
    movies_db = pd.read_csv(movies_db_path)

    # 2. 清洗 movies_db，构造 INFO
    movies_db = movies_db.drop(columns=["durations", "votes"])
    movies_db["INFO"] = (
        movies_db["genres"].fillna("").astype(str) + " " +
        movies_db["countries"].fillna("").astype(str) + " " +
        movies_db["reviews"].fillna("").astype(str)
    )
    movies_db = movies_db.drop(columns=["genres", "countries", "reviews"])
    movies_db["title"] = movies_db["title"].apply(
        lambda x: "".join(re.findall(r"[\u4e00-\u9fff]+", str(x)))
    )

    # 3. 清洗 movies，本体只保留高分电影，保留演员信息
    movies = movies.drop(
        columns=[
            "COVER", "IMDB_ID", "MINS", "OFFICIAL_SITE", "RELEASE_DATE",
            "SLUG", "ACTOR_IDS", "DIRECTOR_IDS", "LANGUAGES", "GENRES",
            "ALIAS"  # 注意：不再 drop "ACTORS"
        ]
    )
    movies = movies[movies["DOUBAN_SCORE"] >= 6.5]

    # 4. 构造 movies_new（评分/人数过滤），保留演员信息
    movies_new_filtered = movies[movies["DOUBAN_VOTES"] >= 3000] \
        .sort_values(by=["DOUBAN_SCORE", "DOUBAN_VOTES"], ascending=[False, False])[
        ["DIRECTORS", "ACTORS", "MOVIE_ID", "NAME", "DOUBAN_SCORE",
         "STORYLINE", "TAGS", "REGIONS", "YEAR"]
    ]

    # 5. 拼接剧情 + 标签 + 地区 作为 INFO
    movies_new_filtered["INFO"] = (
        movies_new_filtered["STORYLINE"].fillna("").astype(str) + " " +
        movies_new_filtered["TAGS"].fillna("").astype(str) + " " +
        movies_new_filtered["REGIONS"].fillna("").astype(str)
    )
    movies_new_filtered = movies_new_filtered.drop(columns=["STORYLINE", "TAGS", "REGIONS"])

    # 6. 拼接 movies_db（爬虫来的数据），保留演员信息
    movies_db_renamed = movies_db.rename(columns={
        "subject_id": "MOVIE_ID",
        "title": "NAME",
        "year": "YEAR",
        "rating": "DOUBAN_SCORE",
        "directors": "DIRECTORS",
        "actors": "ACTORS"
    })
    # 某些爬虫数据可能没有 actors 字段，需防御性处理
    db_cols = ["DIRECTORS", "MOVIE_ID", "NAME", "DOUBAN_SCORE", "YEAR", "INFO"]
    if "ACTORS" in movies_db_renamed.columns:
        db_cols.insert(1, "ACTORS")
    movies_db_renamed = movies_db_renamed[db_cols]

    # 7. 合并两部分数据
    movies_new_combined = pd.concat([movies_new_filtered, movies_db_renamed], ignore_index=True)

    # 8. 加导演标签
    director_label = pd.read_csv(director_label_path)
    director_to_label = dict(zip(director_label["DIRECTOR"], director_label["LABEL"]))
    movies_new_combined["LABEL"] = movies_new_combined["DIRECTORS"].apply(
        lambda x: ",".join(
            {
                director_to_label.get(d.strip())
                for d in str(x).split("/")
                if director_to_label.get(d.strip())
            }
        ) if pd.notna(x) else None
    )

    # 更新全局变量 movies_new
    movies_new = movies_new_combined
    _set_progress(30, "数据清洗完成")
    print("✅ 数据清洗完成")

    # 预计算流派和导演特征，便于后续个性化推荐
    try:
        _set_progress(40, "正在计算流派/导演/演员特征")
        compute_genre_director_actor_features(movies_new)
        _debug_actor_index()
        _set_progress(50, "流派/导演/演员特征预计算完成")
        print("✅ 流派/导演/演员特征预计算完成")
    except Exception as e:
        _set_progress(45, f"流派/导演/演员特征预计算失败: {e}")
        print(f"⚠️ 流派/导演/演员特征预计算失败: {e}")

    # --- BOW + DVAE ---
    # 使用辅助函数创建 CountVectorizer
    cv = _create_count_vectorizer()

    vector = cv.fit_transform(movies_new["INFO"].astype(str)).toarray().astype("float32")
    _set_progress(60, "BOW 向量构建完成")
    print("✅ BOW 向量构建完成")

    # DVAE 参数
    inp_dim = vector.shape[1]
    code_dim = 64
    epochs = 5  # 调试阶段设小，生产可调大
    batch_size = 256
    beta_kl = 1.0

    # 编码器
    inputs = keras.Input(shape=(inp_dim,), name="bow_counts")
    x = keras.layers.GaussianNoise(0.15)(inputs)
    x = keras.layers.Dense(1000, activation="selu")(x)
    x = keras.layers.Dense(256, activation="selu")(x)
    z_mean = keras.layers.Dense(code_dim, name="z_mean")(x)
    z_logvar = keras.layers.Dense(code_dim, name="z_logvar")(x)

    def reparameterize(args):
        mu, logvar = args
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps

    z = keras.layers.Lambda(reparameterize, name="z")([z_mean, z_logvar])
    encoder = keras.Model(inputs, [z_mean, z_logvar, z], name="dvae_encoder")

    # 解码器 (用于训练)
    latent_inputs = keras.Input(shape=(code_dim,), name="z_in")
    d = keras.layers.Dense(256, activation="selu")(latent_inputs)
    d = keras.layers.Dense(1000, activation="selu")(d)
    recons = keras.layers.Dense(inp_dim, activation=None, name="recon")(d)
    decoder = keras.Model(latent_inputs, recons, name="dvae_decoder")

    # KL 正则层
    class KLDivergenceLayer(keras.layers.Layer):
        def __init__(self, beta=1.0, scale=1.0, **kwargs):
            super().__init__(**kwargs)
            self.beta = beta
            self.scale = scale

        def call(self, inputs):
            mu, logvar = inputs
            kl_per_sample = -0.5 * tf.reduce_sum(
                1.0 + logvar - tf.exp(logvar) - tf.square(mu), axis=1
            )
            kl = tf.reduce_mean(kl_per_sample) / float(self.scale)
            self.add_loss(self.beta * kl)
            return tf.zeros_like(mu[:, :1])

    z_mean_out, z_logvar_out, z_out = encoder(inputs)
    _ = KLDivergenceLayer(beta=beta_kl, scale=inp_dim, name="kl_reg")(
        [z_mean_out, z_logvar_out]
    )
    recons_out = decoder(z_out)

    vae = keras.Model(inputs, recons_out, name="dvae")
    vae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    # 训练 VAE
    _set_progress(65, "开始训练 DVAE（可能耗时）")
    history = vae.fit(
        vector, vector,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )
    _set_progress(85, "DVAE 模型训练完成")
    print("✅ DVAE 模型训练完成")

    # 提取电影语义向量 feature（z_mean）
    z_mean_val = encoder.predict(vector, verbose=0)[0]
    feature = z_mean_val.astype("float32")
    _set_progress(88, "电影语义特征提取完成")
    print("✅ 电影语义特征提取完成")

    # 计算余弦相似度矩阵
    similarity = cosine_similarity(feature)
    _set_progress(94, "相似度矩阵计算完成")
    print("✅ 相似度矩阵计算完成")

    # --- 缓存模型和特征 ---
    print("💾 正在缓存模型和特征...")
    # 注意：不再缓存 'cv' 对象，因为它包含了不可 pickle 的 lambda
    cache_to_save = {
        'movies_new': movies_new,     # DataFrame
        # 'cv': cv,                   # <-- 移除此行
        'feature': feature,           # NumPy array
        'similarity': similarity,     # NumPy array
        'inp_dim': inp_dim,           # int (用于重建 encoder 结构)
        'code_dim': code_dim          # int (用于重建 encoder 结构)
        # 如果需要缓存 director_to_label，也可以加上
        # 'director_to_label': director_to_label 
    }
    
    try:
        # 补充要缓存的 G/D/indices 与 cv vocabulary
        try:
            cache_to_save['G'] = G
            cache_to_save['D'] = D
            cache_to_save['genre_index'] = genre_index
            cache_to_save['director_index'] = director_index
        except Exception:
            pass
        try:
            # 保存 cv 的 vocabulary（可用于快速恢复 CountVectorizer 的词表）
            if cv is not None and hasattr(cv, 'vocabulary_'):
                cache_to_save['cv_vocab'] = cv.vocabulary_
        except Exception:
            pass

        with open(model_cache_path, 'wb') as f:
            pickle.dump(cache_to_save, f)
        encoder.save_weights(os.path.join(os.path.dirname(model_cache_path), WEIGHTS_FILENAME))
        _set_progress(98, "缓存保存成功")
        print("✅ 缓存保存成功!")
    except Exception as e:
        _set_progress(97, f"缓存保存失败: {e}")
        print(f"⚠️ 缓存保存失败: {e}")
        # 根据你的需求决定是否要在这里抛出异常
        # raise e # 如果缓存失败是致命错误，取消注释此行

    # 注意：函数结束，cv 已在此函数作用域内创建并赋值给全局变量
    _set_progress(100, "初始化完成")

def _build_encoder_structure(inp_dim, code_dim):
    """重建编码器结构以便加载权重"""
    global encoder
    # local import to avoid top-level dependency
    import tensorflow as tf
    from tensorflow import keras

    inputs = keras.Input(shape=(inp_dim,), name="bow_counts")
    x = keras.layers.GaussianNoise(0.15)(inputs)
    x = keras.layers.Dense(1000, activation="selu")(x)
    x = keras.layers.Dense(256, activation="selu")(x)
    z_mean = keras.layers.Dense(code_dim, name="z_mean")(x)
    z_logvar = keras.layers.Dense(code_dim, name="z_logvar")(x)

    def reparameterize(args):
        mu, logvar = args
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps

    z = keras.layers.Lambda(reparameterize, name="z")([z_mean, z_logvar])
    encoder = keras.Model(inputs, [z_mean, z_logvar, z], name="dvae_encoder")


def get_movie_features():
    """获取电影特征向量"""
    return feature


def get_movies_dataframe():
    """获取处理后的电影DataFrame"""
    return movies_new


def get_similarity_matrix():
    """获取电影相似度矩阵"""
    return similarity


def recommand(movie_name, sample_top=15, pick_n=5):
    """基础推荐函数（只用内容相似）"""
    # 防御性检查：确保数据已初始化
    if movies_new is None or similarity is None:
        logger.warning('recommand called before engine initialized or data missing')
        return pd.DataFrame()
    label_idx = movies_new.index[movies_new["NAME"] == movie_name]
    if len(label_idx) == 0:
        # 尝试模糊匹配
        similar_movies = movies_new[movies_new["NAME"].str.contains(movie_name, na=False, case=False)]
        if len(similar_movies) > 0:
            print(f"未精确找到《{movie_name}》，尝试模糊匹配:")
            for idx, row in similar_movies.head(3).iterrows():
                 print(f"  - {row['NAME']}")
            # 默认使用第一个匹配项
            pos = similar_movies.index[0]
        else:
            print(f"未找到影片：《{movie_name}》")
            return None
    else:
        pos = movies_new.index.get_loc(label_idx[0])

    sims = similarity[pos]
    cand = np.argsort(-sims)  # 降序
    cand = cand[cand != pos]  # 去掉自身
    top_candidates = cand[:sample_top]

    n_pick = min(pick_n, len(top_candidates))
    if n_pick == 0:
        return pd.DataFrame()
    selected = np.random.choice(top_candidates, n_pick, replace=False)

    recs = []
    for j in selected:
        row = movies_new.iloc[j]
        recs.append({
            "MOVIE_ID": row.get('MOVIE_ID'),
            "电影名": row["NAME"],
            "豆瓣评分": row["DOUBAN_SCORE"],
            "流派": row.get("LABEL"),
            "相似度": sims[j],
            "导演": row.get("DIRECTORS"),
        })
    df = pd.DataFrame(recs).sort_values(by="相似度", ascending=False).reset_index(drop=True)
    return df


def itemcf_recommend_for_movie(movie_name, sample_top=50):
    """基于共现的 ItemCF：对给定电影计算与其它电影的协同过滤相似度分数。

    返回长度为 n 的 numpy 数组（与 `movies_new` 行对应），数值为相似度分数（float）。
    当数据库或数据不足时返回全零向量。
    """
    global movies_new
    n = len(movies_new) if movies_new is not None else 0
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # 找到查询电影索引（尝试精确匹配，再模糊匹配）
    matches = movies_new[movies_new['NAME'] == movie_name]
    if matches.empty:
        similar = movies_new[movies_new['NAME'].str.contains(movie_name, na=False, case=False)]
        if similar.empty:
            return np.zeros(n, dtype=np.float32)
        target_idx = similar.index[0]
    else:
        target_idx = matches.index[0]

    # 延迟导入 models，避免循环依赖
    try:
        from models import UserMoviePreference
    except Exception:
        return np.zeros(n, dtype=np.float32)

    # 从数据库加载所有用户-喜欢关系
    try:
        prefs = UserMoviePreference.query.with_entities(UserMoviePreference.user_id, UserMoviePreference.movie_douban_id).all()
    except Exception:
        return _np.zeros(n, dtype=_np.float32)

    # 构建 movie -> set(users) 映射
    movie_users = {}
    for uid, mid in prefs:
        movie_users.setdefault(str(mid), set()).add(int(uid))

    # 目标电影的 MOVIE_ID
    target_mid = str(movies_new.at[target_idx, 'MOVIE_ID'])
    users_target = movie_users.get(target_mid, set())
    if not users_target:
        return np.zeros(n, dtype=np.float32)

    # 计算与每个 movie 的相似度（基于共现 / cosine-like）
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        try:
            mid_i = str(movies_new.at[i, 'MOVIE_ID'])
        except Exception:
            continue
        users_i = movie_users.get(mid_i, set())
        if not users_i:
            continue
        inter = len(users_target & users_i)
        if inter == 0:
            continue
        # cosine-like normalization
        denom = (np.sqrt(len(users_target) * len(users_i)))
        if denom > 0:
            scores[i] = float(inter) / float(denom)

    # 排除自身
    scores[target_idx] = -1.0
    return scores


def hybrid_recommend_for_user(movie_name, user_id=None, weights=None, sample_top=50, pick_n=15):
    """混合推荐：将 DVAE(content) 相似度与 itemCF 共现分数按权重合并并返回推荐 DataFrame。

    参数:
        movie_name: 查询电影名称（字符串）
        user_id: 可选的用户 id（当前实现未直接用到，但保留接口兼容）
        weights: dict, e.g. {'dvae':0.6, 'itemcf':0.4}
    返回: pd.DataFrame 与 `recommand` / `enhanced_recommend_for_user` 相同格式
    """
    global similarity, movies_new
    if weights is None:
        weights = {'dvae': 0.6, 'itemcf': 0.4}

    n = len(movies_new) if movies_new is not None else 0
    if n == 0:
        return pd.DataFrame()

    # 1) DVAE 相似度分量
    try:
        # 寻找查询电影索引
        matches = movies_new[movies_new['NAME'] == movie_name]
        if matches.empty:
            similar = movies_new[movies_new['NAME'].str.contains(movie_name, na=False, case=False)]
            if similar.empty:
                return pd.DataFrame()
            q = similar.index[0]
        else:
            q = matches.index[0]

        if similarity is None:
            sims_dvae = np.zeros(n, dtype=np.float32)
        else:
            sims_dvae = np.array(similarity[q], dtype=np.float32)
    except Exception:
        sims_dvae = np.zeros(n, dtype=np.float32)

    # 2) itemCF 分量（共现）
    try:
        sims_item = itemcf_recommend_for_movie(movie_name, sample_top=sample_top)
        if sims_item.shape[0] != n:
            sims_item = np.zeros(n, dtype=np.float32)
    except Exception:
        sims_item = np.zeros(n, dtype=np.float32)

    # 归一化两个分量（避免尺度差异）
    def _normalize_vec(v):
        vmax = v.max() if v.size > 0 else 0.0
        vmin = v.min() if v.size > 0 else 0.0
        if vmax - vmin > 1e-9:
            return (v - vmin) / (vmax - vmin)
        return v

    nd = _normalize_vec(sims_dvae)
    ni = _normalize_vec(sims_item)

    combined = weights.get('dvae', 0.0) * nd + weights.get('itemcf', 0.0) * ni

    # 排除查询自身（如果能定位到）
    try:
        if 'q' in locals():
            combined[q] = -1.0
    except Exception:
        pass

    # 取 top
    top_idxs = np.argsort(-combined)[:sample_top]
    pick_n = min(pick_n, len(top_idxs))
    if pick_n <= 0:
        return pd.DataFrame()
    picks = np.random.choice(top_idxs, pick_n, replace=False)

    recs = []
    for idx in picks:
        r = movies_new.iloc[idx]
        recs.append({
            'MOVIE_ID': r.get('MOVIE_ID'),
            '电影名': r.get('NAME'),
            '豆瓣评分': r.get('DOUBAN_SCORE'),
            '流派': r.get('LABEL') if 'LABEL' in r.index else None,
            '导演': r.get('DIRECTORS'),
            '相似度': float(combined[idx])
        })
    return pd.DataFrame(recs)


def get_popular_movies(data_folder_path=None, count=20, min_score=8.5, min_votes=100000):
    """
    获取热门电影：随机选择满足条件的电影
    
    Args:
        data_folder_path (str): 数据文件夹路径（如果 None，使用全局 movies_new）
        count (int): 返回电影数量，默认 20
        min_score (float): 最低豆瓣评分，默认 8.5
        min_votes (int): 最低评分人数，默认 100000
    
    Returns:
        pd.DataFrame: 包含热门电影的 DataFrame，或空 DataFrame
    """
    global movies_new
    
    # 如果已初始化且有数据，尝试从全局 movies_new 中获取
    if movies_new is not None and not movies_new.empty:
        # 检查必要的列是否存在
        if 'DOUBAN_SCORE' in movies_new.columns:
            # 从全局数据中筛选
            # 先确保评分和投票数为数值类型，防止字符串比较导致 TypeError
            df = movies_new.copy()
            if 'DOUBAN_SCORE' in df.columns:
                df['DOUBAN_SCORE'] = pd.to_numeric(df['DOUBAN_SCORE'], errors='coerce')
            if 'DOUBAN_VOTES' in df.columns:
                df['DOUBAN_VOTES'] = pd.to_numeric(df['DOUBAN_VOTES'], errors='coerce')

            # 构造过滤条件，使用 .ge/.ge 以避免类型不一致的比较
            cond = df['DOUBAN_SCORE'].ge(min_score)
            if 'DOUBAN_VOTES' in df.columns:
                cond = cond & df['DOUBAN_VOTES'].ge(min_votes)

            popular = df[cond].copy()
            
            if not popular.empty:
                # 随机选择 count 部电影
                sample_count = min(count, len(popular))
                result = popular.sample(n=sample_count, random_state=None).reset_index(drop=True)
                logger.info(f"从全局 movies_new 中获取 {sample_count} 部热门电影")
                return result
    
    # 如果没有初始化或全局数据不足，从原始 CSV 读取
    if data_folder_path is None:
        logger.warning("无法获取热门电影：movies_new 未初始化且未提供 data_folder_path")
        return pd.DataFrame()
    
    movies_csv_path = os.path.join(data_folder_path, 'movies.csv')
    if not os.path.exists(movies_csv_path):
        logger.error(f"movies.csv 不存在: {movies_csv_path}")
        return pd.DataFrame()
    
    try:
        # 直接从 CSV 读取原始数据
        movies = pd.read_csv(movies_csv_path)
        
        # 筛选条件：先将评分和投票数转换成数值型，再应用阈值过滤
        if 'DOUBAN_SCORE' in movies.columns:
            movies['DOUBAN_SCORE'] = pd.to_numeric(movies['DOUBAN_SCORE'], errors='coerce')
        if 'DOUBAN_VOTES' in movies.columns:
            movies['DOUBAN_VOTES'] = pd.to_numeric(movies['DOUBAN_VOTES'], errors='coerce')

        popular = movies[
            (movies['DOUBAN_SCORE'].ge(min_score)) & 
            (movies['DOUBAN_VOTES'].ge(min_votes))
        ].copy()
        
        if popular.empty:
            logger.warning(f"没有找到符合条件的电影 (评分>={min_score}, 评分人数>={min_votes})")
            return pd.DataFrame()
        
        # 随机选择 count 部电影
        sample_count = min(count, len(popular))
        result = popular.sample(n=sample_count, random_state=None).reset_index(drop=True)
        
        # 只保留需要显示的列
        needed_cols = ['MOVIE_ID', 'NAME', 'DOUBAN_SCORE', 'DOUBAN_VOTES', 'YEAR', 'DIRECTORS']
        available_cols = [col for col in needed_cols if col in result.columns]
        result = result[available_cols]
        
        logger.info(f"从 CSV 中获取 {sample_count} 部热门电影 (评分>={min_score}, 评分人数>={min_votes})")
        return result
    
    except Exception as e:
        logger.error(f"获取热门电影时出错: {e}")
        return pd.DataFrame()