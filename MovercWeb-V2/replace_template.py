#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Replace movie_detail.html with Douban-style layout"""

content = '''<!-- templates/movie_detail.html - Douban Style Layout -->
{% extends "base.html" %}

{% block title %}{{ movie.NAME or "电影详情" }}{% endblock %}

{% block content %}
<div class="container movie-detail-container mt-4">
    <h1 class="movie-title">{{ movie.NAME }} 
        {% if movie.YEAR %}<small class="text-muted">({{ movie.YEAR }})</small>{% endif %}
    </h1>

    <div class="row">
        <div class="col-md-3">
            {% if movie.COVER %}
                <img src="{{ movie.COVER }}" alt="{{ movie.NAME }}" class="img-fluid rounded shadow-sm" style="max-width: 100%; max-height: 500px; object-fit: cover;">
            {% else %}
                <div class="bg-light d-flex align-items-center justify-content-center rounded" style="width: 100%; height: 380px; border: 1px solid #ddd;">
                    <span class="text-muted">海报暂无</span>
                </div>
            {% endif %}
        </div>

        <div class="col-md-9">
            <div class="rating-section mb-4">
                {% if movie.DOUBAN_SCORE %}
                    <div class="rating-score">
                        <span class="score-number">{{ movie.DOUBAN_SCORE }}</span>
                        <span class="score-unit">/ 10</span>
                    </div>
                {% else %}
                    <p class="text-muted">暂无评分</p>
                {% endif %}
            </div>

            <div class="info-section">
                {% set director = movie.get("DIRECTORS") or movie.get("directors") or movie.get("DIRECTOR") or "未知" %}
                {% set actors = movie.get("ACTORS") or movie.get("actors") or movie.get("CAST") or "未知" %}
                {% set genres = movie.get("GENRES") or movie.get("genres") or movie.get("LABEL") or "未知" %}

                <p class="info-row">
                    <span class="info-label">导演</span>
                    <span class="info-value">{{ director }}</span>
                </p>

                {% if actors != "未知" %}
                <p class="info-row">
                    <span class="info-label">主演</span>
                    <span class="info-value">{{ actors }}</span>
                </p>
                {% endif %}

                {% if genres != "未知" %}
                <p class="info-row">
                    <span class="info-label">类型</span>
                    <span class="info-value">{{ genres }}</span>
                </p>
                {% endif %}

                {% if movie.get("REGIONS") or movie.get("countries") %}
                <p class="info-row">
                    <span class="info-label">国家</span>
                    <span class="info-value">{{ movie.get("REGIONS") or movie.get("countries") }}</span>
                </p>
                {% endif %}
            </div>

            {% if current_user.is_authenticated %}
            <div class="action-buttons mt-4">
                <button id="like-btn" class="btn btn-custom me-2" data-action="like"
                        style="{% if user_has_liked %}background-color: #10b981; color: white; border: none;{% else %}background-color: #f0f0f0; color: #333; border: 1px solid #ddd;{% endif %}">
                    <span class="button-icon">{% if user_has_liked %}✓{% else %}👍{% endif %}</span>
                    <span class="button-text">{% if user_has_liked %}已喜欢{% else %}喜欢{% endif %}</span>
                </button>
                <button id="dislike-btn" class="btn btn-custom me-2" data-action="dislike"
                        style="{% if user_has_disliked %}background-color: #ef4444; color: white; border: none;{% else %}background-color: #f0f0f0; color: #333; border: 1px solid #ddd;{% endif %}">
                    <span class="button-icon">{% if user_has_disliked %}✗{% else %}👎{% endif %}</span>
                    <span class="button-text">{% if user_has_disliked %}已不喜欢{% else %}不喜欢{% endif %}</span>
                </button>
            </div>
            {% else %}
                <p class="text-muted mt-3"><a href="{{ url_for('login') }}">登录</a> 后可以标记这部电影</p>
            {% endif %}
        </div>
    </div>

    <div class="row mt-5">
        <div class="col-md-12">
            <h3 class="section-title">简介</h3>
            <div class="synopsis-section">
                <p style="line-height: 1.8; color: #555; white-space: pre-wrap;">
                    {{ movie.get("STORYLINE") or movie.get("INFO") or movie.get("PLOT_SUMMARY") or movie.get("DESCRIPTION") or "暂无简介" }}
                </p>
            </div>
        </div>
    </div>

    {% if comparison_tags %}
    <div class="row mt-5">
        <div class="col-md-12">
            <h3 class="section-title">评价排名</h3>
            <div class="comparison-tags">
                {% for tag in comparison_tags[:3] %}
                <span class="badge comparison-badge">{{ tag }}</span>
                {% endfor %}
            </div>
        </div>
    </div>
    {% endif %}
</div>

<script>
document.addEventListener('DOMContentLoaded', function () {
    const buttons = document.querySelectorAll('[data-action]');
    const movieId = "{{ movie.MOVIE_ID }}";
    const csrfToken = "{{ csrf_token() }}";

    buttons.forEach(button => {
        button.addEventListener('click', async function () {
            const action = this.dataset.action;
            const originalHTML = this.innerHTML;
            this.innerHTML = '...';
            this.disabled = true;

            try {
                const response = await fetch('{{ url_for("toggle_preference_optimized") }}', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': csrfToken
                    },
                    body: JSON.stringify({ movie_douban_id: movieId, action: action })
                });

                if (!response.ok) throw new Error('HTTP ' + response.status);

                const data = await response.json();
                if (data.success) {
                    updateButtonUI(data.new_status);
                } else {
                    alert('操作失败: ' + (data.error || '未知错误'));
                    this.innerHTML = originalHTML;
                }
            } catch (error) {
                console.error('Error:', error);
                alert('网络错误，请稍后重试');
                this.innerHTML = originalHTML;
            } finally {
                this.disabled = false;
            }
        });
    });

    function updateButtonUI(newStatus) {
        const likeBtn = document.getElementById('like-btn');
        const dislikeBtn = document.getElementById('dislike-btn');

        likeBtn.style.backgroundColor = '#f0f0f0';
        likeBtn.style.color = '#333';
        likeBtn.style.border = '1px solid #ddd';
        likeBtn.innerHTML = '<span class="button-icon">👍</span><span class="button-text">喜欢</span>';

        dislikeBtn.style.backgroundColor = '#f0f0f0';
        dislikeBtn.style.color = '#333';
        dislikeBtn.style.border = '1px solid #ddd';
        dislikeBtn.innerHTML = '<span class="button-icon">👎</span><span class="button-text">不喜欢</span>';

        if (newStatus === 'liked') {
            likeBtn.style.backgroundColor = '#10b981';
            likeBtn.style.color = 'white';
            likeBtn.style.border = 'none';
            likeBtn.innerHTML = '<span class="button-icon">✓</span><span class="button-text">已喜欢</span>';
        } else if (newStatus === 'disliked') {
            dislikeBtn.style.backgroundColor = '#ef4444';
            dislikeBtn.style.color = 'white';
            dislikeBtn.style.border = 'none';
            dislikeBtn.innerHTML = '<span class="button-icon">✗</span><span class="button-text">已不喜欢</span>';
        }
    }
});
</script>

<style>
    .movie-detail-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .movie-title { font-size: 2rem; font-weight: bold; color: #333; margin-bottom: 2rem; border-bottom: 2px solid #f0f0f0; padding-bottom: 1rem; }
    .rating-section { display: flex; align-items: flex-start; padding-bottom: 1rem; border-bottom: 1px solid #f0f0f0; }
    .rating-score { display: flex; align-items: baseline; gap: 4px; }
    .score-number { font-size: 3rem; font-weight: bold; color: #f59e0b; line-height: 1; }
    .score-unit { font-size: 0.9rem; color: #999; }
    .info-section { margin-top: 1.5rem; }
    .info-row { display: flex; margin-bottom: 0.8rem; font-size: 0.95rem; line-height: 1.6; }
    .info-label { display: inline-block; min-width: 70px; font-weight: bold; color: #333; margin-right: 1rem; }
    .info-value { flex: 1; color: #666; word-break: break-word; }
    .action-buttons { display: flex; gap: 10px; flex-wrap: wrap; }
    .btn-custom { padding: 10px 24px; border-radius: 4px; font-weight: 500; transition: all 0.3s ease; display: flex; align-items: center; gap: 6px; cursor: pointer; }
    .btn-custom:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); }
    .btn-custom:disabled { opacity: 0.6; cursor: not-allowed; }
    .button-icon { font-size: 1.1em; }
    .button-text { font-size: 0.95rem; }
    .section-title { font-size: 1.3rem; font-weight: bold; color: #333; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #f0f0f0; }
    .synopsis-section { background-color: #fafafa; padding: 1.5rem; border-radius: 4px; border-left: 4px solid #f59e0b; }
    .synopsis-section p { margin: 0; color: #666; }
    .comparison-tags { display: flex; flex-wrap: wrap; gap: 10px; }
    .comparison-badge { background-color: #3b82f6; color: white; padding: 8px 14px; font-size: 0.9em; border-radius: 20px; display: inline-block; font-weight: 500; }
    .comparison-badge:nth-child(2) { background-color: #8b5cf6; }
    .comparison-badge:nth-child(3) { background-color: #ec4899; }
    @media (max-width: 768px) {
        .movie-title { font-size: 1.5rem; }
        .score-number { font-size: 2rem; }
        .action-buttons { flex-direction: column; }
        .btn-custom { width: 100%; justify-content: center; }
        .info-row { flex-direction: column; }
        .info-label { min-width: auto; margin-bottom: 0.3rem; }
    }
</style>
{% endblock %}'''

with open('templates/movie_detail.html', 'w', encoding='utf-8') as f:
    f.write(content)
    
print('[INFO] 豆瓣风格 movie_detail.html 已成功更新!')
