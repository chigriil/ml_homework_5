"""Streamlit UI for inference, interactive charts and SHAP analysis."""

from __future__ import annotations

import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from model import Predictor

JOKES = [
    "DS-шутка дня: обещали 100К в наносек, но сначала попросили A/B-тест на 3 квартала.",
    "Собеседование DS: «Знаешь SHAP?» — «Да». «Тогда объясни, почему премии нет».",
    "100К в наносек получили все, кто смог убрать leakage до дедлайна в пятницу 18:59.",
    "Модель бьет SOTA, но бухгалтерия попросила сначала `pip install bonus`.",
    "Говорят, data-driven культура есть, просто колонка с зарплатой закрыта правами доступа.",
    "Если метрика выросла на 0.01, можно просить плюс 10К. Если в проде — плюс уважение.",
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            font-family: "Georgia", "Times New Roman", serif;
            color: #2b1d00;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 8% 12%, rgba(255, 228, 160, 0.55), transparent 35%),
                radial-gradient(circle at 92% 8%, rgba(214, 179, 104, 0.45), transparent 30%),
                linear-gradient(165deg, #fff9ec 0%, #f6e6c6 42%, #efd7a7 100%);
        }
        .main .block-container {
            max-width: 1180px;
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stForm"] {
            background: rgba(255, 252, 244, 0.88);
            border: 1px solid rgba(178, 141, 67, 0.45);
            border-radius: 16px;
            padding: 1rem 1rem 0.35rem 1rem;
            box-shadow: 0 8px 24px rgba(110, 80, 23, 0.12);
        }
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            border: none;
            border-radius: 999px;
            font-weight: 700;
            background: linear-gradient(90deg, #c9a24e 0%, #f0d78a 100%);
            color: #2f2300;
            box-shadow: 0 4px 14px rgba(120, 87, 22, 0.22);
        }
        div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
            background: linear-gradient(90deg, #d9b45f 0%, #f6de97 100%);
        }
        .money-card {
            background: rgba(255, 252, 244, 0.88);
            border: 1px solid rgba(178, 141, 67, 0.35);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin: 0.4rem 0 1rem 0;
        }
        .joke-box {
            background: rgba(255, 246, 223, 0.92);
            border-left: 5px solid #b9913a;
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            font-size: 1rem;
            line-height: 1.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_batch(raw_text: str) -> np.ndarray:
    """Parse multiline text into a 2D float array."""
    if not raw_text.strip():
        raise ValueError("Поле с данными пустое. Введите хотя бы одну строку признаков.")

    rows = []
    for line_idx, line in enumerate(raw_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = [token for token in re.split(r"[,\s;]+", line) if token]
        if not parts:
            continue

        try:
            row = [float(token) for token in parts]
        except ValueError as exc:
            raise ValueError(
                f"Не удалось распарсить строку {line_idx}. Используйте только числа."
            ) from exc

        rows.append(row)

    if not rows:
        raise ValueError("После разбора не найдено ни одной валидной строки с числами.")

    n_features = len(rows[0])
    for idx, row in enumerate(rows, start=1):
        if len(row) != n_features:
            raise ValueError(
                f"Разное количество признаков в строках. "
                f"Ожидалось {n_features}, строка {idx} содержит {len(row)}."
            )
    return np.asarray(rows, dtype=float)


def init_state() -> None:
    if "started" not in st.session_state:
        st.session_state.started = False
    if "just_started" not in st.session_state:
        st.session_state.just_started = False
    if "joke_idx" not in st.session_state:
        st.session_state.joke_idx = 0
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def score_histogram(scores: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=max(8, min(35, int(np.sqrt(len(scores)) * 3))),
            marker_color="#c89e43",
            opacity=0.9,
            name="Score",
        )
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="#8e6a25")
    fig.update_layout(
        title="Распределение decision score",
        xaxis_title="Score",
        yaxis_title="Количество",
        template="plotly_white",
        margin=dict(l=20, r=20, t=48, b=20),
    )
    return fig


def prediction_donut(predictions: np.ndarray) -> go.Figure:
    values, counts = np.unique(predictions, return_counts=True)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[f"class {int(v)}" for v in values],
                values=counts,
                hole=0.55,
                marker=dict(colors=["#9a7a2d", "#e1bb63"][: len(values)]),
            )
        ]
    )
    fig.update_layout(
        title="Баланс предсказанных классов",
        template="plotly_white",
        margin=dict(l=20, r=20, t=48, b=20),
    )
    return fig


def shap_bar_plot(shap_values: np.ndarray, top_k: int) -> tuple[go.Figure, np.ndarray]:
    top_idx = np.argsort(np.abs(shap_values))[::-1][:top_k]
    ordered = top_idx[::-1]
    y_names = [f"f{i}" for i in ordered]
    x_values = shap_values[ordered]
    colors = ["#0f9d58" if val >= 0 else "#c3473a" for val in x_values]

    fig = go.Figure(
        data=[
            go.Bar(
                x=x_values,
                y=y_names,
                orientation="h",
                marker=dict(color=colors),
            )
        ]
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="#7b7b7b")
    fig.update_layout(
        title=f"SHAP values (top-{top_k} признаков)",
        xaxis_title="Вклад в score",
        yaxis_title="Признак",
        template="plotly_white",
        margin=dict(l=20, r=20, t=48, b=20),
    )
    return fig, top_idx


def observation_vs_baseline_plot(
    observation: np.ndarray, baseline: np.ndarray, feature_idx: np.ndarray
) -> go.Figure:
    ordered = feature_idx
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[f"f{i}" for i in ordered],
            y=observation[ordered],
            mode="lines+markers",
            name="Observation",
            line=dict(color="#b2852f", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[f"f{i}" for i in ordered],
            y=baseline[ordered],
            mode="lines+markers",
            name="Baseline",
            line=dict(color="#5b6b8a", width=2, dash="dot"),
        )
    )
    fig.update_layout(
        title="Наблюдение против baseline (по top SHAP признакам)",
        xaxis_title="Признаки",
        yaxis_title="Значение",
        template="plotly_white",
        margin=dict(l=20, r=20, t=48, b=20),
    )
    return fig


st.set_page_config(page_title="Capital Predictor", layout="wide")
inject_styles()
init_state()

st.title("Capital Predictor Desk")
st.caption("Интерактивный инференс, визуализация и объяснения SHAP для конкретных наблюдений.")

if not st.session_state.started:
    st.markdown(
        "<div class='money-card'><b>Перед началом:</b> включите режим работы кнопкой ниже.</div>",
        unsafe_allow_html=True,
    )
    if st.button("Войти в режим капитала"):
        st.session_state.started = True
        st.session_state.just_started = True
        rerun_app()
    st.stop()

if st.session_state.just_started:
    st.success("бабки")
    st.balloons()
    st.session_state.just_started = False

left_joke, right_joke = st.columns([6, 1])
with left_joke:
    st.markdown(
        f"<div class='joke-box'>{JOKES[st.session_state.joke_idx]}</div>",
        unsafe_allow_html=True,
    )
with right_joke:
    if st.button("Новая шутка"):
        st.session_state.joke_idx = (st.session_state.joke_idx + 1) % len(JOKES)
        rerun_app()

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        weights_path = st.text_input("Путь к файлу весов (.npz)", value="weights.npz")
    with col2:
        model_type = st.selectbox("Тип модели", options=["robust", "sklearn"], index=0)

    data_text = st.text_area(
        "Batch-данные (2D): одна строка = один объект, значения через пробел/запятую/точку с запятой",
        height=210,
        placeholder="0.1, 0.2, 0.3\n0.4, 0.5, 0.6",
    )

    submitted = st.form_submit_button("Рассчитать")

if submitted:
    try:
        data = parse_batch(data_text)
        predictor = Predictor(weights_path=weights_path, model_type=model_type)
        predictions = predictor.predict(data)
        scores = predictor.decision_function(data)

        st.session_state.last_result = {
            "weights_path": weights_path,
            "model_type": model_type,
            "data": data,
            "predictions": predictions,
            "scores": scores,
        }
        st.session_state.joke_idx = (st.session_state.joke_idx + 1) % len(JOKES)
        st.success("Расчет выполнен успешно.")
    except FileNotFoundError:
        st.error(f"Файл весов не найден: `{weights_path}`.")
        st.session_state.last_result = None
    except PermissionError:
        st.error(f"Нет доступа к файлу весов: `{weights_path}`.")
        st.session_state.last_result = None
    except (ValueError, KeyError) as exc:
        st.error(f"Ошибка данных или модели: {exc}")
        st.session_state.last_result = None
    except Exception as exc:  # pragma: no cover
        st.error(f"Непредвиденная ошибка: {exc}")
        st.session_state.last_result = None

result = st.session_state.last_result
if result is not None:
    data = result["data"]
    predictions = result["predictions"]
    scores = result["scores"]
    n_samples, n_features = data.shape

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Объектов в batch", f"{n_samples}")
    with m2:
        st.metric("Признаков", f"{n_features}")
    with m3:
        st.metric("Средний score", f"{float(np.mean(scores)):.5f}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(score_histogram(scores), use_container_width=True)
    with c2:
        st.plotly_chart(prediction_donut(predictions), use_container_width=True)

    st.subheader("Предсказания")
    rows = {
        "sample_id": np.arange(n_samples, dtype=int),
        "score": np.round(scores, 6),
        "prediction": predictions.astype(int),
    }
    st.dataframe(rows)

    st.subheader("SHAP для конкретного наблюдения")
    st.caption(
        "Для линейной модели используем точное разложение SHAP по decision function."
    )

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        observation_index = st.slider(
            "Индекс наблюдения",
            min_value=0,
            max_value=n_samples - 1,
            value=0,
            step=1,
        )
    with ctrl2:
        baseline_mode = st.radio(
            "Baseline",
            options=["Среднее по batch", "Нулевой вектор"],
        )
    with ctrl3:
        top_k = st.slider(
            "Топ признаков по |SHAP|",
            min_value=1,
            max_value=min(40, n_features),
            value=min(15, n_features),
            step=1,
        )

    baseline = data.mean(axis=0) if baseline_mode == "Среднее по batch" else np.zeros(n_features)

    try:
        predictor = Predictor(
            weights_path=result["weights_path"], model_type=result["model_type"]
        )
        explanation = predictor.shap_values(
            data,
            observation_index=observation_index,
            baseline=baseline,
        )
        shap_vals = np.asarray(explanation["shap_values"], dtype=float)
        base_value = float(explanation["base_value"])
        score = float(explanation["score"])
        obs = np.asarray(explanation["observation"], dtype=float)
        base_vec = np.asarray(explanation["baseline"], dtype=float)

        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("Base value", f"{base_value:.6f}")
        with e2:
            st.metric("SHAP sum", f"{float(shap_vals.sum()):.6f}")
        with e3:
            st.metric("Final score", f"{score:.6f}")

        bar_fig, selected_idx = shap_bar_plot(shap_vals, top_k=top_k)
        st.plotly_chart(bar_fig, use_container_width=True)
        st.plotly_chart(
            observation_vs_baseline_plot(obs, base_vec, selected_idx),
            use_container_width=True,
        )
    except FileNotFoundError:
        st.error(
            "Не удалось пересчитать SHAP: файл весов больше недоступен. "
            "Укажите актуальный путь и нажмите «Рассчитать» снова."
        )
    except (ValueError, KeyError, RuntimeError) as exc:
        st.error(f"Ошибка при расчете SHAP: {exc}")
