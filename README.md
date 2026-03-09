# Capital Predictor

Streamlit-приложение для инференса бинарной линейной модели (`robust`/`sklearn`) с интерактивными графиками и SHAP-объяснениями.

## Что внутри

- `model.py`:
  - `Predictor(weights_path="weights.npz", model_type="robust")`
  - `predict(data)` -> метки `-1/+1`
  - `decision_function(data)` -> raw score
  - `shap_values(data, observation_index, baseline)` -> SHAP для линейной модели
- `app.py`:
  - UI на Streamlit
  - кнопка старта с сообщением `бабки` и салютом
  - интерактивные графики Plotly
  - SHAP-анализ для выбранного наблюдения
  - обработка ошибок ввода/весов
- `Dockerfile` для контейнерного запуска

## Требования

- Python `3.9+`
- зависимости из `requirements.txt`

## Быстрый запуск локально

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Открой в браузере: `http://localhost:8501`

## Запуск через Docker

```bash
docker build -t capital-predictor .
docker run --rm -p 8501:8501 capital-predictor
```

Открой: `http://localhost:8501`

## Формат файла весов

`weights.npz` должен содержать ключи:

- `robust_w`, `robust_b`
- `sklearn_w`, `sklearn_b`
- `scaler_min`, `scaler_scale`

Препроцессинг в модели:

```text
X_scaled = X * scaler_scale + scaler_min
score = X_scaled @ w + b
```

## Демо-данные

- `sample_batch.txt` — пример batch для вставки в приложение
- `weights.npz` — демо-веса (сгенерированные, не обученная production-модель)
