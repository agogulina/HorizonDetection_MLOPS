# HorizonDetection_MLOPS

Система обнаружения линии горизонта для БПЛА на основе компьютерного зрения и глубокого обучения.
Оценивает ориентацию аппарата (крен и тангаж) относительно горизонта как резервный источник
данных для систем стабилизации полётного контроллера.

**Стек:** TensorFlow/Keras (U-Net), FastAPI, Docker, DVC, Kubernetes (Minikube), Argo CD,
Prometheus + Grafana, TensorBoard.

**Репозиторий:** `https://github.com/agogulina/HorizonDetection_MLOPS`

---

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 1. Датасет и базовая модель

Датасет — кадры с камеры БПЛА (`dataset/images`) с масками (`dataset/masks`).
Базовая модель — U-Net (`src/models/unet.py`), веса в `checkpoints/model.keras`.
Все гиперпараметры — в `configs/train.yaml`.

```bash
# обучение
python main.py train --config configs/train.yaml

# инференс
python main.py predict --ckpt checkpoints/model.keras --input <папка_или_картинка>
```

## 2. Git + DVC

Версионирование кода — Git (conventional commits). Данные — DVC, хранилище — S3.

```bash
dvc pull            # скачать данные из хранилища
dvc add dataset     # зафиксировать новую версию данных
dvc push            # выгрузить данные в хранилище
dvc status -c       # проверить синхронизацию с хранилищем
```

## 3. Шаблонизация (Cookiecutter)

Структура проекта по шаблону cookiecutter-data-science: `src/`, `configs/`, `app/`, `monitoring/`.
Отдельных команд запуска не требует.

## 4. Трекинг экспериментов (TensorBoard)

```bash
tensorboard --logdir logs/
# открыть http://localhost:6006
```

## 5. CI/CD

CI — GitHub Actions (`.github/workflows/ci-cd.yml`): запускается на Pull Request в `main`,
ставит зависимости и прогоняет обучение как проверку пайплайна.
CD — через Argo CD (см. пункт 10). Смотреть прогоны: вкладка **Actions** на GitHub.

## 6. Сервис FastAPI + Docker

```bash
# собрать образ
docker build -t horizon-detection:v0.2 .

# локально через docker-compose (api + Prometheus + Grafana)
docker compose up --build
```

После старта:
- веб-UI: `http://localhost:8001`
- OpenAPI (Swagger): `http://localhost:8001/docs`
- метрики: `http://localhost:8001/metrics`

```bash
docker compose down   # остановить
```

## 7. Дрейф + мониторинг (Prometheus + Grafana)

Поднимаются вместе с сервисом через `docker compose up`:
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin / admin)

Статистика дрейфа отдаётся приложением: `GET http://localhost:8001/api/v1/drift/stats`.

## 8. Отчёт о дрейфе

```bash
python monitoring/drift/drift_report.py
# создаёт HTML-отчёт (Evidently) — открыть в браузере
```

## 9. Веб-UI

`app/static/index.html`. Открывается на `http://localhost:8001`.
Разделы: инференс, последние предсказания, дрейф данных, эксперименты, уведомления,
кнопка запуска переобучения.

## 10. CD через Argo CD (Minikube)

```bash
# 1. кластер + образ
minikube start
docker build -t horizon-detection:v0.2 .
minikube image load horizon-detection:v0.2

# 2. установка Argo CD
kubectl create namespace argocd
kubectl apply --server-side -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl get pods -n argocd          # ждать, пока все 1/1 Running

# 3. доступ к Argo CD (отдельное окно, не закрывать)
kubectl port-forward svc/argocd-server -n argocd 8080:443
# логин admin, пароль:
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo
```

Открыть `https://localhost:8080`. Создать Application: repo
`https://github.com/agogulina/HorizonDetection_MLOPS.git`, revision `main`, path `k8s`,
cluster `https://kubernetes.default.svc`, namespace `default` -> **SYNC**.

```bash
# проверить и открыть приложение
kubectl get pods                                       # ждать 1/1 Running
kubectl port-forward svc/horizon-detection 8001:80     # отдельное окно
# открыть http://localhost:8001
```



