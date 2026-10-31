# CPU LLM API в Kubernetes

OpenAI-совместимый API модели `HuggingFaceTB/SmolLM2-135M-Instruct` на vLLM 0.27.0 CPU. Контейнер запускается от UID/GID 10001, скачивает веса в PVC при первом старте и повторно использует их локально. Запросы требуют Bearer-токен; аудит сохраняет метаданные без содержимого диалога.

## Стенд и настройки

Проверено вручную на Ubuntu 24.04 в WSL2, Intel Core i3-10110U (AVX2, 2 ядра / 4 потока), 8 ГБ физической памяти, Docker Engine и Minikube с Docker-драйвером. WSL выделено 5 ГБ, Minikube — 4 ГБ и 2 CPU. При работе кластера отдельный Docker-контейнер модели остановлен.

| Параметр | Значение |
|---|---|
| Образ | `llm-cpu:0.4` |
| Python / vLLM / PyTorch | 3.12 / 0.27.0 CPU / 2.13.0+cpu |
| Модель | SmolLM2-135M-Instruct, около 135 млн параметров |
| Ревизия модели | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Тип данных / контекст / параллелизм | float32 / 512 токенов / 1 последовательность |
| KV-кэш | 134217728 байт (128 МиБ) |
| Память pod | request 2 GiB / limit 3 GiB |
| Хранилище весов | PVC 2 GiB, ReadWriteOnce |

Модель небольшая, преимущественно англоязычная; выбрана для запуска на CPU с ограниченной памятью. Контекст включает вход и генерируемый ответ. После одного короткого запроса Docker показывал 2.387 GiB при лимите 3 GiB; это не измерение пиковой памяти или нагрузочный тест.

## Сборка и запуск

Нужны Docker Engine с доступом текущего пользователя, Minikube, Python 3 и интернет для сборки и первого скачивания весов. Команды выполняются в Linux из корня репозитория. Отдельный kubectl не требуется. При работе в WSL необходимо заранее проверить доступ к Docker, репозиториям пакетов и Hugging Face из Ubuntu.

```bash
docker build -t llm-cpu:0.4 .
minikube start -p llm-dev --driver=docker --cpus=2 --memory=4096
minikube -p llm-dev image load llm-cpu:0.4
minikube -p llm-dev image ls
```

Импорт образа объёмом несколько гигабайт может занять несколько минут. Deployment использует `imagePullPolicy: Never`, поэтому образ должен быть загружен в этот профиль Minikube.

`requirements.txt` содержит точные версии прямых и транзитивных Python-зависимостей из рабочего окружения. vLLM установлен из CPU wheel релиза 0.27.0. Dockerfile устанавливает список с `--no-deps` и проверяет согласованность через `pip check`. Базовый образ Ubuntu и пакеты apt не закреплены по digest/снимку: побитовая воспроизводимость всей ОС не заявляется.

Создать локальный `.env` с новым случайным API-токеном; команда откажется перезаписывать существующий файл:

```bash
python3 - <<'PY'
import os
import secrets
fd = os.open('.env', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as f:
    f.write('VLLM_API_KEY=' + secrets.token_hex(32) + '\nHF_TOKEN=\n')
print('Created .env')
PY
python3 create_secret.py --profile llm-dev
minikube -p llm-dev kubectl -- apply -k k8s
minikube -p llm-dev kubectl -- -n llm-demo rollout status deployment/llm-api --timeout=20m
minikube -p llm-dev kubectl -- -n llm-demo get pods,pvc,service
```

Для публичной модели `HF_TOKEN` может оставаться пустым. Для закрытой модели указать в `.env` токен с доступом к её репозиторию и повторить `create_secret.py`. Значения передаются в Secret `llm-secrets` через stdin и далее в контейнер через `secretKeyRef`. `.env` исключён из Git; в репозитории есть только `.env.example`. После изменения Secret уже работающий pod нужно перезапустить.

Проверить пользователя контейнера и выполнить запрос внутри pod:

```bash
minikube -p llm-dev kubectl -- -n llm-demo exec deployment/llm-api -c inference -- id
minikube -p llm-dev kubectl -- -n llm-demo exec -i deployment/llm-api -c inference -- python - < check_api_inside.py
```

Ожидается UID 10001, HTTP 200 и ответ о столице Франции. Скрипт проверки печатает ответ в терминал клиента; это не серверный журнал.

## Аутентификация и журналы

В отдельном терминале оставить работающим проброс порта:

```bash
minikube -p llm-dev kubectl -- -n llm-demo port-forward service/llm-api 8000:8000 --address=127.0.0.1
```

Из корня репозитория с `.env`, соответствующим Secret:

```bash
python3 verify_logging.py --kubernetes
```

Проверка выполняет запросы без токена и с неверным токеном (401), обычный и потоковый запросы с верным токеном (200), запрос с некорректными сообщениями (400/422). Затем читает журналы pod, проверяет аудит этих запросов и отсутствие тестовых маркеров, токена и проверяемых фрагментов ответов. Неожиданные тела ответов и необработанные журналы скрипт не печатает.

`privacy_logging.py` содержит ASGI middleware, который не читает тело запроса, заголовки или содержимое ответа. Он записывает `request_received` / `request_finished`, случайный серверный ID, разрешённые метод и категорию пути, статус, длительность и исход запроса. Query string и неизвестные пути не сохраняются. Успешный старт отмечается событием `server_ready`.

`logging.json` подключает JSON-форматтер: для сторонних Python-логгеров свободный текст, аргументы и traceback подавляются, сохраняются только ограниченные диагностические метаданные. Логирование промптов и стандартный access log vLLM отключены. Такая настройка уменьшает полезность диагностики: текст исключений не виден.

Ограничение: форматтер контролирует Python logging, но не произвольные `print`, вывод нативных библиотек и аварийные дампы. Успешные проверки относятся к выполненным сценариям и не доказывают отсутствие утечки при любом сбое. Здесь нет внешнего сборщика или долговременного хранения журналов.

## PVC и повторный запуск без скачивания

`download_model.py` сохраняет safetensors, конфигурацию и токенизатор в `/models` по закреплённой ревизии. Блокировка предотвращает конкурирующие загрузки. После успешной подготовки атомарно создаётся `.ready.json` с SHA-256 файлов. Эти суммы обнаруживают последующее повреждение, но не являются независимой подписью издателя.

При следующем запуске файлы проверяются локально; загрузчик Hugging Face не вызывается. Повреждённый готовый кэш вызывает ошибку без скрытого повторного скачивания. Незавершённая первая загрузка может продолжиться при следующем старте с интернетом. После подготовки `start_server.py` удаляет HF-токен из окружения, включает offline-режим библиотек и запускает vLLM с локальным путём к модели. Статистика использования vLLM отключена.

После успешного первого старта проверить повторное использование:

```bash
minikube -p llm-dev kubectl -- -n llm-demo set env deployment/llm-api HF_HUB_OFFLINE=1
minikube -p llm-dev kubectl -- -n llm-demo rollout status deployment/llm-api --timeout=10m
minikube -p llm-dev kubectl -- -n llm-demo logs deployment/llm-api -c inference | grep model_ready
minikube -p llm-dev kubectl -- -n llm-demo exec -i deployment/llm-api -c inference -- python - < check_api_inside.py
```

Ожидаются `reused: true` и HTTP 200. Это проверка offline-режима приложения; сеть pod этой командой не блокируется. Для возвращения исходного окружения:

```bash
minikube -p llm-dev kubectl -- -n llm-demo set env deployment/llm-api HF_HUB_OFFLINE-
minikube -p llm-dev kubectl -- -n llm-demo rollout status deployment/llm-api --timeout=10m
```

В Docker повторный запуск также проверялся с `--network none` после заполнения volume. Для отдельного повторения вне кластера сначала остановить кластер/освободить память; затем заполнить собственный volume:

```bash
docker run --rm --name llm-api --memory=3g --shm-size=256m \
  --env-file .env -v llm-models:/models llm-cpu:0.4
```

После `server_ready` остановить контейнер из другого терминала и запустить с тем же volume без сети:

```bash
docker stop --time 60 llm-api
docker run --rm --name llm-api --network none --memory=3g --shm-size=256m \
  --env-file .env -e HF_HUB_OFFLINE=1 -e VLLM_HOST_IP=127.0.0.1 \
  -v llm-models:/models llm-cpu:0.4
```

После `server_ready` в другом терминале: `docker exec -i llm-api python - < check_api_inside.py`. Пустой volume впервые требует интернет. Docker volume и Kubernetes PVC — отдельные хранилища.

## Изменение конфигурации

Изменить `k8s/configmap.yaml`, применить и перезапустить pod:

```bash
minikube -p llm-dev kubectl -- apply -k k8s
minikube -p llm-dev kubectl -- -n llm-demo rollout restart deployment/llm-api
minikube -p llm-dev kubectl -- -n llm-demo rollout status deployment/llm-api --timeout=20m
```

Переменные ConfigMap читаются при старте, поэтому применение ConfigMap само по себе не обновляет работающий процесс. Пересборка образа не нужна. Изменение модели/ревизии требует нового первоначального скачивания и достаточного места в PVC; большая модель может превысить лимит памяти. `KV_CACHE_MEMORY_BYTES` имеет приоритет в команде запуска; старый `VLLM_CPU_KVCACHE_SPACE` удаляется из окружения, чтобы он не переопределял CLI в vLLM 0.27.0.

Deployment использует одну реплику и `Recreate`, поэтому обновление сопровождается перерывом в обслуживании. Контейнер имеет read-only корневую ФС, отключённое повышение привилегий и сброшенные capabilities. Записываемые каталоги вынесены в PVC/emptyDir. Пробы `/health` проверяют запуск и работоспособность. API доступен через ClusterIP; внешний Ingress и TLS в стенде не настроены.

## Проверки и состав

```bash
python3 -m unittest -v test_privacy_logging test_download_model
minikube -p llm-dev kubectl -- apply --dry-run=server -k k8s
```

17 модульных тестов проверяют фильтрацию логов, ASGI-аудит, холодную/повторную загрузку, повреждение кэша, прерванную загрузку, пути файлов и параметры запуска. Для них не нужны веса или установленный vLLM. Server dry-run требует доступного кластера и созданного namespace.

Ручные проверки на стенде: ответ модели, non-root UID, авторизация, аудит обычных/ошибочных/потоковых запросов, повторное использование модели и запуск Docker без сети. Нагрузочный тест с TTFT, latency и throughput не выполнялся.

| Файлы | Назначение |
|---|---|
| `Dockerfile`, `requirements.txt`, `.dockerignore` | Сборка CPU-образа и закрепление Python-зависимостей |
| `download_model.py`, `start_server.py` | Подготовка PVC и запуск vLLM |
| `privacy_logging.py`, `logging.json` | Аудит и фильтрация Python-журналов |
| `k8s/` | Namespace, ConfigMap, PVC, Deployment, Service, Kustomize |
| `create_secret.py`, `.env.example` | Создание Secret из локальных значений |
| `check_api_inside.py`, `verify_logging.py` | Проверки работающего API |
| `test_privacy_logging.py`, `test_download_model.py` | Модульные тесты |

Реализован дополнительный пункт с Kustomize. Отдельный initContainer не реализован: загрузка происходит в entrypoint основного non-root контейнера. Удаление namespace/PVC или профиля Minikube может уничтожить сохранённые веса; обычный rollout их сохраняет.
