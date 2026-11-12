# BULFM — LFM3 Browser RL on DGX Spark

Локальное дообучение **LiquidAI/LFM3-1.7B** управлению браузером с помощью **GRPO + LoRA**. Модель видит accessibility tree страницы, выдаёт действие BrowserGym, получает настоящую награду MiniWoB и обновляет адаптеры. Данные генерируются взаимодействием со средой: скачивать обучающий датасет не нужно.

Терминал с `/`-командами, историей и автодополнением напоминает Hermes. Дашборд показывает реальные метрики работающего процесса, обновляется каждые 2 секунды и не требует аккаунтов.

## Быстрый запуск на Spark

Используйте Python 3.12 с **уже работающим NVIDIA CUDA PyTorch 2.14.1–2.15.x**, поддерживающим GB10. Установщик проверяет bf16 matmul и сохраняет этот torch. На ARM нельзя без проверки ставить обычный CPU wheel PyTorch. Нужны `git`, `curl` или `wget`, доступ к GitHub, PyPI и Hugging Face. Установщик сам найдёт `uv` в PATH, `~/.local/bin` или `~/.cargo/bin`; при отсутствии установит его в `.tools/uv` без sudo и изменения профиля оболочки.

```bash
git clone https://github.com/crystalcali/bulfm.git
cd bulfm
# Если CUDA torch находится в другом Python:
# export BULFM_PYTHON=/path/to/cuda/python
bash scripts/setup_spark.sh
./start.sh
```

В терминале:

```text
/doctor
/dashboard
/train configs/spark-smoke.yaml
/status
```

После успешного smoke-запуска:

```text
/train configs/spark.yaml
```

`setup_spark.sh` создаёт `.venv` с доступом к пакетам выбранного Python, устанавливает приложение, Chromium и фиксированную версию страниц MiniWoB. Если Playwright сообщает об отсутствующих системных библиотеках, установите их администратором: `.venv/bin/python -m playwright install-deps chromium`. Скрипт сам не выполняет `sudo` и не останавливает ваши сервисы. Перед обучением освободите память от ненужных inference-серверов самостоятельно.

Если GitHub на Spark недоступен, склонируйте проект и MiniWoB на другой машине и перенесите каталоги. Можно указать `MINIWOB_URL=file:///absolute/path/miniwob-plusplus/miniwob/html/miniwob/` или HTTP URL страниц. Обычного Hugging Face токена для публичной модели не требуется; ключи не зашиты в код.

## Онлайн-дашборд

После `/dashboard` откройте `http://127.0.0.1:7860`. Для доступа с личного компьютера:

```bash
ssh -L 7860:127.0.0.1:7860 userdgx1@SPARK_IP
```

Затем откройте `http://127.0.0.1:7860` на своём компьютере. Это дашборд в реальном времени на вашем Spark, без публикации в облаке.

Отдельный запуск сервера:

```bash
./start.sh dashboard --host 127.0.0.1 --port 7860
```

Показатели: reward, success rate, GRPO loss, время шага, action tokens/s, learning rate, grad norm, пиковая зарезервированная GPU-память, RAM, доля групп с одинаковыми наградами, доля некорректных действий, оценка на отложенных сидах. На Spark CPU/GPU используют общую память; GPU peak и RAM нельзя складывать как независимые объёмы. GRPO loss не обязан монотонно падать: качество оценивайте по held-out success rate.

## Команды

| Команда | Действие |
|---|---|
| `/train [CONFIG]` | Новое обучение в отдельном процессе |
| `/status [RUN_ID]` | Статус, фаза, шаг и последние метрики |
| `/logs RUN_ID` | Последние 50 строк лога |
| `/stop RUN_ID` | Остановить после текущего optimizer step с сохранением |
| `/resume RUN_ID` | Восстановить веса, optimizer и RNG из последней контрольной точки |
| `/eval RUN_ID` | Отдельная оценка сохранённого LoRA-адаптера |
| `/config [FILE]` | Показать проверенные настройки |
| `/dashboard` | Запустить веб-дашборд |
| `/doctor` | Проверить зависимости, CUDA, bf16, Chromium и страницы |
| `/exit` | Закрыть интерфейс; обучение продолжится |

Те же команды работают без интерактивного интерфейса: `./start.sh train configs/spark.yaml`, `./start.sh status`. `--runs /absolute/path` перед командой меняет каталог запусков. Это интерфейс управления обучением, а не подключённый к LLM чат-агент.

## Эксперименты

| Конфигурация | Назначение |
|---|---|
| `configs/spark-smoke.yaml` | 2 optimizer step, группа 2, до 2 действий, проверка полного цикла |
| `configs/spark.yaml` | click-test, 100 шагов, группа 4, до 6 действий, LoRA rank 16 |
| `configs/spark-book-flight.yaml` | book-flight, 300 шагов, до 20 действий, контекст 4096 |

Меняйте `task`, `steps`, `group_size`, `groups_per_step`, `episode_steps`, `context_tokens`, `completion_tokens`, `learning_rate`, `beta`, LoRA-параметры и частоту evaluation в YAML. Параметр `steps` — optimizer steps; `episode_steps` — действия внутри эпизода. Каждый шаг собирает `group_size × groups_per_step` эпизодов. Начальная задача намеренно простая; успешный click-test ещё не означает умение работать с произвольными сайтами.

По умолчанию LoRA использует LFM3-модули `q_proj, k_proj, v_proj, o_proj, in_proj, out_proj, w1, w2, w3`, как в официальном LFM3 GRPO notebook. Для полного fine-tuning задайте `lora: false`, `beta: 0`; расход памяти выше. Ручная `/eval` сейчас поддерживает LoRA; автоматическая оценка до/после работает и при полном обучении.

Генерация идёт через Transformers в том же процессе, по одному эпизоду. `use_cache=False` и eager attention выбраны для простоты совместимости гибридных операторов и GB10. Это консервативный старт, а не максимальная скорость. Нет vLLM, Modal, отдельной копии модели-генератора, FlashAttention или обязательных внешних трекеров.

## Как устроено обучение

Основа — [Liquid4All browser-control](https://github.com/Liquid4All/cookbook/tree/3d0cbcb7dc40b690e3e4996c5b78e4d3020f8232/examples/browser-control) и [официальный LFM3 GRPO notebook](https://github.com/Liquid4All/cookbook/blob/3d0cbcb7dc40b690e3e4996c5b78e4d3020f8232/finetuning/notebooks/lfm3_grpo_with_trl.ipynb). Реализация здесь самостоятельная: сохраняет задачу и алгоритм, но не является побитовой копией эксперимента.

1. BrowserGym локально запускает MiniWoB в headless Chromium. Вместо OpenEnv HTTP-сервера используется прямой Python API той же среды.
2. Все эпизоды одной GRPO-группы получают один seed состояния среды. Разные группы получают разные seeds.
3. LFM3 видит цель, предыдущие действия, ошибку и актуальный accessibility tree. Контекст ограничивается по токенам, сохраняя системную инструкцию и цель.
4. Веса фиксированы на время сбора группы; модель сэмплирует действия с temperature=1, без top-k/top-p отсечения. Награда — сумма фактических наград среды за эпизод.
5. Advantage — `(R - mean(group)) / (std(group) + 1e-4)`. Используется clipped GRPO surrogate с сохранёнными old log probabilities и опциональным KL к исходной модели через отключение LoRA. Нормировка — среднее по action-токенам эпизода, затем по группе. Один on-policy проход на optimizer step.
6. Каждый ход хранит собственные prompt/action tokens. Градиент идёт только по action-токенам; наблюдения среды не склеиваются в искусственное completion. Нет critic-модели.
7. Evaluation использует greedy actions и неизменные seeds из другого диапазона до/после и с заданным интервалом. Метрики success/reward сохраняются отдельно от train.

Группы с одинаковой наградой дают нулевой policy advantage. Если `zero_variance_groups` постоянно 100%, задача может быть уже решена либо слишком сложна для случайного исследования; увеличьте группу или смените задачу. Выходы с неправильным синтаксисом превращаются в `noop()` и сохраняются в traces; инфраструктурные ошибки завершают запуск с явным `failed`, а не маскируются под нулевую награду.

## Файлы запуска и восстановление

```text
runs/RUN_ID/
  config.yaml                 зафиксированные настройки
  runtime.json                версии, GPU, trainable params, resolved model commit
  state.json                  статус и прогресс
  worker.log                  stdout/stderr
  metrics.jsonl               train/eval события
  episodes.jsonl              реальные наблюдения, ответы, действия и награды train
  evaluation-before.json      held-out baseline
  evaluation-after.json       held-out after
  latest.json                 последняя полностью сохранённая контрольная точка
  checkpoints/step-000010/     модель/LoRA, tokenizer, optimizer и RNG
```

Сохранение идёт через временный каталог с атомарным переключением `latest.json`. Хранятся последние 2 точки по умолчанию. `/resume` берёт настройки запуска и последнюю целую точку, восстанавливает optimizer/RNG и продолжает со следующего шага. После аварийного завершения работа с последней точки может повториться; точное побитовое совпадение GPU/браузера не гарантируется. После перезагрузки Spark повторно запустите терминал и `/resume RUN_ID`.

Не запускайте два обучения одновременно на одном Spark без оценки памяти. На один RUN_ID действует файловая блокировка. `/stop` проверяется между optimizer steps; длинный шаг нужно дождаться. Принудительный kill оставляет только ранее сохранённые точки.

## Проверки разработчика

```bash
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

CPU-тесты проверяют GRPO-градиенты, shift log-probability, изоляцию токенов, восстановление optimizer/RNG, управление запуском, API дашборда и полный цикл worker на маленькой модели с тестовой средой. Реальную среду без модели проверяет `./start.sh env-check`; она выполняет валидный клик и проверяет положительную награду. Для проверки именно LFM3 + CUDA нужен `spark-smoke.yaml` на Spark. Успех CPU-тестов не подтверждает качество обученной LFM3 или скорость на GB10.

Источники: [BrowserGym setup](https://github.com/ServiceNow/BrowserGym/tree/main/browsergym/miniwob), [LFM3](https://huggingface.co/LiquidAI/LFM3-1.7B). Лицензия весов и условия компонентов определяются их авторами.
