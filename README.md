# LogSight

Компактный репозиторий для 3 рабочих направлений:

1. `clients` — клиент и анализ логов через YandexGPT
2. `training` — код дообучения/инференса LogBERT
3. `parsing` — подготовка и нормализация логов для обучения

## Структура

```text
LogSight/
├── clients/
│   ├── yandex_gpt_client.py
│   └── log_analyzer.py
├── training/
│   ├── bert_pytorch/              # ядро модели и train/predict логика
│   ├── inference/                 # production-style предиктор
│   └── logbert_inference_client.py
├── parsing/
│   ├── drain.py                   # Drain parser
│   └── dataset/                   # пайплайны подготовки датасета
├── scripts/
│   ├── logsight.py                # CLI для YandexGPT анализа
│   ├── prepare_dataset.py         # подготовка structured csv
│   └── merge_datasets.py          # merge DVWA + Vulhab
├── configs/
│   └── log_analysis_prompt.yaml
├── notebooks/
│   └── finetune_logbert_web_lora.ipynb
├── logs/
├── output/
├── weights/
└── main.py                        # entrypoint инференса LogBERT
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для GPU-сервера установите CUDA-сборку PyTorch под вашу версию CUDA (пример для CUDA 12.4):

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

## Запуск

### 1) YandexGPT client

```bash
python scripts/logsight.py -f logs/your_log_file.log
# опционально: -c configs/log_analysis_prompt.yaml
```

Нужны переменные окружения:

- `YANDEX_CLOUD_API_KEY`
- `YANDEX_CLOUD_FOLDER`

### 2) Парсинг логов

```bash
python scripts/prepare_dataset.py --target vulhab --logs-dir logs --output-dir output
python scripts/prepare_dataset.py --target dvwa --logs-dir logs --output-dir output
python scripts/merge_datasets.py --output-dir output/dvwa_vulhab
```

### 3) Инференс / дообученная модель

```bash
python main.py --logs logs/vulhub_labeled_interleaved.jsonl --state weights/best_bert.pth --vocab weights/vocab.pkl
```

Ноутбук с LoRA-дообучением:

- `notebooks/finetune_logbert_web_lora.ipynb`

Запускать Jupyter лучше из корня репозитория:

```bash
jupyter lab
```
