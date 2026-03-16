# Parsing Pipelines

`parsing/dataset/` contains data-preparation pipelines for LogBERT training.

## Modules
- `loaders.py`: find dataset files and read JSON/JSONL/CSV/TXT into unified records.
- `line_parser.py`: parse raw log prefix fields (`Date`, `Time`, `AdminAddr`, etc.) and extract `Content`.
- `template.py`: normalize `Content` into `EventTemplate` and generate stable `EventId`.
- `builders.py`: build/save structured rows and template tables.
- `pipeline.py`: orchestration for dataset preparation and DVWA+Vulhab merge.
- `cli_prepare.py`: CLI entrypoint for one dataset.
- `cli_merge.py`: CLI entrypoint for merged dataset.

## CLI
```bash
python3 scripts/prepare_dataset.py --target vulhab
python3 scripts/prepare_dataset.py --target dvwa
python3 scripts/merge_datasets.py
```
