# Depth Anything V2 Fine-Tuning on NYU Depth V2

fine-tuning предобученной модели Depth Anything V2 на наборе данных NYU Depth V2.

## Структура проекта

```text
data/
  raw/
    nyu_depth_v2_labeled.mat
  processed/
checkpoints/
outputs/
prepare_data.py
dataset.py
model.py
losses.py
metrics.py
train.py
test.py
predict.py
requirements.txt
README.md
```

## назначение файлов

* `data/raw/nyu_depth_v2_labeled.mat` —  исходный файл датасета NYU Depth V2.
* `data/processed/` будет хранить извлечённые и подготовленные RGB-изображения и карты глубины.
* `checkpoints/` хранит веса дообученной модели.
* `outputs/` хранит предсказанные карты глубины и визуализации.
* `prepare_data.py` подготавливает NYU Depth V2 для обучения.
* `dataset.py` определяет датасет PyTorch для пар RGB-изображений и карт глубины.
* `model.py` загружает предобученную модель Depth Anything V2 и подготавливает её к дообучению.
* `losses.py` содержит функции потерь для оценки глубины.
* `metrics.py` содержит метрики качества предсказания глубины.
* `train.py` дообучает предобученную модель на NYU Depth V2.
* `test.py` оценивает сохранённый чекпоинт.
* `predict.py` генерирует карту глубины по одному RGB-изображению.

## настройка окружения

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## внешний код Depth Anything V2

Официальный репозиторий должен быть расположен так:

```bash
mkdir -p external
git clone https://github.com/DepthAnything/Depth-Anything-V2.git external/Depth-Anything-V2
```

Для metric-depth режима проект импортирует код из:

```text
external/Depth-Anything-V2/metric_depth/
```

Далее поставить дополнительные зависимости внешнего проекта:

```bash
pip install -r external/Depth-Anything-V2/metric_depth/requirements.txt
```

Быстрая диагностика окружения:

```bash
python check_environment.py --config configs/strategies/train_vitb_head_only_mps.yaml
```


## pretrained checkpoint



Для основных ViT-B экспериментов скачайте indoor Hypersim Base checkpoint и положите в папку:

```bash
mkdir -p checkpoints/pretrained
```

```bash
curl -L \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth?download=true" \
  -o checkpoints/pretrained/depth_anything_v2_metric_hypersim_vitb.pth
```


## Датасет NYU Depth V2 ([link](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html))



Перед обучением нужно один раз преобразовать исходный файл
`data/raw/nyu_depth_v2_labeled.mat` в набор PNG-изображений, NumPy depth maps и
split-файлы:

```bash
source .venv/bin/activate
python prepare_data.py
```

Скрипт создаёт:

```text
data/processed/images/*.png
data/processed/depths/*.npy
data/processed/train.txt
data/processed/val.txt
data/processed/test.txt
data/processed/dataset_stats.json
```


## Fine-tuning strategy experiment

В `train.py` поддерживаются пять стратегий адаптации предобученной Depth Anything V2:

| Стратегия | `finetune_strategy` | Что обучается |
|---|---|---|
| Full fine-tuning | `full` | все параметры модели |
| Head-only fine-tuning | `head_only` | только depth head / decoder |
| Partial fine-tuning | `partial` | depth head / decoder и последние блоки encoder |
| Layer-wise learning rate | `layerwise_lr` | вся модель, но encoder получает меньший LR |
| LoRA fine-tuning | `lora` | depth head / decoder и LoRA-адаптеры в attention encoder |

Основные параметры конфига:

```yaml
device: auto          # auto, cuda, mps, cpu
finetune_strategy: partial
unfreeze_last_blocks: 2
encoder_learning_rate: 0.0000005
head_learning_rate: 0.000005
```

## Скрипты
 
```bash
python train.py --config configs/strategies/train_vitb_lora_mps.yaml
```

После обучения оценка лучшего checkpoint:

```bash
python test.py \
  --config configs/strategies/train_vitb_lora_mps.yaml \
  --checkpoint outputs/experiments/vitb_lora_mps/checkpoints/best_absrel.pth
```

