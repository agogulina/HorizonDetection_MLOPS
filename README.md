# HorizonDetection_MLOPS

Система обнаружения линии горизонта для БПЛА на основе компьютерного зрения и глубокого обучения. <br>
Может быть использована для оценки ориентации аппарата (крен и тангаж) относительно горизонта Земли в качестве резервного/дополнительного источника данных для IMU-датчиков и систем стабилизации полетного контроллера.<br>
В дальнейшем планируется оптимизировать для запуска на одноплатных компьютерах.

## Архитеткура пайплайна

Система состоит из 4 последовательных этапов обработки видеопотока с фронтальной камеры:

1. Семантическая сегментация

Входной кадр обрабатывается U-Net. Классифицирует каждый пиксель как sky или land.

2. Выделение границы

Морфологические операции применяются к маскам для подавления шума и точного выделения линии перехода небо/земля.

3. Линейная/полномиальная регрессия

Метод наименьших квадратов аппроксимирует точки границы

4. Расчет крена и тангажа* 

Геометрическое преобразование параметров линии в углы ориентации.


## Требования и установка

```bash
Python 3.8+
TensorFlow 2.x
OpenCV 4.x
```

## Установка (Linux)

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Подготовка датасета

1. Извлеките кадры из видеопотока БПЛА
2. Разметьте изображения в CVAT и сохраните маски
3. Для анализа геометрии горизонта или валидации датасета: 
```bash
python utils/generate_dataset_slope.py
```
Скрипт создаст label.csv с параметрами filename, slope и offset для каждого кадра.

4. Обучение модели:
```bash
python main.py train --config configs/train.yaml
```
Веса сохранятся в checkpoints/, логи в logs/.

5. Инференс и оптимизация (в разработке)
```bash
python main.py predict --config <cfg> --ckpt <ckpt_path> --input <visdeo_src>
```

## Config

Все гиперпараметры управляются через configs/train.yaml:
```yaml
seed: 1234
paths:
  dataset: ./dataset
  logs: ./logs
  checkpoints: ./checkpoints
data:
  image_size: [128, 128]
  batch_size: 32
  test_size: 0.2
  num_classes: 2
training:
  epochs: 50
  learning_rate: 0.0001
  n_encoder_decoder: 3
  initial_filters: 8
```

## Логирование и мониторинг
Проект использует TensorBoard для отслеживания метрик и визуализации предиктов в реальном времени:
```bash
tensorboard --logdir logs/
```


## Инференс
```bash
python main.py predict --ckpt <model_ckpt> --input <folder or image>
```
Все гиперпараметры управляются через configs/train.yaml:
```yaml
inference:
  threshold: 0.5
  save_mask: false
  save_visualization: true 
  output_dir: ./logs/predictions 
  ```
