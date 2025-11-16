# Detección de Acciones Humanas en Videos UCF101

Proyecto de Deep Learning para clasificación de acciones humanas usando esqueletos 2D del dataset UCF101.

## Modelos Implementados

- **Baseline**: LSTM bidireccional
- **ST-GCN**: Spatial-Temporal Graph Convolutional Network


## Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Uso

### Entrenamiento
```bash
# Baseline
python src/train.py --model baseline --split train1 --epochs 50

# ST-GCN
python src/train.py --model st_gcn --split train1 --epochs 100
```

### Evaluación
```bash
python src/evaluate.py --model baseline --checkpoint results/baseline_split1/BaselineLSTM_best.pth --split train1
```

### Predicción
```bash
python src/predict.py --model baseline --checkpoint results/baseline_split1/BaselineLSTM_best.pth --split test1
```

### Comparar Modelos
```bash
python compare_models.py \
    --baseline_checkpoint results/baseline_split1/BaselineLSTM_best.pth \
    --stgcn_checkpoint results/st_gcn_split1/STGCN_best.pth
```

## Resultados

- **Baseline LSTM**: 34.76% accuracy (20 épocas)
- **ST-GCN**: 14.06% accuracy (1 época - limitaciones computacionales)

Los resultados completos están documentados en `latex/report.pdf`.

## Dataset

UCF101 con 101 clases, 17 keypoints COCO por persona, ~13,320 videos anotados.

