# DFKI SLT at GermEval 2021 

Code for the paper: **Multilingual Pre-training and Data Augmentation for the Classification of Toxicity in Social Media Comments**

## Datasets

The `data` folder contains all the files used to train the models. Some files might not be used for the final versions of the models.

## Training

To train the models, please run:

```
python train.py <parameters>
```

Run `python train.py --help` to see all the posibilities.

## Models

All the final models are available on the HuggingFace Hub.

| Model                                                               | Link                                                                                                                   |
|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| GBERT (Fine-tuning only)                                            | https://huggingface.co/airKlizz/gbert-base-germeval21-toxic                                                            |
| XLM-RoBERTa (Fine-tuning only)                                      | https://huggingface.co/airKlizz/xlm-roberta-base-germeval21-toxic                                                      |
| mT5 (Fine-tuning only)                                              | https://huggingface.co/airKlizz/mt5-base-germeval21-toxic                                                              |
| XLM-RoBERTa (With task-specific pre-training)                       | https://huggingface.co/airKlizz/xlm-roberta-base-germeval21-toxic-with-task-specific-pretraining                       |
| mT5 (With task-specific pre-training)                               | https://huggingface.co/airKlizz/mt5-base-germeval21-toxic-with-task-specific-pretraining                               |
| GBERT (With data augmentation)                                      | https://huggingface.co/airKlizz/gbert-base-germeval21-toxic-with-data-augmentation                                     |
| XLM-RoBERTa (With data augmentation)                                | https://huggingface.co/airKlizz/xlm-roberta-base-germeval21-toxic-with-data-augmentation                               |
| mT5 (With data augmentation)                                        | https://huggingface.co/airKlizz/mt5-base-germeval21-toxic-with-data-augmentation                                       |
| XLM-RoBERTa (With task-specific pre-training and data augmentation) | https://huggingface.co/airKlizz/xlm-roberta-base-germeval21-toxic-with-task-specific-pretraining-and-data-augmentation |
| mT5 (With task-specific pre-training and data augmentation)         | https://huggingface.co/airKlizz/mt5-base-germeval21-toxic-with-task-specific-pretraining-and-data-augmentation         |
