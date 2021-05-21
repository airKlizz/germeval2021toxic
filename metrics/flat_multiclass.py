import datasets
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


class FlatMultiClass(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="_DESCRIPTION",
            citation="_CITATION",
            inputs_description="_KWARGS_DESCRIPTION",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
            ),
            reference_urls=[""],
        )

    def _compute(
        self, predictions, references, normalize=True, sample_weight=None, labels=None, pos_label=1, average="binary"
    ):
        predictions = [v for b in predictions for v in b]
        references = [v for b in references for v in b]
        return {
            "accuracy": accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight),
            "f1": f1_score(
                references,
                predictions,
                labels=labels,
                pos_label=pos_label,
                average=average,
                sample_weight=sample_weight,
            ),
            "precision": precision_score(
                references,
                predictions,
                labels=labels,
                pos_label=pos_label,
                average=average,
                sample_weight=sample_weight,
            ),
            "recall": recall_score(
                references,
                predictions,
                labels=labels,
                pos_label=pos_label,
                average=average,
                sample_weight=sample_weight,
            ),
        }
