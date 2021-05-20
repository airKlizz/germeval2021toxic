import datasets
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


class MultiClass(datasets.Metric):
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
        assert len(predictions) == len(references)
        assert len(predictions[0]) == len(references[0])
        nb_classes = len(predictions[0])
        data = []
        for c in range(nb_classes):
            data.append(
                {
                    "predictions": [],
                    "references": [],
                }
            )
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                data[j]["predictions"].append(predictions[i][j])
                data[j]["references"].append(references[i][j])

        results = {}
        for i, d in enumerate(data):
            results[f"class{i}"] = {
                "accuracy": accuracy_score(
                    d["references"], d["predictions"], normalize=normalize, sample_weight=sample_weight
                ),
                "f1": f1_score(
                    d["references"],
                    d["predictions"],
                    labels=labels,
                    pos_label=pos_label,
                    average=average,
                    sample_weight=sample_weight,
                ),
                "precision": precision_score(
                    d["references"],
                    d["predictions"],
                    labels=labels,
                    pos_label=pos_label,
                    average=average,
                    sample_weight=sample_weight,
                ),
                "recall": recall_score(
                    d["references"],
                    d["predictions"],
                    labels=labels,
                    pos_label=pos_label,
                    average=average,
                    sample_weight=sample_weight,
                ),
            }

        return results
