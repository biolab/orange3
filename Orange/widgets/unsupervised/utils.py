from typing import NamedTuple, Type, Dict

from Orange import distance


Euclidean, EuclideanNormalized, Manhattan, ManhattanNormalized, Cosine, \
    Mahalanobis, Hamming, \
    Pearson, PearsonAbsolute, Spearman, SpearmanAbsolute, Jaccard = range(12)


class MetricDef(NamedTuple):
    id: int  # pylint: disable=invalid-name
    name: str
    tooltip: str
    metric: Type[distance.Distance]
    normalize: bool = False


MetricDefs: Dict[int, MetricDef] = {
    metric.id: metric for metric in (
        MetricDef(EuclideanNormalized, "Euclidean (normalized)",
                  "Square root of summed difference between normalized values",
                  distance.Euclidean, normalize=True),
        MetricDef(Euclidean, "Euclidean",
                  "Square root of summed difference between values",
                  distance.Euclidean),
        MetricDef(ManhattanNormalized, "Manhattan (normalized)",
                  "Sum of absolute differences between normalized values",
                  distance.Manhattan, normalize=True),
        MetricDef(Manhattan, "Manhattan",
                  "Sum of absolute differences between values",
                  distance.Manhattan),
        MetricDef(Mahalanobis, "Mahalanobis",
                  "Mahalanobis distance",
                  distance.Mahalanobis),
        MetricDef(Hamming, "Hamming", "Hamming distance",
                  distance.Hamming),
        MetricDef(Cosine, "Cosine", "Cosine distance",
                  distance.Cosine),
        MetricDef(Pearson, "Pearson",
                  "Pearson correlation; distance = 1 - ρ/2",
                  distance.PearsonR),
        MetricDef(PearsonAbsolute, "Pearson (absolute)",
                  "Absolute value of Pearson correlation; distance = 1 - |ρ|",
                  distance.PearsonRAbsolute),
        MetricDef(Spearman, "Spearman",
                  "Spearman correlation; distance = 1 - ρ/2",
                  distance.SpearmanR),
        MetricDef(SpearmanAbsolute, "Spearman (absolute)",
                  "Absolute value of Pearson correlation; distance = 1 - |ρ|",
                  distance.SpearmanRAbsolute),
        MetricDef(Jaccard, "Jaccard", "Jaccard distance",
                  distance.Jaccard)
    )
}
