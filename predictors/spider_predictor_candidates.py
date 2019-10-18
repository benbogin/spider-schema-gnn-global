from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register("spider_candidates")
class SpiderCandidatesParserPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        model._beam_search._beam_size = 40
        model._beam_search._per_node_beam_size = 40

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        if instance:
            outputs = self._model.forward_on_instance(instance)
            if 'candidates' in outputs:
                return sanitize(outputs['candidates'])
            else:
                return sanitize([])
        else:
            return sanitize([])
