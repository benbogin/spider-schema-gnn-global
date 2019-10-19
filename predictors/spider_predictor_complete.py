from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model, load_archive
from allennlp.predictors.predictor import Predictor

from dataset_readers.spider_rerank import SpiderRerankDatasetReader
from models.semantic_parsing.spider_reranker import SpiderReranker
from semparse.contexts.spider_db_context import SpiderDBContext


@Predictor.register("spider_predict_complete")
class SpiderParserPredictor(Predictor):
    count = 1

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        model._beam_search._beam_size = 40
        model._beam_search._per_node_beam_size = 40

        self._num_rank_candidates = 10

        archive_path = 'experiments/experiment_rerank'

        reranker_archive = load_archive(archive_path, weights_file=archive_path + '/best.th')
        self._reranker_model: SpiderReranker = reranker_archive.model
        self._reranker_model.eval()

        config = reranker_archive.config.duplicate()
        dataset_reader_params = config["dataset_reader"]
        self._reranker_dataset_reader: SpiderRerankDatasetReader = DatasetReader.from_params(dataset_reader_params)
        self._reranker_dataset_reader._sub_sample_candidates = False
        self._reranker_dataset_reader._unique_sub_graphs = False
        self._reranker_dataset_reader._keep_if_unparsable = True
        self._reranker_dataset_reader._tables_file = dataset_reader._tables_file
        SpiderDBContext.tables_file = dataset_reader._tables_file

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        json_output = {}
        predicted_sql_query = None

        del instance.fields['action_sequence']

        outputs = self._model.forward_on_instance(instance)

        candidates = outputs['candidates'][:self._num_rank_candidates]

        rerank_instance = self._reranker_dataset_reader.process_instance(instance, candidates=candidates)

        if rerank_instance is not None:
            rerank_outputs = self._reranker_model.forward_on_instance(rerank_instance)
            predicted_sql_query = rerank_outputs['candidates'][0]

        if not predicted_sql_query:
            # line must not be empty for the evaluator to consider it
            predicted_sql_query = 'NO PREDICTION'
        json_output['predicted_sql_query'] = predicted_sql_query
        print(SpiderParserPredictor.count, predicted_sql_query)
        SpiderParserPredictor.count += 1
        return sanitize(json_output)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return outputs['predicted_sql_query'] + "\n"
