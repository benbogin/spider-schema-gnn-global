import json
import logging
from random import random
from typing import Dict, List

from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer, Token
from allennlp.data.fields import ListField, LabelField, MetadataField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

from dataset_readers.dataset_util.spider_utils import ent_key_to_name
from dataset_readers.spider import SpiderDatasetReader
from semparse.worlds.spider_world import SpiderWorld

logger = logging.getLogger(__name__)


@DatasetReader.register("spider_rerank")
class SpiderRerankDatasetReader(SpiderDatasetReader):
    """
    Having both dataset reader classes inheriting from a base class may have been cleaner,
    but this way it is easier to share cache files which take time to create.

    This reader goes through each example in the spider dataset and loads the output parser candidates that were saved.
    It creates the sub-graphs (the subset of all schema items) from each one of the candidate queries.

    Not that if sampling is required, `lazy` should be False, so that this reader will be called at each step (this
    could probably be optimized better)
    """
    def __init__(self,
                 sub_graphs_candidates_path: str,
                 max_candidates: int,
                 sub_sample_candidates: bool,
                 lazy: bool = False,
                 unique_sub_graphs: bool = True,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 keep_if_unparsable: bool = True,
                 tables_file: str = None):
        """
        :param sub_graphs_candidates_path: path to the created json file holding the candidates for each example
        :param max_candidates: limit the number of candidates returned per example
        :param sub_sample_candidates: if True and `max_candidates` is smaller than the number of candidates, a subset
            will be sampled at each step
        :param unique_sub_graphs: should equal sub-graphs be treated as one? This should generally be True for training,
            False for validation
        """
        super().__init__(lazy, question_token_indexers, keep_if_unparsable, tables_file)

        self._unique_sub_graphs = unique_sub_graphs
        self._sub_graphs_candidates_path = sub_graphs_candidates_path
        self._sub_graphs_candidates = []
        self._max_candidates = max_candidates
        self._sub_sample_candidates = sub_sample_candidates
        self._query_token_indexers = {'query_tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        if not self._sub_graphs_candidates:
            with open(self._sub_graphs_candidates_path, "r") as data_file:
                for line in data_file:
                    self._sub_graphs_candidates.append(json.loads(line))
        yield from self._read_examples_file(file_path)

    @overrides
    def process_instance(self, instance: Instance, index: int = None, candidates: List = None):
        """
        This function is called after the instance was loaded with the basic dataset reader, and adds the query
        candidates data.
        """
        assert index is not None or candidates is not None

        if instance is None:
            return instance

        fields = instance.fields
        world: SpiderWorld = fields['world'].metadata
        del fields['valid_actions']
        if 'action_sequence' in fields:
            del fields['action_sequence']

        # Get the correct sub-graph (our supervision)
        if world.query is not None:
            correct_sub_graph = set()
            for token in world.query:
                if token in world.entities_names:
                    correct_sub_graph.add(token)

        original_candidates = candidates or self._sub_graphs_candidates[index]
        if self._sub_sample_candidates:
            shuffled_candidates = list(original_candidates)
            # make sure there is a correct candidate, so put them on top
            shuffled_candidates = sorted(shuffled_candidates, key=lambda x: (x['correct'], random()), reverse=True)
        else:
            shuffled_candidates = original_candidates

        sub_graphs = []
        label_fields = []
        sub_graphs_label_fields = []

        entities_names = [ent_key_to_name(e) for e in world.db_context.knowledge_graph.entities
                          if e.startswith('column:') or e.startswith('table:')]

        unique_sub_graphs = set()
        kept_candidates = []

        # go through the candidate queries and extract their sub-graph
        for candidate in shuffled_candidates:
            if self._sub_sample_candidates and len(kept_candidates) == self._max_candidates:
                break

            query_tokens = candidate['query'].split()

            sub_graph = set()
            candidate_entities = []
            for i, token in enumerate(query_tokens):
                ent_id = -1
                potential_ent = token.replace('.', '@')
                if potential_ent in entities_names:
                    sub_graph.add(potential_ent)
                    ent_id = world.entities_names[potential_ent]
                candidate_entities.append(LabelField(ent_id, skip_indexing=True))

            if not sub_graph:
                continue

            if not self._unique_sub_graphs:
                sub_graphs.append(sub_graph)
            else:
                sub_graph_hash = tuple(sorted(sub_graph))
                if sub_graph_hash in unique_sub_graphs:
                    continue
                unique_sub_graphs.add(sub_graph_hash)
                sub_graphs.append(sub_graph)
            kept_candidates.append(candidate)

            if candidate['correct'] is not None:
                label_fields.append(LabelField(int(candidate['correct']), skip_indexing=True))
                sub_graphs_label_fields.append(LabelField(int(sub_graph == correct_sub_graph), skip_indexing=True))

        # check if we should return all examples (even when no correct answers found)
        if not self._keep_if_unparsable:
            if self._unique_sub_graphs and not any([sg == correct_sub_graph for sg in sub_graphs]):
                return None
            if not self._unique_sub_graphs and not any([l.label for l in label_fields]):
                return None

        sub_graph_candidates = []
        for sub_graph in sub_graphs:
            entities_ids = []
            for ent in sub_graph:
                ent_id = world.entities_names[ent]
                entities_ids.append(LabelField(ent_id, skip_indexing=True))
            if not entities_ids:
                continue
            sub_graph_candidates.append(ListField(entities_ids))

        # we give both the subgraphs and the actual query candidates
        fields['sub_graphs'] = ListField(sub_graph_candidates)
        fields['candidates'] = MetadataField(kept_candidates)

        if sub_graphs_label_fields:
            fields['sub_graphs_labels'] = ListField(sub_graphs_label_fields)
        if label_fields:
            fields['candidates_labels'] = ListField(label_fields)

        return Instance(fields)
