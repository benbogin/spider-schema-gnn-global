from typing import Dict, List, Tuple, Any, Set

import torch
import torch.nn.functional as F
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, Embedding, TimeDistributed, Attention
from allennlp.nn import util
from allennlp.nn.util import masked_softmax
from allennlp.training.metrics import Average
from ordered_set import OrderedSet
from overrides import overrides
from torch_geometric.data import Data, Batch

from models.semantic_parsing.spider_base import SpiderBase
from modules.gated_graph_conv import GatedGraphConv
from semparse.worlds.spider_world import SpiderWorld
from state_machines.states.grammar_based_state import GrammarBasedState


@Model.register("spider_reranker")
class SpiderReranker(SpiderBase):
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,
                 entity_encoder: Seq2VecEncoder,
                 question_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 attention: Attention,
                 gnn_timesteps: int = 2,
                 dropout: float = 0.0,
                 rule_namespace: str = 'rule_labels') -> None:
        super().__init__(vocab, encoder, entity_encoder, question_embedder, gnn_timesteps, dropout, rule_namespace)
        self.vocab = vocab
        self._encoder = encoder
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace
        self._question_embedder = question_embedder
        self._entity_encoder = TimeDistributed(entity_encoder)

        self._action_padding_index = -1  # the padding value used by IndexField

        self._metric_num_candidates = 10
        self._accuracy_metrics = dict()

        self._accuracy_metrics['accuracy'] = Average()
        self._accuracy_metrics['accuracy_single'] = Average()
        self._accuracy_metrics['accuracy_multi'] = Average()
        self._accuracy_metrics['query_accuracy'] = Average()
        self._accuracy_metrics['query_accuracy_single'] = Average()
        self._accuracy_metrics['query_accuracy_multi'] = Average()

        self._num_entity_types = 9
        self._embedding_dim = question_embedder.get_output_dim()

        self._entity_type_encoder_embedding = Embedding(self._num_entity_types, action_embedding_dim)

        self._attention = attention
        self._q_att_query = torch.nn.Parameter(torch.FloatTensor(1, encoder.get_output_dim()+action_embedding_dim))
        torch.nn.init.xavier_uniform_(self._q_att_query)

        self._q_att_proj = torch.nn.Linear(encoder.get_output_dim(), action_embedding_dim)
        self._linking_params = torch.nn.Linear(16, 1)
        torch.nn.init.uniform_(self._linking_params.weight, 0, 1)

        self._gnn = GatedGraphConv(self._embedding_dim, gnn_timesteps, num_edge_types=4, dropout=dropout)

        self._graph_global_node_init_emb = torch.nn.Parameter(torch.FloatTensor(1, self._embedding_dim))
        torch.nn.init.xavier_uniform_(self._graph_global_node_init_emb)

        initial_embedding_size = action_embedding_dim + encoder.get_output_dim()
        self._ent_initial_proj = torch.nn.Linear(initial_embedding_size, self._embedding_dim)
        torch.nn.init.xavier_uniform_(self._ent_initial_proj.weight)

        self._final_graph_proj = torch.nn.Linear(action_embedding_dim*2 + encoder.get_output_dim(), self._embedding_dim)
        torch.nn.init.xavier_uniform_(self._final_graph_proj.weight)

        self._neighbor_params = torch.nn.Linear(self._embedding_dim, self._embedding_dim)

        self._score_ff = torch.nn.Linear(self._embedding_dim, 1)
        torch.nn.init.xavier_uniform_(self._score_ff.weight)

    @overrides
    def forward(self,  # type: ignore
                utterance: Dict[str, torch.LongTensor],
                world: List[SpiderWorld],
                schema: Dict[str, torch.LongTensor],
                sub_graphs,
                candidates,
                sub_graphs_labels=None,
                candidates_labels=None
                ) -> Dict[str, torch.Tensor]:
        batch_size = len(world)
        num_candidates = sub_graphs.size(1)

        # get the score for each of the queries for this examples
        candidates_scores, debug_info = self._score_candidates(utterance, world, schema, sub_graphs)
        candidates_scores = candidates_scores.view(batch_size, num_candidates)

        outputs: Dict[str, Any] = {}

        if sub_graphs_labels is not None:
            candidates_labels = candidates_labels.view(batch_size, num_candidates)
            sub_graphs_labels = sub_graphs_labels.view(batch_size, num_candidates)

            # mask questions with no positive candidate, or all positive candidates
            unanswerable_mask = sub_graphs_labels.max(dim=1)[0] == 0
            unanswerable_mask2 = sub_graphs_labels.min(dim=1)[0] == 1
            num_unanswerable = unanswerable_mask.sum() + unanswerable_mask2.sum()

            if batch_size - num_unanswerable > 0:
                candidates_mask = sub_graphs.max(dim=-1)[0] != -1

                normalized_scores = masked_softmax(candidates_scores, candidates_mask)
                loss = -torch.log((normalized_scores * sub_graphs_labels.float()).sum(dim=1))

                loss[unanswerable_mask] = 0
                loss[unanswerable_mask2] = 0
                loss = loss.sum() / (batch_size - num_unanswerable)
            else:
                # hack to reutrn zero loss without getting "does not require grad" error
                loss = (candidates_scores - candidates_scores).sum()

            outputs['loss'] = loss

        self._compute_validation_outputs(world,
                                         sub_graphs,
                                         candidates_scores,
                                         sub_graphs_labels,
                                         candidates,
                                         outputs)
        return outputs

    def _score_candidates(self,
                          utterance: Dict[str, torch.LongTensor],
                          worlds: List[SpiderWorld],
                          schema: Dict[str, torch.LongTensor],
                          sub_graphs: List[List[List[str]]],
                          ) -> Tuple[GrammarBasedState, Dict]:
        schema_text = schema['text']
        embedded_schema = self._question_embedder(schema_text, num_wrapping_dims=1)
        schema_mask = util.get_text_field_mask(schema_text, num_wrapping_dims=1).float()

        embedded_utterance = self._question_embedder(utterance)
        utterance_mask = util.get_text_field_mask(utterance).float()

        batch_size, num_entities, num_entity_tokens, _ = embedded_schema.size()
        num_entities = max([len(world.db_context.knowledge_graph.entities) for world in worlds])
        num_question_tokens = utterance['tokens'].size(1)

        # entity_types: tensor with shape (batch_size, num_entities), where each entry is the
        # entity's type id.
        # entity_type_dict: Dict[int, int], mapping flattened_entity_index -> type_index
        # These encode the same information, but for efficiency reasons later it's nice
        # to have one version as a tensor and one that's accessible on the cpu.
        entity_types, entity_type_dict = self._get_type_vector(worlds, num_entities, embedded_schema.device)

        entity_type_embeddings = self._entity_type_encoder_embedding(entity_types)

        # Compute entity and question word similarity.  We tried using cosine distance here, but
        # because this similarity is the main mechanism that the model can use to push apart logit
        # scores for certain actions (like "n -> 1" and "n -> -1"), this needs to have a larger
        # output range than [-1, 1].
        question_entity_similarity = torch.bmm(embedded_schema.view(batch_size,
                                                                    num_entities * num_entity_tokens,
                                                                    self._embedding_dim),
                                               torch.transpose(embedded_utterance, 1, 2))

        question_entity_similarity = question_entity_similarity.view(batch_size,
                                                                     num_entities,
                                                                     num_entity_tokens,
                                                                     num_question_tokens)
        # (batch_size, num_entities, num_question_tokens)
        question_entity_similarity_max_score, _ = torch.max(question_entity_similarity, 2)

        # (batch_size, num_entities, num_question_tokens, num_features)
        linking_features = schema['linking']

        feature_scores = self._linking_params(linking_features).squeeze(3)

        linking_scores = question_entity_similarity_max_score + feature_scores

        # (batch_size, num_question_tokens, num_entities)
        linking_probabilities = self._get_linking_probabilities(worlds, linking_scores.transpose(1, 2),
                                                                utterance_mask, entity_type_dict)

        # (batch_size, num_entities, embedding_dim)
        entity_embeddings = torch.tanh(entity_type_embeddings)

        link_embedding = util.weighted_sum(entity_embeddings, linking_probabilities)
        encoder_input = torch.cat([link_embedding, embedded_utterance], 2)

        # (batch_size, utterance_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._encoder(encoder_input, utterance_mask))

        # (batch_size, num_entities, encoder_output_dim)
        q_aligned = util.weighted_sum(encoder_outputs, linking_probabilities.transpose(1, 2))

        graph_initial_embedding = torch.cat((entity_type_embeddings, q_aligned), dim=-1)

        # (batch_size, num_entities, graph_embedding)
        graph_initial_embedding_proj = self._dropout(self._ent_initial_proj(graph_initial_embedding))

        global_node_initial_embedding = self._graph_global_node_init_emb.repeat(batch_size, 1)

        graph_initial_embedding_with_global = torch.cat((graph_initial_embedding_proj,
                                             global_node_initial_embedding.unsqueeze(1)), dim=1)

        # run the GNN over the sub graphs with the initial graph embeddings
        global_schema_encoding, entities_graph_encoding = \
            self._get_schema_graph_encoding(worlds, graph_initial_embedding_with_global, sub_graphs)

        num_candidates = global_schema_encoding.size(0) // batch_size

        # we now create q_aligned
        linking_probabilities = linking_probabilities.unsqueeze(1).expand(-1, num_candidates, -1, -1)\
            .contiguous().view(batch_size*num_candidates, num_question_tokens, num_entities)
        link_embedding = util.weighted_sum(entities_graph_encoding, linking_probabilities)

        encoder_outputs = encoder_outputs.unsqueeze(1).expand(-1, num_candidates, -1, -1)\
            .contiguous().view(batch_size*num_candidates, num_question_tokens, -1)

        utterance_mask = utterance_mask.unsqueeze(1).expand(-1, num_candidates, -1)\
            .contiguous().view(batch_size*num_candidates, num_question_tokens)

        encoder_augmented_outputs = torch.cat([encoder_outputs, link_embedding], -1)

        q_att_weights = self._attention(self._q_att_query.expand(batch_size * num_candidates, -1),
                                        encoder_augmented_outputs,
                                        utterance_mask)

        q_aligned = util.weighted_sum(encoder_augmented_outputs, q_att_weights)

        q_aligned = self._dropout(q_aligned)

        encoding_comps = torch.cat((
            global_schema_encoding,
            q_aligned
        ), dim=-1)

        final_encoding = self._final_graph_proj(encoding_comps)

        scores = self._score_ff(final_encoding)

        debug_info = {
            'linking_features': linking_features,
            'linking_probabilities': linking_probabilities,
            'linking_scores': linking_scores,
            'question_entity_similarity': question_entity_similarity_max_score,
            'feature_scores': feature_scores
        }

        return scores, debug_info

    @staticmethod
    def _get_neighbor_indices(worlds: List[SpiderWorld],
                              num_entities: int,
                              device: torch.device) -> torch.LongTensor:
        """
        This method returns the indices of each entity's neighbors. A tensor
        is accepted as a parameter for copying purposes.

        Parameters
        ----------
        worlds : ``List[SpiderWorld]``
        num_entities : ``int``
        tensor : ``torch.Tensor``
            Used for copying the constructed list onto the right device.

        Returns
        -------
        A ``torch.LongTensor`` with shape ``(batch_size, num_entities, num_neighbors)``. It is padded
        with -1 instead of 0, since 0 is a valid neighbor index. If all the entities in the batch
        have no neighbors, None will be returned.
        """

        num_neighbors = 0
        for world in worlds:
            for entity in world.db_context.knowledge_graph.entities:
                if len(world.db_context.knowledge_graph.neighbors[entity]) > num_neighbors:
                    num_neighbors = len(world.db_context.knowledge_graph.neighbors[entity])

        batch_neighbors = []
        no_entities_have_neighbors = True
        for world in worlds:
            # Each batch instance has its own world, which has a corresponding table.
            entities = world.db_context.knowledge_graph.entities
            entity2index = {entity: i for i, entity in enumerate(entities)}
            entity2neighbors = world.db_context.knowledge_graph.neighbors
            neighbor_indexes = []
            for entity in entities:
                entity_neighbors = [entity2index[n] for n in entity2neighbors[entity]]
                if entity_neighbors:
                    no_entities_have_neighbors = False
                # Pad with -1 instead of 0, since 0 represents a neighbor index.
                padded = pad_sequence_to_length(entity_neighbors, num_neighbors, lambda: -1)
                neighbor_indexes.append(padded)
            neighbor_indexes = pad_sequence_to_length(neighbor_indexes,
                                                      num_entities,
                                                      lambda: [-1] * num_neighbors)
            batch_neighbors.append(neighbor_indexes)
        # It is possible that none of the entities has any neighbors, since our definition of the
        # knowledge graph allows it when no entities or numbers were extracted from the question.
        if no_entities_have_neighbors:
            return None
        return torch.tensor(batch_neighbors, device=device, dtype=torch.long)

    def _get_schema_graph_encoding(self,
                                   worlds: List[SpiderWorld],
                                   initial_graph_embeddings: torch.Tensor,
                                   sub_graphs) -> Tuple[torch.Tensor, torch.Tensor]:
        max_num_entities = max([len(world.db_context.knowledge_graph.entities) for world in worlds])
        batch_size = initial_graph_embeddings.size(0)
        num_candidates = sub_graphs.size(1)

        graph_data_list = []

        sub_graphs = sub_graphs.tolist()

        for batch_index, world in enumerate(worlds):
            x = initial_graph_embeddings[batch_index]

            for sub_graph in sub_graphs[batch_index]:
                adj_list = self._get_graph_adj_lists(initial_graph_embeddings.device,
                                                     world, initial_graph_embeddings.size(1)-1, sub_graph)
                graph_data = Data(x)
                for i, l in enumerate(adj_list):
                    graph_data[f'edge_index_{i}'] = l
                graph_data_list.append(graph_data)

        batch = Batch.from_data_list(graph_data_list)

        gnn_output = self._gnn(batch.x, [batch[f'edge_index_{i}'] for i in range(4)])
        gnn_output = gnn_output.view(batch_size*num_candidates, max_num_entities+1, -1)
        entities_encodings = gnn_output[:, :max_num_entities]
        global_node_encodings = gnn_output[:, max_num_entities]

        return global_node_encodings, entities_encodings

    def _get_graph_adj_lists(self, device, world, global_entity_id, sub_graph: Set = None):
        entity_mapping = {}
        for i, entity in enumerate(world.db_context.knowledge_graph.entities):
            entity_mapping[entity] = i
        entity_mapping['_global_'] = global_entity_id
        adj_list_own = []  # column--table
        adj_list_link = []  # table->table / foreign->primary
        adj_list_linked = []  # table<-table / foreign<-primary
        adj_list_global = []  # node->global
        # TODO: Prepare in advance?
        for key, neighbors in world.db_context.knowledge_graph.neighbors.items():
            idx_source = entity_mapping[key]

            # skip if not in sub graph
            if idx_source not in sub_graph and not key.startswith('string:'):
                continue

            # if not key.startswith('string:'):
            adj_list_global.append((idx_source, entity_mapping['_global_']))

            for n_key in neighbors:
                idx_target = entity_mapping[n_key]

                # skip if not in sub graph
                if idx_target not in sub_graph and not key.startswith('string:'):
                    continue

                if n_key.startswith("table") or key.startswith("table"):
                    adj_list_own.append((idx_source, idx_target))
                elif n_key.startswith("string") or key.startswith("string"):
                    adj_list_own.append((idx_source, idx_target))
                elif key.startswith("column:foreign"):
                    adj_list_link.append((idx_source, idx_target))
                    src_table_key = f"table:{key.split(':')[2]}"
                    tgt_table_key = f"table:{n_key.split(':')[2]}"
                    idx_source_table = entity_mapping[src_table_key]
                    idx_target_table = entity_mapping[tgt_table_key]
                    adj_list_link.append((idx_source_table, idx_target_table))
                elif n_key.startswith("column:foreign"):
                    adj_list_linked.append((idx_source, idx_target))
                    src_table_key = f"table:{key.split(':')[2]}"
                    tgt_table_key = f"table:{n_key.split(':')[2]}"
                    idx_source_table = entity_mapping[src_table_key]
                    idx_target_table = entity_mapping[tgt_table_key]
                    adj_list_linked.append((idx_source_table, idx_target_table))
                else:
                    assert False

        all_adj_types = [adj_list_own, adj_list_link, adj_list_linked, adj_list_global]
        return [torch.tensor(l, device=device, dtype=torch.long).transpose(0,1) if l
                else torch.tensor(l, device=device, dtype=torch.long)
                for l in all_adj_types]

    @staticmethod
    def _query_difficulty(query_tokens, entities):
        number_tables = len(set([token for token in query_tokens if 'table:' + token in entities]))
        return number_tables > 1


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {key: metric.get_metric(reset) for key, metric in self._accuracy_metrics.items()}

    def _update_metric(self, key, val, difficulty=None):
        self._accuracy_metrics[key](val)

        if difficulty is not None:
            key = key + '_' + ('multi' if difficulty else 'single')
            self._accuracy_metrics[key](val)

    def _compute_validation_outputs(self,
                                    world: List[SpiderWorld],
                                    sub_graphs,
                                    sub_graphs_scores,
                                    sub_graphs_labels,
                                    candidates,
                                    outputs: Dict[str, Any]) -> None:
        batch_size = len(world)

        outputs['predicted_sql_query'] = []
        outputs['candidates'] = []

        for i in range(batch_size):
            if world[i].query is not None:
                gold_sql_query = ' '.join(world[i].query)
                difficulty = self._query_difficulty(query_tokens=gold_sql_query.split(),
                                                    entities=set(world[i].db_context.knowledge_graph.entities))

            num_candidates = self._metric_num_candidates
            example_sub_graphs = sub_graphs[i, :num_candidates]
            example_sub_graphs_scores = sub_graphs_scores[i, :num_candidates]
            example_candidates = candidates[i][:num_candidates]
            if sub_graphs_labels is not None:
                example_sub_graphs_labels = sub_graphs_labels[i, :num_candidates]

            candidate_to_sub_graph_id = {}
            sub_graphs_ids = []
            for sub_graph in example_sub_graphs:
                entities_ids = sub_graph[sub_graph > -1].tolist()
                if len(entities_ids) == 0:
                    continue
                sub_graph = tuple(sorted(entities_ids))
                if sub_graph not in candidate_to_sub_graph_id:
                    candidate_to_sub_graph_id[sub_graph] = len(candidate_to_sub_graph_id)
                sub_graphs_ids.append(candidate_to_sub_graph_id[sub_graph])

            sorted_candidates_ids = example_sub_graphs_scores.sort(descending=True)[1].tolist()
            sorted_sub_graphs = OrderedSet([sub_graphs_ids[j] for j in sorted_candidates_ids if j < len(sub_graphs_ids)])

            candidates_for_final_sort = []
            for original_rank, c in enumerate(example_candidates):
                sub_graph_id = sub_graphs_ids[original_rank]
                if sub_graphs_labels is not None:
                    sg_correct = int(example_sub_graphs_labels[original_rank] == 1)
                else:
                    sg_correct = None
                candidates_for_final_sort.append({
                    'query': c['query'],
                    'original_rank': original_rank,
                    'reranker_sg_rank': sorted_sub_graphs.index(sub_graph_id),
                    'reranker_cand_rank': sorted_candidates_ids.index(original_rank),
                    'sub_graph_correct': sg_correct,
                    'correct': c['correct']
                })

            # sorting sub graphs, then inner-ranking by original beam search order
            candidates_sg_sort = sorted(candidates_for_final_sort,
                                               key=lambda x: (x['reranker_sg_rank'], x['original_rank']))

            if sub_graphs_labels is not None:
                sg_tsk_query_correct = candidates_sg_sort[0]['correct']

                if max(example_sub_graphs_labels) > 0:
                    self._update_metric('accuracy', int(sg_tsk_query_correct))

                self._update_metric('query_accuracy', int(sg_tsk_query_correct), difficulty)

            outputs['candidates'].append([c['query'] for c in candidates_sg_sort])
