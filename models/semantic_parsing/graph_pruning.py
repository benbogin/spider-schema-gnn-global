from typing import List

import torch

from torch import nn
from allennlp.nn import util
from torch_geometric.data import Data, Batch

from modules.gated_graph_conv import GatedGraphConv
from semparse.worlds.spider_world import SpiderWorld


class GraphPruning(nn.Module):
    def __init__(self, num_edge_types, embedding_dim, encoder_output_dim, rnn_dropout, timesteps=2):
        super().__init__()

        self._ent_initial_proj = torch.nn.Linear(embedding_dim+encoder_output_dim + 1, embedding_dim)
        self.dim = embedding_dim

        self._gnn = GatedGraphConv(self.dim, timesteps, num_edge_types=num_edge_types, dropout=rnn_dropout)

        self._graph_global_node_init_emb = torch.nn.Parameter(torch.FloatTensor(1, embedding_dim))
        torch.nn.init.xavier_uniform_(self._graph_global_node_init_emb)

        self._predict_rel_ff = torch.nn.Linear(self.dim, 1)

    def forward(self,
                worlds: List[SpiderWorld],
                encoder_outputs: torch.Tensor,
                entity_type_embeddings: torch.Tensor,
                linking_scores: torch.Tensor,
                utterance_mask: torch.Tensor,
                get_graph_adj_lists):
        # This index of 0 is for the null entity for each type, representing the case where a
        # word doesn't link to any entity.
        linking_scores_with_null = torch.cat((
            linking_scores.new_zeros(*linking_scores.size()[:2], 1),
            linking_scores
        ), dim=-1)
        utterance_mask_with_null = torch.cat((
            utterance_mask.new_zeros(utterance_mask.size(0), 1),
            utterance_mask
        ), dim=-1)

        # (batch_size, num_entities, num_question_tokens)
        linking_probabilities = util.masked_softmax(linking_scores_with_null, utterance_mask_with_null)
        linking_probabilities = linking_probabilities[:, :, 1:]

        r0 = linking_probabilities.max(dim=-1, keepdim=True)[0]

        entity_type_embeddings = torch.cat((entity_type_embeddings, r0), dim=-1)

        q_aligned = util.weighted_sum(encoder_outputs, linking_probabilities)
        initial_embedding = torch.cat((entity_type_embeddings, q_aligned), dim=-1)

        initial_embedding2 = self._ent_initial_proj(initial_embedding)

        relevance, relevance_logits = self._calculate_relevance(worlds,
                                                                initial_embedding2,
                                                                get_graph_adj_lists)

        return relevance, relevance_logits, linking_probabilities

    def _calculate_relevance(self,
                             worlds: List[SpiderWorld],
                             initial_embedding: torch.Tensor,
                             get_graph_adj_lists):
        max_num_entities = max([len(world.db_context.knowledge_graph.entities) for world in worlds])
        batch_size = initial_embedding.size(0)

        initial_embedding = torch.cat((
            initial_embedding,
            self._graph_global_node_init_emb.repeat(batch_size, 1).unsqueeze(1)
        ), dim=1)
        max_num_entities += 1

        graph_data_list = []

        for batch_index, world in enumerate(worlds):
            x = initial_embedding[batch_index]

            adj_list = get_graph_adj_lists(initial_embedding.device, world, initial_embedding.size(1) - 1,
                                           global_node=True)
            graph_data = Data(x)
            for i, l in enumerate(adj_list):
                graph_data[f'edge_index_{i}'] = l
            graph_data_list.append(graph_data)

        batch = Batch.from_data_list(graph_data_list)

        gnn_output = self._gnn(batch.x, [batch[f'edge_index_{i}'] for i in range(self._gnn.num_edge_types)])
        gnn_output = gnn_output.view(batch_size, max_num_entities, -1)

        # take only the embedding of the global node
        gnn_output = gnn_output[:, :max_num_entities-1]

        predicted_relevance_logits = self._predict_rel_ff(gnn_output)
        return torch.sigmoid(predicted_relevance_logits), predicted_relevance_logits
