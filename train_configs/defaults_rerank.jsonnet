{
  "dataset_reader": {
    "type": "spider_rerank",
    "tables_file": "dataset/tables.json",
    "lazy": true,
    "keep_if_unparsable": false,
    "max_candidates": 10,
    "sub_sample_candidates": true,
    "unique_sub_graphs": true,
    "sub_graphs_candidates_path": "experiments/t33_wd_mltsk/candidates_train.json",
  },
  "validation_dataset_reader": {
    "type": "spider_rerank",
    "tables_file": "dataset/tables.json",
    "lazy": false,
    "keep_if_unparsable": true,
    "sub_sample_candidates": false,
    "unique_sub_graphs": false,
    "max_candidates": 10,
    "sub_graphs_candidates_path": "experiments/t33_wd_mltsk/candidates_dev.json"
  },
  "train_data_path": "dataset/train_spider.json",
  "validation_data_path": "dataset/dev.json",
  "model": {
    "type": "spider_reranker",
    "gnn_timesteps": 2,
    "question_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 200,
        "trainable": true
      }
    },
    "action_embedding_dim": 200,
    "encoder": {
      "type": "lstm",
      "input_size": 400,
      "hidden_size": 200,
      "bidirectional": true,
      "num_layers": 1
    },
    "entity_encoder": {
      "type": "boe",
      "embedding_dim": 200,
      "averaged": true
    },
    "attention": {"type": "dot_product"},
    "dropout": 0.5,
  },
  "iterator": {
    "type": "basic",
    "batch_size" : 5,
    "max_instances_in_memory": 10000
  },
  "trainer": {
    "num_epochs": 300,
    "cuda_device": 0,
    "patience": 100,
    "validation_metric": "+query_acc",
    "optimizer": {
      "type": "adam",
      "lr": 0.001,
      "weight_decay": 5e-4
    },
    "learning_rate_scheduler": {
      "type": "exponential",
      "gamma": 0.99
    },
    "predict_output_file": "predict.json",
    "num_serialized_models_to_keep": 2
  }
}
