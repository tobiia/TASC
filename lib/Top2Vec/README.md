# Top2Vec Fork 1.0.36a

This fork extends the original Top2Vec implementation with additional flexibility for modern embedding workflows.

## Credits

### Top2Vec
- Version: 1.0.36
- Author: Dimo Angelov \<dimo.angelov@gmail.com>
- Repo URL: https://github.com/ddangelov/Top2Vec
- License: BSD

## Features

### MPNet Support

Added support for MPNet-based sentence embedding models in the classic Top2Vec pipeline, enabling higher-quality document representations and topic discovery.

```python
from top2vec import Top2Vec

model = Top2Vec(
    documents,
    embedding_model="all-mpnet-base-v2"
)
```

### Configurable Embedding Layer

Users can now specify which transformer layer should be used when generating document embeddings. This makes it possible to experiment with different representation levels and tailor embeddings to specific use cases.

```python
model = Top2Vec(
    documents,
    embedding_layer=-1
)
```

### Pre-computed Embeddings

Top2Vec can now accept pre-computed embeddings directly. This allows users to:

* Generate embeddings using external pipelines
* Reuse existing embeddings without recomputation
* Experiment with custom embedding models not natively supported by Top2Vec
* Reduce processing time when iterating on topic modeling parameters

```python
model = Top2Vec(
    documents,
    precomputed_embeddings=embeddings
)
```

## Compatibility

This fork aims to remain compatible with the original Top2Vec API while providing additional options for embedding generation and integration.
