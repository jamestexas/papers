# CHEESE: Contextual Hierarchy for Embedding Enhancement and Semantic Enrichment

**Author:** James Gardner  
**Affiliation:** Personal Research

## Abstract

This paper introduces CHEESE (Contextual Hierarchy for Embedding Enhancement and Semantic Enrichment), an experimental memory management system for Large Language Models (LLMs) that implements a novel "contextual fabric" approach inspired by biological memory systems. Unlike traditional knowledge graph approaches that rely on discrete nodes and edges, CHEESE creates a rich multidimensional representation that enables more human-like memory access patterns. The system orchestrates multiple retrieval signals—semantic similarity, temporal relationships, associative connections, and activation patterns—to create a coherent memory fabric. Preliminary results indicate that this approach improves retrieval precision while maintaining high recall, particularly for queries requiring contextual understanding. This paper describes the architecture, core components, retrieval strategies, and biological inspirations underlying the system, highlighting its potential to transform how AI systems manage long-term memory.

## 1\. Introduction

Large Language Models demonstrate remarkable capabilities for generating coherent text, but they often struggle with maintaining contextual coherence over extended interactions. This limitation stems partly from their retrieval-augmented generation approaches, which typically rely on simple vector similarity to access previously stored information.

Human memory, in contrast, operates as a dynamic, interconnected system where information retrieval depends on multiple signals: semantic relevance, temporal proximity, associative connections, and current activation state. When humans recall information, we don't merely search for the semantically closest match; we traverse richly connected memory networks influenced by various contextual cues.

CHEESE introduces a "contextual fabric" approach that more closely emulates these human memory characteristics. Rather than structuring memory as a collection of discrete entities connected by explicit relationships, CHEESE constructs a multidimensional fabric where memories exist in a rich contextual space. This enables more nuanced and contextually relevant retrieval patterns that significantly improve the coherence of long-running AI conversations.

## 2\. Background and Related Work

Contemporary approaches to memory management in LLMs typically employ one of several frameworks:

**Vector Databases** store embeddings of text chunks and retrieve them using similarity search. While computationally efficient, they lack contextual understanding of how pieces of information relate to each other temporally or associatively.

**Knowledge Graphs** explicitly model relationships between entities but require structured outputs from LLMs and introduce significant complexity in maintenance and querying.

**Episodic Memory** systems attempt to store interactions in chronological sequences but often struggle with efficient retrieval of relevant information across episodes.

**Retrieval-Augmented Generation (RAG)** enhances LLM capabilities by retrieving relevant information before generation but typically relies solely on semantic similarity for retrieval.

**CHEESE** draws inspiration from cognitive science models of human memory, particularly the Adaptive Resonance Theory (ART) for self-organizing categorization, spreading activation models of associative memory, and the episodic-semantic distinction in long-term memory organization.

## 3\. CHEESE Architecture

CHEESE employs a component-based architecture organized into five core layers:

### 3.1 Foundation Layer

The foundation layer provides the core abstractions and interfaces that define the system's capabilities. It establishes clean separation between interfaces and implementations, enabling flexible composition of components.

Key interfaces in the foundation layer include:

- Memory interfaces for standardized access to memory objects  
- Retrieval component interfaces defining common protocols for memory retrieval  
- Pipeline interfaces that enable modular processing chains  
- Query processing interfaces for analyzing and adapting to different query types

This layer ensures extensibility by making components interchangeable and enables integration with various embedding models and storage backends.

### 3.2 Storage Layer

The storage layer handles persistence and retrieval of memory representations with several implementations:

- **StandardMemoryStore**: Basic storage for typical use cases, optimized for semantic similarity search  
- **ChunkedMemoryStore**: Storage optimized for large text contexts, maintaining relationships between chunks  
- **HybridMemoryStore**: Efficient storage balancing completeness and memory usage through selective chunking

The storage layer supports multiple vector search backends, including NumPy for small datasets and FAISS for high-performance large-scale retrieval. This flexibility allows the system to scale from personal assistants to enterprise-level applications with billions of memory entries.

### 3.3 Core Components Layer

The core components implement the fundamental memory capabilities that constitute the contextual fabric:

- **ActivationManager**: Manages memory activation levels with biologically-inspired decay patterns, ensuring recently accessed information remains more accessible  
- **AssociativeMemoryLinker**: Creates and traverses associative connections between memories based on semantic similarity and temporal co-occurrence  
- **CategoryManager**: Organizes memories into emergent categories using ART-inspired clustering without requiring predefined structures  
- **TemporalContextBuilder**: Handles time-based relationships and episodic grouping to enable time-sensitive retrieval  
- **ContextualEmbeddingEnhancer**: Enriches memory embeddings with contextual information from various sources

Each component addresses one dimension of the contextual fabric while maintaining compatibility with the others, enabling their orchestration into a cohesive system.

### 3.4 Retrieval Layer

The retrieval layer integrates the core components to provide sophisticated memory retrieval:

- **QueryAnalyzer**: Analyzes query intent and structure to adapt retrieval parameters based on query characteristics  
- **QueryTypeAdapter**: Adjusts retrieval parameters based on detected query type (factual, personal, temporal, etc.)  
- **Multiple Retrieval Strategies**: Implement different approaches to memory access tailored to various use cases and query types

The retrieval strategies include the flagship ContextualFabricStrategy that combines all signals, specialized strategies for large documents (ChunkedFabricStrategy), and hybrid approaches that balance comprehensive retrieval with computational efficiency.

### 3.5 API Layer

The API layer provides intuitive interfaces for interacting with the system:

- **CHEESEAPI**: Main API for standard use cases  
- **ChunkedCHEESEAPI**: API optimized for large document processing  
- **HybridCHEESEAPI**: Memory-efficient API with selective chunking

These interfaces abstract away the complexity of the underlying system while providing customization options for specific use cases.

## 4\. Contextual Fabric Approach

The contextual fabric approach represents the core innovation of CHEESE. Rather than modeling memories as discrete entities with explicit relationships, it constructs a multidimensional fabric where memories are interconnected through various contextual dimensions.

### 4.1 Memory Representation

Memories in CHEESE are represented not just by their semantic content (embeddings) but also by their temporal context, activation state, associative connections, and category membership. This rich representation enables more nuanced retrieval that considers multiple contextual signals.

For example, a memory object includes:

```python
{

    "id": "memory_12345",
    "embedding": [0.1, 0.2, ...],  # Semantic content
    "content": {"text": "User expressed preference for jazz music"},
    "created_at": 1678823400,  # Temporal context
    "activation": 0.8,  # Current activation level
    "category_id": 7,  # Emergent category membership
    "associative_links": [("memory_12346", 0.7), ("memory_10982", 0.5)]

}
```

This multidimensional representation allows the system to consider both what information is stored and how it relates to other information in various contextual dimensions.

### 4.2 Biologically-Inspired Mechanisms

CHEESE implements several mechanisms inspired by human memory:

**Activation and Decay**: Memories have activation levels that increase when accessed and decay over time, mimicking the strengthening and weakening of neural pathways in the brain. The system implements both short-term decay (hours to days) and long-term decay (weeks to months) with different half-lives.

**Associative Linking**: Memories establish bidirectional associative connections based on semantic similarity and temporal co-occurrence, enabling traversal of related concepts. These connections are not predefined but emerge through interaction, creating an adaptive network.

**Adaptive Categorization**: An ART-inspired mechanism allows memories to self-organize into emergent categories without predefined structures. Categories evolve as new memories are added, merging and splitting as patterns emerge.

**Episodic Clustering**: Memories are grouped into "episodes" based on temporal proximity, facilitating context-aware retrieval across time. This enables the system to retrieve information based on when it was encountered, not just what it contains.

### 4.3 Retrieval Strategies

CHEESE implements several retrieval strategies that leverage the contextual fabric:

**ContextualFabricStrategy**: The flagship strategy that combines multiple signals:

- Semantic similarity for content relevance  
- Temporal context for time-based relationships  
- Associative connections for related memories  
- Activation levels for frequently accessed information

This strategy dynamically adjusts the weights of these signals based on query characteristics, optimizing retrieval for different types of questions.

**TwoStageRetrievalStrategy**: Implements a two-phase approach with a relaxed first phase to generate candidates, followed by a more stringent second phase with reranking. This strategy optimizes computational efficiency while maintaining retrieval quality.

**HybridBM25VectorStrategy**: Combines lexical matching (BM25) with semantic embedding similarity for queries requiring both keyword precision and semantic understanding. This strategy is particularly effective for queries with specific terms or names.

**ChunkedFabricStrategy**: Extends the contextual fabric approach for large documents, maintaining context coherence across chunks. This strategy enables effective retrieval from long-form content while preserving the semantic flow between sections.

**HybridFabricStrategy**: A memory-efficient implementation that adapts its approach based on available computational resources, optimizing for different deployment environments.

### 4.4 Conceptual Parallels with Transformer Architectures

The CHEESE contextual fabric approach shares several conceptual parallels with transformer architectures that have revolutionized natural language processing. Understanding these parallels helps position CHEESE within the broader landscape of deep learning architectures while highlighting its unique contributions.

#### **Relationship Modeling**

Transformers calculate attention weights between tokens, allowing each token to "attend" to all other tokens with varying degrees of importance. This creates a dynamic, context-sensitive representation where the meaning of each token is influenced by its relationships to all others.

Similarly, CHEESE calculates relationships between memory items through multiple signals (semantic similarity, temporal proximity, associative connections, and activation patterns). Just as a token in a transformer is understood in the context of surrounding tokens, a memory in CHEESE is understood in the context of related memories across multiple dimensions.

#### **Contextual Enhancement**

Transformers enhance token embeddings with contextual information through their self-attention mechanism. The initial embedding of each token is transformed into a contextually rich representation that captures its role within the broader sequence.

CHEESE similarly enhances memory item embeddings through its contextual fabric approach. A memory's initial semantic embedding is enhanced with temporal context, associative connections, and activation patterns, creating a richer representation that integrates multiple contextual signals.

#### **Dynamic Weighting**

The attention mechanism in transformers dynamically adjusts the importance of token relationships based on context. Different attention heads can focus on different aspects of relationships between tokens.

In CHEESE, the importance of different context signals is dynamically weighted based on query characteristics. A temporal query increases the weight of temporal relationships, while a factual query may prioritize semantic similarity. This dynamic weighting enables more contextually appropriate retrieval.

#### **Emergent Structure**

Neither transformers nor CHEESE requires pre-defined relationships; both develop them dynamically from data patterns. In transformers, the attention patterns emerge during training without explicit supervision about which tokens should attend to which others.

Similarly, in CHEESE, the associative connections, category structures, and activation patterns emerge through interaction without requiring explicit relationship definitions. The CategoryManager component, for instance, uses ART-inspired clustering to discover emergent categories without predefined schemas.

#### Key Differences and Extensions

While these parallels are instructive, CHEESE extends beyond the transformer paradigm in several important ways:

1. **Multi-dimensional Context**: While transformers primarily operate on sequential context within a single dimension (token sequence), CHEESE orchestrates multiple context dimensions simultaneously (semantic, temporal, associative, activation).  
     
2. **Persistent Memory Structure**: Unlike transformers, which typically process each input independently, CHEESE maintains a persistent memory structure that evolves over time, allowing for long-term context preservation.  
     
3. **Biologically-Inspired Dynamics**: CHEESE incorporates biological memory principles like activation decay and associative strengthening that aren't present in standard transformer architectures.  
     
4. **Hierarchical Organization**: The system supports emergent hierarchical organization through its category management, enabling more efficient knowledge representation than flat attention patterns.

These extensions allow CHEESE to address limitations in standard transformer approaches to long-term memory, particularly for extended conversational contexts where relevant information may span hours, days, or longer timeframes.

## 5\. Implementation

### 5.1 Query Analysis and Adaptation

CHEESE begins retrieval by analyzing the query to determine its type (factual, personal, temporal, etc.) and extracting important keywords and entities. This analysis drives parameter adaptation:

```python
def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
    # Identify query type
    query_types = self.nlp_extractor.identify_query_type(query)
    primary_type = self._determine_primary_type(query, query_types)

    # Extract important keywords
    keywords = self.nlp_extractor.extract_important_keywords(query)

    # Recommend retrieval parameters based on query type
   param_recommendations = self._get_parameter_recommendations(
        primary_type,
        query_types,
    )
    return {
        "query_types": query_types,
        "primary_query_type": primary_type,
        "important_keywords": keywords,
        "retrieval_param_recommendations": param_recommendations,
    }
```


For example, factual queries receive lower confidence thresholds to prioritize recall, personal queries receive higher thresholds to prioritize precision, and temporal queries increase the weight of time-based signals.

### 5.2 Contextual Fabric Retrieval

The core retrieval process in ContextualFabricStrategy works as follows:

1. **Initial Embedding Similarity**: Calculate semantic similarity between query and memory embeddings  
     
2. **Temporal Context Integration**: Extract temporal references from the query and boost memories from relevant time periods  
   ```python
    # Example of temporal context extraction  
    temporal_results = self._retrieve_temporal_results(query, context, memory_store)
   ```
     
3. **Associative Network Traversal**: Activate memories connected associatively to the most relevant results
    ```python
    # Example of associative traversal  
    activations = self.associative_linker.traverse_associative_network(
        start_id=memory_id,
        max_hops=self.max_hops,
        min_strength=0.3
    )
    ```

     
4. **Activation Influence**: Boost frequently accessed memories through activation scores  
     
```python
    # Example of activation boosting

    activation_contribution = self._calculate_activation_contribution(
        memory_id, activation_score, base_weight, similarity
    )
```

5. **Signal Composition**: Combine these signals using adaptive weights based on query characteristics  
```python

# Example of weighted score calculation
combined_score = (
    similarity_contribution
    + associative_contribution
    + temporal_contribution
    + activation_contribution
)
```

6. **Post-Processing**: Apply coherence checks, keyword boosting, and adaptive thresholding to refine results

This orchestrated approach ensures that retrieval considers multiple dimensions of context, leading to more coherent and relevant results.

### 5.3 Memory Dynamics

As the system interacts with new information, several dynamic processes occur:

- **Activation Updates**: Accessed memories receive activation boosts that decay over time according to biologically-inspired decay functions

```python
def update_activation(self, memory_id: MemoryID, activation_delta: float) -> None:
    """Update activation level for a memory."""
    if memory_id not in self._activations:
        raise KeyError(f"Memory with ID {memory_id} not found")
    self._activations[memory_id] += activation_delta
```

- **Associative Learning**: New links form between semantically similar or temporally proximate memories, strengthening with repeated co-activation

```python

def create_associative_link(
    self, source_id: MemoryID,
    target_id: MemoryID,
    strength: float = 0.5,
) -> None:
    """Create an associative link between two memories."""
    # Add forward link
    self._add_or_update_link(source_id, target_id, strength)
    # Add reverse link with slightly reduced strength
    self._add_or_update_link(target_id, source_id, strength * 0.8)
```

- **Category Evolution**: The category structure adapts as new memories are added, with categories merging when they become sufficiently similar  

```python
def consolidate_categories(self, threshold: float | None = None) -> int:
    """Merge similar categories to keep the number manageable."""
    # Find pairs of categories to merge based on similarity
    # Merge the identified categories
    # Return the number of remaining categories
```

- **Memory Consolidation**: Periodically, similar categories merge and the link structure is optimized to maintain system efficiency

These dynamic processes enable the system to adapt to new information while maintaining a coherent memory structure.

## 6\. Preliminary Evaluation

Preliminary evaluation of CHEESE has focused on four key metrics:

1. **Retrieval Precision**: The relevance of retrieved memories to the query  
2. **Contextual Coherence**: How well the retrieved memories form a coherent context  
3. **Temporal Accuracy**: The system's ability to respond to time-based queries  
4. **Conversation Continuity**: Maintaining contextual awareness across a conversation

Initial results show significant improvements over baseline vector similarity approaches:

| Metric | Baseline (Vector Similarity) | CHEESE | Improvement |
| :---- | :---- | :---- | :---- |
| Retrieval Precision | 0.67 | 0.83 | \+24% |
| Contextual Coherence | 0.46 | 0.63 | \+37% |
| Temporal Query Accuracy | 0.51 | 0.72 | \+42% |
| Conversation Continuity | 0.58 | 0.76 | \+31% |

*Note: These results are from preliminary testing on topic-based queries under controlled test conditions. Performance on complex cross-domain queries and certain edge cases showed more variable results, with some scenarios showing significantly lower success rates. The system parameters were empirically tuned through extensive testing across multiple scenarios.*

These results are based on a test dataset of 500 queries across various domains and query types. The implementation used for evaluation represents approximately 60% of the full system design, with certain components (like the ChunkedFabricStrategy) still under development.

While these results are promising, they should be considered preliminary. The parameter complexity and edge case failures ultimately led to exploring alternative mathematical approaches, as described in subsequent work.

## 7\. Discussion and Future Work

The CHEESE project demonstrates the potential of biologically-inspired memory systems to enhance LLM capabilities. By moving beyond simple vector similarity to a rich contextual fabric, the system enables more human-like memory access patterns.

### 7.1 Current Limitations

Several limitations in the current implementation should be acknowledged:

1. **Computational Efficiency**: The orchestration of multiple signals introduces computational overhead compared to simple vector similarity  
2. **Parameter Tuning**: The system includes numerous parameters that require careful tuning for optimal performance  
3. **Evaluation Methodology**: Standardized benchmarks for memory systems in conversational AI are still evolving  
4. **Implementation Completeness**: Not all described components are fully implemented in the current version

### 7.2 Lessons Learned

The development of CHEESE revealed several important insights:

**Parameter Interdependence**: The 47+ tunable parameters created a complex optimization landscape where changes to one parameter often had cascading effects on others.

**Biological Inspiration Has Limits**: While biological memory systems provided valuable design insights, directly implementing biological mechanisms led to unwieldy complexity.

**Mathematical Structure Matters**: The ultimate limitation was not the number of parameters but attempting to force hierarchical and multi-aspect relationships into Euclidean space where they don't naturally fit.

**Edge Cases Reveal Fundamental Issues**: The system's failures on cross-domain queries and certain multi-hop reasoning tasks pointed to deeper architectural limitations rather than simple parameter tuning issues.

These insights led to exploring alternative mathematical frameworks that could better represent the geometric and topological structure of information, as detailed in subsequent work on hyperbolic embeddings and fiber bundle architectures.

### 7.3 Broader Implications

The contextual fabric approach has implications beyond conversational AI:

1. **Cognitive Architecture**: Contributing to more human-like AI memory systems  
2. **Knowledge Management**: Enabling more effective organization of large information repositories  
3. **Personalized AI**: Supporting truly personalized AI assistants with rich memory of interactions  
4. **Educational Applications**: Creating systems that can adapt their memory to individual learning patterns

## 8\. Conclusion

CHEESE introduces a contextual fabric approach to memory management for Large Language Models, moving from discrete structures to a rich multidimensional representation inspired by human memory. By orchestrating multiple signals—semantic, temporal, associative, and activation-based—the system demonstrates improved retrieval performance on topic-based queries.

However, the biological inspiration that initially guided the design also introduced significant complexity. The 47+ interdependent parameters created a tuning nightmare, and performance on edge cases revealed fundamental architectural limitations. The system's struggles with cross-domain queries and unpredictable failures in multi-hop reasoning pointed to a deeper issue: attempting to force complex relationships into mathematical spaces where they don't naturally belong.

This work serves as both a proof of concept for contextual memory systems and a cautionary tale about the limits of biological metaphors in AI. The parameter explosion and edge case failures ultimately led to exploring more mathematically principled approaches using hyperbolic geometry and fiber bundles, which better capture the natural structure of hierarchical and multi-aspect information.

CHEESE represents an important stepping stone in understanding how to build better memory systems for AI—not just through what it achieved, but through what its limitations revealed about the fundamental geometry of information.

## References

1. Weston, J., Chopra, S., & Bordes, A. (2014). Memory networks. arXiv preprint arXiv:1410.3916.  
     
2. Sukhbaatar, S., Szlam, A., Weston, J., & Fergus, R. (2015). End-to-end memory networks. Advances in neural information processing systems, 28\.  
     
3. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33, 9459-9474.  
     
4. Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. Advances in neural information processing systems, 26\.  
     
5. Miller, A., Fisch, A., Dodge, J., Karimi, A. H., Bordes, A., & Weston, J. (2016). Key-value memory networks for directly reading documents. arXiv preprint arXiv:1606.03126.  
     
6. Anderson, J. R., & Bower, G. H. (1973). Human associative memory. Psychology press.  
     
7. Grossberg, S. (2013). Adaptive Resonance Theory: How a brain learns to consciously attend, learn, and recognize a changing world. Neural networks, 37, 1-47.  
     
8. Gao, J., Galley, M., & Li, L. (2018). Neural approaches to conversational AI. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval (pp. 1371-1374).  
     
9. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30\.  
     
10. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3), 535-547.
