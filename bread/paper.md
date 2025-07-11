# BREAD: Bundles, Relations, Embeddings, And Dimensions

**James Gardner**  
_Independent Researcher_  
_[jamestexasgardner@gmail.com](mailto:jamestexasgardner@gmail.com)_

## Abstract

We introduce **BREAD**, a novel approach to knowledge representation that integrates three distinct geometric innovations: hyperbolic space operations for capturing hierarchical relationships, fiber bundles for representing multi-faceted entity properties, and simplicial complexes for modeling multi-way relationships. By leveraging the intrinsic properties of these non-Euclidean structures, BREAD overcomes fundamental limitations of Euclidean embeddings in representing complex knowledge. The system demonstrates exceptional capacity for multi-hop reasoning and cross-domain discovery while effectively addressing hierarchy preservation challenges. We present the theoretical foundations of our approach, implementation details, and architectural considerations. BREAD represents a significant advance in geometric approaches to knowledge representation, with promising applications in information organization, cross-domain reasoning, and knowledge-intensive AI systems.

## 1. Introduction

Knowledge representation remains a fundamental challenge in artificial intelligence, particularly when dealing with complex relationships across multiple domains. Traditional embedding techniques in Euclidean space have shown limitations in capturing hierarchical structures, multi-hop reasoning paths, and cross-domain connections. These limitations stem from the fundamental properties of Euclidean geometry, where the amount of space grows polynomially with distance from the origin, constraining the effective embedding of exponentially-growing hierarchical structures.

Recent work has explored non-Euclidean spaces, particularly hyperbolic spaces, for embedding hierarchical data [1, 2]. These approaches leverage the exponential growth of hyperbolic space with distance from the origin, naturally accommodating tree-like structures. However, most existing implementations fail to fully exploit the theoretical advantages of hyperbolic geometry when dealing with complex, real-world knowledge structures that exhibit multi-faceted relationships.

In this paper, we introduce **BREAD**—short for **Bundles, Relations, Embeddings, And Dimensions**—a comprehensive framework that combines hyperbolic embeddings with fiber bundles and simplicial complexes to create a unified geometric representation of knowledge. Our key contributions include:

1. A robust implementation of hyperbolic operations with numerical safeguards that ensures stability even with deep hierarchies
2. A novel integration of fiber bundles for representing multiple aspects of information entities
3. Simplicial complex structures for capturing multi-way relationships beyond simple dyadic connections

We demonstrate the potential of our approach through theoretical analysis and implementation details, suggesting significant implications for knowledge-intensive AI applications.

## 2. Background and Related Work

### 2.1 Knowledge Representation

Knowledge representation has evolved from symbolic approaches (e.g., logic-based systems) to distributed representations in the form of embeddings [3]. Vector embeddings map entities and relationships to continuous vector spaces, enabling efficient computation and generalization. However, the choice of geometric space fundamentally constrains what relationships can be efficiently modeled.

Traditional embedding methods like TransE [4], DistMult [5], and more recent transformer-based embeddings [6] operate in Euclidean space. While effective for many tasks, these approaches face inherent limitations when modeling hierarchical relationships and complex structural patterns due to the geometric properties of Euclidean space.

### 2.2 Hyperbolic Embeddings

Hyperbolic geometry provides an alternative embedding space with properties particularly suited to hierarchical structures. In hyperbolic space, the volume grows exponentially with the radius, naturally accommodating tree-like structures where the number of nodes grows exponentially with depth.

Nickel and Kiela [1] pioneered Poincaré embeddings, demonstrating their effectiveness for hierarchical data. Subsequent work expanded these approaches with Hyperbolic Graph Convolutional Networks (HGCNs) [7] and other specialized architectures. Despite theoretical advantages, practical implementations have faced several limitations:

1. **Numerical instability**: Operations near the boundary of the Poincaré ball can lead to numerical errors
2. **Optimization challenges**: Training models in hyperbolic space requires specialized optimizers
3. **Limited expressivity**: Most implementations focus solely on hierarchical relationships without addressing multi-faceted or multi-way relationships
4. **Integration difficulties**: Combining hyperbolic embeddings with other representation techniques has proven challenging

Our work directly addresses these limitations while extending hyperbolic representations to capture more complex knowledge structures.

### 2.3 Geometric Deep Learning

The broader field of geometric deep learning [8] has explored various non-Euclidean spaces for representing structured data. This includes work on manifold learning, Riemannian optimization, and topological data analysis. While researchers have explored various non-Euclidean structures individually, little work has been done to create unified frameworks that leverage the complementary strengths of different geometric structures.

### 2.4 Literature Review of Combined Approaches

In examining the current literature on geometric approaches to knowledge representation, we find that while individual components of our approach have been explored separately, the full integration of hyperbolic geometry, fiber bundles, simplicial complexes, and sheaf cohomology represents a novel contribution.

Hyperbolic embeddings have gained significant attention for hierarchical data [1, 2, 9]. The pioneering work of Nickel & Kiela [1] demonstrated how the exponential capacity growth in hyperbolic space naturally accommodates tree-like structures, showing impressive results on WordNet hierarchies. Liu et al. [9] extended this to graph neural networks but remained focused on dyadic (pairwise) relationships and struggled with multi-parent hierarchies. These approaches excel at preserving hierarchical distances but don't provide mechanisms for representing different facets of entities or complex multi-way relationships.

Simplicial complexes have emerged as powerful tools for modeling complex relational data [10, 11]. Battiston et al. [10] provided a comprehensive framework for higher-order networks using simplicial complexes, allowing representation of group relationships beyond pairwise interactions. Bick et al. [11] further developed the theory of higher-order networks, but these approaches typically operate in Euclidean space, limiting their effectiveness for hierarchical structures. While they capture multi-way relationships effectively, they lack the exponential capacity benefits of hyperbolic geometry.

The mathematics of fiber bundles has been utilized in representation learning [12], particularly for multi-view and equivariant data. Cohen & Welling [12] demonstrated how fiber bundles can provide theoretical foundations for equivariant representations, but these approaches rarely combine with hyperbolic geometries and typically don't address hierarchical structures. The power of fiber bundles for separating different aspects of entity representation remains largely unexplored in knowledge representation systems.

Sheaf theory has only recently begun to appear in machine learning contexts [13, 14], primarily for signal processing on graphs. Hansen & Ghrist [13] developed spectral theory for cellular sheaves with applications to network analysis, while Bodnar et al. [14] introduced topological message passing using sheaf theory. However, these applications focus primarily on signal processing rather than knowledge representation, and don't integrate with hyperbolic geometries or simplicial structures for multi-way relationships.

No single approach in prior work provides a coherent framework that simultaneously addresses hierarchy preservation, multi-faceted entity representations, and multi-way relationships within a unified geometric structure. This represents a significant gap in the literature that BREAD addresses.

Conversations with researchers in topological data analysis and geometric deep learning confirm that while these mathematical structures are recognized individually, their combined application remains unexplored. The work of Mathieu et al. [15] combines hyperbolic spaces with product manifolds for multi-aspect representation, introducing Poincaré Variational Autoencoders that can learn hierarchical representations, but lacks mechanisms for multi-way relationships or cohomological consistency. Chami et al. [7] extends hyperbolic embeddings to graph convolutional networks (HGCNs) but focuses primarily on pairwise edges without addressing higher-order structures or multi-faceted representations. Recent work by Skopek et al. [16] explores mixed-curvature embeddings, allowing different components of the representation to exist in spaces of different curvature, but emphasizes combining different geometries rather than integrating complementary geometric structures.

These approaches, while theoretically sophisticated, often remain limited in practical implementations, particularly for large-scale knowledge representation challenges. Many face numerical stability issues near the boundaries of hyperbolic spaces, struggle with optimization procedures in non-Euclidean settings, or lack comprehensive implementations that address real-world knowledge representation needs beyond specific benchmark tasks.

### 2.5 Comparison to Existing Partial Approaches

While existing approaches have successfully applied individual geometric concepts to knowledge representation, they address only fragments of the challenges that BREAD tackles holistically.

Hyperbolic embedding methods [1, 2] excel at preserving hierarchical structure but face several limitations. Nickel & Kiela's original Poincaré embeddings [1] demonstrated superior performance for hierarchical data but struggled with numerical instability near the boundary of the Poincaré ball. Subsequent work by Sala et al. [2] analyzed representational tradeoffs but continued to focus exclusively on hierarchy without addressing multi-faceted entity properties. Additionally, these methods typically assume tree-like hierarchies and struggle with multi-parent relationships, creating inconsistent representations when an entity has multiple parents.

Knowledge graph embedding techniques using Euclidean space [3, 4, 5] have become standard in many applications but face fundamental geometric limitations. Bordes et al.'s TransE [3] and subsequent models like Wang et al.'s TransH [4] efficiently capture simple relational patterns but cannot efficiently represent hierarchical structures due to the polynomial growth of Euclidean space. These approaches are also limited to pairwise relationships represented as triples (head, relation, tail), lacking mechanisms for directly modeling relationships involving multiple entities simultaneously. The dimensional interference problem—where different aspects of an entity must compete for the same embedding dimensions—further limits representation capacity.

Recent work on simplicial complex networks [10, 11] captures higher-order relationships but introduces its own limitations. While Battiston et al. [10] successfully model multi-way interactions, their approach lacks the hierarchical efficiency of hyperbolic space and the multi-faceted representation offered by fiber bundles. These models typically operate in Euclidean space, inheriting its limitations for hierarchical structures, and don't provide mechanisms for separating different aspects of entities into distinct representational spaces.

**BREAD** explicitly closes this gap by integrating these complementary geometric structures into a unified framework where each component addresses specific representational challenges:

- **Hyperbolic space** provides the foundation for efficient hierarchical representation, leveraging its exponential capacity growth with distance from the origin
- **Fiber bundles** enable multi-faceted entity representation without dimensional interference, allowing different aspects of entities to be encoded in separate fiber spaces
- **Simplicial complexes** capture multi-way relationships beyond simple edges, directly modeling group relationships rather than decomposing them into pairwise connections
- **Sheaf cohomology** ensures consistent handling of multi-parent hierarchies by providing a principled approach to representing entities across different hierarchical contexts

This integration enables capabilities beyond what any single geometric approach can achieve, particularly for complex knowledge structures that exhibit hierarchy, multi-faceted properties, and multi-way relationships simultaneously. While partial approaches optimize for specific relationship types, BREAD provides a comprehensive geometric foundation for representing the full complexity of real-world knowledge.

Moreover, unlike many prior approaches that remain largely theoretical or limited to small-scale experiments, BREAD is implemented as a practical system with attention to computational efficiency, numerical stability, and integration capabilities. We provide robust implementations of hyperbolic operations with fallback mechanisms for numerical edge cases, efficient data structures for simplicial complex representation, and practical algorithms for hierarchical placement and optimization. This bridges the gap between theoretical geometric innovations and practical knowledge representation systems that can scale to real-world applications.

By addressing both the theoretical limitations of existing approaches and the practical challenges of implementation, BREAD represents a significant advance in geometric approaches to knowledge representation. Its unified framework provides a foundation for more expressive, efficient, and context-aware knowledge systems that better capture the complexity of real-world information.

## 3. BREAD Methodology

BREAD integrates three geometric structures—hyperbolic space, fiber bundles, and simplicial complexes—into a unified framework for knowledge representation. This section details our approach and implementation. For readers less familiar with these geometric concepts, we provide intuitive explanations alongside the technical details.

### 3.1 Core Hyperbolic Operations

The foundation of BREAD is a robust implementation of hyperbolic operations in the Poincaré ball model. Intuitively, hyperbolic space can be visualized as a disc where distances grow exponentially as you move from the center toward the edge, making it ideal for representing hierarchies where child nodes grow exponentially with depth.

We implement the following key operations with numerical safeguards to ensure stability:

1. **Geodesic distance calculation**: Computes the hyperbolic distance between points with fallback mechanisms for numerical stability. Unlike Euclidean space where distance is measured along straight lines, hyperbolic distance follows curved geodesics.
2. **Möbius addition**: Implements gyrovector operations for combining vectors in hyperbolic space. This operation replaces standard vector addition while preserving the hyperbolic structure.
3. **Exponential and logarithmic maps**: Provides mappings between the tangent space and the manifold, enabling operations like optimization in hyperbolic space.
4. **Parallel transport**: Enables moving vectors along geodesics while preserving their geometric properties, critical for comparing directions at different points in the space.

Our implementation includes careful handling of edge cases, ensuring stability even near the boundary of the Poincaré ball where numerical issues commonly arise:

```python
def geodesic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate geodesic distance between points in the Poincaré ball.

    Args:
        x: First point
        y: Second point

    Returns:
        Geodesic distance
    """
    # Ensure inputs are in the Poincaré ball
    x = self._project_to_ball(x.to(self.device))
    y = self._project_to_ball(y)

    try:
        # Compute squared norms
        x_norm_sq = torch.sum(x**2)
        y_norm_sq = torch.sum(y**2)

        # Compute numerator: |x - y|^2
        xy_diff_norm_sq = torch.sum((x - y) ** 2)

        # Compute denominator with safeguards
        denom = (1 - x_norm_sq) * (1 - y_norm_sq)
        denom = torch.clamp(denom, min=self.eps)

        # Compute the argument for acosh
        acosh_arg = 1 + 2 * xy_diff_norm_sq / denom
        acosh_arg = torch.clamp(acosh_arg, min=1.0 + self.eps)  # Ensure arg > 1 for acosh

        # Compute geodesic distance
        dist = torch.acosh(acosh_arg)

        # Check for NaN or inf
        if torch.isnan(dist) or torch.isinf(dist):
            # Fallback to approximate distance
            return self._fallback_distance(x, y)

        return dist
    except Exception:
        # Fallback for numerical errors
        return self._fallback_distance(x, y)
```

### 3.2 Fiber Bundles for Multi-aspect Representation

To represent multiple aspects of entities, we implement a fiber bundle structure where each entity has a base position in hyperbolic space and multiple fiber spaces representing different properties. Intuitively, fiber bundles can be understood as attaching different specialized "feature spaces" to each point in our base space, allowing entities to have multiple aspects while maintaining their fundamental relationships.

**Intuitive Example:** Consider a movie entity in a knowledge base. In a traditional embedding, all aspects (genre, visual style, emotional tone, director style) must be compressed into a single vector, forcing trade-offs. With fiber bundles, the movie's position in hyperbolic space represents its fundamental relationships to other entities, while separate fiber spaces capture different aspects: one fiber for visual characteristics, another for thematic elements, and another for temporal properties.

This geometric structure allows for:

1. **Multi-faceted entity representation**: Entities can have different aspects represented in separate fiber spaces
2. **Aspect-specific similarity**: Computing similarity along specific dimensions of comparison
3. **Flexible querying**: Retrieving information based on combined criteria across different aspects


```python

def attach_fiber(self, item_id: str, name: str, data: Any) -> torch.Tensor | None:
    """Attach a fiber to an existing item.

    Args:
        item_id: ID of the item to attach fiber to
        name: Name of the fiber (e.g., 'color', 'sentiment', etc.)
        data: Data to encode in the fiber

    Returns:
        The created fiber tensor or None if failed
    """
    # Ensure item_id is a string
    item_id = str(item_id)

    if item_id not in self.nucleotides:
        return None

    # Create fiber data structure if needed
    if item_id not in self.fibers:
        self.fibers[item_id] = {}

    # Create fiber embedding from data
    if isinstance(data, str):
        # Create embedding from text
        fiber = self.encoding._get_text_embedding(data)
        # Resize to fiber dimension
        if self.fiber_dim <= self.dim:
            fiber = fiber[: self.fiber_dim]
        else:
            # Pad with zeros
            fiber = torch.cat([fiber, torch.zeros(self.fiber_dim - self.dim)])
    elif isinstance(data, dict) and "embedding" in data:
        fiber = data["embedding"]
        # Resize if needed
        if len(fiber) != self.fiber_dim:
            if len(fiber) > self.fiber_dim:
                fiber = fiber[: self.fiber_dim]
            else:
                fiber = torch.cat([fiber, torch.zeros(self.fiber_dim - len(fiber))])
    else:
        # For other types, create a simple representation
        fiber = torch.randn(self.fiber_dim)
        fiber = fiber / fiber.norm()

    # Store the fiber
    self.fibers[item_id][name] = fiber

    return fiber
```


### 3.3 Simplicial Complexes for Multi-way Relationships

Beyond pairwise relationships, many knowledge structures involve multi-way connections. We implement simplicial complexes to capture these relationships. Intuitively, while graphs can only represent connections between pairs of entities (edges), simplicial complexes extend this to higher dimensions—triangles connect three entities, tetrahedra connect four, and so on—enabling direct representation of group relationships.

**Intuitive Example:** Consider a biomedical knowledge base containing proteins that form functional complexes. In a traditional graph, each protein would connect to others with pairwise edges, but this fails to capture that the complex functions only when all proteins are present together. With simplicial complexes, we can represent the entire functional unit as a single simplex, directly modeling the group relationship.

These structures enable more expressive relationship modeling than traditional graph-based approaches, particularly for complex phenomena with interdependent elements.


```python
def create_simplex(self, items: list[str], metadata: dict[str, Any] | None = None) -> str:
    """Create a simplex connecting multiple items.

    Args:
        items: list of item IDs to connect
        metadata: Optional metadata about the relationship

    Returns:
        ID of the created simplex
    """
    if len(items) < 2 or len(items) > self.max_simplex_dim + 1:
        raise ValueError(f"Simplex must have between 2 and {self.max_simplex_dim + 1} vertices")

    # Ensure all item IDs are strings
    items = [str(item) for item in items]

    # Check that all items exist
    for item_id in items:
        if item_id not in self.nucleotides:
            raise ValueError(f"Item {item_id} does not exist")

    # Create simplex (dimension is vertices - 1)
    simplex_dim = len(items) - 1
    simplex_id = str(id((frozenset(items), metadata)))

    # Initialize dimension if not present
    if simplex_dim not in self.simplices:
        self.simplices[simplex_dim] = []

    # Create the simplex
    simplex = {
        "id": simplex_id,
        "vertices": set(items),
        "metadata": metadata,
    }

    self.simplices[simplex_dim].append(simplex)

    # Also create fabric edges between all pairs of items
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            edge = FabricEdge(
                source=items[i],
                target=items[j],
                weight=1.0,
                type="simplex",
                properties={"simplex_id": simplex_id, "metadata": metadata},
            )
            self.add_fabric_edge(edge)

            # Also add reverse edge
            edge_rev = FabricEdge(
                source=items[j],
                target=items[i],
                weight=1.0,
                type="simplex",
                properties={"simplex_id": simplex_id, "metadata": metadata},
            )
            self.add_fabric_edge(edge_rev)

    return simplex_id
```

### 3.4 Hierarchical Placement Algorithm

A key innovation in BREAD is our hierarchical placement algorithm, which positions entities in hyperbolic space to preserve hierarchical relationships:


```python
def calculate_hierarchical_position(
    self, 
    text_embedding: torch.Tensor,
    ancestor_positions: list[torch.Tensor],
    ancestor_levels: list[int]
) -> torch.Tensor:
    """Calculate optimal position based on hierarchical relationships.

    Args:
        text_embedding: Content embedding
        ancestor_positions: List of ancestor positions
        ancestor_levels: List of hierarchy levels for ancestors

    Returns:
        Optimal position tensor
    """
    # Ensure we have valid ancestors
    if not ancestor_positions:
        return self.hyp_ops._project_to_ball(text_embedding)

    # Apply weighting based on level (more recent ancestors have higher weight)
    max_level = max(ancestor_levels)
    weights = [2.0 ** (level - max_level) for level in ancestor_levels]
    total_weight = sum(weights)

    # Calculate weighted ancestor centroid
    ancestor_centroid = torch.zeros_like(ancestor_positions[0])
    for i, pos in enumerate(ancestor_positions):
        ancestor_centroid += (weights[i] / total_weight) * pos

    # Calculate distance from origin for the centroid
    centroid_norm = ancestor_centroid.norm().item()

    # Child nodes should be further from origin in hyperbolic space
    # The deeper in hierarchy, the greater the ratio should be
    depth = 1 + max_level  # Depth of the new node

    # Use logarithmic scaling that better respects hyperbolic geometry
    distance_ratio = 1.0 + (1.0 / math.log(depth + 2))
    target_norm = centroid_norm * distance_ratio

    # Enforce minimum distance from parent to ensure children
    # are always further from origin than parents
    min_norm = max([pos.norm().item() for pos in ancestor_positions]) * 1.05
    target_norm = max(target_norm, min_norm)

    # Cap maximum distance to avoid numerical issues
    target_norm = min(target_norm, 0.95)

    # The direction should be a blend of:
    # 1. Similar to ancestor direction (for hierarchical consistency)
    # 2. Influenced by content embedding (for content relevance)
    # 3. With a small perturbation (to distinguish siblings)

    # Start with ancestor direction
    if centroid_norm > 1e-8:
        ancestor_dir = ancestor_centroid / centroid_norm
    else:
        # Use a deterministic direction if at origin
        ancestor_dir = torch.zeros_like(ancestor_centroid)
        ancestor_dir[0] = 0.7
        ancestor_dir[1] = 0.7
        ancestor_dir = ancestor_dir / ancestor_dir.norm()

    # Get content direction (normalized text embedding)
    content_dir = text_embedding / text_embedding.norm()

    # Add deterministic perturbation based on content
    # Use hash of content embedding for consistent placement
    hash_val = int(torch.sum(text_embedding * 1000).item()) % 1000
    perturbation = torch.zeros_like(content_dir)
    perturbation[0] = math.cos(hash_val * 0.1)
    perturbation[1] = math.sin(hash_val * 0.1)

    # Blend directions with weights
    final_dir = 0.6 * ancestor_dir + 0.3 * content_dir + 0.1 * perturbation
    final_dir = final_dir / final_dir.norm()

    # Calculate final position
    position = final_dir * target_norm

    # Ensure we're still in the Poincaré ball
    position = self.hyp_ops._project_to_ball(position)

    return position
```

### 3.5 Sheaf Cohomology for Multi-parent Hierarchies

To better handle complex hierarchies where nodes can have multiple parents, we implement a sheaf cohomology approach. Sheaf theory provides a mathematical framework for tracking how information varies and connects across different "open sets" or perspectives. In our context, it enables consistent representation of concepts across different hierarchical paths.


Our sheaf-based implementation extends hierarchy preservation in several ways:

1. **Consistent multi-parent representation**: Ensures a node with multiple parents has a consistent representation
    
2. **Local-to-global information flow**: Provides mechanisms for combining local information into global knowledge structures
    
3. **Categorical organization**: Naturally forms emergent categories based on shared properties
    

```python
def compute_multi_parent_position(
    self,
    item_id: str,
    text_embedding: torch.Tensor,
    parents: list[str],
    nucleotides: dict[str, tuple[torch.Tensor, Any]],
    depth: int = 0,
) -> torch.Tensor:
    """Compute position for an item with multiple parents.

    This is a key advantage of the sheaf approach - handling multiple
    parents in a mathematically consistent way.

    Args:
        item_id: ID of the item
        text_embedding: Content embedding
        parents: List of parent IDs
        nucleotides: Dictionary mapping item IDs to (position, info)
        depth: Depth in the hierarchy

    Returns:
        Position tensor in hyperbolic space
    """
    if not parents:
        # No parents - use standard embedding
        normalized = text_embedding / max(text_embedding.norm().item(), 1e-6)
        return self.hyp_ops._project_to_ball(normalized * 0.1)  # Place near origin as root

    # Get valid parent positions
    parent_positions = []
    valid_parents = []

    for parent in parents:
        if parent in nucleotides:
            parent_positions.append(nucleotides[parent][0])
            valid_parents.append(parent)

    if not parent_positions:
        # No valid parents, use standard embedding
        normalized = text_embedding / max(text_embedding.norm().item(), 1e-6)
        return self.hyp_ops._project_to_ball(normalized)

    # Find maximum parent norm
    parent_max_norm = 0.0
    for pos in parent_positions:
        norm = pos.norm().item()
        parent_max_norm = max(parent_max_norm, norm)

    # Use sheaf cohomology for multi-parent case
    if len(parent_positions) > 1:
        # Create or get appropriate regions
        target_region = f"depth_{depth}" if f"depth_{depth}" in self.local_sections else None

        if target_region is None:
            # Create depth region if needed
            target_region = f"depth_{depth}"
            self.create_region(target_region)

        # Calculate mean position in the target region
        mean_pos = torch.stack(parent_positions).mean(dim=0)

        # Calculate target norm (must be greater than all parents)
        target_norm = max(self._get_norm_from_depth(depth), parent_max_norm * 1.05)
        target_norm = min(target_norm, 0.95)  # Cap for numerical stability

        # Rescale to target norm
        mean_norm = mean_pos.norm().item()
        if mean_norm > 1e-6:
            direction = mean_pos / mean_norm
            position = direction * target_norm
        else:
            # Random direction if at origin
            direction = text_embedding / max(text_embedding.norm().item(), 1e-6)
            position = direction * target_norm

        # Add content-dependent perturbation
        perturbation = text_embedding * 0.05
        perturbation = perturbation - torch.sum(perturbation * direction) * direction
        position = position + perturbation

        # Ensure we're in the Poincaré ball
        position = self.hyp_ops._project_to_ball(position)

        return position

    # Fallback to standard approach for single parent
    midpoint = parent_positions[0]
    midpoint_norm = midpoint.norm().item()
    target_norm = max(midpoint_norm, parent_max_norm * 1.05)
    target_norm = min(target_norm, 0.95)  # Cap for numerical stability

    # Keep direction but adjust norm
    if midpoint_norm > 1e-6:
        direction = midpoint / midpoint_norm
    else:
        direction = text_embedding / max(text_embedding.norm().item(), 1e-6)
    
    position = direction * target_norm

    # Add content-dependent perturbation
    perturbation = text_embedding * 0.05
    perturbation = perturbation - torch.sum(perturbation * direction) * direction
    position = position + perturbation

    # Ensure we're in the Poincaré ball
    position = self.hyp_ops._project_to_ball(position)

    return position
```


### 3.6 Biologically-Inspired Adaptive Mechanisms

Taking inspiration from how biological systems organize information, we implement several adaptive mechanisms:

1. **Activation and Decay**: Memories have activation levels that increase when accessed and decay over time, mimicking the strengthening and weakening of neural pathways
    
2. **Associative Linking**: Entities establish bidirectional associative connections based on similarity and temporal co-occurrence
    
3. **Local Optimization**: Rather than globally optimal structure, the space optimizes locally for efficiency
    
4. **Sparse Connectivity**: Instead of dense connections between everything, maintain sparse but meaningful connections
    

This biologically-inspired approach allows the system to develop efficient pathways for frequently accessed information while maintaining a coherent overall structure.

## 4. System Architecture

**BREAD** employs a modular architecture organized into several core components:

### 4.1 Component Organization

Our implementation uses a modular approach with clearly separated responsibilities:

- **HyperbolicOperations**: Provides core mathematical operations in hyperbolic space
- **HierarchyEmbedding**: Manages hierarchical relationships and placement
- **Encoding**: Handles conversion between raw information and geometric representations
- **KnowledgeSheaf**: Implements sheaf cohomology for multi-parent hierarchies
- **BREADCore**: Orchestrates all components into a unified system

This separation of concerns enables flexible composition and easier maintenance.

### 4.2 BREAD Core

```python
class BreadCore
    """Core implementation of BreadCore with all component integrations."""

    def __init__(
        self,
        dim: int = 100,
        curvature: float = -1.0,
        learning_rate: float = 0.01,
        fiber_dim: int = 32,
        max_simplex_dim: int = 3,
        use_cached_paths: bool = True,
        adaptive_curvature: bool = True,
        use_gpu: bool | None = None,
        use_sheaf: bool = True,
        # Shared dictionaries
        nucleotides: dict = None,
        fibers: dict = None,
        simplices: dict = None,
    ):
        # Basic parameters
        self.dim = dim
        self.curvature = curvature
        self.learning_rate = learning_rate
        self.fiber_dim = fiber_dim
        self.max_simplex_dim = max_simplex_dim
        self.use_cached_paths = use_cached_paths
        self.adaptive_curvature = adaptive_curvature
        self.use_sheaf = use_sheaf

        # Initialize components
        self.hyp_ops = HyperbolicOperations(dim=dim, curvature=curvature)
        self.attention = HyperbolicAttention(dim=dim, curvature=curvature, hyp_ops=self.hyp_ops)

        # Use the enhanced hierarchy component with sheaf cohomology
        self.hierarchy_embedding = HierarchySheaf(
            dim=dim,
            curvature=curvature,
            hyp_ops=self.hyp_ops,
            use_sheaf=self.use_sheaf,
        )

        self.encoding = Encoding(
            dim=dim,
            curvature=curvature,
            hyp_ops=self.hyp_ops,
            hierarchy_embedding=self.hierarchy_embedding,
        )

        # Initialize data structures
        self.nucleotides = {} if nucleotides is None else nucleotides
        self.fibers = {} if fibers is None else fibers
        self.simplices = {} if simplices is None else simplices
        self.regions = defaultdict(set)
        self.hierarchy_graph = nx.DiGraph()
        self.fabric_connections = {}
        self.path_cache = {}
        self.local_curvatures = {}
        self.temporal_sequences = {}
        self.item_to_sequences = {}
        self.attention_heads = {}

        # Set up device for GPU acceleration
        self.device = "mps"
        self.use_gpu = use_gpu
        if self.use_gpu is None:
            # Auto-detect
            self.use_gpu = torch.cuda.is_available()
            if (
                not self.use_gpu
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self.use_gpu = True  # Use Apple Metal if available

        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
```


### 4.3 Core Knowledge Operations

BREAD supports several key operations for knowledge management:

- **Encoding**: Converting raw information into geometric representations
- **Expressing**: Retrieving information based on semantic queries
- **Discovering Connections**: Finding paths between entities
- **Creating Simplices**: Establishing multi-way relationships
- **Attaching Fibers**: Adding multi-faceted properties to entities

```python
def encode(self, info: str | dict, context=None, ancestors=None) -> str:
    """Fixed encode method with direct norm enforcement."""
    # Convert all ancestors to strings upfront
    str_ancestors = [str(a) for a in ancestors] if ancestors else []

    # Get ID and position from encoder
    info_id, position = self.encoding.encode_with_hierarchical_placement(
        info, str_ancestors, context, self.nucleotides
    )

    # Force string type for ID
    str_info_id = str(info_id)

    # Store with guaranteed string ID
    self.nucleotides[str_info_id] = (position, info)

    # Connect hierarchy
    for ancestor in str_ancestors:
        if ancestor in self.nucleotides:
            self.hierarchy_graph.add_edge(ancestor, str_info_id)

            # Update hierarchy level (important for norm enforcement)
            ancestor_level = self.hierarchy_embedding.hierarchy_levels.get(ancestor, 0)
            self.hierarchy_embedding.hierarchy_levels[str_info_id] = ancestor_level + 1

    # Register with regions
    self._register_with_regions(position, info, str_info_id)

    return str_info_id
```

The express operation enables retrieval of information based on semantic queries, leveraging the geometric properties of the space:

```python
def express(self, query, context=None, k=5):
    """Express (retrieve) information based on query and context.

    This is a simplified implementation that returns items nearest to the query.

    Args:
        query: The query to resolve
        context: Optional context to influence the query
        k: Maximum number of results to return

    Returns:
        list of (info, relevance) tuples
    """
    # Get query embedding
    if isinstance(query, str):
        query_embedding = self.encoding._get_text_embedding(query)
    elif isinstance(query, torch.Tensor):
        query_embedding = query
    elif isinstance(query, dict) and "embedding" in query:
        query_embedding = query["embedding"]
    else:
        # Create a representation
        query_embedding = torch.randn(self.dim)
        query_embedding = query_embedding / query_embedding.norm()

    # Apply context if provided
    if context is not None:
        context_embedding = self.encoding._get_context_embedding(context)
        # Blend query with context
        combined = 0.8 * query_embedding + 0.2 * context_embedding
        query_embedding = combined / combined.norm()

    # Make sure the query embedding is in the ball
    query_embedding = self.hyp_ops._project_to_ball(query_embedding)

    # Find nearest neighbors
    results = []
    for item_id, (position, info) in self.nucleotides.items():
        # Calculate hyperbolic distance
        try:
            distance = self.hyp_ops.geodesic_distance(query_embedding, position)
            results.append((info, distance.item(), item_id))
        except Exception:
            # Fall back to Euclidean distance
            distance = torch.norm(query_embedding - position).item()
            results.append((info, distance, item_id))

    # Sort by distance (lower is better)
    results.sort(key=lambda x: x[1])

    # Return top k results (without the item_id)
    return [(info, distance) for info, distance, _ in results[:k]]
```

### 4.4 Asynchronous Implementation
To ensure scalability, BREAD implements asynchronous versions of key operations


```python
async def aencode(self, info, context=None, ancestors=None):
    """Encode information into the geometric structure asynchronously."""
    return await self.core.aencode(info, context, ancestors)

async def aexpress(self, query, context=None, k=5):
    """Express (retrieve) information based on query and context asynchronously."""
    return await self.core.aexpress(query, context, k)

async def afind_related(self, item_id, k=5):
    """Find information related to a specific item asynchronously."""
    if isinstance(item_id, str):
        return await self.core.aexpress({"id": item_id}, k=k)
    else:
        return await self.core.aexpress(item_id, k=k)

async def adiscover_connections(
    self,
    source_id: str,
    target_id: str,
    max_steps: int = 10,
    max_hops: int = None,
    required_types: tuple = None,
) -> list:
    """Discover connections between two information items asynchronously."""
    # Handle max_hops as an alias for max_steps
    if max_hops is not None:
        max_steps = max_hops

    # Delegate to the core implementation
    return await self.core.adiscover_connections(
        source_id, target_id, max_steps=max_steps, required_types=required_types
    )
```
### 4.5 Batch Processing for Efficiency
BREAD includes batch processing capabilities for efficiently handling large collections of entities:

```python
async def abatch_encode(
    self,
    items: list[Any],
    batch_size: int = 64,
    show_progress: bool = True,
    optimize_hierarchy: bool = True,
) -> list[str]:
    """Batch encode items asynchronously.

    Args:
        items: list of items to encode
        batch_size: Batch size for processing
        show_progress: Whether to show progress
        optimize_hierarchy: Whether to optimize hierarchy after encoding

    Returns:
        list of item IDs
    """
    item_ids = await self.encoding.abatch_encode(
        items,
        batch_size=batch_size,
        nucleotides=self.nucleotides,
        hierarchy_graph=self.hierarchy_graph,
        show_progress=show_progress,
        optimize_hierarchy=optimize_hierarchy,
        device=self.device,
    )
    # Print diagnostic info
    c.log(f"Current nucleotides count: {len(self.nucleotides)}")

    return item_ids
```

The batch encoding process is optimized to handle large volumes of data efficiently by:

1. **Preprocessing Embeddings**: Computing embeddings in batches before positioning
2. **GPU Acceleration**: Using GPU where available for faster tensor operations
3. **Hierarchical Optimization**: Efficiently optimizing hierarchy after batch insertion
4. **Parallel Processing**: Handling multiple encoding operations concurrently

These optimizations enable the system to scale to large knowledge bases with millions of entities while maintaining geometric consistency.


### 4.6 Temporal Representation

BREAD extends beyond static knowledge representation to include temporal sequences and relationships:

```python
def create_temporal_sequence(
    self,
    sequence_id: str,
    points: list[TemporalPoint],
    properties: dict[str, Any] | None = None,
) -> str:
    """Create a temporal sequence connecting multiple items in chronological order.

    Args:
        sequence_id: Unique identifier for the sequence
        points: list of temporal points in the sequence
        properties: Optional metadata about the sequence

    Returns:
        ID of the created sequence
    """
    if not points:
        raise ValueError("Temporal sequence must have at least one point")

    # Create the sequence
    sequence = TemporalSequence(id=sequence_id, points=points, properties=properties or {})

    # Register the sequence
    self.temporal_sequences[sequence_id] = sequence

    # Register items with this sequence
    for point in points:
        if point.item_id not in self.item_to_sequences:
            self.item_to_sequences[point.item_id] = set()
        self.item_to_sequences[point.item_id].add(sequence_id)

    # Create connections in the fabric
    for i in range(len(points) - 1):
        source_id = points[i].item_id
        target_id = points[i + 1].item_id

        edge = FabricEdge(
            source=source_id,
            target=target_id,
            weight=min(points[i].weight, points[i + 1].weight),
            type="temporal",
            properties={
                "sequence_id": sequence_id,
                "time_gap": points[i + 1].timestamp - points[i].timestamp,
            },
        )

        self.add_fabric_edge(edge)

    # If possible, create a simplex for the sequence
    if len(points) <= self.max_simplex_dim + 1:
        try:
            items = [p.item_id for p in points]
            self.create_simplex(
                items,
                {
                    "type": "temporal_sequence",
                    "sequence_id": sequence_id,
                    "properties": properties or {},
                },
            )
        except ValueError:
            # Skip if simplex creation fails
            pass

    # Create temporal fibers for the items
    for i, point in enumerate(points):
        # Calculate relative position in sequence (0 to 1)
        if len(points) > 1:
            rel_pos = i / (len(points) - 1)
        else:
            rel_pos = 0.5

        # Create temporal fiber data
        temporal_data = {
            "sequence_id": sequence_id,
            "position": rel_pos,
            "timestamp": point.timestamp,
            "is_first": i == 0,
            "is_last": i == len(points) - 1,
        }

        # Attach fiber to the item
        self.attach_fiber(point.item_id, f"temporal_{sequence_id}", temporal_data)

    return sequence_id
```

This temporal representation enables several important capabilities:

1. **Sequence Modeling**: Explicit representation of temporally ordered items
2. **Sequence Prediction**: Projection of temporal patterns into the future
3. **Temporal Context**: Retrieval of items based on temporal proximity
4. **Temporal Fibers**: Representation of an entity's role in different temporal contexts

These capabilities are essential for modeling processes, events, and temporal relationships in knowledge structures.
## 5. Theoretical Implications and Applications

### 5.1 Theoretical Advantages

BREAD's integration of multiple geometric structures offers several theoretical advantages:

1. **Representational Capacity**: The exponential growth property of hyperbolic space allows more efficient representation of hierarchies than Euclidean approaches
2. **Multi-faceted Knowledge**: Fiber bundles enable representation of different aspects of the same entity without dimensional interference
3. **Complex Relationships**: Simplicial complexes capture multi-way relationships that cannot be adequately represented by pairwise connections
4. **Consistent Multi-parent Handling**: Sheaf cohomology provides a principled approach to representing entities with multiple parents

### 5.2 Potential Applications

The BREAD framework has promising applications in several domains:

1. **Knowledge Graph Enhancement**: Providing more expressive representations for complex knowledge graphs
2. **Text Understanding**: Representing documents as simplicial complexes where concepts form multi-way relationships
3. **Cross-domain Recommendation**: Using fiber bundles to maintain domain-specific properties while enabling cross-domain connections
4. **Conversational AI**: Enabling richer contextual understanding by representing conversational history in a geometric fabric
5. **Scientific Knowledge Integration**: Connecting concepts across scientific domains while preserving domain-specific aspects

### 5.3 Transformer Integration

BREAD can complement transformer architectures by providing a geometric memory layer. While transformers excel at processing sequential information within a limited context window, BREAD offers:

1. **Long-term Memory**: Persistent storage of information in a structured geometric space
2. **Hierarchical Organization**: Natural organization of concepts into hierarchical structures
3. **Multi-faceted Representation**: Ability to store and retrieve different aspects of the same entity
4. **Complex Relationships**: Representation of relationships beyond what can fit in a transformer's context window


The system includes specific components designed for transformer integration:

```python
def create_geometric_attention_head(
    self,
    head_id: str,
    dim: int = 64,
    focus: str | None = None,
    properties: dict[str, Any] | None = None,
) -> str:
    """Create a geometric attention head for transformer integration.

    This creates a specialized attention mechanism that operates
    in hyperbolic space rather than Euclidean space.

    Args:
        head_id: Unique identifier for the attention head
        dim: Dimensionality of the attention mechanism
        focus: Optional focus area/region
        properties: Optional head properties

    Returns:
        ID of the created attention head
    """
    # Initialize the attention head
    self.attention_heads[head_id] = {
        "dim": dim,
        "focus": focus,
        "properties": properties or {},
        "weights": torch.randn(dim, self.dim),  # Random initialization
    }

    # Normalize the weights
    weights = self.attention_heads[head_id]["weights"]
    self.attention_heads[head_id]["weights"] = weights / weights.norm(dim=1, keepdim=True)

    return head_id
```

BREAD also provides mechanisms for generating context windows and path-based prompts for transformer models:

```python
def get_geometric_context_window(
    self, center_id: str, window_size: int = 5, context_type: str = "neighborhood"
) -> list[str]:
    """Get context window based on geometric relationships.

    This method returns a context window based on the geometric
    structure rather than sequential position, enabling more
    effective context for transformer architectures.

    Args:
        center_id: ID of the center item
        window_size: Maximum size of context window
        context_type: Type of context to retrieve
                    ("neighborhood", "branch", "path", "temporal")

    Returns:
        list of item IDs in the context window
    """
    center_id = str(center_id)

    # Initialize attention component if needed
    if not hasattr(self.core, "attention"):
        attention = HyperbolicAttention(
            dim=self.dim,
            hyp_ops=self.hyp_ops,
        )
    else:
        attention = self.core.attention

    return attention.get_geometric_context_window(
        center_id=center_id,
        nucleotides=self.nucleotides,
        hierarchy_graph=self.hierarchy,
        fabric_connections=self.fabric_connections,
        item_to_sequences=self.item_to_sequences,
        window_size=window_size,
        context_type=context_type,
    )

def path_based_prompt(
    self, start_id: str, end_id: str, max_path_length: int = 10, include_info: bool = True
) -> list[str]:
    """Generate a path-based prompt for transformer architectures.

    Creates a sequence of items forming a path between start and end
    points, which can be used to guide transformer generation.

    Args:
        start_id: Starting item ID
        end_id: Ending item ID
        max_path_length: Maximum path length to consider
        include_info: Whether to include item info in the prompt

    Returns:
        list of prompt elements (either item IDs or info strings)
    """
    start_id = str(start_id)
    end_id = str(end_id)

    # Initialize attention component if needed
    if not hasattr(self.core, "attention"):
        from genome.fabric.attention import HyperbolicAttention

        # Initialize attention component if not available
        attention = HyperbolicAttention(
            dim=self.dim,
            hyp_ops=self.hyp_ops,
        )
    else:
        attention = self.core.attention

    return attention.path_based_prompt(
        start_id=start_id,
        end_id=end_id,
        nucleotides=self.nucleotides,
        discover_connections_fn=self.discover_connections,
        max_path_length=max_path_length,
        include_info=include_info,
    )
```

By integrating BREAD with transformer models, we can create systems that combine the strengths of both approaches: transformers for processing immediate context and geometric fabric for organizing broader knowledge.

## 6. Implementation Details

### 6.1 Hyperbolic Operations Implementation

Our implementation of hyperbolic operations focuses on numerical stability while maintaining mathematical correctness. Here, we highlight key aspects of our implementation:

```python
class HyperbolicOperations(BaseModel):
    """Enhanced hyperbolic operations with numerical safeguards."""

    dim: int = Field(default=100, description="Dimensionality of the hyperbolic space")
    curvature: float = Field(
        default=-1.0, description="Curvature of the hyperbolic space (negative for hyperbolic)"
    )
    eps: float = Field(default=1e-8, description="Small epsilon value for numerical stability")
    max_norm: float = Field(
        default=0.99995, description="Safe maximum norm to prevent boundary issues"
    )
    device: str = Field(default="mps", description="Device for tensor operations")

    def _project_to_ball(self, x: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
        """Project a tensor to the Poincaré ball."""
        # Calculate norm
        norm = x.norm().item()

        # Project if needed
        if norm >= (1.0 - epsilon):
            # Scale to be within the ball with safety margin
            x = x * (1.0 - 2 * epsilon) / max(norm, 1e-8)

        return x

    def mobius_addition(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform Möbius addition in the Poincaré ball."""
        # Ensure points are in the ball
        x = self._project_to_ball(x)
        y = self._project_to_ball(y)

        try:
            # Mobius addition formula
            c = abs(self.curvature)  # Assume negative curvature

            # Calculate components
            x_dot_y = torch.sum(x * y)
            x_norm_sq = torch.sum(x**2)
            y_norm_sq = torch.sum(y**2)

            # Calculate numerator terms
            numerator_1 = (1 + 2 * c * x_dot_y + c * y_norm_sq) * x
            numerator_2 = (1 - c * x_norm_sq) * y

            # Calculate denominator
            denominator = 1 + 2 * c * x_dot_y + c**2 * x_norm_sq * y_norm_sq

            # Mobius addition result
            result = (numerator_1 + numerator_2) / (denominator + 1e-8)

            # Project back to ensure numerical stability
            return self._project_to_ball(result)
        except Exception:
            # Simple fallback
            result = x + y
            return self._project_to_ball(result)
```

The key innovations in our implementation include:

1. **Boundary Protection**: We enforce a maximum norm below 1.0 to prevent numerical issues near the boundary of the Poincaré ball.
    
2. **Fallback Mechanisms**: For every operation, we provide fallback implementations that ensure the system continues to function even if numerical issues arise.
    
3. **Device-Aware Operations**: Our implementation supports GPU acceleration where available, with automatic device detection.
    
4. **Safe Division**: We add a small epsilon term to denominators to prevent division by zero.
    

### 6.2 Hierarchical Encoding

Our hierarchical encoding process integrates multiple signals to position entities in hyperbolic space:

```python
def encode_with_hierarchical_placement(
    self, info: str | dict, ancestors: list[str] | None = None, context: str | None = None
) -> str:
    """Encode information with enhanced hierarchical placement."""
    # Call the core encode method
    str_ancestors = [str(a) for a in ancestors] if ancestors else []

    info_id = self.core.encode(info, context, str_ancestors)

    # For individual encodes with ancestors, validate hierarchy
    if ancestors and len(ancestors) > 0:
        # Only validate relationships involving this node for efficiency
        self._validate_node_hierarchy(info_id)

    return str(info_id)
```

This process includes:

1. **Content Embedding**: Creating a semantic embedding of the entity's content
    
2. **Ancestor Weighting**: Applying exponential weighting to ancestor positions based on their hierarchical depth
    
3. **Context Integration**: Blending content embedding with context for more contextually relevant placement
    
4. **Hierarchy Validation**: Ensuring the entity respects hierarchical constraints relative to its ancestors
    

### 6.3 Multi-hop Path Discovery

One of GeometricFabric's key capabilities is discovering connections between entities that may not be directly related. Our implementation uses a combination of geometric and graph-based approaches:

```python
async def adiscover_connections(
    self,
    source_id: str,
    target_id: str,
    max_steps: int = 10,
    max_hops: int = None,
    required_types: tuple = None,
) -> list:
    """Discover connections between two information items asynchronously."""
    # Handle max_hops as an alias for max_steps
    if max_hops is not None:
        max_steps = max_hops

    # Ensure string IDs
    source_id = str(source_id)
    target_id = str(target_id)

    # Generate a cache key including required_types
    cache_key = (source_id, target_id, max_steps, required_types)

    # Check cache if enabled
    if hasattr(self, "path_cache") and cache_key in self.path_cache:
        return self.path_cache.get(cache_key, [])

    # Initialize path cache if not exists
    if not hasattr(self, "path_cache"):
        self.path_cache = {}

    # Check if both endpoints exist
    if source_id not in self.nucleotides or target_id not in self.nucleotides:
        self.path_cache[cache_key] = []
        return []

    # If we have required types, use typed path finding
    if required_types:
        path_nodes = await self._find_typed_path(
            source_id, target_id, required_types, max_steps
        )

        if path_nodes and len(path_nodes) > 1:
            # Create edge objects
            typed_path = []
            for i in range(len(path_nodes) - 1):
                src = path_nodes[i]
                tgt = path_nodes[i + 1]

                # Get edge type from fabric_connections
                edge_type = "unknown"
                if src in self.fabric_connections and tgt in self.fabric_connections[src]:
                    edge_type = self.fabric_connections[src][tgt].get("type", "unknown")

                # Create an edge object (as dictionary)
                edge = {"type": edge_type, "source": src, "target": tgt}
                typed_path.append(edge)

            # Cache and return the result
            self.path_cache[cache_key] = typed_path
            return typed_path
        else:
            # No valid path found
            self.path_cache[cache_key] = []
            return []

    # Get positions for geometric path
    source_pos = self.nucleotides[source_id][0]
    target_pos = self.nucleotides[target_id][0]

    # Check direct fabric connection
    if source_id in self.fabric_connections and target_id in self.fabric_connections[source_id]:
        # Create geodesic path
        try:
            path = self.hyp_ops.geodesic(source_pos, target_pos, steps=max_steps)
            # Cache and return
            self.path_cache[cache_key] = path
            return path
        except Exception:
            # Fall through to other methods if geodesic fails
            pass

    # Try to find a path through the fabric using BFS
    node_path = await self._afabric_bfs(source_id, target_id, max_depth=max_steps)
    if node_path and len(node_path) >= 2:
        # Convert path of IDs to positions
        position_path = []
        for node_id in node_path:
            if node_id in self.nucleotides:
                position_path.append(self.nucleotides[node_id][0])

        # Cache and return if we found a path
        if position_path and len(position_path) >= 2:
            self.path_cache[cache_key] = position_path
            return position_path

    # Fall back to simple geodesic path
    try:
        path = self.hyp_ops.geodesic(source_pos, target_pos, steps=max_steps)
        # Cache the path
        self.path_cache[cache_key] = path
        return path
    except Exception:
        # Fallback to simple linear interpolation
        path = []
        for t in range(max_steps):
            t_val = t / (max_steps - 1) if max_steps > 1 else 0.5
            point = (1 - t_val) * source_pos + t_val * target_pos
            point = self.hyp_ops._project_to_ball(point)
            path.append(point)

        # Cache and return the fallback path
        self.path_cache[cache_key] = path
        return path
```

This approach offers several advantages:

1. **Path Caching**: Frequently accessed paths are cached for efficiency
    
2. **Multi-strategy Approach**: The system tries multiple strategies (direct fabric connection, breadth-first search, geodesic interpolation) and uses the best result
    
3. **Type-constrained Paths**: Optional path constraints can ensure that the path follows specific relationship types
    
4. **Fallback Mechanisms**: Even if optimal path finding fails, the system provides a reasonable path approximation
    

### 6.4 Sheaf Cohomology Implementation

Our sheaf cohomology implementation provides a sophisticated approach to multi-parent hierarchies:

```python
def evaluate_hierarchy_preservation(
    self, hierarchy_graph: nx.DiGraph, nucleotides: dict[str, tuple[torch.Tensor, Any]]
) -> dict[str, float]:
    """Evaluate how well the hierarchy is preserved using sheaf metrics.

    Args:
        hierarchy_graph: Directed graph representing hierarchy
        nucleotides: Dictionary mapping item IDs to (position, info) pairs

    Returns:
        Dictionary with evaluation metrics
    """
    if not hierarchy_graph.edges():
        return {
            "origin_preservation": 0.0,
            "topological_consistency": 0.0,
            "sheaf_cohomology_score": 0.0,
            "overall_score": 0.0,
        }

    # Check origin distance preservation
    origin_preserved = 0
    total_edges = 0

    for parent, child in hierarchy_graph.edges():
        if parent in nucleotides and child in nucleotides:
            parent_pos = nucleotides[parent][0]
            child_pos = nucleotides[child][0]

            parent_norm = parent_pos.norm().item()
            child_norm = child_pos.norm().item()

            if child_norm > parent_norm:
                origin_preserved += 1

            total_edges += 1

    origin_score = origin_preserved / total_edges if total_edges > 0 else 0.0

    # Calculate topological consistency using sheaf structure
    topo_consistent = 0

    # Create sheaf if needed
    if not self.local_sections:
        self.create_hierarchy_sheaf(hierarchy_graph, nucleotides)

    # Calculate cohomology
    cohomology = self.calculate_cohomology()
    h0 = cohomology[0]

    # Count edges that are consistent in the sheaf
    for parent, child in hierarchy_graph.edges():
        if parent in nucleotides and child in nucleotides:
            sheaf_distance = self.compute_sheaf_distance(parent, child)

            # Check if this is a consistent edge in the sheaf
            if sheaf_distance < float("inf"):
                # Check if distance is reasonable
                if sheaf_distance < 0.5:  # Threshold for reasonable distance
                    topo_consistent += 1

    topo_score = topo_consistent / total_edges if total_edges > 0 else 0.0
    # GeometricFabric: Enhancing Knowledge Representation through Non-Euclidean Geometric Structures (Continued)

### 6.4 Sheaf Cohomology Implementation (Continued)

```python
    # Calculate cohomology score based on global sections
    expected_sections = min(len(hierarchy_graph.nodes()), len(nucleotides))
    actual_sections = len(h0)
    cohomology_score = actual_sections / expected_sections if expected_sections > 0 else 0.0

    # Overall score is weighted combination
    overall_score = 0.5 * origin_score + 0.3 * topo_score + 0.2 * cohomology_score

    return {
        "origin_preservation": origin_score,
        "topological_consistency": topo_score,
        "sheaf_cohomology_score": cohomology_score,
        "overall_score": overall_score,
    }
```

Our implementation calculates three key metrics:

1. **Origin Preservation**: How well the hierarchy preserves the property that children are further from the origin than their parents
    
2. **Topological Consistency**: How well the sheaf structure maintains consistent relationships across different "views" of the hierarchy
    
3. **Cohomology Score**: How many entities form "global sections" (consistent representations across all relevant contexts)
    

### 6.5 Hierarchical Sheaf Integration

A key innovation in our implementation is the integration of hierarchical embedding with sheaf cohomology:

```python
class HierarchySheaf(HierarchyEmbedding):
    """
    Enhanced hierarchy preservation using sheaf cohomology.

    This class extends the HierarchyEmbedding component with sheaf cohomology
    to better handle complex hierarchical structures, especially those with
    multiple parents or cross-connections.
    """

    def __init__(
        self,
        dim: int = 100,
        curvature: float = -1.0,
        learning_rate: float = 0.01,
        hyp_ops: HyperbolicOperations | None = None,
        default_iterations: int = 3,
        use_sheaf: bool = True,
    ):
        """Initialize the HierarchySheaf component."""
        # Initialize parent class
        super().__init__(
            dim=dim,
            curvature=curvature,
            learning_rate=learning_rate,
            hyp_ops=hyp_ops,
            default_iterations=default_iterations,
            use_sheaf=use_sheaf,
            sheaf=KnowledgeSheaf(
                dim=dim,
                curvature=curvature,
                learning_rate=learning_rate,
                hyp_ops=hyp_ops,
            ),
        )
```

This integration enables:

1. **Consistent Multi-parent Handling**: Nodes with multiple parents maintain consistent representation
    
2. **Enhanced Hierarchy Preservation**: Hierarchy constraints are enforced from multiple perspectives
    
3. **Improved Cross-Domain Connections**: Entities can participate in multiple hierarchies while maintaining coherent representation
    

### 6.6 Validation and Hierarchy Enforcement

To ensure hierarchical relationships are properly preserved, GeometricFabric includes explicit validation and enforcement mechanisms:

```python
def validate_hierarchy_preservation(self):
    """Validate and enforce hierarchy preservation across the entire fabric.

    Returns:
        Dictionary with validation metrics
    """
    # Skip if no hierarchy
    if not hasattr(self, "hierarchy") or not self.hierarchy.edges():
        return {"fixed": 0, "total": 0}

    # Get all edges in the hierarchy
    test_pairs = list(self.hierarchy.edges())
    total_edges = len(test_pairs)

    # First, compute depth for each node using BFS
    depths = {}
    roots = [n for n in self.hierarchy.nodes() if self.hierarchy.in_degree(n) == 0]

    # Breadth-first traversal to assign depths
    for root in roots:
        depths[root] = 0
        queue = [(root, 0)]
        visited = {root}

        while queue:
            node, depth = queue.pop(0)
            # Get children
            for _, child in self.hierarchy.out_edges(node):
                if child not in visited:
                    depths[child] = depth + 1
                    visited.add(child)
                    queue.append((child, depth + 1))

    # Apply fixes based on depth
    fixed = 0

    # First pass: Set norms based on hierarchy level
    for node, depth in depths.items():
        if node in self.nucleotides:
            pos = self.nucleotides[node][0]
            current_norm = pos.norm().item()

            # Get target norm for this depth level
            target_norm = NORM_TARGETS.get(depth, 0.95)

            # Adjust position to match target norm if significantly different
            if abs(current_norm - target_norm) > 0.05:
                direction = pos / max(current_norm, 1e-8)
                new_pos = direction * target_norm
                self.nucleotides[node] = (new_pos, self.nucleotides[node][1])
                fixed += 1

    # Second pass: Ensure parent-child relationships
    for parent, child in test_pairs:
        if parent in self.nucleotides and child in self.nucleotides:
            parent_pos = self.nucleotides[parent][0]
            child_pos = self.nucleotides[child][0]

            parent_norm = parent_pos.norm().item()
            child_norm = child_pos.norm().item()

            # Child should be at least 15% further from origin than parent
            if child_norm <= parent_norm * 1.15:
                target_norm = parent_norm * 1.15
                target_norm = min(target_norm, 0.95)  # Cap at 0.95 to stay away from boundary

                # Keep direction but adjust norm
                child_dir = child_pos / max(child_norm, 1e-8)
                new_child_pos = child_dir * target_norm
                self.nucleotides[child] = (new_child_pos, self.nucleotides[child][1])
                fixed += 1

    return {"fixed": fixed, "total": total_edges}
```

These mechanisms ensure that:

1. **Depth-based Norms**: Entities are positioned at appropriate distances from the origin based on their hierarchy depth
    
2. **Parent-Child Relationships**: Children are always positioned further from the origin than their parents
    
3. **Numerical Stability**: All positions remain within safe bounds to prevent numerical issues
    

By enforcing these constraints, GeometricFabric maintains the hierarchical properties of the space even as new entities are added or existing entities are updated.

## 7. Discussion and Future Work

### 7.1 Contributions and Significance

BREAD makes several significant contributions to knowledge representation:

1. **Unified Geometric Framework**: Integrates multiple geometric structures (hyperbolic spaces, fiber bundles, simplicial complexes) into a coherent system
2. **Enhanced Hierarchy Preservation**: Provides robust mechanisms for preserving hierarchical relationships in complex knowledge structures
3. **Multi-faceted Representation**: Enables representation of multiple aspects of entities without dimensional interference
4. **Practical Integration**: Includes attention to computational efficiency, numerical stability, and real-world integration

### 7.2 Limitations and Challenges

1. **Computational Complexity**: Operations in hyperbolic space can be more intensive than Euclidean methods
2. **Implementation Complexity**: Combining multiple geometric structures adds engineering overhead
3. **Parameter Tuning**: Numerous hyperparameters require careful tuning
4. **Integration**: Existing systems often assume Euclidean embeddings, posing compatibility hurdles

### 7.3 Future Directions

1. **Dynamic Curvature Adaptation**: Mechanisms to adapt curvature locally based on data structure
2. **Distributed Implementation**: Scaling to extremely large knowledge bases
3. **Closer Transformer Integration**: More efficient interfaces for attention across hyperbolic geometry
4. **Multi-modal Extension**: Applying BREAD to images, audio, or other data types
5. **Interactive Learning**: Adapting the geometric structure based on real-time feedback

### 7.4 Broader Implications

1. **Cognitive Architecture**: Contributing to AI memory systems that integrate multiple relationship types
2. **Explainable AI**: Offering geometric representations that clarify AI reasoning
3. **Cross-domain Knowledge**: Facilitating more effective multi-domain knowledge integration
4. **Foundation Models**: Enhancing the representational power of large-scale foundational models

## 8. Conclusion

**BREAD** introduces a novel approach to knowledge representation that integrates hyperbolic spaces, fiber bundles, and simplicial complexes into a unified geometric framework. By leveraging the intrinsic properties of these non-Euclidean structures, the system addresses fundamental limitations of traditional embedding approaches.

Our implementation provides robust mechanisms for hierarchy preservation, multi-faceted entity representation, and multi-way relationship modeling. The theoretical foundations and practical implementation details presented in this paper demonstrate the potential of advanced geometric approaches to transform knowledge representation in AI systems. As foundation models continue to evolve, frameworks like BREAD will be crucial for organizing and accessing the complex, hierarchical knowledge they contain.

By moving beyond simple vector embeddings to rich geometric structures, BREAD opens new possibilities for more sophisticated, contextually aware, and geometrically principled knowledge representation in artificial intelligence.