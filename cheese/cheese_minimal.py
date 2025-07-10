#!/usr/bin/env python3
"""
CHEESE: Contextual Hierarchy for Embedding Enhancement and Semantic Enrichment
A minimal implementation showing the 47-parameter nightmare that led to BREAD.

This is a simplified version of the full CHEESE system, demonstrating
the core concepts and why the parameter tuning became impossible.

Author: James Gardner
Date: 2025
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Memory:
    """A memory with biological-inspired properties."""

    id: str
    content: str
    embedding: np.ndarray
    timestamp: float = field(default_factory=time.time)
    activation_level: float = 0.1
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    episode_id: str | None = None

    def decay(self, current_time: float, decay_rate: float = 0.9):
        """Apply time-based decay to activation."""
        time_delta = current_time - self.last_access
        self.activation_level *= decay_rate ** (time_delta / 3600)  # hourly decay


@dataclass
class AssociativeLink:
    """A bidirectional link between memories."""

    source_id: str
    target_id: str
    strength: float = 0.5
    link_type: str = "semantic"  # semantic, temporal, causal


class CHEESE:
    """
    The 47-parameter nightmare in all its glory.

    This minimal implementation shows why biological inspiration
    led to parameter explosion and tuning hell.
    """

    def __init__(self):
        # The infamous 47 parameters (showing the main culprits)

        # Similarity weights
        self.similarity_weight = 0.5  # Weight for embedding similarity
        self.associative_weight = 0.3  # Weight for associative connections
        self.temporal_weight = 0.1  # Weight for temporal proximity
        self.activation_weight = 0.1  # Weight for activation levels

        # Thresholds
        self.similarity_threshold = 0.7  # Min similarity to consider
        self.activation_threshold = 0.2  # Min activation to retrieve
        self.link_formation_threshold = 0.8  # When to create associations
        self.clustering_threshold = 0.65  # For episodic clustering

        # Decay parameters
        self.base_decay_rate = 0.9  # How fast memories decay
        self.access_boost = 0.2  # Activation boost on access
        self.interference_factor = 0.15  # How much memories interfere

        # Spreading activation
        self.spreading_factor = 0.5  # How much activation spreads
        self.max_spreading_hops = 2  # How far activation spreads
        self.spreading_decay = 0.7  # Decay per hop

        # Temporal parameters
        self.temporal_window = 300  # Seconds for temporal proximity
        self.episode_threshold = 600  # Seconds to separate episodes
        self.recency_bias = 0.3  # Boost for recent memories

        # Retrieval parameters
        self.first_stage_k = 20  # Candidates in first stage
        self.final_k = 10  # Final results to return
        self.diversity_threshold = 0.3  # Minimum diversity in results

        # Category parameters
        self.category_similarity_threshold = 0.75
        self.max_categories = 50
        self.category_merge_threshold = 0.85

        # And 20+ more parameters...
        # This is where the nightmare begins

        # Storage
        self.memories: dict[str, Memory] = {}
        self.associations: dict[str, list[AssociativeLink]] = defaultdict(list)
        self.episodes: dict[str, list[str]] = defaultdict(list)
        self.categories: dict[str, set[str]] = defaultdict(set)

    def add_memory(self, content: str, embedding: np.ndarray) -> str:
        """Add a memory and update all the interconnected systems."""
        memory_id = f"mem_{len(self.memories)}"
        memory = Memory(memory_id, content, embedding)

        # Update episodic context
        self._update_episodes(memory)

        # Create associative links
        self._create_associations(memory)

        # Update categories
        self._update_categories(memory)

        # Apply interference to existing memories
        self._apply_interference(memory)

        self.memories[memory_id] = memory
        return memory_id

    def retrieve(
        self, query_embedding: np.ndarray, query_text: str = ""
    ) -> list[tuple[Memory, float]]:
        """
        The retrieval process with all its complexity.
        This is where the 47 parameters dance together.
        """
        results = []
        current_time = time.time()

        # Apply decay to all memories
        for memory in self.memories.values():
            memory.decay(current_time, self.base_decay_rate)

        # Stage 1: Similarity-based candidate selection
        candidates = []
        for memory in self.memories.values():
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            if similarity > self.similarity_threshold:
                candidates.append((memory, similarity))

        # Sort and take top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[: self.first_stage_k]

        # Stage 2: Multi-signal scoring
        scored_results = []
        for memory, base_similarity in candidates:
            # Calculate all the signals
            similarity_score = base_similarity * self.similarity_weight

            # Activation score (with threshold)
            activation_score = 0
            if memory.activation_level > self.activation_threshold:
                activation_score = memory.activation_level * self.activation_weight

            # Temporal score (recency)
            time_delta = current_time - memory.timestamp
            temporal_score = np.exp(-time_delta / 86400) * self.temporal_weight

            # Associative score (spreading activation)
            assoc_score = self._calculate_associative_score(memory.id, query_embedding, candidates)

            # Combined score (where parameters interact)
            total_score = similarity_score + activation_score + temporal_score + assoc_score

            # Apply boost factors
            if memory.access_count > 5:  # Frequently accessed
                total_score *= 1.2

            scored_results.append((memory, total_score))

        # Stage 3: Post-processing
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Diversity filtering
        final_results = []
        for memory, score in scored_results:
            if self._is_diverse_enough(memory, final_results):
                final_results.append((memory, score))

                # Update activation
                memory.activation_level += self.access_boost
                memory.last_access = current_time
                memory.access_count += 1

                # Spread activation
                self._spread_activation(memory.id)

            if len(final_results) >= self.final_k:
                break

        return final_results

    def _calculate_associative_score(
        self, memory_id: str, query_embedding: np.ndarray, candidates: list[tuple[Memory, float]]
    ) -> float:
        """Calculate score from associative connections."""
        score = 0.0
        visited = set()

        # Get candidate memory IDs for faster lookup
        candidate_ids = {m[0].id for m in candidates}

        # Traverse associative network
        queue = [(memory_id, 1.0, 0)]  # (id, strength, depth)

        while queue and len(visited) < 50:  # Limit traversal
            current_id, strength, depth = queue.pop(0)

            if current_id in visited or depth > self.max_spreading_hops:
                continue

            visited.add(current_id)

            # Add contribution from connected memories
            for link in self.associations[current_id]:
                if link.target_id in candidate_ids:
                    contrib = strength * link.strength * (self.spreading_decay**depth)
                    score += contrib * self.associative_weight

                # Continue spreading
                if depth < self.max_spreading_hops:
                    queue.append((link.target_id, strength * link.strength, depth + 1))

        return score

    def _spread_activation(self, memory_id: str):
        """Spread activation through associative network."""
        memory = self.memories[memory_id]

        # BFS through associations
        queue = [(memory_id, memory.activation_level)]
        visited = set()

        while queue:
            current_id, activation = queue.pop(0)

            if current_id in visited:
                continue
            visited.add(current_id)

            for link in self.associations[current_id]:
                if link.target_id not in visited:
                    # Spread reduced activation
                    spread_amount = activation * link.strength * self.spreading_factor
                    if spread_amount > 0.01:  # Threshold to stop spreading
                        target = self.memories.get(link.target_id)
                        if target:
                            target.activation_level += spread_amount
                            queue.append((link.target_id, spread_amount))

    def _create_associations(self, new_memory: Memory):
        """Create associative links based on similarity and temporal proximity."""
        current_time = new_memory.timestamp

        for existing_id, existing_memory in self.memories.items():
            # Semantic similarity
            similarity = self._cosine_similarity(new_memory.embedding, existing_memory.embedding)

            if similarity > self.link_formation_threshold:
                # Create bidirectional link
                link1 = AssociativeLink(new_memory.id, existing_id, similarity)
                link2 = AssociativeLink(existing_id, new_memory.id, similarity * 0.9)
                self.associations[new_memory.id].append(link1)
                self.associations[existing_id].append(link2)

            # Temporal proximity
            time_delta = abs(current_time - existing_memory.timestamp)
            if time_delta < self.temporal_window:
                strength = 1.0 - (time_delta / self.temporal_window)
                link1 = AssociativeLink(new_memory.id, existing_id, strength, link_type="temporal")
                link2 = AssociativeLink(
                    existing_id, new_memory.id, strength * 0.9, link_type="temporal"
                )
                self.associations[new_memory.id].append(link1)
                self.associations[existing_id].append(link2)

    def _update_episodes(self, memory: Memory):
        """Cluster memories into episodes based on temporal proximity."""
        # Find recent episode
        recent_episode = None
        for episode_id, memory_ids in self.episodes.items():
            if memory_ids:
                last_memory = self.memories[memory_ids[-1]]
                if memory.timestamp - last_memory.timestamp < self.episode_threshold:
                    recent_episode = episode_id
                    break

        if recent_episode:
            self.episodes[recent_episode].append(memory.id)
            memory.episode_id = recent_episode
        else:
            # Create new episode
            episode_id = f"episode_{len(self.episodes)}"
            self.episodes[episode_id] = [memory.id]
            memory.episode_id = episode_id

    def _update_categories(self, memory: Memory):
        """Update category clusters (simplified ART-inspired clustering)."""
        best_category = None
        best_similarity = 0

        # Find best matching category
        for cat_id, member_ids in self.categories.items():
            if member_ids:
                # Calculate average similarity to category members
                similarities = []
                for member_id in list(member_ids)[:5]:  # Sample for efficiency
                    member = self.memories.get(member_id)
                    if member:
                        sim = self._cosine_similarity(memory.embedding, member.embedding)
                        similarities.append(sim)

                avg_similarity = np.mean(similarities) if similarities else 0
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_category = cat_id

        # Add to category or create new one
        if best_similarity > self.category_similarity_threshold:
            self.categories[best_category].add(memory.id)
        else:
            new_category = f"cat_{len(self.categories)}"
            self.categories[new_category] = {memory.id}

    def _apply_interference(self, new_memory: Memory):
        """Apply interference effects to existing memories."""
        for existing in self.memories.values():
            similarity = self._cosine_similarity(new_memory.embedding, existing.embedding)

            # Similar memories interfere with each other
            if similarity > 0.9:  # Very similar
                existing.activation_level *= 1 - self.interference_factor

    def _is_diverse_enough(self, memory: Memory, selected: list[tuple[Memory, float]]) -> bool:
        """Check if memory adds diversity to results."""
        if not selected:
            return True

        for selected_memory, _ in selected:
            similarity = self._cosine_similarity(memory.embedding, selected_memory.embedding)
            if similarity > (1 - self.diversity_threshold):
                return False
        return True

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_statistics(self) -> dict:
        """Get system statistics showing the complexity."""
        total_associations = sum(len(links) for links in self.associations.values())
        avg_activation = np.mean([m.activation_level for m in self.memories.values()])

        return {
            "total_memories": len(self.memories),
            "total_associations": total_associations,
            "total_episodes": len(self.episodes),
            "total_categories": len(self.categories),
            "avg_activation_level": float(avg_activation),
            "parameters": {
                "similarity_weight": self.similarity_weight,
                "associative_weight": self.associative_weight,
                "temporal_weight": self.temporal_weight,
                "activation_weight": self.activation_weight,
                # ... and 43 more
            },
        }


def demo():
    """Demonstrate the system and its parameter sensitivity."""
    print("CHEESE: The 47-Parameter Nightmare Demo\n")

    # Initialize system
    cheese = CHEESE()

    # Add some memories with time delays to show temporal effects
    memories_data = [
        ("Python is a programming language", [0.8, 0.2, 0.1, 0.9]),
        ("Machine learning uses Python", [0.7, 0.3, 0.8, 0.9]),
        ("The weather is nice today", [0.1, 0.9, 0.2, 0.1]),
        ("Python is a type of snake", [0.6, 0.1, 0.1, 0.5]),
        ("I learned Python yesterday", [0.7, 0.2, 0.3, 0.8]),
        ("JavaScript is also a programming language", [0.75, 0.15, 0.05, 0.85]),
        ("Deep learning requires Python knowledge", [0.65, 0.35, 0.85, 0.88]),
    ]

    print("Adding memories...")
    for content, embedding in memories_data:
        embedding = np.array(embedding) / np.linalg.norm(embedding)
        cheese.add_memory(content, embedding)
        time.sleep(0.1)  # Simulate time passing

    # Test retrieval
    query = "Tell me about Python programming"
    query_embedding = np.array([0.75, 0.25, 0.2, 0.85])
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    print(f"\nQuery: {query}")
    print("Retrieved memories:")
    results = cheese.retrieve(query_embedding, query)

    for i, (memory, score) in enumerate(results[:5]):
        print(f"{i + 1}. [{score:.3f}] {memory.content}")
        print(f"   (activation: {memory.activation_level:.3f}, accesses: {memory.access_count})")

    # Show statistics
    stats = cheese.get_statistics()
    print("\nSystem Statistics:")
    print(f"Total memories: {stats['total_memories']}")
    print(f"Total associations: {stats['total_associations']}")
    print(f"Average activation: {stats['avg_activation_level']:.3f}")

    # Demonstrate parameter sensitivity
    print("\n--- Parameter Sensitivity Demo ---")
    print("Watch how small parameter changes break everything:\n")

    # Show original parameters working well
    print("Config 1: Balanced parameters")
    print(f"  similarity_weight: {cheese.similarity_weight}")
    print(f"  associative_weight: {cheese.associative_weight}")
    print(f"  temporal_weight: {cheese.temporal_weight}")
    print(f"  activation_weight: {cheese.activation_weight}")

    # Now break it
    print("\nConfig 2: Boost similarity_weight to 0.7")
    cheese.similarity_weight = 0.7  # Seems reasonable, right?
    # But now associative connections dominate unless we rebalance...

    results2 = cheese.retrieve(query_embedding, query)
    print("Results:", [f"{m.content[:20]}..." for m, _ in results2[:3]])

    print("\nConfig 3: Try to fix by reducing associative_weight")
    cheese.associative_weight = 0.1
    # But now temporal effects are too strong...

    results3 = cheese.retrieve(query_embedding, query)
    print("Results:", [f"{m.content[:20]}..." for m, _ in results3[:3]])

    print("\nConfig 4: Reduce temporal_weight to compensate")
    cheese.temporal_weight = 0.05
    # But now activation patterns are wrong...

    results4 = cheese.retrieve(query_embedding, query)
    print("Results:", [f"{m.content[:20]}..." for m, _ in results4[:3]])

    print("\nConfig 5: Fine, let's try different decay parameters")
    cheese.base_decay_rate = 0.95  # Slower decay
    cheese.spreading_factor = 0.3  # Less spreading
    # And now the whole system behaves differently!

    # Clear activation to show the effect
    for m in cheese.memories.values():
        m.activation_level = 0.1

    results5 = cheese.retrieve(query_embedding, query)
    print("Results:", [f"{m.content[:20]}..." for m, _ in results5[:3]])

    print("\n🤯 Every 'fix' created new problems!")
    print("This is the 47-parameter whack-a-mole that drove me to madness.")
    print("\nThe same query returns completely different results based on tiny parameter tweaks.")
    print("And this is just 5 parameters - imagine tuning 47 of them!")


if __name__ == "__main__":
    demo()
