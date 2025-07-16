# SPDX-License-Identifier: CC-BY-4.0
# Copyright (c) 2025 James Gardner
# Part of: Hyperbolic Helices (DOI: 10.5281/zenodo.15983944)

"""
Simplified semantic uncertainty detector matching paper's Appendix A.
This is a pedagogical implementation for clarity - see PRODUCTION_NOTES.md for deployment guidance.
"""

import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Constants (in practice, tune these empirically)
ROUGHNESS_THRESH = 0.5
OSCILLATION_THRESH = 0.7
JUMP_THRESH = 3.0

def measure_semantic_uncertainty(text: str, model=None) -> Dict[str, float]:
    """
    Computes semantic uncertainty metrics from text trajectory.
    
    NOTE: Production systems should use tokenizer-consistent segmentation
    and optimized hyperbolic distance calculations.
    
    Args:
        text: Input text to analyze
        model: Embedding model (defaults to MiniLM)
        
    Returns: 
        {'uncertainty_score': float, 'pattern': str, 'metrics': dict}
    """
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Simplified trajectory extraction (word-level for clarity)
    tokens = text.split()
    if len(tokens) < 3:
        return {'uncertainty_score': 0.0, 'pattern': 'insufficient_data'}
    
    embeddings = [model.encode(token) for token in tokens]
    
    # Core hyperbolic distance (Poincaré ball model)
    def hyper_dist(x, y):
        """Hyperbolic distance in Poincaré ball"""
        # Normalize to unit sphere first
        x_norm = x / (np.linalg.norm(x) + 1e-10)
        y_norm = y / (np.linalg.norm(y) + 1e-10)
        
        # Map to Poincaré ball (radius 0.99)
        scale = 0.99
        x_ball = scale * x_norm / (1 + np.sqrt(1 + 1e-10))
        y_ball = scale * y_norm / (1 + np.sqrt(1 + 1e-10))
        
        # Ensure within ball
        x_ball = x_ball / max(1.0, np.linalg.norm(x_ball) / 0.99)
        y_ball = y_ball / max(1.0, np.linalg.norm(y_ball) / 0.99)
        
        # Calculate distance
        norm_x = np.linalg.norm(x_ball)
        norm_y = np.linalg.norm(y_ball)
        norm_diff = np.linalg.norm(x_ball - y_ball)
        
        denom = (1 - norm_x**2) * (1 - norm_y**2)
        if denom <= 1e-10:
            return 20.0
            
        arg = 1 + 2 * norm_diff**2 / denom
        return np.arccosh(max(1.0, arg))
    
    # Calculate trajectory metrics (illustrative implementations)
    roughness = compute_roughness(embeddings, hyper_dist)
    oscillation = compute_oscillation(embeddings)
    jump_score = compute_jump_score(embeddings, hyper_dist)
    
    # Pattern detection based on geometric signatures
    if oscillation > OSCILLATION_THRESH and jump_score > JUMP_THRESH:
        pattern = "iterative"  # Helical trajectory (e.g., counting)
    elif roughness > ROUGHNESS_THRESH:
        pattern = "conceptual"  # Complex reasoning
    elif jump_score > JUMP_THRESH:
        pattern = "bridging"   # Connecting distant concepts
    else:
        pattern = "stable"     # Normal text flow
    
    # Combined uncertainty score
    uncertainty_score = 0.4*roughness + 0.3*oscillation + 0.3*jump_score
    
    return {
        'uncertainty_score': float(uncertainty_score),
        'pattern': pattern,
        'metrics': {
            'roughness': float(roughness),
            'oscillation': float(oscillation),
            'jump_score': float(jump_score)
        }
    }

def compute_roughness(embeddings: List[np.ndarray], dist_func) -> float:
    """Path roughness: deviation from geodesic in hyperbolic space"""
    if len(embeddings) < 3:
        return 0.0
        
    roughness = 0.0
    for i in range(len(embeddings) - 2):
        # In hyperbolic space, check reverse triangle inequality
        d_i_i2 = dist_func(embeddings[i], embeddings[i+2])
        d_i_i1 = dist_func(embeddings[i], embeddings[i+1])
        d_i1_i2 = dist_func(embeddings[i+1], embeddings[i+2])
        
        # Deviation from geodesic (shortcut factor)
        indirect = d_i_i1 + d_i1_i2
        shortcut = d_i_i2 / (indirect + 1e-10)
        
        # In hyperbolic space, shortcuts are expected (reverse triangle)
        # High roughness when path doesn't take shortcuts
        roughness += (1 - shortcut)
    
    return roughness / (len(embeddings) - 2)

def compute_oscillation(embeddings: List[np.ndarray]) -> float:
    """Directional changes along trajectory"""
    if len(embeddings) < 3:
        return 0.0
        
    direction_changes = 0
    for i in range(1, len(embeddings) - 1):
        v1 = embeddings[i] - embeddings[i-1]
        v2 = embeddings[i+1] - embeddings[i]
        
        # Normalized dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        if cos_angle < 0.5:  # >60° turn
            direction_changes += 1
    
    return direction_changes / (len(embeddings) - 2)

def compute_jump_score(embeddings: List[np.ndarray], dist_func) -> float:
    """Maximum distance jump relative to average"""
    if len(embeddings) < 2:
        return 1.0
        
    distances = [dist_func(embeddings[i], embeddings[i+1]) 
                 for i in range(len(embeddings) - 1)]
    
    return max(distances) / (np.mean(distances) + 1e-10)

# Example usage
if __name__ == "__main__":
    examples = [
        "Count the r's in strawberry",
        "The weather is nice today",
        "Consciousness emerges from quantum phenomena"
    ]
    
    for text in examples:
        result = measure_semantic_uncertainty(text)
        print(f"\nText: {text}")
        print(f"Pattern: {result['pattern']}")
        print(f"Uncertainty: {result['uncertainty_score']:.3f}")