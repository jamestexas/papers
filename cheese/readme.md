# CHEESE: Contextual Hierarchy for Embedding Enhancement and Semantic Enrichment

## The 47-Parameter Nightmare That Led to Mathematical Enlightenment

CHEESE was my first attempt at building a better memory system for LLMs. Inspired by biological memory systems, it implements spreading activation, episodic clustering, and associative networks. It works... but at the cost of 47+ interdependent parameters that made it impossible to tune.

## What's Here

- `paper.md` - The formal write-up of CHEESE's architecture and approach
- `cheese_minimal.py` - A working implementation that demonstrates the parameter explosion problem
- `CHEESE.md` - The blog post telling the real story of building this

## The Core Idea

Instead of treating memories as isolated vectors in a database, CHEESE creates a "contextual fabric" where memories are connected through:

- **Semantic similarity** - How similar are the embeddings?
- **Temporal proximity** - When were they created/accessed?
- **Associative links** - What memories activate together?
- **Activation levels** - How "hot" is this memory right now?

## The Problem

Every biological mechanism I added introduced new parameters:
- Spreading activation? Add `spreading_factor`, `max_hops`, `hop_decay`...
- Memory decay? Add `base_decay_rate`, `long_term_decay_rate`, `decay_time_scale`...
- Episode formation? Add `episode_threshold`, `temporal_window`, `recency_bias`...

By the end, I had 47 parameters that all affected each other. Change one, break three others. Fix those three, break five more. It was parameter whack-a-mole.

## Running the Demo

```bash
python cheese_minimal.py
```

This will show you:
1. The system successfully retrieving relevant memories
2. How changing just a few parameters completely changes the results
3. Why biological inspiration led to complexity explosion

## Key Lessons

1. **Biological metaphors are seductive but dangerous** - Real biological systems evolved over millions of years. Implementing them directly leads to unmanageable complexity.

2. **Parameter interdependence is a design smell** - When you can't tune parameters independently, you've probably chosen the wrong abstraction.

3. **Sometimes you need math, not metaphors** - The failure of CHEESE led me to discover that information has geometric structure that's better captured by mathematical frameworks (hyperbolic spaces, fiber bundles) than biological analogies.

## Why This Matters

CHEESE worked well enough to prove that contextual memory systems could improve on simple vector similarity. But its parameter explosion revealed a fundamental issue: I was trying to force hierarchical and multi-aspect relationships into Euclidean space where they don't naturally fit.

This realization led to [BREAD](../bread/) and the mathematical approaches that followed.

## Performance

When properly tuned (good luck with that), CHEESE achieved:
- 74% precision on topic queries
- 8,697 memories/second throughput  
- 76% success on multi-hop reasoning

But on edge cases:
- Cross-domain queries: 0% success rate
- Unpredictable failures on seemingly simple queries
- Required constant re-tuning for different domains

## The Code

`cheese_minimal.py` implements the core concepts:
- Multi-signal retrieval (all 4 signals)
- Spreading activation through networks
- Biological decay and interference
- Episode and category formation
- The parameter sensitivity that drove me insane

It's a real, working system - not a simulation. Run it and experience the same parameter hell I did.

## What's Next

After spending a week trying to tune 47 parameters, I asked an AI in frustration: "I have 47 parameters and they all affect each other and I can't tune this thing without going insane."

Its response changed everything: "Your problem isn't too many parameters. Your problem is that you're trying to force hierarchical relationships into Euclidean space where they don't naturally fit."

That led to BREAD, hyperbolic embeddings, and the realization that information has inherent geometric structure.

---

*CHEESE: Sometimes you have to build the wrong thing to discover what the right thing looks like.*