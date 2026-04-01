// Genesys Module - The Soul of Apeiron
// Contains Metacognition (Critic), Reasoning (MCTS), and Self-Editing (Architect) components.

pub mod critic;
#[cfg(feature = "std")]
pub mod architect;
pub mod mcts;
