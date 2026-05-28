use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineNode {
    pub id: String,
    pub node_type: String,
    pub engine_id: Option<String>,
    pub params: HashMap<String, serde_json::Value>,
    pub position: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineEdge {
    pub id: String,
    pub source: String,
    pub source_port: String,
    pub target: String,
    pub target_port: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineGraph {
    pub nodes: Vec<PipelineNode>,
    pub edges: Vec<PipelineEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
}

impl PipelineGraph {
    pub fn validate(&self) -> ValidationResult {
        let mut errors = Vec::new();
        let ids: HashSet<&str> = self.nodes.iter().map(|n| n.id.as_str()).collect();

        for edge in &self.edges {
            if !ids.contains(edge.source.as_str()) {
                errors.push(format!("Edge references unknown source: {}", edge.source));
            }
            if !ids.contains(edge.target.as_str()) {
                errors.push(format!("Edge references unknown target: {}", edge.target));
            }
        }

        // check for cycles via DFS
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for edge in &self.edges {
            adj.entry(&edge.source).or_default().push(&edge.target);
        }
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();
        for node in &self.nodes {
            if Self::has_cycle(node.id.as_str(), &adj, &mut visited, &mut stack) {
                errors.push(format!("Cycle detected involving node: {}", node.id));
                break;
            }
        }

        ValidationResult {
            valid: errors.is_empty(),
            errors,
        }
    }

    fn has_cycle<'a>(
        node: &'a str,
        adj: &HashMap<&'a str, Vec<&'a str>>,
        visited: &mut HashSet<&'a str>,
        stack: &mut HashSet<&'a str>,
    ) -> bool {
        if stack.contains(node) {
            return true;
        }
        if visited.contains(node) {
            return false;
        }
        visited.insert(node);
        stack.insert(node);
        if let Some(neighbors) = adj.get(node) {
            for &n in neighbors {
                if Self::has_cycle(n, adj, visited, stack) {
                    return true;
                }
            }
        }
        stack.remove(node);
        false
    }

    pub fn topological_order(&self) -> Result<Vec<&PipelineNode>, String> {
        let mut in_degree: HashMap<&str, usize> = self.nodes.iter().map(|n| (n.id.as_str(), 0)).collect();
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for edge in &self.edges {
            adj.entry(edge.source.as_str()).or_default().push(edge.target.as_str());
            *in_degree.entry(edge.target.as_str()).or_insert(0) += 1;
        }

        let mut queue: Vec<&str> = in_degree.iter().filter(|(_, &d)| d == 0).map(|(&k, _)| k).collect();
        let mut order = Vec::new();
        let node_map: HashMap<&str, &PipelineNode> = self.nodes.iter().map(|n| (n.id.as_str(), n)).collect();

        while let Some(id) = queue.pop() {
            order.push(node_map[id]);
            if let Some(neighbors) = adj.get(id) {
                for &n in neighbors {
                    let deg = in_degree.get_mut(n).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(n);
                    }
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err("Cycle detected in pipeline graph".to_string());
        }
        Ok(order)
    }
}
