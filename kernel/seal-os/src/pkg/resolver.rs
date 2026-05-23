// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Greedy dependency resolver with cycle detection.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

pub struct DependencyResolver {
    graph: BTreeMap<String, Vec<String>>,
}

impl DependencyResolver {
    pub const fn new() -> Self {
        Self {
            graph: BTreeMap::new(),
        }
    }

    pub fn register(&mut self, name: &str, deps: &[String]) {
        self.graph.insert(String::from(name), deps.to_vec());
    }

    /// Resolve dependencies in installation order (leaves first).
    /// Returns Err if a cycle is detected.
    pub fn resolve(&self, name: &str) -> Result<Vec<String>, String> {
        let mut visited = BTreeMap::new();
        let mut stack = Vec::new();
        let mut order = Vec::new();

        fn dfs(
            graph: &BTreeMap<String, Vec<String>>,
            node: &str,
            visited: &mut BTreeMap<String, u8>, // 0=unvisited, 1=visiting, 2=done
            stack: &mut Vec<String>,
            order: &mut Vec<String>,
        ) -> Result<(), String> {
            match visited.get(node) {
                Some(2) => return Ok(()),
                Some(1) => {
                    return Err(format!("dependency cycle detected at '{}'", node));
                }
                _ => {}
            }
            visited.insert(String::from(node), 1);
            stack.push(String::from(node));

            if let Some(deps) = graph.get(node) {
                for dep in deps {
                    dfs(graph, dep, visited, stack, order)?;
                }
            }

            stack.pop();
            visited.insert(String::from(node), 2);
            order.push(String::from(node));
            Ok(())
        }

        dfs(&self.graph, name, &mut visited, &mut stack, &mut order)?;
        // Remove root itself from order — caller installs it last
        order.retain(|n| n != name);
        Ok(order)
    }

    pub fn all_deps(&self, name: &str) -> Result<Vec<String>, String> {
        let mut visited = BTreeMap::new();
        let mut result = Vec::new();

        fn collect(
            graph: &BTreeMap<String, Vec<String>>,
            node: &str,
            visited: &mut BTreeMap<String, bool>,
            result: &mut Vec<String>,
        ) {
            if visited.get(node) == Some(&true) {
                return;
            }
            visited.insert(String::from(node), true);
            if let Some(deps) = graph.get(node) {
                for dep in deps {
                    collect(graph, dep, visited, result);
                    result.push(String::from(dep));
                }
            }
        }

        collect(&self.graph, name, &mut visited, &mut result);
        // Deduplicate while preserving order
        let mut seen = BTreeMap::new();
        let mut unique = Vec::new();
        for dep in result {
            if seen.insert(dep.clone(), true).is_none() {
                unique.push(dep);
            }
        }
        Ok(unique)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_deps() {
        let mut r = DependencyResolver::new();
        r.register("a", &[String::from("b"), String::from("c")]);
        r.register("b", &[String::from("c")]);
        r.register("c", &[]);
        let order = r.resolve("a").unwrap();
        assert_eq!(order, vec!["c", "b"]);
    }

    #[test]
    fn test_cycle_detected() {
        let mut r = DependencyResolver::new();
        r.register("a", &[String::from("b")]);
        r.register("b", &[String::from("a")]);
        assert!(r.resolve("a").is_err());
    }

    #[test]
    fn test_all_deps_dedup() {
        let mut r = DependencyResolver::new();
        r.register("a", &[String::from("b"), String::from("c")]);
        r.register("b", &[String::from("c")]);
        let deps = r.all_deps("a").unwrap();
        assert_eq!(deps, vec!["c", "b"]);
    }
}
