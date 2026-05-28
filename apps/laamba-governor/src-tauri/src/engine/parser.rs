use regex::Regex;
use std::collections::HashMap;

pub struct MetricParser {
    patterns: HashMap<String, Regex>,
}

impl MetricParser {
    pub fn new(patterns: HashMap<String, String>) -> Self {
        let compiled: HashMap<String, Regex> = patterns
            .into_iter()
            .filter_map(|(k, v)| Regex::new(&v).ok().map(|r| (k, r)))
            .collect();
        Self { patterns: compiled }
    }

    pub fn parse_line(&self, line: &str) -> Option<(String, f64)> {
        for (name, re) in &self.patterns {
            if let Some(caps) = re.captures(line) {
                if let Some(m) = caps.get(1) {
                    if let Ok(val) = m.as_str().parse::<f64>() {
                        return Some((name.clone(), val));
                    }
                }
            }
        }
        None
    }
}
