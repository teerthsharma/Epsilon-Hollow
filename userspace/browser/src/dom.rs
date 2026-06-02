use std::string::String;
use std::vec::Vec;

#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Document,
    Element { tag: String },
    Text(String),
    Comment(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Attr {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub node_type: NodeType,
    pub attrs: Vec<Attr>,
    pub children: Vec<Node>,
}

impl Node {
    pub fn new_document() -> Self {
        Node {
            node_type: NodeType::Document,
            attrs: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn new_element(tag: impl Into<String>) -> Self {
        Node {
            node_type: NodeType::Element { tag: tag.into() },
            attrs: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn new_text(text: impl Into<String>) -> Self {
        Node {
            node_type: NodeType::Text(text.into()),
            attrs: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn new_comment(text: impl Into<String>) -> Self {
        Node {
            node_type: NodeType::Comment(text.into()),
            attrs: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn append_child(&mut self, child: Node) {
        self.children.push(child);
    }

    pub fn tag_name(&self) -> Option<&str> {
        match &self.node_type {
            NodeType::Element { tag } => Some(tag.as_str()),
            _ => None,
        }
    }

    pub fn text_content(&self) -> String {
        match &self.node_type {
            NodeType::Text(t) => t.clone(),
            _ => {
                let mut out = String::new();
                for child in &self.children {
                    out.push_str(&child.text_content());
                }
                out
            }
        }
    }

    pub fn query_selector(&self, tag: &str) -> Vec<&Node> {
        let mut results = Vec::new();
        self.query_selector_rec(tag, &mut results);
        results
    }

    fn query_selector_rec<'a>(&'a self, tag: &str, acc: &mut Vec<&'a Node>) {
        if let NodeType::Element { tag: t } = &self.node_type {
            if t.eq_ignore_ascii_case(tag) {
                acc.push(self);
            }
        }
        for child in &self.children {
            child.query_selector_rec(tag, acc);
        }
    }

    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|a| a.name.eq_ignore_ascii_case(name)).map(|a| a.value.as_str())
    }
}
