use std::string::{String, ToString};
use std::vec::Vec;

use crate::dom::{Attr, Node};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Doctype(String),
    StartTag { name: String, attrs: Vec<Attr>, self_closing: bool },
    EndTag { name: String },
    Text(String),
    Comment(String),
    Eof,
}

pub struct Tokenizer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        Tokenizer { input, pos: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn next_char(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn consume_while<F>(&mut self, pred: F) -> String
    where
        F: Fn(char) -> bool,
    {
        let mut out = String::new();
        while let Some(ch) = self.peek() {
            if pred(ch) {
                self.pos += ch.len_utf8();
                out.push(ch);
            } else {
                break;
            }
        }
        out
    }

    fn skip_whitespace(&mut self) {
        self.consume_while(|c| c.is_ascii_whitespace());
    }

    pub fn next_token(&mut self) -> Token {
        if self.pos >= self.input.len() {
            return Token::Eof;
        }

        if self.peek() == Some('<') {
            self.pos += 1; // consume '<'
            if self.peek() == Some('!') {
                self.pos += 1;
                if self.input[self.pos..].starts_with("--") {
                    self.pos += 2;
                    return self.read_comment();
                } else if self.input[self.pos..].to_ascii_lowercase().starts_with("doctype") {
                    return self.read_doctype();
                } else {
                    // Bogus declaration, skip until >
                    self.consume_while(|c| c != '>');
                    self.next_char(); // skip '>'
                    return self.next_token();
                }
            } else if self.peek() == Some('/') {
                self.pos += 1;
                let name = self.read_tag_name();
                self.consume_while(|c| c != '>');
                self.next_char(); // skip '>'
                return Token::EndTag { name };
            } else {
                let (name, attrs, self_closing) = self.read_start_tag();
                return Token::StartTag { name, attrs, self_closing };
            }
        }

        // Text node
        let mut text = String::new();
        while let Some(ch) = self.peek() {
            if ch == '<' {
                break;
            }
            self.pos += ch.len_utf8();
            text.push(ch);
        }
        Token::Text(text)
    }

    fn read_tag_name(&mut self) -> String {
        self.consume_while(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == ':')
    }

    fn read_start_tag(&mut self) -> (String, Vec<Attr>, bool) {
        let name = self.read_tag_name();
        let mut attrs = Vec::new();
        let mut self_closing = false;

        loop {
            self.skip_whitespace();
            if let Some('/') = self.peek() {
                if self.input[self.pos..].starts_with("/>") {
                    self_closing = true;
                    self.pos += 2;
                    break;
                }
            }
            if let Some('>') = self.peek() {
                self.pos += 1;
                break;
            }
            if let Some(attr) = self.read_attr() {
                attrs.push(attr);
            } else {
                // Stuck; skip one char to avoid infinite loop on garbage
                self.next_char();
            }
        }

        (name, attrs, self_closing)
    }

    fn read_attr(&mut self) -> Option<Attr> {
        let name = self.consume_while(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == ':' || c == '.');
        if name.is_empty() {
            return None;
        }
        self.skip_whitespace();
        let mut value = String::new();
        if self.peek() == Some('=') {
            self.pos += 1;
            self.skip_whitespace();
            value = self.read_attr_value();
        }
        Some(Attr { name, value })
    }

    fn read_attr_value(&mut self) -> String {
        let quote = self.peek();
        if quote == Some('"') || quote == Some('\'') {
            self.pos += 1;
            let mut val = String::new();
            while let Some(ch) = self.peek() {
                self.pos += ch.len_utf8();
                if ch == quote.unwrap() {
                    break;
                }
                val.push(ch);
            }
            val
        } else {
            self.consume_while(|c| !c.is_ascii_whitespace() && c != '>' && c != '/')
        }
    }

    fn read_comment(&mut self) -> Token {
        let mut text = String::new();
        while let Some(ch) = self.peek() {
            if self.input[self.pos..].starts_with("-->") {
                self.pos += 3;
                break;
            }
            self.pos += ch.len_utf8();
            text.push(ch);
        }
        Token::Comment(text)
    }

    fn read_doctype(&mut self) -> Token {
        let start = self.pos;
        self.consume_while(|c| c != '>');
        let text = self.input[start..self.pos].to_string();
        self.next_char(); // skip '>'
        Token::Doctype(text)
    }
}

pub fn tokenize(input: &str) -> Vec<Token> {
    let mut tokenizer = Tokenizer::new(input);
    let mut tokens = Vec::new();
    loop {
        let tok = tokenizer.next_token();
        if tok == Token::Eof {
            break;
        }
        tokens.push(tok);
    }
    tokens
}

/// Simple tree builder that constructs a DOM from tokens.
/// Handles mismatched tags by auto-closing open elements.
pub fn parse(input: &str) -> Node {
    let tokens = tokenize(input);
    let mut root = Node::new_document();
    let mut stack: Vec<Node> = Vec::new();

    for token in tokens {
        match token {
            Token::Doctype(_) => {}
            Token::StartTag { name, attrs, self_closing } => {
                let mut el = Node::new_element(name.clone());
                el.attrs = attrs;
                if self_closing || is_void_element(&name) {
                    if let Some(parent) = stack.last_mut() {
                        parent.append_child(el);
                    } else {
                        root.append_child(el);
                    }
                } else {
                    stack.push(el);
                }
            }
            Token::EndTag { name } => {
                // Pop until matching tag found; if not found, ignore.
                let mut found = false;
                let mut temp = Vec::new();
                while let Some(mut node) = stack.pop() {
                    if let Some(tag) = node.tag_name() {
                        if tag.eq_ignore_ascii_case(&name) {
                            found = true;
                            // Append temp nodes (mismatched) to this node first, then append to parent
                            for child in temp.into_iter().rev() {
                                node.append_child(child);
                            }
                            if let Some(parent) = stack.last_mut() {
                                parent.append_child(node);
                            } else {
                                root.append_child(node);
                            }
                            break;
                        } else {
                            temp.push(node);
                        }
                    } else {
                        temp.push(node);
                    }
                }
                if !found {
                    // No matching start tag; discard temp nodes to keep it simple
                }
            }
            Token::Text(text) => {
                if text.chars().all(|c| c.is_ascii_whitespace()) {
                    // Optional: ignore pure whitespace text nodes for cleaner tree
                    // But keep if inside <style> or <script>
                    if let Some(parent) = stack.last() {
                        if let Some(tag) = parent.tag_name() {
                            if tag.eq_ignore_ascii_case("style") || tag.eq_ignore_ascii_case("script") {
                                if let Some(parent) = stack.last_mut() {
                                    parent.append_child(Node::new_text(text));
                                }
                            }
                        }
                    }
                    continue;
                }
                if let Some(parent) = stack.last_mut() {
                    parent.append_child(Node::new_text(text));
                } else {
                    root.append_child(Node::new_text(text));
                }
            }
            Token::Comment(text) => {
                if let Some(parent) = stack.last_mut() {
                    parent.append_child(Node::new_comment(text));
                } else {
                    root.append_child(Node::new_comment(text));
                }
            }
            Token::Eof => break,
        }
    }

    // Remaining stack: auto-close everything
    while let Some(node) = stack.pop() {
        if let Some(parent) = stack.last_mut() {
            parent.append_child(node);
        } else {
            root.append_child(node);
        }
    }

    root
}

fn is_void_element(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "area" | "base" | "br" | "col" | "embed" | "hr" | "img" | "input"
        | "link" | "meta" | "param" | "source" | "track" | "wbr"
    )
}
