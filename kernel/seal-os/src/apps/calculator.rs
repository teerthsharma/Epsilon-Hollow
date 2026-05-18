// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Calculator — supports basic arithmetic, trig, log, and expression evaluation.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

pub struct Calculator {
    history: Vec<(String, f64)>,
    last_result: f64,
}

impl Calculator {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            last_result: 0.0,
        }
    }

    pub fn evaluate(&mut self, expr: &str) -> String {
        let expr = expr.trim();
        if expr.is_empty() {
            return String::from("Usage: calc <expression>\nExamples: calc 2+3, calc sin(1.57), calc sqrt(144)");
        }

        if expr == "history" {
            return self.show_history();
        }

        match self.parse_and_eval(expr) {
            Ok(result) => {
                self.last_result = result;
                self.history.push((String::from(expr), result));
                if result == (result as i64) as f64 && result.abs() < 1e15 {
                    format!("= {}", result as i64)
                } else {
                    format!("= {:.6}", result)
                }
            }
            Err(e) => format!("Error: {}", e),
        }
    }

    fn parse_and_eval(&self, expr: &str) -> Result<f64, &'static str> {
        let expr = expr.replace(' ', "");
        let expr = expr.replace("ans", &format!("{}", self.last_result));

        if let Some(inner) = try_func_call(&expr, "sin") {
            return self.parse_and_eval(inner).map(|v| libm::sin(v));
        }
        if let Some(inner) = try_func_call(&expr, "cos") {
            return self.parse_and_eval(inner).map(|v| libm::cos(v));
        }
        if let Some(inner) = try_func_call(&expr, "tan") {
            return self.parse_and_eval(inner).map(|v| libm::tan(v));
        }
        if let Some(inner) = try_func_call(&expr, "sqrt") {
            return self.parse_and_eval(inner).map(|v| libm::sqrt(v));
        }
        if let Some(inner) = try_func_call(&expr, "abs") {
            return self.parse_and_eval(inner).map(|v| libm::fabs(v));
        }
        if let Some(inner) = try_func_call(&expr, "ln") {
            return self.parse_and_eval(inner).map(|v| libm::log(v));
        }
        if let Some(inner) = try_func_call(&expr, "log") {
            return self.parse_and_eval(inner).map(|v| libm::log10(v));
        }
        if let Some(inner) = try_func_call(&expr, "exp") {
            return self.parse_and_eval(inner).map(|v| libm::exp(v));
        }
        if let Some(inner) = try_func_call(&expr, "ceil") {
            return self.parse_and_eval(inner).map(|v| libm::ceil(v));
        }
        if let Some(inner) = try_func_call(&expr, "floor") {
            return self.parse_and_eval(inner).map(|v| libm::floor(v));
        }

        if &expr == "pi" {
            return Ok(core::f64::consts::PI);
        }
        if &expr == "e" {
            return Ok(core::f64::consts::E);
        }

        self.eval_additive(&expr)
    }

    fn eval_additive(&self, expr: &str) -> Result<f64, &'static str> {
        let bytes = expr.as_bytes();
        let mut depth = 0i32;
        let mut last_op = None;

        for i in (0..bytes.len()).rev() {
            match bytes[i] {
                b')' => depth += 1,
                b'(' => depth -= 1,
                b'+' if depth == 0 && i > 0 => { last_op = Some((i, b'+')); break; }
                b'-' if depth == 0 && i > 0 && !is_op_char(bytes[i - 1]) => {
                    last_op = Some((i, b'-'));
                    break;
                }
                _ => {}
            }
        }

        if let Some((pos, op)) = last_op {
            let left = self.eval_multiplicative(&expr[..pos])?;
            let right = self.eval_multiplicative(&expr[pos + 1..])?;
            return Ok(if op == b'+' { left + right } else { left - right });
        }

        self.eval_multiplicative(expr)
    }

    fn eval_multiplicative(&self, expr: &str) -> Result<f64, &'static str> {
        let bytes = expr.as_bytes();
        let mut depth = 0i32;
        let mut last_op = None;

        for i in (0..bytes.len()).rev() {
            match bytes[i] {
                b')' => depth += 1,
                b'(' => depth -= 1,
                b'*' if depth == 0 => { last_op = Some((i, b'*')); break; }
                b'/' if depth == 0 => { last_op = Some((i, b'/')); break; }
                b'%' if depth == 0 => { last_op = Some((i, b'%')); break; }
                _ => {}
            }
        }

        if let Some((pos, op)) = last_op {
            let left = self.eval_power(&expr[..pos])?;
            let right = self.eval_power(&expr[pos + 1..])?;
            return match op {
                b'*' => Ok(left * right),
                b'/' => {
                    if right == 0.0 { Err("division by zero") } else { Ok(left / right) }
                }
                b'%' => {
                    if right == 0.0 { Err("modulo by zero") } else { Ok(libm::fmod(left, right)) }
                }
                _ => unreachable!(),
            };
        }

        self.eval_power(expr)
    }

    fn eval_power(&self, expr: &str) -> Result<f64, &'static str> {
        let bytes = expr.as_bytes();
        let mut depth = 0i32;

        for i in 0..bytes.len() {
            match bytes[i] {
                b'(' => depth += 1,
                b')' => depth -= 1,
                b'^' if depth == 0 => {
                    let base = self.eval_unary(&expr[..i])?;
                    let exp = self.eval_power(&expr[i + 1..])?;
                    return Ok(libm::pow(base, exp));
                }
                _ => {}
            }
        }

        self.eval_unary(expr)
    }

    fn eval_unary(&self, expr: &str) -> Result<f64, &'static str> {
        if expr.starts_with('-') {
            return self.eval_atom(&expr[1..]).map(|v| -v);
        }
        if expr.starts_with('+') {
            return self.eval_atom(&expr[1..]);
        }
        self.eval_atom(expr)
    }

    fn eval_atom(&self, expr: &str) -> Result<f64, &'static str> {
        let expr = expr.trim();
        if expr.is_empty() {
            return Err("empty expression");
        }

        if expr.starts_with('(') && expr.ends_with(')') {
            let inner = &expr[1..expr.len() - 1];
            let mut depth = 0i32;
            let mut valid = true;
            for b in inner.bytes() {
                match b {
                    b'(' => depth += 1,
                    b')' => {
                        depth -= 1;
                        if depth < 0 { valid = false; break; }
                    }
                    _ => {}
                }
            }
            if valid && depth == 0 {
                return self.parse_and_eval(inner);
            }
        }

        if expr == "pi" { return Ok(core::f64::consts::PI); }
        if expr == "e" { return Ok(core::f64::consts::E); }

        parse_number(expr).ok_or("invalid number or expression")
    }

    fn show_history(&self) -> String {
        if self.history.is_empty() {
            return String::from("(no calculations yet)");
        }
        let mut out = String::from("Calculator History\n══════════════════\n");
        for (i, (expr, result)) in self.history.iter().enumerate() {
            out.push_str(&format!("  {:>3}. {} = {}\n", i + 1, expr, result));
        }
        out
    }

    pub fn render_to_window(&self, win: &mut crate::wm::window::Window) {
        use crate::apps::game_engine;
        game_engine::clear_window(win);
        game_engine::render_text(win, 10, 10, "Calculator", 0xE94560);
        game_engine::render_text(win, 10, 30, "Type 'calc <expr>' in SealShell", 0xAAAAAA);
        game_engine::render_text(win, 10, 50, "Supports: + - * / ^ % sqrt sin cos tan ln log", 0x888888);
        game_engine::render_text(win, 10, 70, "Constants: pi, e, ans (last result)", 0x888888);

        let start = if self.history.len() > 8 { self.history.len() - 8 } else { 0 };
        for (i, (expr, result)) in self.history[start..].iter().enumerate() {
            let y = 100 + (i as u32) * 20;
            game_engine::render_text(win, 10, y, &format!("{} = {}", expr, result), 0xCCCCCC);
        }
    }
}

fn try_func_call<'a>(expr: &'a str, name: &str) -> Option<&'a str> {
    if expr.starts_with(name) && expr.as_bytes().get(name.len()) == Some(&b'(') && expr.ends_with(')') {
        Some(&expr[name.len() + 1..expr.len() - 1])
    } else {
        None
    }
}

fn is_op_char(b: u8) -> bool {
    matches!(b, b'+' | b'-' | b'*' | b'/' | b'^' | b'%' | b'(')
}

fn parse_number(s: &str) -> Option<f64> {
    if s.is_empty() { return None; }
    let mut result: f64 = 0.0;
    let mut frac = false;
    let mut frac_div: f64 = 1.0;
    let mut negative = false;
    let mut i = 0;
    let bytes = s.as_bytes();

    if bytes[0] == b'-' { negative = true; i = 1; }
    else if bytes[0] == b'+' { i = 1; }

    if i >= bytes.len() { return None; }

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'.' {
            if frac { return None; }
            frac = true;
        } else if b.is_ascii_digit() {
            if frac {
                frac_div *= 10.0;
                result += (b - b'0') as f64 / frac_div;
            } else {
                result = result * 10.0 + (b - b'0') as f64;
            }
        } else {
            return None;
        }
        i += 1;
    }

    if negative { result = -result; }
    Some(result)
}
