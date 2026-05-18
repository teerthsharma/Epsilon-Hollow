// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Calculator — scientific calculator with high-tech graphical display, expression history,
//! gradient buttons, anti-aliased text, and glow effects.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::graphics::htek;
use crate::wm::window::Window;

const BG_TOP: u32 = 0x00101020;
const BG_BOTTOM: u32 = 0x00080810;
const DISPLAY_TOP: u32 = 0x00141428;
const DISPLAY_BOTTOM: u32 = 0x000A0A18;
const DISPLAY_GLOW: u32 = 0x00003322;
const RESULT_COLOR: u32 = 0x0000FFAA;
const RESULT_GLOW: u32 = 0x00004433;
const EXPR_COLOR: u32 = 0x00558877;
const ANS_COLOR: u32 = 0x00336655;
const BTN_TOP_NUM: u32 = 0x002A2A48;
const BTN_BOT_NUM: u32 = 0x001E1E38;
const BTN_TOP_OP: u32 = 0x00442030;
const BTN_BOT_OP: u32 = 0x00331828;
const BTN_TOP_FN: u32 = 0x00203044;
const BTN_BOT_FN: u32 = 0x00182038;
const BTN_TOP_EQ: u32 = 0x00204838;
const BTN_BOT_EQ: u32 = 0x00183828;
const BTN_BORDER: u32 = 0x00404060;
const TEXT_NUM: u32 = 0x00DDDDEE;
const TEXT_OP: u32 = 0x00FF5577;
const TEXT_FN: u32 = 0x0099BBFF;
const TEXT_EQ: u32 = 0x0066FFAA;
const HISTORY_FG: u32 = 0x00606080;
const LABEL_FG: u32 = 0x00808098;
const SEPARATOR: u32 = 0x00303050;

pub struct Calculator {
    history: Vec<(String, f64)>,
    last_result: f64,
    input_buf: String,
}

impl Calculator {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            last_result: 0.0,
            input_buf: String::new(),
        }
    }

    pub fn key_press(&mut self, ch: u8) {
        match ch {
            b'0'..=b'9' | b'.' | b'+' | b'-' | b'*' | b'/' | b'^' | b'(' | b')' => {
                self.input_buf.push(ch as char);
            }
            b'\n' | b'=' => {
                let expr = self.input_buf.clone();
                let _result = self.evaluate(&expr);
                self.input_buf.clear();
            }
            0x08 => {
                self.input_buf.pop();
            }
            b'c' | b'C' => {
                self.input_buf.clear();
            }
            _ => {}
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
        let mut out = String::from("Calculator History\n");
        for (i, (expr, result)) in self.history.iter().enumerate() {
            out.push_str(&format!("  {:>3}. {} = {}\n", i + 1, expr, result));
        }
        out
    }

    pub fn render_to_window(&self, win: &mut Window) {
        let cw = win.client_width();
        let ch = win.client_height();

        // Gradient background
        htek::fill_gradient_v(win, 0, 0, cw, ch, BG_TOP, BG_BOTTOM);

        let margin = 10u32;
        let disp_h = 100u32;

        // Display area with rounded corners and subtle glow
        htek::glow_rect(win, margin, margin, cw - margin * 2, disp_h, 6, DISPLAY_GLOW);
        htek::fill_rounded_rect_gradient(
            win, margin, margin, cw - margin * 2, disp_h, 8, DISPLAY_TOP, DISPLAY_BOTTOM,
        );
        htek::stroke_rounded_rect(
            win, margin, margin, cw - margin * 2, disp_h, 8, 1, SEPARATOR,
        );

        // Display content
        if let Some((expr, result)) = self.history.last() {
            htek::render_text_small(win, margin + 12, margin + 10, expr, EXPR_COLOR);

            let result_str = if *result == (*result as i64) as f64 && result.abs() < 1e15 {
                format!("= {}", *result as i64)
            } else {
                format!("= {:.6}", result)
            };
            htek::render_text_glow(win, margin + 12, margin + 32, &result_str, RESULT_COLOR, RESULT_GLOW);
        } else {
            htek::render_text_glow(win, margin + 12, margin + 32, "0", RESULT_COLOR, RESULT_GLOW);
        }

        // ans indicator
        let ans_str = format!("ans = {:.4}", self.last_result);
        htek::render_text_small(win, margin + 12, margin + disp_h - 20, &ans_str, ANS_COLOR);

        // Separator line
        htek::draw_line_h(win, margin, margin + disp_h + 4, cw - margin * 2, SEPARATOR, 80);

        // Button grid
        let grid_top = margin + disp_h + 10;
        let cols = 4u32;
        let gap = 5u32;
        let btn_w = (cw - margin * 2 - gap * (cols - 1)) / cols;
        let btn_h = 34u32;

        let buttons: &[&[(&str, u32, u32, u32)]] = &[
            &[("sin", BTN_TOP_FN, BTN_BOT_FN, TEXT_FN), ("cos", BTN_TOP_FN, BTN_BOT_FN, TEXT_FN),
              ("tan", BTN_TOP_FN, BTN_BOT_FN, TEXT_FN), ("C", BTN_TOP_OP, BTN_BOT_OP, TEXT_OP)],
            &[("sqrt", BTN_TOP_FN, BTN_BOT_FN, TEXT_FN), ("log", BTN_TOP_FN, BTN_BOT_FN, TEXT_FN),
              ("ln", BTN_TOP_FN, BTN_BOT_FN, TEXT_FN), ("^", BTN_TOP_OP, BTN_BOT_OP, TEXT_OP)],
            &[("7", BTN_TOP_NUM, BTN_BOT_NUM, TEXT_NUM), ("8", BTN_TOP_NUM, BTN_BOT_NUM, TEXT_NUM),
              ("9", BTN_TOP_NUM, BTN_BOT_NUM, TEXT_NUM), ("/", BTN_TOP_OP, BTN_BOT_OP, TEXT_OP)],
            &[("4", BTN_TOP_NUM, BTN_BOT_NUM, TEXT_NUM), ("5", BTN_TOP_NUM, BTN_BOT_NUM, TEXT_NUM),
              ("6", BTN_TOP_NUM, BTN_BOT_NUM, TEXT_NUM), ("*", BTN_TOP_OP, BTN_BOT_OP, TEXT_OP)],
            &[("1", BTN_TOP_NUM, BTN_BOT_NUM, TEXT_NUM), ("2", BTN_TOP_NUM, BTN_BOT_NUM, TEXT_NUM),
              ("3", BTN_TOP_NUM, BTN_BOT_NUM, TEXT_NUM), ("-", BTN_TOP_OP, BTN_BOT_OP, TEXT_OP)],
            &[("0", BTN_TOP_NUM, BTN_BOT_NUM, TEXT_NUM), (".", BTN_TOP_NUM, BTN_BOT_NUM, TEXT_NUM),
              ("=", BTN_TOP_EQ, BTN_BOT_EQ, TEXT_EQ), ("+", BTN_TOP_OP, BTN_BOT_OP, TEXT_OP)],
            &[("(", BTN_TOP_FN, BTN_BOT_FN, TEXT_FN), (")", BTN_TOP_FN, BTN_BOT_FN, TEXT_FN),
              ("pi", BTN_TOP_FN, BTN_BOT_FN, TEXT_FN), ("e", BTN_TOP_FN, BTN_BOT_FN, TEXT_FN)],
        ];

        for (row_idx, row) in buttons.iter().enumerate() {
            let by = grid_top + row_idx as u32 * (btn_h + gap);
            if by + btn_h > ch { break; }

            for (col_idx, &(label, top_c, bot_c, text_c)) in row.iter().enumerate() {
                let bx = margin + col_idx as u32 * (btn_w + gap);
                if bx + btn_w > cw - margin { break; }

                htek::fill_rounded_rect_gradient(win, bx, by, btn_w, btn_h, 6, top_c, bot_c);
                htek::stroke_rounded_rect(win, bx, by, btn_w, btn_h, 6, 1, BTN_BORDER);

                // Centered label (small text for compactness)
                let text_px_w = label.len() as u32 * htek::TEXT_CHAR_W;
                let tx = bx + (btn_w.saturating_sub(text_px_w)) / 2;
                let ty = by + (btn_h.saturating_sub(htek::TEXT_CHAR_H)) / 2;
                htek::render_text_small(win, tx, ty, label, text_c);
            }
        }

        // History panel
        let hist_top = grid_top + 7 * (btn_h + gap) + 4;
        if hist_top + 20 < ch && !self.history.is_empty() {
            htek::draw_line_h(win, margin, hist_top - 2, cw - margin * 2, SEPARATOR, 60);
            htek::render_text_small(win, margin, hist_top, "History", LABEL_FG);
            let start = if self.history.len() > 3 { self.history.len() - 3 } else { 0 };
            for (i, (expr, result)) in self.history[start..].iter().enumerate() {
                let hy = hist_top + (i as u32 + 1) * (htek::TEXT_CHAR_H + 3);
                if hy + htek::TEXT_CHAR_H > ch { break; }
                let line = format!("{} = {:.4}", expr, result);
                htek::render_text_small(win, margin + 4, hy, &line, HISTORY_FG);
            }
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
