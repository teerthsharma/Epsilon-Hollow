// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! SealShell - the human-friendly shell. Seal-native commands only.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::drivers::bluetooth::BluetoothDriver;
use crate::drivers::wifi::WifiDriver;
use crate::fs::manifold_fs::ManifoldFS;
use crate::lang::AetherRuntime;
use crate::security::mac::Permissions;

use super::calculator::Calculator;
use super::clipboard::Clipboard;
use super::help;
use super::media_player::MediaPlayer;

pub struct Shell {
    fs: ManifoldFS,
    cwd: u64,
    cwd_path: String,
    history: Vec<String>,
    clipboard: Clipboard,
    calculator: Calculator,
    media_player: MediaPlayer,
    last_model: Option<Vec<u8>>,
    vars: Vars,
    /// Scripts open right now; see `MAX_SOURCE_DEPTH`.
    source_depth: usize,
}

/// Absolute path named by a shell argument, or `None` when the argument is
/// not already in the canonical form the access policy is keyed on.
///
/// `Shell` holds its own `ManifoldFS` (see `Shell::new`), whose inherent
/// methods carry no permission logic at all — only `Vfs` calls
/// `security::mac::check_file_permission`. Every command therefore has to
/// name the file to the policy itself, and the string it names must denote
/// the same inode the filesystem will open.
///
/// This validates rather than rewrites. `Vfs::canonicalize_path`
/// (fs/vfs.rs:172) is private, and a second canonicalizer here is exactly
/// the defect that let `//root/secret` past a `Deny /root` rule: two
/// components normalizing one string differently authorize one file and open
/// another. Rejecting non-canonical components instead — empty (`//`), `.`,
/// or `..` — cannot disagree in the permissive direction. What remains is a
/// literal component list, and `ManifoldFS::resolve_path`
/// (fs/manifold_fs.rs:646) walks components as literal directory-entry
/// lookups with no symlink substitution, so the path checked and the inode
/// reached are the same file.
///
/// `cwd_path` is the form `Shell::update_cwd_path` produces: `/`, or
/// `/a/b/` with a trailing separator. An empty `arg` means that directory
/// itself.
///
/// Free-standing rather than a method so it is exercisable without building
/// a `Shell`, which mounts — and on a blank disk formats — the AHCI block
/// store.
fn abs_path_in(cwd_path: &str, arg: &str) -> Option<String> {
    let cwd = if cwd_path == "/" {
        "/"
    } else {
        cwd_path.trim_end_matches('/')
    };
    if arg.is_empty() {
        return Some(String::from(cwd));
    }
    if arg == "/" {
        return Some(String::from("/"));
    }
    let joined = if arg.starts_with('/') {
        String::from(arg)
    } else if cwd == "/" {
        format!("/{}", arg)
    } else {
        format!("{}/{}", cwd, arg)
    };
    for part in joined.split('/').skip(1) {
        if part.is_empty() || part == "." || part == ".." {
            return None;
        }
    }
    Some(joined)
}

/// Largest output one pipeline stage may hand the next, and the largest file
/// `<` will read in.
///
/// Stages run to completion one after another (see `run_pipeline`), so the
/// whole of a stage's output is resident at once and a runaway producer would
/// otherwise be bounded only by the heap. Over the cap is an error, not a
/// truncation: a downstream stage handed a silently cut buffer would report a
/// confident wrong answer.
const PIPE_CAPACITY: usize = 64 * 1024;

/// One pipeline stage: its command text with the redirections removed, plus
/// the files those redirections named.
///
/// `output` carries the append flag — `true` for `>>`, `false` for `>`.
struct Stage<'a> {
    cmd: &'a str,
    input: Option<&'a str>,
    output: Option<(&'a str, bool)>,
}

/// Split one pipeline stage into its command and its redirections.
///
/// The shell has no quoting, so `<` and `>` are metacharacters everywhere in
/// a line and cannot be escaped; see the ceiling note on `Shell::run_line`.
/// A redirection with no file after it is a syntax error rather than a
/// no-op, and so is a redirection with no command before it.
///
/// Free-standing so the parse is exercisable without building a `Shell`,
/// which mounts — and on a blank disk formats — the AHCI block store.
fn parse_stage(stage: &str) -> Result<Stage<'_>, String> {
    let is_redirect = |c: char| c == '<' || c == '>';
    let split = stage.find(is_redirect).unwrap_or(stage.len());
    let mut parsed = Stage {
        cmd: stage[..split].trim(),
        input: None,
        output: None,
    };
    let mut rest = &stage[split..];
    while let Some(first) = rest.chars().next() {
        let (op, after) = if let Some(a) = rest.strip_prefix(">>") {
            (">>", a)
        } else if first == '>' {
            (">", &rest[1..])
        } else {
            ("<", &rest[1..])
        };
        let end = after.find(is_redirect).unwrap_or(after.len());
        let target = after[..end].trim();
        if target.is_empty() {
            return Err(format!("seal: syntax error: '{}' needs a file name", op));
        }
        if op == "<" {
            parsed.input = Some(target);
        } else {
            parsed.output = Some((target, op == ">>"));
        }
        rest = &after[end..];
    }
    if parsed.cmd.is_empty() {
        return Err(format!(
            "seal: syntax error: missing command in '{}'",
            stage.trim()
        ));
    }
    Ok(parsed)
}

/// Split a command line into its pipeline stages.
///
/// Redirection is confined to the ends of the pipeline: `<` on the first
/// stage, `>`/`>>` on the last. A `>` in the middle would send that stage's
/// output to a file and leave the next stage with nothing, which is `sh`
/// behaviour but is far more likely a typo here; refusing it costs the user a
/// retry and never silently empties a stage.
///
/// Free-standing for the same reason as `parse_stage`: no `Shell`, so no
/// AHCI mount.
fn parse_line(input: &str) -> Result<Vec<Stage<'_>>, String> {
    let raw: Vec<&str> = input.split('|').collect();
    let last = raw.len() - 1;
    let mut stages = Vec::with_capacity(raw.len());
    for (i, part) in raw.iter().enumerate() {
        let stage = parse_stage(part)?;
        if stage.input.is_some() && i != 0 {
            return Err(String::from(
                "seal: '<' is only allowed on the first stage of a pipeline",
            ));
        }
        if stage.output.is_some() && i != last {
            return Err(String::from(
                "seal: '>' is only allowed on the last stage of a pipeline",
            ));
        }
        stages.push(stage);
    }
    Ok(stages)
}

/// Run `commands` in order, each one's output becoming the next one's stdin.
///
/// Sequential, not concurrent: a stage runs to completion into a buffer and
/// the next stage starts on that buffer. There is no process-level stdin or
/// stdout to plumb — commands are `&mut Shell` methods returning a `String`
/// (`Shell::dispatch`) — and a kernel shell has no interactive pipeline to
/// keep responsive, so the only thing concurrency would buy is unbounded
/// buffering complexity.
///
/// A stage feeding another stage more than `PIPE_CAPACITY` bytes aborts the
/// pipeline with an error; the last stage is uncapped because its output goes
/// to the terminal, which is what a bare command already does today.
///
/// Takes the runner as a closure so the composition is testable without a
/// `Shell` and its AHCI mount.
fn run_pipeline<F>(commands: &[&str], stdin: String, mut run: F) -> Result<String, String>
where
    F: FnMut(&str, &str) -> String,
{
    let mut buffer = stdin;
    for (i, cmd) in commands.iter().enumerate() {
        let produced = run(cmd, &buffer);
        if i + 1 < commands.len() && produced.len() > PIPE_CAPACITY {
            return Err(format!(
                "seal: stage {} produced {} bytes, over the {}-byte pipe buffer",
                i + 1,
                produced.len(),
                PIPE_CAPACITY
            ));
        }
        buffer = produced;
    }
    Ok(buffer)
}

/// Lines `head` and `tail` take when no `-n` says otherwise.
const DEFAULT_LINES: usize = 10;

/// Leading single-letter flags of a filter's arguments, and the rest of the
/// argument text.
///
/// The shell has no argv tokenizer — every command arm is built on
/// `splitn(3, ' ')` (`Shell::dispatch`) and the ceiling note on
/// `Shell::run_line` covers why adding one is a separate change — so each
/// filter parses its own flags. Leading tokens beginning with `-` are flags,
/// letters may be bundled (`-ru` is `-r -u`), and the first token that is not
/// a flag ends them: the remainder comes back verbatim, inner spaces and all,
/// so `grep two words` keeps its pattern.
///
/// A letter no filter here accepts is refused rather than ignored, because a
/// dropped flag answers a different question than the one asked. A lone `-`,
/// and a `-` before a digit, are not flags but the start of the remainder: so
/// `grep -` searches for a hyphen, and `head -n -3` reports a bad line count
/// rather than an unknown option.
fn split_flags<'a>(cmd: &str, args: &'a str, allowed: &str) -> Result<(String, &'a str), String> {
    let mut flags = String::new();
    let mut rest = args.trim();
    while let Some(token) = rest.split_whitespace().next() {
        let after_dash = token.chars().nth(1);
        if !token.starts_with('-') || matches!(after_dash, None | Some('0'..='9')) {
            break;
        }
        for c in token.chars().skip(1) {
            if !allowed.contains(c) {
                return Err(format!(
                    "{}: unknown option '-{}' (accepted: -{})",
                    cmd, c, allowed
                ));
            }
            flags.push(c);
        }
        rest = rest[token.len()..].trim_start();
    }
    Ok((flags, rest))
}

/// Lines of `stdin` matching `pattern`, each keeping its terminator.
///
/// Flags: `i` matches without regard to case, `v` keeps the lines that do not
/// match, `n` prefixes each kept line with its 1-based number in the input,
/// and `c` reports how many lines were kept instead of the lines themselves.
///
/// ponytail: `i` lowercases the pattern and every line it tests, one
/// allocation per line. Upgrade path if a large stream ever makes that show
/// up: compare `char` iterators with `to_lowercase` applied per character.
fn grep_matches(flags: &str, pattern: &str, stdin: &str) -> String {
    let fold = flags.contains('i');
    let invert = flags.contains('v');
    let needle = if fold {
        pattern.to_lowercase()
    } else {
        String::from(pattern)
    };
    let mut out = String::new();
    let mut kept = 0usize;
    for (i, line) in stdin.lines().enumerate() {
        let hay = if fold {
            line.to_lowercase()
        } else {
            String::from(line)
        };
        if hay.contains(&needle) == invert {
            continue;
        }
        kept += 1;
        if !flags.contains('c') {
            if flags.contains('n') {
                out.push_str(&format!("{}:", i + 1));
            }
            out.push_str(line);
            out.push('\n');
        }
    }
    if flags.contains('c') {
        format!("{}\n", kept)
    } else {
        out
    }
}

/// Lines, words and bytes of `stdin`, in that order, whichever of `l`, `w`
/// and `c` are set — all three when none is.
///
/// A line is what `str::lines` yields, so a final line with no terminator
/// still counts; a word is a run of non-whitespace; a byte is a byte.
fn wc_counts(flags: &str, stdin: &str) -> String {
    let all = flags.is_empty();
    let mut fields: Vec<String> = Vec::new();
    if all || flags.contains('l') {
        fields.push(format!("{}", stdin.lines().count()));
    }
    if all || flags.contains('w') {
        fields.push(format!("{}", stdin.split_whitespace().count()));
    }
    if all || flags.contains('c') {
        fields.push(format!("{}", stdin.len()));
    }
    format!("{}\n", fields.join(" "))
}

/// The first `count` lines of `stdin`, or the last `count` when `from_end`,
/// each terminated.
fn take_lines(count: usize, from_end: bool, stdin: &str) -> String {
    let lines: Vec<&str> = stdin.lines().collect();
    let skip = if from_end {
        lines.len().saturating_sub(count)
    } else {
        0
    };
    let mut out = String::new();
    for line in lines.iter().skip(skip).take(count) {
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// The lines of `stdin` in byte order, each terminated; `unique` drops
/// repeats, `reverse` turns the order around afterwards so `-ru` is the
/// unique lines descending.
///
/// Every line comes back terminated whether or not the input ended in a
/// newline, because the result is a list of lines and a filter downstream
/// reading it with `str::lines` must see the same count. An empty stream has
/// no lines and so produces nothing at all, not a blank line.
fn sort_lines(reverse: bool, unique: bool, stdin: &str) -> String {
    let mut lines: Vec<&str> = stdin.lines().collect();
    lines.sort_unstable();
    if unique {
        lines.dedup();
    }
    if reverse {
        lines.reverse();
    }
    let mut out = String::new();
    for line in lines {
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// `stdin` with each run of identical adjacent lines collapsed to one,
/// prefixed by the length of the run when `with_count`.
///
/// Only adjacent lines: duplicates that are not neighbours both survive,
/// which is what makes `sort | uniq` different from `uniq` alone.
fn uniq_lines(with_count: bool, stdin: &str) -> String {
    let mut out = String::new();
    let mut run: Option<(&str, usize)> = None;
    for line in stdin.lines() {
        run = match run {
            Some((prev, n)) if prev == line => Some((prev, n + 1)),
            Some((prev, n)) => {
                push_run(&mut out, prev, n, with_count);
                Some((line, 1))
            }
            None => Some((line, 1)),
        };
    }
    if let Some((prev, n)) = run {
        push_run(&mut out, prev, n, with_count);
    }
    out
}

/// Emit one `uniq` run: the line, terminated, behind its count when asked.
fn push_run(out: &mut String, line: &str, n: usize, with_count: bool) {
    if with_count {
        out.push_str(&format!("{} ", n));
    }
    out.push_str(line);
    out.push('\n');
}

/// `stdin` with every `from` character replaced by `to`.
///
/// Characters, not lines: whatever the stream ended with, it still ends with.
fn tr_chars(from: char, to: char, stdin: &str) -> String {
    stdin
        .chars()
        .map(|c| if c == from { to } else { c })
        .collect()
}

/// The single `char` of `s`, or `None` when it is not exactly one.
fn one_char(s: &str) -> Option<char> {
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Some(c),
        _ => None,
    }
}

/// Run the stage text `input` as a filter over `stdin`, or `None` when its
/// command is not a filter.
///
/// Takes the whole stage text, splits the command from its arguments, and
/// owns the table: all of it lives here rather than in `Shell::dispatch` so
/// that the name-to-behaviour mapping and every filter's argument parsing are
/// exercisable without building a `Shell` — which mounts, and on a blank disk
/// formats, the AHCI block store. It is the same reason `parse_stage` and
/// `run_pipeline` are free functions.
///
/// Everything after the first space is the argument text, verbatim, so a
/// filter can be handed an argument containing spaces (`grep two words`).
///
/// A filter that cannot parse its arguments returns the message as its
/// output. There is no exit status to fail with (see `Shell::run_line`), and
/// a message on the terminal is what the operator needs; what it must never
/// do is proceed on a guessed argument and report a confident wrong answer.
fn run_filter(input: &str, stdin: &str) -> Option<String> {
    let (cmd, args) = input.split_once(' ').unwrap_or((input, ""));
    let result = match cmd {
        "grep" => filter_grep(args, stdin),
        "wc" => filter_wc(args, stdin),
        "head" | "tail" => filter_head_tail(cmd, args, stdin),
        "sort" => filter_sort(args, stdin),
        "uniq" => filter_uniq(args, stdin),
        "tr" => filter_tr(args, stdin),
        // `echo` reads no stdin and produces its argument text, verbatim and
        // already expanded, with a terminator. It lives here because it needs
        // no `Shell` either, and because a stage that ignores its input is
        // still a stage: `echo a b c | wc -w` works like any other pipeline.
        "echo" => Ok(format!("{}\n", args)),
        // A named file is `Shell::cmd_peek`'s job, and needs the filesystem.
        "cat" if args.trim().is_empty() => Ok(String::from(stdin)),
        _ => return None,
    };
    Some(result.unwrap_or_else(|e| e))
}

fn filter_grep(args: &str, stdin: &str) -> Result<String, String> {
    let (flags, pattern) = split_flags("grep", args, "cinv")?;
    if pattern.is_empty() {
        return Err(String::from(
            "grep: what text? Usage: <command> | grep [-c] [-i] [-n] [-v] <text>",
        ));
    }
    Ok(grep_matches(&flags, pattern, stdin))
}

fn filter_wc(args: &str, stdin: &str) -> Result<String, String> {
    let (flags, rest) = split_flags("wc", args, "clw")?;
    if !rest.is_empty() {
        return Err(format!(
            "wc: unexpected argument '{}'. Usage: <command> | wc [-l] [-w] [-c]",
            rest
        ));
    }
    Ok(wc_counts(&flags, stdin))
}

/// `head`/`tail`, which differ only in the end of the stream they keep.
///
/// A count that is zero, negative or not a number is refused: silently
/// falling back to the default would print ten lines to someone who asked
/// for something else.
fn filter_head_tail(cmd: &str, args: &str, stdin: &str) -> Result<String, String> {
    let (flags, rest) = split_flags(cmd, args, "n")?;
    let count = if flags.contains('n') {
        match rest.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => {
                return Err(format!(
                    "{}: -n needs a line count of 1 or more, not '{}'",
                    cmd, rest
                ))
            }
        }
    } else if rest.is_empty() {
        DEFAULT_LINES
    } else {
        return Err(format!(
            "{}: unexpected argument '{}'. Usage: <command> | {} [-n <lines>]",
            cmd, rest, cmd
        ));
    };
    Ok(take_lines(count, cmd == "tail", stdin))
}

fn filter_sort(args: &str, stdin: &str) -> Result<String, String> {
    let (flags, rest) = split_flags("sort", args, "ru")?;
    if !rest.is_empty() {
        return Err(format!(
            "sort: unexpected argument '{}'. Usage: <command> | sort [-r] [-u]",
            rest
        ));
    }
    Ok(sort_lines(flags.contains('r'), flags.contains('u'), stdin))
}

fn filter_uniq(args: &str, stdin: &str) -> Result<String, String> {
    let (flags, rest) = split_flags("uniq", args, "c")?;
    if !rest.is_empty() {
        return Err(format!(
            "uniq: unexpected argument '{}'. Usage: <command> | uniq [-c]",
            rest
        ));
    }
    Ok(uniq_lines(flags.contains('c'), stdin))
}

/// `tr <from> <to>`, one character to one character.
///
/// ponytail: no ranges (`a-z`) and no classes (`[:upper:]`), which is why
/// both arguments are checked to be exactly one `char` rather than truncated
/// — `tr a-z A-Z` asks for something this does not do and must say so instead
/// of translating `a` to `A` and dropping the rest. Upgrade path: expand a
/// `x-y` argument into a character set and map by position.
fn filter_tr(args: &str, stdin: &str) -> Result<String, String> {
    let mut parts = args.split_whitespace();
    let from = parts.next().and_then(one_char);
    let to = parts.next().and_then(one_char);
    let extra = parts.next();
    match (from, to, extra) {
        (Some(from), Some(to), None) => Ok(tr_chars(from, to, stdin)),
        _ => Err(String::from(
            "tr: usage: <command> | tr <from> <to> — one character each, no ranges or classes",
        )),
    }
}

/// Bytes a redirection target must hold afterwards: `produced` alone for `>`,
/// or whatever was there followed by `produced` for `>>`.
fn redirected_content(existing: Option<String>, produced: &str, append: bool) -> String {
    match existing {
        Some(prior) if append => prior + produced,
        _ => String::from(produced),
    }
}

/// The shell's variables: name to value, with the flag `export` sets.
///
/// Ordered rather than hashed, so `set` and `env` list in name order without
/// sorting anything and two runs of the same input print the same lines.
type Vars = BTreeMap<String, (String, bool)>;

/// Most variables the store holds, and the most bytes their names and values
/// may total.
///
/// The kernel heap is fixed (`memory::heap_stats`) and a variable lives until
/// the machine is reset, so an uncapped store is a way to exhaust the heap
/// from the terminal. Both caps refuse the assignment rather than evicting
/// something: a variable that quietly vanished would make every later `$NAME`
/// expand to empty, which reads as a wrong answer instead of a full store.
const MAX_VARS: usize = 64;
const MAX_VAR_BYTES: usize = 4096;

/// Whether `s` is a variable name: `[A-Za-z_][A-Za-z0-9_]*`.
fn is_var_name(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The name and the unexpanded value of an assignment statement, or `None`
/// when `stmt` is not one and belongs on the normal command path.
///
/// The name is everything before the first `=`, so `A=b=c` gives `A` the value
/// `b=c`. Anything that is not a name — `a-b=c`, `1a=c`, `=c`, `set k v` —
/// is not an assignment at all and stays an ordinary command line, which is
/// what makes `a-b=c` an unknown command rather than a variable no `$a-b`
/// could ever name.
fn parse_assignment(stmt: &str) -> Option<(&str, &str)> {
    let (name, value) = stmt.split_once('=')?;
    if !is_var_name(name) {
        return None;
    }
    Some((name, value))
}

/// Bytes the store is holding: every name plus every value.
fn var_bytes(vars: &Vars) -> usize {
    vars.iter().map(|(k, (v, _))| k.len() + v.len()).sum()
}

/// Give `name` the value `value`, keeping whatever `export` flag it already
/// had, or say why it cannot.
///
/// A value may not contain whitespace. This shell has no quoting (see the
/// ceiling note on `Shell::run_line`), so `A=1 look` cannot be told apart from
/// a value that happens to contain a space, and the two readings do opposite
/// things: `sh` would set `A` for one command and run `look`, while taking the
/// rest of the line as the value silently stores `1 look` and runs nothing.
/// Refusing is the only answer that cannot do something the operator did not
/// ask for, and it buys a second property the expander depends on — since no
/// value holds a space, `$NAME` always expands to exactly one argument and
/// expansion can never change how `Shell::dispatch` splits a line.
fn set_var(vars: &mut Vars, name: &str, value: &str) -> Result<(), String> {
    if !is_var_name(name) {
        return Err(format!(
            "seal: '{}' is not a variable name: a letter or '_' then letters, digits or '_'",
            name
        ));
    }
    if value.chars().any(char::is_whitespace) {
        return Err(format!(
            "seal: {}=: a value may not contain a space — this shell has no quoting, so '{}=<value> <command>' cannot be told from one value",
            name, name
        ));
    }
    let prior = vars.get(name).map(|(v, e)| (v.len(), *e));
    if prior.is_none() && vars.len() >= MAX_VARS {
        return Err(format!(
            "seal: {} variables is the limit; 'unset <name>' frees one",
            MAX_VARS
        ));
    }
    // Reassigning replaces the old value's bytes; a new name adds its own too.
    let after = match prior {
        Some((prior_len, _)) => var_bytes(vars) - prior_len + value.len(),
        None => var_bytes(vars) + name.len() + value.len(),
    };
    if after > MAX_VAR_BYTES {
        return Err(format!(
            "seal: variables may total {} bytes; that assignment needs {}",
            MAX_VAR_BYTES, after
        ));
    }
    let exported = prior.map(|(_, e)| e).unwrap_or(false);
    vars.insert(String::from(name), (String::from(value), exported));
    Ok(())
}

/// `text` with every `$NAME` and `${NAME}` replaced by that variable's value,
/// or the syntax error that stopped it.
///
/// A name that is not set expands to nothing. A `$` that no name follows is a
/// literal `$`, which is what keeps `write price.txt $5` and `$?` — this shell
/// has no exit status — the text they look like rather than silent empties.
/// Inside braces there is no such reading: `${` must be closed and must hold a
/// name, so `${A` and `${a-b}` are errors the operator sees.
///
/// One pass, left to right: what a variable expands to is copied to the output
/// and never looked at again, so the recursion bound is exactly one
/// substitution deep and no pair of variables can make this run twice over the
/// same text. `A=$B` with `B=$A` terminates because each assignment expands
/// its right-hand side once, when it is made.
fn expand_vars(text: &str, vars: &Vars) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find('$') {
        out.push_str(&rest[..at]);
        let after = &rest[at + 1..];
        let braced = after.starts_with('{');
        let (name, tail) = if braced {
            match after[1..].find('}') {
                Some(end) => (&after[1..1 + end], &after[end + 2..]),
                None => {
                    return Err(String::from(
                        "seal: syntax error: '${' has no closing '}'",
                    ))
                }
            }
        } else {
            let end = after
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(after.len());
            (&after[..end], &after[end..])
        };
        if !is_var_name(name) {
            if braced {
                return Err(format!(
                    "seal: syntax error: '${{{}}}' does not name a variable",
                    name
                ));
            }
            out.push('$');
            rest = after;
            continue;
        }
        if let Some((value, _)) = vars.get(name) {
            out.push_str(value);
        }
        rest = tail;
    }
    out.push_str(rest);
    Ok(out)
}

/// `text` expanded, refused when there is nothing left of it.
///
/// Used for a stage's command and for a redirection's file name, the two
/// places where an empty string is not an empty argument but a missing one:
/// `peek notes > $OUT` with `OUT` unset would otherwise create a file called
/// `$OUT`, and `$CMD` alone would report an unknown command named nothing.
fn expand_required(text: &str, vars: &Vars) -> Result<String, String> {
    let expanded = expand_vars(text, vars)?;
    if expanded.trim().is_empty() {
        return Err(format!("seal: '{}' expanded to nothing", text.trim()));
    }
    Ok(expanded)
}

/// `export NAME` or `export NAME=value`: mark a variable for the environment,
/// creating it empty if it does not exist yet.
///
/// Arguments reach this already expanded (`Shell::run_line`), and no value
/// holds a space, so exactly one token is the whole of a well-formed argument
/// and a second one is a mistake rather than a second name to export.
///
/// ponytail: the mark is a mark and nothing more — `env` reports the set and
/// no command receives it. Seal OS commands are `Shell` methods rather than
/// processes (see `Shell::dispatch`), so there is no environment block to put
/// anything in; inventing one would mean a process boundary that does not
/// exist. Upgrade path: when `run` or `install` spawns a real task, hand the
/// exported pairs to whatever builds its address space.
fn cmd_export(vars: &mut Vars, args: &str) -> String {
    let mut tokens = args.split_whitespace();
    let (name, value) = match (tokens.next(), tokens.next()) {
        (Some(arg), None) => match parse_assignment(arg) {
            Some((name, value)) => (name, Some(value)),
            None => (arg, None),
        },
        _ => return String::from("export: usage: export <NAME> or export <NAME>=<value>"),
    };
    if !is_var_name(name) {
        return format!(
            "export: '{}' is not a variable name: a letter or '_' then letters, digits or '_'",
            name
        );
    }
    if let Some(value) = value {
        if let Err(e) = set_var(vars, name, value) {
            return e;
        }
    } else if !vars.contains_key(name) {
        if let Err(e) = set_var(vars, name, "") {
            return e;
        }
    }
    if let Some(entry) = vars.get_mut(name) {
        entry.1 = true;
    }
    String::new()
}

/// `unset NAME`: forget a variable, whether or not it was set.
fn cmd_unset(vars: &mut Vars, args: &str) -> String {
    let mut tokens = args.split_whitespace();
    let name = match (tokens.next(), tokens.next()) {
        (Some(name), None) => name,
        _ => return String::from("unset: usage: unset <NAME>"),
    };
    if !is_var_name(name) {
        return format!(
            "unset: '{}' is not a variable name: a letter or '_' then letters, digits or '_'",
            name
        );
    }
    vars.remove(name);
    String::new()
}

/// Every variable as `NAME=value`, one per line in name order, or only those
/// `export` marked.
fn list_vars(vars: &Vars, exported_only: bool) -> String {
    let mut out = String::new();
    for (name, (value, exported)) in vars {
        if exported_only && !exported {
            continue;
        }
        out.push_str(&format!("{}={}\n", name, value));
    }
    if out.is_empty() {
        return String::from("(none)");
    }
    out
}

/// Nested `source` calls a script may reach before the shell refuses.
///
/// A script that sources itself has no other stop: `Shell::cmd_source` runs
/// each line through `Shell::run_line`, which re-enters `dispatch` and so this
/// again, on the kernel stack. Eight levels is deep enough for a script that
/// pulls in a settings file that pulls in another, and shallow enough that the
/// chain of `run_line` → `run_pipeline` → `dispatch` → `run_script` frames
/// cannot run the boot stack out. `expand_vars` bounded the same class of
/// problem one layer down by expanding exactly once.
const MAX_SOURCE_DEPTH: usize = 8;

/// Largest script the shell will run, in bytes and in lines.
///
/// A script is a file from the filesystem and is read whole into the heap
/// before the first line runs, so both are trust-boundary bounds rather than
/// style limits. The byte cap is `PIPE_CAPACITY` because that is already what
/// `< file` may read in — one number for "the largest file this shell takes at
/// once". The line cap is separate because a file of empty lines is cheap in
/// bytes and still walks the whole `run_line` path once per line.
const MAX_SCRIPT_BYTES: usize = PIPE_CAPACITY;
const MAX_SCRIPT_LINES: usize = 1024;

/// `line` with any comment cut off and the surrounding whitespace trimmed.
///
/// A `#` opens a comment only where a word opens — at the start of the line or
/// after whitespace. This shell has no quoting, so the alternative rule, "cut
/// at every `#`", would quietly shorten `A=v#1` to `A=v` with no way at all to
/// ask for the character back. With this rule the value keeps its `#`, and the
/// cost is stated: `grep #tag` is a comment, and the literal is reached the
/// same way any other metacharacter is, through a variable (`H=#`, then
/// `grep $H` + `tag`).
///
/// Free-standing, and applied by `Shell::run_line` to every line from either
/// source, so a comment behaves the same typed at the prompt as read from a
/// file.
fn strip_comment(line: &str) -> &str {
    for (i, c) in line.char_indices() {
        if c == '#' && (i == 0 || line[..i].ends_with(char::is_whitespace)) {
            return line[..i].trim();
        }
    }
    line.trim()
}

/// The lines of a script, or why it will not be run at all.
///
/// Refuses rather than truncates: half a script is not a shorter script, it is
/// a different one, and the operator would have no way to tell which half ran.
///
/// Free-standing so both caps are exercisable without a `Shell`, which mounts
/// — and on a blank disk formats — the AHCI block store.
fn script_lines(text: &str) -> Result<Vec<&str>, String> {
    if text.len() > MAX_SCRIPT_BYTES {
        return Err(format!(
            "seal: source: script is {} bytes, over the {}-byte limit",
            text.len(),
            MAX_SCRIPT_BYTES
        ));
    }
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() > MAX_SCRIPT_LINES {
        return Err(format!(
            "seal: source: script is {} lines, over the {}-line limit",
            lines.len(),
            MAX_SCRIPT_LINES
        ));
    }
    Ok(lines)
}

/// Run every line of `text` through `run`, in order, and collect what they
/// printed.
///
/// `depth` counts the scripts already open, this one included, and past
/// `MAX_SOURCE_DEPTH` nothing runs at all — that is what terminates a script
/// which sources itself.
///
/// A line stops the script when `run` reports `Err`, which is what
/// `Shell::run_line` returns for a line it could not run at all: a syntax
/// error, an expansion that left nothing, a redirection that was refused, an
/// over-capacity pipe. Everything else continues, including a command that ran
/// and printed an error, because this shell has no exit status — every arm of
/// `Shell::dispatch` returns a `String` whether it worked or not, so `peek:
/// 'x' not found` and a file's contents are the same thing here. Giving a
/// script a real failure rule means giving the command table one first, which
/// is a rewrite of ~60 arms rather than a change to script running.
///
/// Every line is handed on, blank and comment lines included, so the line
/// number in the message is the line's number in the file. Skipping them is
/// `Shell::run_line`'s job, in one place, for both a script and the prompt.
///
/// Takes the runner as a closure so the bound, the stop rule and the numbering
/// are testable without a `Shell` and its AHCI mount, the way `run_pipeline`
/// does.
fn run_script<F>(text: &str, depth: usize, mut run: F) -> Result<String, String>
where
    F: FnMut(&str) -> Result<String, String>,
{
    if depth > MAX_SOURCE_DEPTH {
        return Err(format!(
            "seal: source: more than {} nested scripts — a script that sources itself?",
            MAX_SOURCE_DEPTH
        ));
    }
    let lines = script_lines(text)?;
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        let produced = match run(line) {
            Ok(produced) => produced,
            // Whatever the script printed before it stopped goes with the
            // message. There is one string on the way to the terminal, and
            // dropping the earlier lines would leave the operator holding an
            // error with no sight of how far the script got.
            Err(e) => return Err(format!("{}seal: source: line {}: {}", out, i + 1, e)),
        };
        if produced.is_empty() {
            continue;
        }
        out.push_str(&produced);
        if !produced.ends_with('\n') {
            out.push('\n');
        }
    }
    Ok(out)
}

impl Shell {
    pub fn new() -> Self {
        Self {
            fs: ManifoldFS::new(),
            cwd: 0,
            cwd_path: String::from("/"),
            history: Vec::new(),
            clipboard: Clipboard::new(),
            calculator: Calculator::new(),
            media_player: MediaPlayer::new(),
            last_model: None,
            vars: Vars::new(),
            source_depth: 0,
        }
    }

    pub fn execute(&mut self, input: &str) -> String {
        let input = input.trim();
        if input.is_empty() {
            return String::new();
        }

        self.history.push(String::from(input));

        match self.run_line(input) {
            Ok(output) => output,
            Err(e) => e,
        }
    }

    /// Run one command line: an assignment, or a pipeline of one or more
    /// stages with optional `< file` on the first and `> file` / `>> file` on
    /// the last.
    ///
    /// A comment is cut first, before anything else reads the line, so a `#`
    /// can never be a stage separator's neighbour or a redirection's file
    /// name; expansion is last, so a `#` that comes out of a variable is text
    /// and cannot retroactively comment out the rest of the line. A line with
    /// nothing left — blank, or all comment — does nothing, which is what lets
    /// a script hold both.
    ///
    /// `NAME=value` is a whole statement, tested before the line is split, so
    /// an assignment is never a pipeline and its value may hold any character
    /// at all. Everything else is split into stages first and expanded second,
    /// each stage's command and each redirection's file name separately. That
    /// order is what keeps a variable's contents data: a value holding `|`,
    /// `<` or `>` arrives after the only place those are read as operators and
    /// is passed to the command as text. Expanding first would let `V=a|b`
    /// turn `grep $V` into two stages, which is a shell where no value can be
    /// trusted to stay one argument — and with no quoting there would be no
    /// way to ask for the literal.
    ///
    /// Nothing here can observe that a command failed. Every arm of
    /// `Shell::dispatch` returns a `String` whether it succeeded or not, so a
    /// stage's error text is indistinguishable from its output and flows
    /// downstream as stdin like anything else — only the parse errors and the
    /// redirect I/O errors raised here abort the line. Giving the pipeline a
    /// real exit status means turning ~60 dispatch arms into
    /// `Result<String, String>`, which is a rewrite of the command table
    /// rather than a change to composition.
    ///
    /// ponytail: no quoting, so `|`, `<` and `>` are metacharacters
    /// everywhere and cannot appear literally in an argument — `write note a
    /// > b` redirects instead of writing `a > b`. The shell had no quoting
    /// before this change either, and adding it means a real tokenizer in
    /// front of the `splitn(3, ' ')` every command arm is built on. Upgrade
    /// path: tokenize once into `Vec<String>` with `'`/`"` handling, split
    /// the pipeline on unquoted `|`, and hand each arm its argument vector.
    fn run_line(&mut self, input: &str) -> Result<String, String> {
        // Comments go first, before the split and long before expansion: a
        // '#' typed on the line ends it, and a '#' a variable expands to is
        // text the command receives. Blank lines — a script's, and what a
        // comment-only line leaves behind — do nothing at all.
        let input = strip_comment(input);
        if input.is_empty() {
            return Ok(String::new());
        }

        if let Some((name, value)) = parse_assignment(input) {
            let value = expand_vars(value, &self.vars)?;
            set_var(&mut self.vars, name, &value)?;
            return Ok(String::new());
        }

        let stages = parse_line(input)?;
        let last = stages.len() - 1;

        let stdin = match stages[0].input {
            Some(file) => {
                let file = expand_required(file, &self.vars)?;
                self.read_stdin(&file)?
            }
            None => String::new(),
        };
        let redirect = match stages[last].output {
            Some((file, append)) => Some((expand_required(file, &self.vars)?, append)),
            None => None,
        };
        let expanded: Vec<String> = stages
            .iter()
            .map(|s| expand_required(s.cmd, &self.vars))
            .collect::<Result<_, _>>()?;
        let commands: Vec<&str> = expanded.iter().map(|s| s.as_str()).collect();
        let output = run_pipeline(&commands, stdin, |cmd, stdin| self.dispatch(cmd, stdin))?;

        match redirect {
            Some((file, append)) => {
                self.write_redirect(&file, &output, append)?;
                Ok(String::new())
            }
            None => Ok(output),
        }
    }

    /// Contents of the file named by `< file`, refused if the access policy
    /// denies the read, if it does not exist, or if it is larger than one
    /// pipe buffer.
    fn read_stdin(&self, file: &str) -> Result<String, String> {
        if let Some(e) = self.deny(file, Permissions::R) {
            return Err(format!("seal: {}", e));
        }
        let text = self
            .fs
            .read_text(file, self.cwd)
            .ok_or_else(|| format!("seal: '{}' not found", file))?;
        if text.len() > PIPE_CAPACITY {
            return Err(format!(
                "seal: '{}' is {} bytes, over the {}-byte pipe buffer",
                file,
                text.len(),
                PIPE_CAPACITY
            ));
        }
        Ok(text)
    }

    /// Write a pipeline's output to the file named by `>` or `>>`.
    ///
    /// `ManifoldFS::store` refuses a name that already exists
    /// (fs/manifold_fs.rs:364), so both forms rewrite the whole entry: the
    /// full contents are assembled first, and only then is the old entry
    /// removed. Nothing is deleted unless the write is authorized.
    fn write_redirect(&mut self, file: &str, output: &str, append: bool) -> Result<(), String> {
        if let Some(e) = self.deny(file, Permissions::W) {
            return Err(format!("seal: {}", e));
        }
        let existed = self.fs.exists(file, self.cwd);
        let existing = if existed {
            self.fs.read_text(file, self.cwd)
        } else {
            None
        };
        let content = redirected_content(existing, output, append);
        if existed {
            self.fs
                .delete(file, self.cwd)
                .map_err(|e| format!("seal: {}: {}", file, e))?;
        }
        self.fs
            .store_text(file, &content, self.cwd)
            .map(|_| ())
            .map_err(|e| format!("seal: {}: {}", file, e))
    }

    /// Run a single command with `stdin` as its standard input.
    fn dispatch(&mut self, input: &str, stdin: &str) -> String {
        let parts: Vec<&str> = input.splitn(3, ' ').collect();
        let cmd = parts[0];
        let arg1 = parts.get(1).copied().unwrap_or("");
        let arg2 = parts.get(2).copied().unwrap_or("");

        match cmd {
            "help" => {
                if arg1.is_empty() {
                    help::handbook()
                } else {
                    help::help_for(arg1)
                }
            }

            "look" => self.cmd_look(arg1),
            "open" => self.cmd_open(arg1),
            "back" => self.cmd_open(".."),
            "home" => self.cmd_open("/"),
            "peek" => self.cmd_peek(arg1),
            "create" => self.cmd_create(arg1),
            "write" => self.cmd_write(arg1, arg2),
            "delete" => self.cmd_delete(arg1),
            "rename" => self.cmd_rename(arg1, arg2),
            "copy" => self.cmd_copy(arg1),
            "paste" => self.cmd_paste(),
            "move" => self.cmd_move(arg1, arg2),
            "search" => {
                let query = if arg2.is_empty() { arg1 } else {
                    input.split_once(' ').map(|x| x.1).unwrap_or(arg1)
                };
                self.cmd_search(query)
            }
            // `cat` with no argument is the stdin filter, below.
            "cat" if !arg1.is_empty() => self.cmd_peek(arg1),
            "info" => self.cmd_info(arg1),
            "seal" => self.cmd_seal(),
            "tasks" => self.cmd_tasks(),
            "stats" => self.cmd_stats(),
            "gpu_benchmark" => {
                crate::apps::gpu_benchmark::main();
                String::from("[gpu_benchmark] Done. See serial log for results.")
            }
            "race" => self.cmd_race(arg1),
            "history" => self.cmd_history(),
            "clear" => String::from("\x1b[clear]"),

            // Package manager
            "install" => self.cmd_install(arg1),
            "remove" => self.cmd_remove(arg1),
            "packages" => self.cmd_packages(),
            "update" => self.cmd_update(),

            // Network
            "wifi" => self.cmd_wifi(arg1, arg2),
            "bluetooth" => self.cmd_bluetooth(arg1, arg2),

            // Scripting
            "run" => self.cmd_run(arg1),
            "source" | "." => self.cmd_source(arg1),
            "aether" => String::from("[Aether-Lang] REPL mode — type expressions terminated with ~\nType 'exit~' to return to SealShell."),

            // Games
            "snake" => String::from("[Snake] Starting Snake game... Use arrow keys to move!"),
            "breakout" => String::from("[Breakout] Starting Breakout... Arrow keys to move paddle!"),
            "warp" => String::from("[Warp Racer] Starting Warp Racer... Fly through the data stream!"),

            // System
            "whoami" => {
                match crate::security::passwd::get_current_user() {
                    Some(user) => user.username,
                    None => String::from("unknown"),
                }
            }
            "memory" => self.cmd_memory(),

            // ML
            "ml" => self.cmd_ml(arg1, arg2),

            // Variables. `NAME=value` itself is a statement, handled by
            // `Shell::run_line` before the line is ever split into stages.
            "export" => cmd_export(&mut self.vars, input.split_once(' ').map_or("", |x| x.1)),
            "unset" => cmd_unset(&mut self.vars, input.split_once(' ').map_or("", |x| x.1)),
            "env" => list_vars(&self.vars, true),

            // Settings
            "theme" => self.cmd_theme(arg1),
            // A bare `set` lists the variables; with a key it is a setting.
            "set" if arg1.is_empty() => list_vars(&self.vars, false),
            "set" => self.cmd_set(arg1, arg2),

            // Prefetch
            "prefetch" => self.cmd_prefetch(arg1),

            // TopCrypt / Lypnos Guard
            "topcrypt" => {
                let rest = input.strip_prefix("topcrypt ").unwrap_or("");
                self.cmd_topcrypt(rest)
            }

            // Tensor visualization
            "tensor" => self.cmd_tensor(arg1, arg2),
            "audit" => crate::security::audit_runtime::audit_probe(arg1),

            // Calculator
            "calc" => {
                let expr = if arg2.is_empty() { arg1 } else {
                    input.split_once(' ').map(|x| x.1).unwrap_or(arg1)
                };
                self.calculator.evaluate(expr)
            }

            // Media player
            "play" => self.cmd_play(arg1, arg2),
            "pause" => self.media_player.pause(),
            "stop" => self.media_player.stop(),
            "formats" => MediaPlayer::supported_formats(),

            // The stdin filters: `grep`, `wc`, `head`, `tail`, `sort`,
            // `uniq`, `tr`, a bare `cat`, and `echo`, which reads no stdin.
            // They need no `Shell`, so they live in `run_filter` where a test
            // can reach them.
            _ => match run_filter(input, stdin) {
                Some(output) => output,
                None => format!("seal: unknown command '{}' — type 'help' for the handbook", cmd),
            },
        }
    }

    /// Absolute path named by `arg`, relative to `self.cwd_path`.
    fn abs_path(&self, arg: &str) -> Option<String> {
        abs_path_in(&self.cwd_path, arg)
    }

    /// Message to return when the current task may not touch `arg` with
    /// `perms`; `None` when the operation is authorized.
    ///
    /// Fails closed: an argument that cannot be resolved to a canonical path
    /// is refused rather than passed through to the filesystem.
    fn deny(&self, arg: &str, perms: Permissions) -> Option<String> {
        let path = match self.abs_path(arg) {
            Some(p) => p,
            None => {
                return Some(format!(
                    "'{}': path must be a plain name or absolute, with no empty, '.' or '..' components",
                    arg
                ))
            }
        };
        if crate::security::mac::check_file_permission(
            crate::process::scheduler::current_uid(),
            &path,
            perms,
        ) {
            None
        } else {
            Some(format!("permission denied: {}", path))
        }
    }

    fn cmd_look(&self, arg: &str) -> String {
        if let Some(e) = self.deny(arg, Permissions::R) {
            return format!("look: {}", e);
        }
        let dir = if arg.is_empty() {
            self.cwd
        } else {
            match self.fs.resolve_path(arg) {
                Ok(id) => id,
                Err(e) => return format!("look: {}", e),
            }
        };
        match self.fs.ls(dir) {
            Ok(entries) => {
                if entries.is_empty() {
                    return String::from("(empty)");
                }
                let mut out = String::new();
                for e in &entries {
                    let icon = if e.kind == "dir" { "📁" } else { "📄" };
                    out.push_str(&format!(
                        " {} {:<20} {:>8} pts={} cell={}\n",
                        icon, e.name, e.original_size, e.payload_points, e.voronoi_cell
                    ));
                }
                out
            }
            Err(e) => format!("look: {}", e),
        }
    }

    fn cmd_open(&mut self, arg: &str) -> String {
        if arg.is_empty() || arg == "/" {
            self.cwd = 0;
            self.cwd_path = String::from("/");
            return String::new();
        }
        if arg == ".." {
            if let Some(inode) = self.fs.inode(self.cwd) {
                self.cwd = inode.parent;
                self.update_cwd_path();
            }
            return String::new();
        }
        if let Some(e) = self.deny(arg, Permissions::R) {
            return format!("open: {}", e);
        }
        match self.fs.resolve_path_from(arg, self.cwd) {
            Ok(id) => {
                if !self.fs.is_dir(id) {
                    return format!("open: '{}' is not a folder", arg);
                }
                self.cwd = id;
                self.update_cwd_path();
                String::new()
            }
            Err(e) => format!("open: {}", e),
        }
    }

    fn cmd_peek(&self, arg: &str) -> String {
        if arg.is_empty() {
            return String::from("peek: which file? Usage: peek <filename>");
        }
        if let Some(e) = self.deny(arg, Permissions::R) {
            return format!("peek: {}", e);
        }
        match self.fs.read_text(arg, self.cwd) {
            Some(text) => text,
            None => format!("peek: '{}' not found", arg),
        }
    }

    fn cmd_create(&mut self, arg: &str) -> String {
        if arg.is_empty() {
            return String::from("create: what name? Usage: create <foldername>");
        }
        if let Some(e) = self.deny(arg, Permissions::W) {
            return format!("create: {}", e);
        }
        match self.fs.mkdir(arg, self.cwd) {
            Ok(id) => format!("Created folder '{}' (inode {})", arg, id),
            Err(e) => format!("create: {}", e),
        }
    }

    fn cmd_write(&mut self, name: &str, content: &str) -> String {
        if name.is_empty() {
            return String::from("write: usage: write <filename> <content>");
        }
        if let Some(e) = self.deny(name, Permissions::W) {
            return format!("write: {}", e);
        }
        let content = if content.is_empty() { "" } else { content };
        match self.fs.store_text(name, content, self.cwd) {
            Ok(id) => format!(
                "Wrote '{}' ({} bytes → 64 pts on S², inode {})",
                name,
                content.len(),
                id
            ),
            Err(e) => format!("write: {}", e),
        }
    }

    fn cmd_delete(&mut self, arg: &str) -> String {
        if arg.is_empty() {
            return String::from("delete: which file? Usage: delete <filename>");
        }
        if let Some(e) = self.deny(arg, Permissions::W) {
            return format!("delete: {}", e);
        }
        match self.fs.delete(arg, self.cwd) {
            Ok(()) => format!("Deleted '{}'", arg),
            Err(e) => format!("delete: {}", e),
        }
    }

    fn cmd_rename(&mut self, old: &str, new: &str) -> String {
        if old.is_empty() || new.is_empty() {
            return String::from("rename: usage: rename <old> <new>");
        }
        for target in [old, new] {
            if let Some(e) = self.deny(target, Permissions::W) {
                return format!("rename: {}", e);
            }
        }
        match self.fs.rename(old, new, self.cwd) {
            Ok(()) => format!("Renamed '{}' → '{}'", old, new),
            Err(e) => format!("rename: {}", e),
        }
    }

    fn cmd_copy(&mut self, arg: &str) -> String {
        if arg.is_empty() {
            return String::from("copy: which file? Usage: copy <filename>");
        }
        if let Some(e) = self.deny(arg, Permissions::R) {
            return format!("copy: {}", e);
        }
        if !self.fs.exists(arg, self.cwd) {
            return format!("copy: '{}' not found", arg);
        }
        self.clipboard.copy(arg, self.cwd);
        format!("Copied '{}' to clipboard", arg)
    }

    fn cmd_paste(&mut self) -> String {
        let entry = match self.clipboard.take() {
            Some(e) => e,
            None => return String::from("paste: clipboard is empty — use 'copy <file>' first"),
        };
        // The source was authorized for read when `copy` put it on the
        // clipboard; this is the write half, at the destination.
        if let Some(e) = self.deny(&entry.name, Permissions::W) {
            return format!("paste: {}", e);
        }
        match self.fs.duplicate(&entry.name, entry.source_dir, self.cwd) {
            Ok(id) => format!("Pasted '{}' here (inode {})", entry.name, id),
            Err(e) => format!("paste: {}", e),
        }
    }

    fn cmd_move(&mut self, name: &str, dest: &str) -> String {
        if name.is_empty() || dest.is_empty() {
            return String::from("move: usage: move <file> <destination_folder>");
        }
        // Source removal, destination directory, and the entry created inside
        // it are three separate writes; a rule may cover any one of them.
        let dest_child = format!("{}/{}", dest.trim_end_matches('/'), name);
        for target in [name, dest, dest_child.as_str()] {
            if let Some(e) = self.deny(target, Permissions::W) {
                return format!("move: {}", e);
            }
        }
        let dst = match self.fs.resolve_path(dest) {
            Ok(id) => id,
            Err(e) => return format!("move: destination: {}", e),
        };
        match self.fs.teleport(name, self.cwd, dst) {
            Ok(r) => format!(
                "Teleported '{}' ({} bytes) in {} ticks — O(1)!\n\
                 Payload: {} points, governor ε: {:.4}",
                name, r.original_size, r.elapsed_ticks, r.payload_points, r.governor_epsilon
            ),
            Err(e) => format!("move: {}", e),
        }
    }

    fn cmd_search(&mut self, query: &str) -> String {
        if query.is_empty() {
            return String::from("search: what? Usage: search <query>");
        }
        // ponytail: `search` is not filtered by the access policy. It names no
        // path to check — `ManifoldFS::find` (fs/manifold_fs.rs:531) scores
        // every inode by payload similarity and returns `FindResult { name,
        // similarity, cell, original_size }` with no parent and no path, so
        // there is nothing to hand `check_file_permission`. Ceiling: a name
        // and a byte count from a directory the caller may not read can still
        // leak. Upgrade path: carry the resolved path on `FindResult` and
        // filter here — that field lives in fs/manifold_fs.rs, outside this
        // unit's scope.
        let results = self.fs.find(query);
        if results.is_empty() {
            return String::from("No matches found.");
        }
        let mut out = String::new();
        for r in &results {
            out.push_str(&format!(
                "  {} (similarity={:.3}, cell={}, {} bytes)\n",
                r.name, r.similarity, r.cell, r.original_size
            ));
        }
        out
    }

    fn cmd_info(&self, arg: &str) -> String {
        if arg.is_empty() {
            return String::from("info: which file? Usage: info <filename>");
        }
        if let Some(e) = self.deny(arg, Permissions::R) {
            return format!("info: {}", e);
        }
        match self.fs.file_info(arg, self.cwd) {
            Some(info) => info,
            None => format!("info: '{}' not found", arg),
        }
    }

    fn cmd_seal(&self) -> String {
        let mut out = format!(
            "Seal OS v{} (x86_64)\n\
             The Geometrical Operating System\n\
             All data = geometry on S²\n\
             File moves = metadata topology; persistent bytes still scale\n\
             ═══════════════════════════════════\n",
            crate::VERSION
        );
        for (name, status) in self.fs.theorem_status() {
            out.push_str(&format!("  {} = {}\n", name, status));
        }
        let stats = self.fs.stats();
        out.push_str(&format!("\n  Governor ε: {:.4}\n", stats.governor_epsilon));
        out.push_str(&format!("  Entropy: {:.3} bits\n", stats.current_entropy));
        out.push_str(&format!(
            "  Hyperbolic ratio: {:.1}x (depth={})\n",
            stats.hyperbolic_ratio, stats.max_depth
        ));
        out
    }

    fn cmd_tasks(&self) -> String {
        let tasks = crate::process::scheduler::list_all_tasks();
        let mut out = String::from("  PID  STATE    NAME             PRI  CELL\n");
        if tasks.is_empty() {
            out.push_str("  (no tasks scheduled)\n");
        } else {
            for (id, name, state, prio, cell) in &tasks {
                out.push_str(&format!(
                    "  {:<4} {:<8} {:<16} {:<4} {}\n",
                    id, state, name, prio, cell
                ));
            }
        }
        out.push_str(&format!(
            "  Total: {} tasks",
            crate::process::scheduler::task_count()
        ));
        out
    }

    fn cmd_stats(&self) -> String {
        let s = self.fs.stats();
        format!(
            "ManifoldFS Statistics\n\
             ═════════════════════\n\
             Files:           {}\n\
             Directories:     {}\n\
             Teleports:       {}\n\
             Governor ε:      {:.4}\n\
             Entropy:         {:.3} bits\n\
             Max depth:       {}\n\
             Hyperbolic ratio: {:.1}x",
            s.total_files,
            s.total_dirs,
            s.total_teleports,
            s.governor_epsilon,
            s.current_entropy,
            s.max_depth,
            s.hyperbolic_ratio
        )
    }

    fn cmd_memory(&self) -> String {
        let (allocated, total) = crate::memory::heap_stats();
        let allocated_mb = allocated / (1024 * 1024);
        let total_mb = total / (1024 * 1024);
        format!(
            "Memory Status\n\
             ═════════════\n\
             Allocated: {} bytes ({} MB)\n\
             Total:     {} bytes ({} MB)\n\
             Free:      {} bytes ({} MB)",
            allocated,
            allocated_mb,
            total,
            total_mb,
            total.saturating_sub(allocated),
            total_mb.saturating_sub(allocated_mb)
        )
    }

    fn cmd_ml(&mut self, subcmd: &str, arg: &str) -> String {
        match subcmd {
            "status" => {
                let status = crate::ml_engine::MlStatus::detect();
                let gpu_line = match &status.gpu_detected {
                    Some((name, _, _)) => format!("GPU:          {} (detected)", name),
                    None => String::from("GPU:          not detected (PCI probe done)"),
                };
                format!(
                    "ML Runtime Status\n\
                     ══════════════════\n\
                     AEGIS Engine: aether-core (no_std)\n\
                     Tensor ops:   {}\n\
                     Neural nets:  {}\n\
                     AVX2:         {}\n\
                     AVX-512:      {}\n\
                     {}\n\
                     Use: ml train [epochs] | ml matmul | ml tensor",
                    if status.tensor_ops_available {
                        "available"
                    } else {
                        "unavailable"
                    },
                    if status.neural_net_available {
                        "available"
                    } else {
                        "unavailable"
                    },
                    if status.avx2_detected {
                        "detected"
                    } else {
                        "not detected"
                    },
                    if status.avx512_detected {
                        "detected"
                    } else {
                        "not detected"
                    },
                    gpu_line,
                )
            }
            "devices" => String::from(
                "ML Compute Devices\n\
                     ═══════════════════\n\
                     CPU:  x86_64 (AVX2/AVX-512 detection pending)\n\
                     GPU:  Not detected (PCI probe pending)\n\
                     NPU:  Not detected",
            ),
            "train" => {
                let epochs: usize = arg.parse().unwrap_or(1000);
                let (report, bytes) = crate::ml_engine::demo_train_mlp(epochs);
                self.last_model = Some(bytes);
                report
            }
            "save" => {
                let name = if arg.is_empty() { "model.sealml" } else { arg };
                match &self.last_model {
                    Some(bytes) => match crate::ml_engine::save_model_bytes(name, bytes) {
                        Ok(msg) => msg,
                        Err(e) => format!("Error: {}", e),
                    },
                    None => String::from("No model to save. Run 'ml train' first."),
                }
            }
            "load" => {
                let name = if arg.is_empty() { "model.sealml" } else { arg };
                match crate::ml_engine::load_model(name) {
                    Ok(_mlp) => format!("Model '{}' loaded successfully", name),
                    Err(e) => format!("Error: {}", e),
                }
            }
            "generate" => {
                let seed = if arg.is_empty() { "Seal" } else { arg };
                let text = crate::ml_engine::demo_generate_text(seed, 120);
                format!(
                    "Markov Generation (seed: '{}')\n\
                         ══════════════════════════════════\n\
                         {}",
                    seed, text
                )
            }
            "tensor" => {
                // Create a simple 2x3 tensor for demo
                let t = crate::ml_engine::tensor_from_data(
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    vec![2, 3],
                );
                match t {
                    Ok(tensor) => crate::ml_engine::format_tensor(&tensor),
                    Err(e) => format!("Tensor error: {}", e),
                }
            }
            "matmul" => {
                let a = match crate::ml_engine::tensor_from_data(
                    vec![1.0, 2.0, 3.0, 4.0],
                    vec![2, 2],
                ) {
                    Ok(t) => t,
                    Err(e) => return format!("Tensor A error: {}", e),
                };
                let b = match crate::ml_engine::tensor_from_data(
                    vec![5.0, 6.0, 7.0, 8.0],
                    vec![2, 2],
                ) {
                    Ok(t) => t,
                    Err(e) => return format!("Tensor B error: {}", e),
                };
                match crate::ml_engine::tensor_matmul(&a, &b) {
                    Ok(result) => crate::ml_engine::format_tensor(&result),
                    Err(e) => format!("Matmul error: {}", e),
                }
            }
            _ => String::from(
                "ML Subcommands:\n\
                 ml status    — Show ML runtime status\n\
                 ml devices   — List compute devices\n\
                 ml train     — Train MLP on XOR dataset\n\
                 ml save      — Save trained model to ManifoldFS\n\
                 ml load      — Load model from ManifoldFS\n\
                 ml generate  — Generate text with Markov chain\n\
                 ml tensor    — Create a demo tensor\n\
                 ml matmul    — Demo matrix multiply",
            ),
        }
    }

    fn cmd_race(&self, arg: &str) -> String {
        let size: usize = arg.parse().unwrap_or(1_000_000);
        let label = if size >= 1_000_000_000 {
            format!("{}GB", size / 1_000_000_000)
        } else if size >= 1_000_000 {
            format!("{}MB", size / 1_000_000)
        } else if size >= 1_000 {
            format!("{}KB", size / 1_000)
        } else {
            format!("{}B", size)
        };

        // Real benchmark: create file, measure duplicate vs teleport.
        // `ManifoldFS::new` mounts the same AHCI block store the rest of the
        // system is on (fs/manifold_fs.rs:112), so these two directories land
        // at the filesystem root for real and need authorizing like any other
        // write.
        for target in ["/race_src", "/race_dst"] {
            if let Some(e) = self.deny(target, Permissions::W) {
                return format!("race: {}", e);
            }
        }
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let src_dir = fs.mkdir("race_src", root).unwrap_or(1);
        let dst_dir = fs.mkdir("race_dst", root).unwrap_or(2);

        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let _file = fs.store("race_file", &data, src_dir);

        // Warm-up
        let _ = fs.duplicate("race_file", src_dir, dst_dir);
        let _ = fs.teleport("race_file", dst_dir, src_dir);

        // Benchmark duplicate (traditional copy)
        let dup_start = crate::drivers::interrupts::ticks();
        let _ = fs.duplicate("race_file", src_dir, dst_dir);
        let dup_ticks = crate::drivers::interrupts::ticks().saturating_sub(dup_start);

        // Benchmark metadata teleport separately from byte persistence.
        let tel_start = crate::drivers::interrupts::ticks();
        let _ = fs.teleport("race_file", dst_dir, src_dir);
        let tel_ticks = crate::drivers::interrupts::ticks().saturating_sub(tel_start);

        let speedup = if tel_ticks > 0 {
            dup_ticks as f64 / tel_ticks as f64
        } else {
            9999.0
        };

        format!(
            "Race: {} payload\n\
             ════════════════════════\n\
             Traditional duplicate:  {} ticks\n\
             Manifold metadata move: {} ticks (topological surgery)\n\
             Speedup:                {:.1}x\n\
             \n\
             The teleport rewrites one BTreeMap entry.\n\
             The duplicate clones geometry metadata + raw bytes.",
            label, dup_ticks, tel_ticks, speedup
        )
    }

    fn cmd_history(&self) -> String {
        let mut out = String::new();
        for (i, cmd) in self.history.iter().enumerate() {
            out.push_str(&format!("  {:>4}  {}\n", i + 1, cmd));
        }
        if out.is_empty() {
            String::from("(no history)")
        } else {
            out
        }
    }

    fn cmd_install(&mut self, pkg: &str) -> String {
        if pkg.is_empty() {
            return String::from("install: which package? Usage: install <package>");
        }

        if pkg.ends_with(".eph") {
            if let Some(e) = self.deny(pkg, Permissions::R) {
                return format!("[ManifoldPkg] local install: {}", e);
            }
            let Some(data) = self.fs.read_bytes(pkg, self.cwd) else {
                return format!("[ManifoldPkg] local package '{}' not found", pkg);
            };
            return match crate::pkg::GLOBAL_PKG.lock().install_bytes(&data, None) {
                Ok(msg) => format!("[ManifoldPkg] {}", msg),
                Err(e) => format!("[ManifoldPkg] local install: {}", e),
            };
        }

        match crate::pkg::GLOBAL_PKG.lock().install(pkg) {
            Ok(msg) => format!("[ManifoldPkg] {}", msg),
            Err(e) => format!("[ManifoldPkg] install: {}", e),
        }
    }

    fn cmd_remove(&mut self, pkg: &str) -> String {
        if pkg.is_empty() {
            return String::from("remove: which package? Usage: remove <package>");
        }
        match crate::pkg::GLOBAL_PKG.lock().remove(pkg) {
            Ok(msg) => format!("[ManifoldPkg] {}", msg),
            Err(e) => format!("[ManifoldPkg] remove: {}", e),
        }
    }

    fn cmd_packages(&self) -> String {
        let pkg = crate::pkg::GLOBAL_PKG.lock();
        let packages = pkg.list();
        if packages.is_empty() {
            return String::from("[ManifoldPkg] No packages installed");
        }
        let mut out = String::from("[ManifoldPkg] Installed packages:\n");
        for manifest in packages {
            out.push_str(&format!(
                "  {} v{} [{}]\n",
                manifest.name,
                manifest.version,
                manifest.carrier.name()
            ));
        }
        out
    }

    fn cmd_update(&self) -> String {
        if crate::net::has_nic() {
            String::from(
                "ManifoldPkg: signed registry-index fixture is boot-gated; public refresh needs hosted release channel",
            )
        } else {
            String::from(
                "ManifoldPkg: no network device; use install <local.eph> for local packages",
            )
        }
    }

    fn cmd_wifi(&self, sub: &str, arg: &str) -> String {
        let mut driver = WifiDriver::new();
        match sub {
            "connect" => {
                if arg.is_empty() {
                    String::from("wifi connect: usage: wifi connect <ssid> [password]")
                } else {
                    match driver.connect(arg, "") {
                        Ok(()) => format!("WiFi: connected to '{}'", arg),
                        Err(e) => e,
                    }
                }
            }
            "disconnect" => {
                driver.disconnect();
                String::from("WiFi: disconnected")
            }
            "scan" => {
                let networks = driver.scan();
                if networks.is_empty() {
                    return String::from("WiFi: no wireless networks found");
                }
                let mut out =
                    String::from("WiFi Networks:\n  SSID              SIGNAL  SECURITY\n");
                for n in &networks {
                    out.push_str(&format!(
                        "  {:<18}  {:>4}dBm  {}\n",
                        n.ssid,
                        n.signal_dbm,
                        n.security.name()
                    ));
                }
                out
            }
            _ => String::from(
                "WiFi Status: no hardware detected\n\
                 Commands: wifi scan, wifi connect <ssid>, wifi disconnect",
            ),
        }
    }

    fn cmd_bluetooth(&self, sub: &str, arg: &str) -> String {
        let mut driver = BluetoothDriver::new();
        match sub {
            "scan" => {
                let devices = driver.scan();
                if devices.is_empty() {
                    return String::from("Bluetooth: no devices found");
                }
                let mut out =
                    String::from("Bluetooth Devices:\n  NAME                 TYPE   RSSI\n");
                for d in &devices {
                    out.push_str(&format!(
                        "  {:<20}  {:<6}  {:>4}dBm\n",
                        d.name,
                        d.device_type.name(),
                        d.rssi_dbm
                    ));
                }
                out
            }
            "pair" => {
                if arg.is_empty() {
                    String::from("bluetooth pair: usage: bluetooth pair <device>")
                } else {
                    match driver.pair(arg) {
                        Ok(()) => format!("Bluetooth: paired with '{}'", arg),
                        Err(e) => e,
                    }
                }
            }
            _ => String::from(
                "Bluetooth Status: no adapter detected\n\
                 Commands: bluetooth scan, bluetooth pair <device>",
            ),
        }
    }

    fn cmd_run(&self, file: &str) -> String {
        if file.is_empty() {
            return String::from("run: which script? Usage: run <file.aether>");
        }
        let mut runtime = AetherRuntime::new();
        match runtime.execute_file(file, "") {
            Ok(output) => output,
            Err(e) => format!("[Aether-Lang] {}: {}", file, e),
        }
    }

    /// `source <file>` — run a file of shell lines in this shell.
    ///
    /// In this shell, not a new one: each line goes through `Shell::run_line`,
    /// the same path a typed line takes, so a script's assignments, exports
    /// and working directory are still there when it ends. That is the whole
    /// point of `source` and the reason it is not `run`, which hands the file
    /// to a separate Aether-Lang runtime.
    ///
    /// Reading the file needs the filesystem and so needs `&mut self`; the
    /// bounds, the stop rule and the depth check do not, and live in
    /// `script_lines` and `run_script` where a test without an AHCI mount can
    /// reach them.
    ///
    /// ponytail: a failing script is reported to whatever sourced it as
    /// ordinary output, because `dispatch` returns a `String` — so a script
    /// that sources a script that fails carries on. Upgrade path is the same
    /// one the `Shell::run_line` note names: an exit status for the command
    /// table, at which point this returns it.
    fn cmd_source(&mut self, file: &str) -> String {
        if file.is_empty() {
            return String::from("source: which file? Usage: source <file>");
        }
        if let Some(e) = self.deny(file, Permissions::R) {
            return format!("source: {}", e);
        }
        let text = match self.fs.read_text(file, self.cwd) {
            Some(t) => t,
            None => return format!("source: '{}' not found", file),
        };
        self.source_depth += 1;
        let depth = self.source_depth;
        let result = run_script(&text, depth, |line| self.run_line(line));
        self.source_depth -= 1;
        match result {
            Ok(out) => out,
            Err(e) => e,
        }
    }

    fn cmd_theme(&self, name: &str) -> String {
        if name.is_empty() {
            return String::from("theme: which theme? Available: dark, light, seal, matrix");
        }
        match name {
            "dark" | "light" | "seal" | "matrix" => {
                format!(
                    "[Theme] Switched to '{}' theme (visual change requires re-render)",
                    name
                )
            }
            _ => format!(
                "[Theme] Unknown theme '{}'. Available: dark, light, seal, matrix",
                name
            ),
        }
    }

    fn cmd_set(&mut self, key: &str, val: &str) -> String {
        if key.is_empty() {
            return String::from("set: usage: set <key> <value>");
        }
        // Settings persist at the filesystem root (`store_text(.., 0)`), not
        // in the working directory.
        if let Some(e) = self.deny(&format!("/.setting_{}", key), Permissions::W) {
            return format!("[Settings] {}", e);
        }
        let entry = format!("{}={}", key, val);
        match self.fs.store_text(&format!(".setting_{}", key), &entry, 0) {
            Ok(_) => format!("[Settings] {} = {} (persisted to ManifoldFS)", key, val),
            Err(e) => format!("[Settings] {} = {} (memory only: {})", key, val, e),
        }
    }

    fn cmd_play(&mut self, sub: &str, arg: &str) -> String {
        match sub {
            "" => self.media_player.play(),
            "status" => self.media_player.status(),
            "pause" => self.media_player.pause(),
            "stop" => self.media_player.stop(),
            "next" => self.media_player.next_track(),
            "prev" => self.media_player.prev_track(),
            "list" | "playlist" => self.media_player.show_playlist(),
            "formats" => MediaPlayer::supported_formats(),
            "add" => {
                if arg.is_empty() {
                    String::from("play add: usage: play add <file>")
                } else {
                    self.media_player.add_to_playlist(arg);
                    format!("[SealPlayer] Added '{}' to playlist", arg)
                }
            }
            "vol" | "volume" => {
                let vol: u8 = arg.parse().unwrap_or(80);
                self.media_player.volume_set(vol)
            }
            "seek" => {
                let secs: u64 = arg.parse().unwrap_or(0);
                self.media_player.seek(secs)
            }
            file => match self.media_player.open(file) {
                Ok(s) => s,
                Err(e) => e,
            },
        }
    }

    fn cmd_topcrypt(&mut self, rest: &str) -> String {
        let parts: Vec<&str> = rest.splitn(3, ' ').collect();
        let sub = parts.first().copied().unwrap_or("");
        let arg1 = parts.get(1).copied().unwrap_or("");
        let arg2 = parts.get(2).copied().unwrap_or("");

        match sub {
            "encode" => {
                if arg1.is_empty() {
                    return String::from("topcrypt encode: usage: topcrypt encode <file>");
                }
                let data = match self.read_file_bytes(arg1) {
                    Ok(d) => d,
                    Err(e) => return format!("topcrypt encode: {}", e),
                };
                let topo = crate::fs::topcrypt::encode_bytes(&data, 0x5EA1);
                let serialized = crate::fs::topcrypt::export_to_bytes(&topo);
                let topo_name = format!("{}.topo", arg1);
                if let Some(e) = self.deny(&topo_name, Permissions::W) {
                    return format!("topcrypt encode: {}", e);
                }
                if self.fs.exists(&topo_name, self.cwd) {
                    let _ = self.fs.delete(&topo_name, self.cwd);
                }
                match self.fs.store(&topo_name, &serialized, self.cwd) {
                    Ok(id) => format!(
                        "[TopCrypt] Encoded {} bytes → {} blocks on S² (inode {})",
                        data.len(),
                        topo.block_count,
                        id
                    ),
                    Err(e) => format!("topcrypt encode: {}", e),
                }
            }
            "decode" => {
                if arg1.is_empty() {
                    return String::from("topcrypt decode: usage: topcrypt decode <file.topo>");
                }
                let data = match self.read_file_bytes(arg1) {
                    Ok(d) => d,
                    Err(e) => return format!("topcrypt decode: {}", e),
                };
                let topo = crate::fs::topcrypt::import_from_bytes(&data, 0);
                let decoded = crate::fs::topcrypt::decode_bytes(&topo);
                let out_name = arg1.strip_suffix(".topo").unwrap_or(arg1);
                if let Some(e) = self.deny(out_name, Permissions::W) {
                    return format!("topcrypt decode: {}", e);
                }
                if self.fs.exists(out_name, self.cwd) {
                    let _ = self.fs.delete(out_name, self.cwd);
                }
                match self.fs.store(out_name, &decoded, self.cwd) {
                    Ok(id) => format!(
                        "[TopCrypt] Decoded {} blocks → {} bytes (inode {})",
                        topo.block_count,
                        decoded.len(),
                        id
                    ),
                    Err(e) => format!("topcrypt decode: {}", e),
                }
            }
            "lock" => {
                if arg1.is_empty() {
                    return String::from("topcrypt lock: usage: topcrypt lock <file.topo>");
                }
                let data = match self.read_file_bytes(arg1) {
                    Ok(d) => d,
                    Err(e) => return format!("topcrypt lock: {}", e),
                };
                if let Some(e) = self.deny(arg1, Permissions::W) {
                    return format!("topcrypt lock: {}", e);
                }
                let mut topo = crate::fs::topcrypt::import_from_bytes(&data, 0);
                let key = crate::security::topcrypt_guard::LYPNOS_KEY;
                crate::fs::topcrypt::lock_file(&mut topo, key);
                let serialized = crate::fs::topcrypt::export_to_bytes(&topo);
                if self.fs.exists(arg1, self.cwd) {
                    let _ = self.fs.delete(arg1, self.cwd);
                }
                match self.fs.store(arg1, &serialized, self.cwd) {
                    Ok(_) => format!(
                        "[Lypnos Guard] File locked. Entropy = {:.3} bits. Sweet dreams.",
                        topo.lock_entropy
                    ),
                    Err(e) => format!("topcrypt lock: {}", e),
                }
            }
            "unlock" => {
                if arg1.is_empty() {
                    return String::from("topcrypt unlock: usage: topcrypt unlock <file.topo>");
                }
                let data = match self.read_file_bytes(arg1) {
                    Ok(d) => d,
                    Err(e) => return format!("topcrypt unlock: {}", e),
                };
                if let Some(e) = self.deny(arg1, Permissions::W) {
                    return format!("topcrypt unlock: {}", e);
                }
                let mut topo = crate::fs::topcrypt::import_from_bytes(&data, 0);
                let key = crate::security::topcrypt_guard::LYPNOS_KEY;
                if !crate::fs::topcrypt::unlock_file(&mut topo, key) {
                    return format!(
                        "topcrypt unlock: incorrect key or corrupted file '{}'",
                        arg1
                    );
                }
                let serialized = crate::fs::topcrypt::export_to_bytes(&topo);
                if self.fs.exists(arg1, self.cwd) {
                    let _ = self.fs.delete(arg1, self.cwd);
                }
                match self.fs.store(arg1, &serialized, self.cwd) {
                    Ok(_) => String::from("[Lypnos Guard] File awakened. Welcome back."),
                    Err(e) => format!("topcrypt unlock: {}", e),
                }
            }
            "info" => {
                if arg1.is_empty() {
                    return String::from("topcrypt info: usage: topcrypt info <file.topo>");
                }
                let data = match self.read_file_bytes(arg1) {
                    Ok(d) => d,
                    Err(e) => return format!("topcrypt info: {}", e),
                };
                let topo = crate::fs::topcrypt::import_from_bytes(&data, 0);
                format!(
                    "Name: {}
Blocks: {}
Locked: {}
Entropy: {:.3} bits
Seed: {:016x}",
                    arg1,
                    topo.block_count,
                    if topo.locked { "yes" } else { "no" },
                    topo.lock_entropy,
                    topo.embedding_seed
                )
            }
            "render" => {
                if arg1.is_empty() {
                    return String::from("topcrypt render: usage: topcrypt render <file.csv>");
                }
                if let Some(e) = self.deny(arg1, Permissions::R) {
                    return format!("topcrypt render: {}", e);
                }
                let text = match self.fs.read_text(arg1, self.cwd) {
                    Some(t) => t,
                    None => return format!("topcrypt render: '{}' not found", arg1),
                };
                let tensor = crate::ml_engine::tensor_viz::parse_csv(&text);
                let data = &tensor.data;
                let len = data.len();
                let mut peaks = 0;
                let mut valleys = 0;
                for w in data.windows(3) {
                    if w[1] > w[0] && w[1] > w[2] {
                        peaks += 1;
                    }
                    if w[1] < w[0] && w[1] < w[2] {
                        valleys += 1;
                    }
                }
                format!(
                    "[Lypnos Guard] Rendering {} tensor... peaks: {}, valleys: {}",
                    len, peaks, valleys
                )
            }
            "export" => {
                if arg1.is_empty() {
                    return String::from(
                        "topcrypt export: usage: topcrypt export <file.topo> [path]",
                    );
                }
                let data = match self.read_file_bytes(arg1) {
                    Ok(d) => d,
                    Err(e) => return format!("topcrypt export: {}", e),
                };
                let topo = crate::fs::topcrypt::import_from_bytes(&data, 0);
                let flat = crate::fs::topcrypt::decode_bytes(&topo);
                let dest = if arg2.is_empty() {
                    format!("{}.flat", arg1)
                } else {
                    String::from(arg2)
                };
                let dest_name = dest.rsplit('/').next().unwrap_or("export.flat");
                if let Some(e) = self.deny(dest_name, Permissions::W) {
                    return format!("topcrypt export: {}", e);
                }
                if self.fs.exists(dest_name, self.cwd) {
                    let _ = self.fs.delete(dest_name, self.cwd);
                }
                match self.fs.store(dest_name, &flat, self.cwd) {
                    Ok(_) => format!(
                        "[TopCrypt] Flattened {} blocks → {} bytes → {}",
                        topo.block_count,
                        flat.len(),
                        dest
                    ),
                    Err(e) => format!("topcrypt export: {}", e),
                }
            }
            "import" => {
                if arg1.is_empty() {
                    return String::from("topcrypt import: usage: topcrypt import <file>");
                }
                let data = match self.read_file_bytes(arg1) {
                    Ok(d) => d,
                    Err(e) => return format!("topcrypt import: {}", e),
                };
                let topo = crate::fs::topcrypt::encode_bytes(&data, 0x5EA1);
                let serialized = crate::fs::topcrypt::export_to_bytes(&topo);
                let topo_name = format!("{}.topo", arg1.rsplit('/').next().unwrap_or(arg1));
                if let Some(e) = self.deny(&topo_name, Permissions::W) {
                    return format!("topcrypt import: {}", e);
                }
                if self.fs.exists(&topo_name, self.cwd) {
                    let _ = self.fs.delete(&topo_name, self.cwd);
                }
                match self.fs.store(&topo_name, &serialized, self.cwd) {
                    Ok(id) => format!(
                        "[TopCrypt] Imported {} bytes → {} blocks on S² (inode {})",
                        data.len(),
                        topo.block_count,
                        id
                    ),
                    Err(e) => format!("topcrypt import: {}", e),
                }
            }
            _ => String::from(
                "TopCrypt / Lypnos Guard commands:
\
                 topcrypt encode <file>       — Encode file to topological format
\
                 topcrypt decode <file.topo>  — Decode topological file
\
                 topcrypt lock <file.topo>    — Lock file with Lypnos Guard
\
                 topcrypt unlock <file.topo>  — Unlock file
\
                 topcrypt info <file.topo>    — Show topological metadata
\
                 topcrypt render <file.csv>   — Render CSV as tensor
\
                 topcrypt export <file.topo> [path] — Export to flat file
\
                 topcrypt import <file>       — Import flat file to manifold",
            ),
        }
    }

    fn cmd_tensor(&mut self, subcmd: &str, arg: &str) -> String {
        match subcmd {
            "render" => {
                if arg.is_empty() {
                    crate::wm::app_launcher::launch_app(9);
                    String::from("[Tensor] Launching tensor viewer with demo pattern")
                } else {
                    if let Some(e) = self.deny(arg, Permissions::R) {
                        return format!("tensor render: {}", e);
                    }
                    match self.fs.read_text(arg, self.cwd) {
                        Some(text) => {
                            crate::apps::tensor_viewer::set_pending_csv(&text);
                            crate::wm::app_launcher::launch_app(9);
                            format!("[Tensor] Launching viewer with '{}'", arg)
                        }
                        None => format!("tensor render: '{}' not found", arg),
                    }
                }
            }
            "info" => {
                if arg.is_empty() {
                    return String::from("tensor info: usage: tensor info <file.csv>");
                }
                if let Some(e) = self.deny(arg, Permissions::R) {
                    return format!("tensor info: {}", e);
                }
                match self.fs.read_text(arg, self.cwd) {
                    Some(text) => {
                        let tensor = crate::ml_engine::tensor_viz::parse_csv(&text);
                        if tensor.data.is_empty() {
                            return format!("tensor info: '{}' contains no numeric data", arg);
                        }
                        let mut min = f32::MAX;
                        let mut max = f32::MIN;
                        let mut sum = 0.0f32;
                        for &v in &tensor.data {
                            if v < min {
                                min = v;
                            }
                            if v > max {
                                max = v;
                            }
                            sum += v;
                        }
                        let mean = sum / tensor.data.len() as f32;
                        let shape_str = tensor
                            .shape
                            .iter()
                            .map(|s| format!("{}", s))
                            .collect::<Vec<_>>()
                            .join("x");
                        format!("Tensor: {}\nshape: [{}]\nmin: {:.4}\nmax: {:.4}\nmean: {:.4}\nelements: {}",
                            arg, shape_str, min, max, mean, tensor.data.len())
                    }
                    None => format!("tensor info: '{}' not found", arg),
                }
            }
            _ => String::from(
                "Tensor commands:\n\
                 tensor render [file.csv] — Launch 3D tensor viewer\n\
                 tensor info <file.csv>   — Show tensor statistics",
            ),
        }
    }

    /// Read a file's raw bytes by path, relative to the working directory.
    ///
    /// Returns the denial message rather than `None` so a refused read is not
    /// reported to the operator as a missing file.
    fn read_file_bytes(&self, name: &str) -> Result<Vec<u8>, String> {
        if let Some(e) = self.deny(name, Permissions::R) {
            return Err(e);
        }
        let inode_id = self
            .fs
            .resolve_path_from(name, self.cwd)
            .map_err(|_| format!("'{}' not found", name))?;
        let inode = self
            .fs
            .inode(inode_id)
            .ok_or_else(|| format!("'{}' not found", name))?;
        Ok(inode.data.clone())
    }

    fn cmd_prefetch(&self, sub: &str) -> String {
        match sub {
            "status" => {
                let engine = crate::fs::prefetch::PrefetchEngine::new_gaming();
                format!(
                    "[aether-link] Prefetch Engine Status\n\
                     ════════════════════════════════════\n\
                     Preset:          {:?} (ε={:.2}, φ={:.2})\n\
                     Decisions:       {}\n\
                     Hit ratio:       {:.1}%\n\
                     Note: prefetch counters are in-memory only (no DMA hardware)",
                    engine.preset(),
                    engine.epsilon(),
                    engine.phi(),
                    engine.total_decisions(),
                    engine.hit_ratio() * 100.0
                )
            }
            _ => String::from("prefetch: usage: prefetch status"),
        }
    }

    fn update_cwd_path(&mut self) {
        let mut parts = Vec::new();
        let mut id = self.cwd;
        while id != 0 {
            if let Some(inode) = self.fs.inode(id) {
                parts.push(inode.name.clone());
                id = inode.parent;
            } else {
                break;
            }
        }
        parts.reverse();
        if parts.is_empty() {
            self.cwd_path = String::from("/");
        } else {
            self.cwd_path = format!("/{}/", parts.join("/"));
        }
    }

    pub fn prompt(&self) -> String {
        format!("seal:{}$ ", self.cwd_path)
    }
}

impl Default for Shell {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::security::mac::{check_file_permission, init_default_policy};
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// The decision `Shell::deny` makes, for an explicit uid.
    ///
    /// `deny` itself reads `scheduler::current_uid()`, which needs a live
    /// scheduler with a current task; this chains the same two calls in the
    /// same order without one, the way `fs/vfs.rs`'s own MAC tests do.
    fn denied(uid: u32, cwd_path: &str, arg: &str, perms: Permissions) -> bool {
        match abs_path_in(cwd_path, arg) {
            Some(path) => !check_file_permission(uid, &path, perms),
            None => true, // unresolvable argument is refused, not passed through
        }
    }

    /// A plain name is resolved against the working directory, so `peek
    /// secret` after `open /root` names `/root/secret` — the string the
    /// `Deny /root` rule is keyed on. Before this gate existed, `ManifoldFS`
    /// looked the name up in the cwd with no check anywhere on the path.
    fn test_plain_name_resolves_against_cwd() -> TestResult {
        test_assert_eq!(abs_path_in("/root/", "secret").unwrap(), "/root/secret");
        test_assert_eq!(abs_path_in("/", "secret").unwrap(), "/secret");
        test_assert_eq!(abs_path_in("/a/b/", "c").unwrap(), "/a/b/c");
        TestResult::Pass
    }

    /// An empty argument means the working directory itself, with the
    /// trailing separator `update_cwd_path` leaves on it removed — MAC rule
    /// matching is keyed on the un-terminated form (`security/mac.rs:150`).
    fn test_empty_arg_is_the_cwd_without_trailing_slash() -> TestResult {
        test_assert_eq!(abs_path_in("/root/", "").unwrap(), "/root");
        test_assert_eq!(abs_path_in("/", "").unwrap(), "/");
        test_assert_eq!(abs_path_in("/a/b/", "/").unwrap(), "/");
        TestResult::Pass
    }

    /// Non-canonical arguments are refused outright. `ManifoldFS::resolve_path`
    /// filters empty components, so `//root/secret` reaches the same inode as
    /// `/root/secret`; `resolve_path_from` follows real `..` directory
    /// entries. Either would let a checked string and an opened file diverge,
    /// so neither is accepted.
    fn test_non_canonical_arguments_are_refused() -> TestResult {
        for arg in ["//root", "/root//secret", "../root", "a/../../root", "./x", "x/"] {
            test_assert!(
                abs_path_in("/data/", arg).is_none(),
                "non-canonical argument must not resolve"
            );
        }
        TestResult::Pass
    }

    /// The bypass itself: an unprivileged shell reading `/root/secret`, by
    /// absolute path or by name from that directory, is denied both ways.
    fn test_root_secret_denied_for_unprivileged_shell() -> TestResult {
        init_default_policy();
        test_assert!(denied(1000, "/", "/root/secret", Permissions::R));
        test_assert!(denied(1000, "/root/", "secret", Permissions::R));
        test_assert!(denied(1000, "/", "/root", Permissions::W));
        TestResult::Pass
    }

    /// `..` and `//` cannot walk out of an allowed directory into a denied
    /// one — the refusal is on the argument form, before any lookup.
    fn test_traversal_out_of_allowed_directory_denied() -> TestResult {
        init_default_policy();
        for arg in ["../root/secret", "//root/secret", "../../root/secret"] {
            test_assert!(denied(1000, "/data/", arg, Permissions::R));
        }
        TestResult::Pass
    }

    /// uid 0 keeps its `security/mac.rs:110` bypass, and the allowed prefixes
    /// stay usable for everyone — the gate subtracts only what a rule covers.
    fn test_allowed_paths_and_root_are_not_broken() -> TestResult {
        init_default_policy();
        test_assert!(!denied(0, "/root/", "secret", Permissions::R));
        test_assert!(!denied(1000, "/data/", "file", Permissions::R));
        test_assert!(!denied(1000, "/tmp/", "file", Permissions::W));
        test_assert!(!denied(1000, "/", "/data/file", Permissions::W));
        TestResult::Pass
    }

    /// `/rootkit` is not inside `/root`; the gate must not over-deny either,
    /// or it would report a lie about which files are protected.
    fn test_sibling_prefix_is_not_denied() -> TestResult {
        init_default_policy();
        test_assert!(!denied(1000, "/", "/rootkit", Permissions::R));
        test_assert_eq!(abs_path_in("/", "rootkit").unwrap(), "/rootkit");
        TestResult::Pass
    }

    /// Stage N's output is stage N+1's stdin, in order, with the `< file`
    /// contents seeding stage 1. The recorded `(command, stdin)` pairs are the
    /// whole contract of the pipeline: getting the buffer from the wrong
    /// stage, or dropping it, shows up here and nowhere else.
    fn test_pipeline_forwards_each_stage_output_as_the_next_stdin() -> TestResult {
        let mut seen: Vec<String> = Vec::new();
        let out = run_pipeline(&["one", "two", "three"], String::from("seed"), |cmd, stdin| {
            seen.push(format!("{}<{}", cmd, stdin));
            format!("{}-out", cmd)
        });
        test_assert!(out.is_ok());
        test_assert_eq!(out.unwrap_or_default(), "three-out");
        test_assert_eq!(seen.len(), 3);
        test_assert_eq!(seen[0], "one<seed");
        test_assert_eq!(seen[1], "two<one-out");
        test_assert_eq!(seen[2], "three<two-out");
        TestResult::Pass
    }

    /// Three filters compose, and each one narrows the stream: dropping any
    /// single stage leaves a different set of lines.
    fn test_three_stage_pipeline_composes() -> TestResult {
        let text = "alpha.txt\nalpha.log\nbeta.log\nnotes\n";
        let filter = |cmd: &str, stdin: &str| {
            grep_matches("", cmd.split_once(' ').map(|x| x.1).unwrap_or(""), stdin)
        };
        let out = run_pipeline(
            &["grep .", "grep alpha", "grep log"],
            String::from(text),
            filter,
        );
        test_assert_eq!(out.unwrap_or_default(), "alpha.log\n");

        // Each stage is load-bearing: without the middle one, beta.log stays.
        let two = run_pipeline(&["grep .", "grep log"], String::from(text), filter);
        test_assert_eq!(two.unwrap_or_default(), "alpha.log\nbeta.log\n");
        TestResult::Pass
    }

    /// A stage that matches nothing hands the next stage an empty stdin
    /// rather than its own input, and the pipeline still completes.
    fn test_stage_with_no_output_feeds_empty_stdin() -> TestResult {
        let mut downstream_stdin = String::from("unset");
        let out = run_pipeline(
            &["grep zzz", "grep alpha"],
            String::from("alpha.txt\nbeta.log\n"),
            |cmd, stdin| {
                if cmd == "grep alpha" {
                    downstream_stdin = String::from(stdin);
                }
                grep_matches("", cmd.split_once(' ').map(|x| x.1).unwrap_or(""), stdin)
            },
        );
        test_assert_eq!(out.unwrap_or_default(), "");
        test_assert_eq!(downstream_stdin, "");
        TestResult::Pass
    }

    /// A stage feeding another stage more than `PIPE_CAPACITY` bytes aborts
    /// the line before the next stage runs. The last stage is uncapped — its
    /// output goes to the terminal, which a bare command already does.
    fn test_pipeline_stage_over_capacity_is_refused() -> TestResult {
        let over = "x".repeat(PIPE_CAPACITY + 1);
        let mut reached_second = false;
        let refused = run_pipeline(&["big", "grep x"], String::new(), |cmd, _| {
            if cmd == "grep x" {
                reached_second = true;
            }
            over.clone()
        });
        test_assert!(refused.is_err(), "over-capacity stage must abort the line");
        test_assert!(
            !reached_second,
            "an over-capacity buffer must not reach the next stage"
        );

        let alone = run_pipeline(&["big"], String::new(), |_, _| over.clone());
        test_assert_eq!(alone.unwrap_or_default().len(), PIPE_CAPACITY + 1);

        let exact = "x".repeat(PIPE_CAPACITY);
        let at_cap = run_pipeline(&["big", "grep x"], String::new(), |_, _| exact.clone());
        test_assert!(at_cap.is_ok(), "exactly one buffer must be allowed");
        TestResult::Pass
    }

    /// `>` and `>>` name the same file and differ only in the append flag,
    /// `<` is a separate channel, and both may appear on one stage.
    fn test_stage_parse_separates_command_from_redirection() -> TestResult {
        let truncate = match parse_stage("peek notes.txt > out.txt") {
            Ok(s) => s,
            Err(_) => return TestResult::Fail("'>' stage must parse"),
        };
        test_assert_eq!(truncate.cmd, "peek notes.txt");
        test_assert!(truncate.input.is_none());
        test_assert_eq!(truncate.output, Some(("out.txt", false)));

        let append = match parse_stage("peek notes.txt >> out.txt") {
            Ok(s) => s,
            Err(_) => return TestResult::Fail("'>>' stage must parse"),
        };
        test_assert_eq!(append.cmd, "peek notes.txt");
        test_assert_eq!(append.output, Some(("out.txt", true)));

        let both = match parse_stage("grep alpha < in.txt >> out.txt") {
            Ok(s) => s,
            Err(_) => return TestResult::Fail("'<' with '>>' must parse"),
        };
        test_assert_eq!(both.cmd, "grep alpha");
        test_assert_eq!(both.input, Some("in.txt"));
        test_assert_eq!(both.output, Some(("out.txt", true)));

        let plain = match parse_stage(" look ") {
            Ok(s) => s,
            Err(_) => return TestResult::Fail("a bare command must parse"),
        };
        test_assert_eq!(plain.cmd, "look");
        test_assert!(plain.input.is_none() && plain.output.is_none());
        TestResult::Pass
    }

    /// `>` replaces what the file held; `>>` keeps it and adds. The two must
    /// not produce the same bytes for the same output.
    fn test_truncate_replaces_and_append_extends() -> TestResult {
        let prior = || Some(String::from("old"));
        test_assert_eq!(redirected_content(prior(), "new", false), "new");
        test_assert_eq!(redirected_content(prior(), "new", true), "oldnew");
        test_assert!(
            redirected_content(prior(), "new", false) != redirected_content(prior(), "new", true)
        );
        // A file that does not exist yet holds exactly the output either way.
        test_assert_eq!(redirected_content(None, "new", true), "new");
        test_assert_eq!(redirected_content(None, "new", false), "new");
        TestResult::Pass
    }

    /// A line splits on `|` into one stage each, and the redirections land on
    /// the stages that may carry them. Without the split the whole line is
    /// one command and `| grep x` is an argument, which is what the shell did
    /// before pipelines existed.
    fn test_line_splits_into_stages() -> TestResult {
        let three = match parse_line("look | grep alpha | grep .txt") {
            Ok(s) => s,
            Err(_) => return TestResult::Fail("a three-stage line must parse"),
        };
        test_assert_eq!(three.len(), 3);
        test_assert_eq!(three[0].cmd, "look");
        test_assert_eq!(three[1].cmd, "grep alpha");
        test_assert_eq!(three[2].cmd, "grep .txt");

        let ends = match parse_line("grep alpha < in.txt | grep log >> out.txt") {
            Ok(s) => s,
            Err(_) => return TestResult::Fail("end-stage redirections must parse"),
        };
        test_assert_eq!(ends.len(), 2);
        test_assert_eq!(ends[0].input, Some("in.txt"));
        test_assert!(ends[0].output.is_none());
        test_assert!(ends[1].input.is_none());
        test_assert_eq!(ends[1].output, Some(("out.txt", true)));

        // A bare command is a one-stage pipeline, unchanged from before.
        let one = match parse_line("look") {
            Ok(s) => s,
            Err(_) => return TestResult::Fail("a bare command must parse"),
        };
        test_assert_eq!(one.len(), 1);
        test_assert_eq!(one[0].cmd, "look");

        // Redirection away from the ends, and empty stages, are refused.
        for line in [
            "look > f.txt | grep a",
            "look | grep a < f.txt",
            "| look",
            "look |",
            "look | | grep a",
        ] {
            test_assert!(parse_line(line).is_err(), "malformed line must be refused");
        }
        TestResult::Pass
    }

    /// A stage with no command, or a redirection with no file, is an error
    /// the operator sees — never a silently skipped stage or a lost write.
    fn test_malformed_stage_is_a_syntax_error() -> TestResult {
        for stage in [
            "", "   ", "> out.txt", ">> out.txt", "< in.txt", "peek f >", "peek f >>",
            "grep a <", "peek f > ",
        ] {
            test_assert!(
                parse_stage(stage).is_err(),
                "malformed stage must be refused"
            );
        }
        test_assert!(parse_stage("look").is_ok());
        test_assert!(parse_stage("look > f").is_ok());
        TestResult::Pass
    }

    /// Five lines, one adjacent duplicate pair, mixed case, unsorted, and no
    /// terminator on the last line: 5 lines, 5 words, 32 bytes.
    const FRUIT: &str = "banana\napple\napple\nCherry\nbanana";

    /// What the shell prints for the stage text `line` handed `stdin`, with a
    /// command that is not a filter named rather than silently empty.
    fn filtered(line: &str, stdin: &str) -> String {
        run_filter(line, stdin).unwrap_or_else(|| String::from("<not a filter>"))
    }

    /// Every filter name reaches its own filter, and nothing else is one. A
    /// table entry pointing at the wrong function — `head` at `tail`, `sort`
    /// at `uniq` — shows up here as the neighbouring filter's answer.
    fn test_filter_table_routes_each_name() -> TestResult {
        test_assert_eq!(filtered("wc -l", FRUIT), "5\n");
        // Everything after the first space is the argument text.
        test_assert_eq!(filtered("wc  -l", FRUIT), "5\n");
        test_assert_eq!(filtered("grep two words", "two words\nno\n"), "two words\n");
        test_assert_eq!(filtered("head -n 1", FRUIT), "banana\n");
        test_assert_eq!(filtered("tail -n 1", FRUIT), "banana\n");
        test_assert_eq!(filtered("head -n 2", FRUIT), "banana\napple\n");
        test_assert_eq!(filtered("tail -n 2", FRUIT), "Cherry\nbanana\n");
        test_assert_eq!(filtered("sort -u", FRUIT), "Cherry\napple\nbanana\n");
        test_assert_eq!(filtered("uniq", FRUIT), "banana\napple\nCherry\nbanana\n");
        test_assert_eq!(filtered("grep Cherry", FRUIT), "Cherry\n");
        test_assert_eq!(filtered("tr a X", "banana"), "bXnXnX");
        // A bare `cat` is the pipeline unchanged; with a file it is `peek`,
        // which needs the filesystem and so is not a filter.
        test_assert_eq!(filtered("cat", FRUIT), FRUIT);
        test_assert!(run_filter("cat notes.txt", FRUIT).is_none());
        for line in ["look", "peek notes.txt", "write a b", "seal", "nosuchcmd", ""] {
            test_assert!(
                run_filter(line, FRUIT).is_none(),
                "only the filters may claim a command name"
            );
        }
        TestResult::Pass
    }

    /// `wc` counts lines, words and bytes, and a flag selects one of the
    /// three. The three counts differ on this stream, so a count read from
    /// the wrong one cannot pass.
    fn test_wc_counts_lines_words_and_bytes() -> TestResult {
        test_assert_eq!(filtered("wc", FRUIT), "5 5 32\n");
        test_assert_eq!(filtered("wc -l", FRUIT), "5\n");
        test_assert_eq!(filtered("wc -w", FRUIT), "5\n");
        test_assert_eq!(filtered("wc -c", FRUIT), "32\n");
        test_assert_eq!(filtered("wc", "one two three\n"), "1 3 14\n");
        test_assert_eq!(filtered("wc", ""), "0 0 0\n");
        test_assert_eq!(filtered("wc -lc", "ab\ncd\n"), "2 6\n");
        TestResult::Pass
    }

    /// `head` and `tail` take opposite ends, default to ten lines, and never
    /// invent lines the stream does not have.
    fn test_head_and_tail_take_opposite_ends() -> TestResult {
        test_assert_eq!(filtered("head -n 2", FRUIT), "banana\napple\n");
        test_assert_eq!(filtered("tail -n 2", FRUIT), "Cherry\nbanana\n");
        test_assert!(filtered("head -n 2", FRUIT) != filtered("tail -n 2", FRUIT));
        // No -n is ten lines, which this five-line stream is inside of.
        test_assert_eq!(filtered("head", FRUIT), "banana\napple\napple\nCherry\nbanana\n");
        test_assert_eq!(filtered("tail", FRUIT), filtered("head", FRUIT));
        test_assert_eq!(filtered("head -n 99", FRUIT), filtered("head", FRUIT));
        test_assert_eq!(filtered("tail -n 99", FRUIT), filtered("head", FRUIT));
        test_assert_eq!(filtered("head -n 3", ""), "");
        // The default is ten, not the whole stream.
        let eleven = "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n";
        test_assert_eq!(filtered("head", eleven), "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n");
        test_assert_eq!(filtered("tail", eleven), "2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n");
        TestResult::Pass
    }

    /// A line count of zero, a negative one, a word, or a stray argument is
    /// refused with a message. Falling back to the default would print ten
    /// lines to an operator who asked for something else.
    fn test_head_and_tail_refuse_a_bad_count() -> TestResult {
        for cmd in ["head", "tail"] {
            for args in ["-n 0", "-n -3", "-n two", "-n", "-n 2 extra", "3"] {
                let out = filtered(&format!("{} {}", cmd, args), FRUIT);
                test_assert!(
                    out.starts_with(cmd) && !out.contains("banana"),
                    "a bad count must be refused, not defaulted"
                );
            }
        }
        test_assert_eq!(
            filtered("head -n 0", FRUIT),
            "head: -n needs a line count of 1 or more, not '0'"
        );
        test_assert_eq!(
            filtered("tail -n -3", FRUIT),
            "tail: -n needs a line count of 1 or more, not '-3'"
        );
        TestResult::Pass
    }

    /// `sort` orders by byte, `-u` drops repeats, `-r` turns the result
    /// around, and every line comes back terminated whether or not the input
    /// ended in a newline. An empty stream has no lines, so it produces
    /// nothing rather than a blank one.
    fn test_sort_orders_dedups_and_reverses() -> TestResult {
        test_assert_eq!(filtered("sort", FRUIT), "Cherry\napple\napple\nbanana\nbanana\n");
        test_assert_eq!(filtered("sort -u", FRUIT), "Cherry\napple\nbanana\n");
        test_assert_eq!(filtered("sort -r", FRUIT), "banana\nbanana\napple\napple\nCherry\n");
        test_assert_eq!(filtered("sort -ru", FRUIT), "banana\napple\nCherry\n");
        test_assert_eq!(filtered("sort -u -r", FRUIT), "banana\napple\nCherry\n");
        // Unterminated in, terminated out; the line count is unchanged.
        test_assert_eq!(filtered("sort", "b\na"), "a\nb\n");
        test_assert_eq!(filtered("sort", "b\na\n"), "a\nb\n");
        test_assert_eq!(filtered("wc -l", &filtered("sort", "b\na")), "2\n");
        // Empty in, empty out — not "\n".
        test_assert_eq!(filtered("sort", ""), "");
        test_assert_eq!(filtered("sort -ru", ""), "");
        TestResult::Pass
    }

    /// `uniq` collapses only neighbours, which is what makes `sort | uniq`
    /// different from `uniq` alone, and `-c` reports the length of each run.
    fn test_uniq_collapses_only_adjacent_duplicates() -> TestResult {
        test_assert_eq!(filtered("uniq", FRUIT), "banana\napple\nCherry\nbanana\n");
        test_assert_eq!(filtered("uniq -c", FRUIT), "1 banana\n2 apple\n1 Cherry\n1 banana\n");
        // The two bananas are not neighbours until the stream is sorted.
        let sorted = filtered("sort", FRUIT);
        test_assert_eq!(filtered("uniq -c", &sorted), "1 Cherry\n2 apple\n2 banana\n");
        test_assert_eq!(filtered("uniq", ""), "");
        test_assert_eq!(filtered("uniq -c", "x\nx\nx\n"), "3 x\n");
        TestResult::Pass
    }

    /// `grep` keeps its old behaviour, and each flag changes the answer:
    /// `-v` inverts, `-i` folds case, `-n` numbers, `-c` counts.
    fn test_grep_flags_invert_fold_number_and_count() -> TestResult {
        // Unflagged, exactly what it did before the flags existed.
        test_assert_eq!(filtered("grep banana", FRUIT), "banana\nbanana\n");
        test_assert_eq!(grep_matches("", "banana", FRUIT), "banana\nbanana\n");
        test_assert_eq!(filtered("grep cherry", FRUIT), "");
        test_assert_eq!(filtered("grep -i cherry", FRUIT), "Cherry\n");
        test_assert_eq!(filtered("grep -v banana", FRUIT), "apple\napple\nCherry\n");
        test_assert_eq!(filtered("grep -n banana", FRUIT), "1:banana\n5:banana\n");
        test_assert_eq!(filtered("grep -c banana", FRUIT), "2\n");
        test_assert_eq!(filtered("grep -vc banana", FRUIT), "3\n");
        test_assert_eq!(filtered("grep -cv banana", FRUIT), "3\n");
        test_assert_eq!(filtered("grep -in CHERRY", FRUIT), "4:Cherry\n");
        // The pattern keeps its spaces, and a missing one is refused.
        test_assert_eq!(filtered("grep two words", "two words\nother\n"), "two words\n");
        test_assert!(filtered("grep", FRUIT).starts_with("grep: what text?"));
        test_assert!(filtered("grep -v", FRUIT).starts_with("grep: what text?"));
        TestResult::Pass
    }

    /// `tr` replaces one character with one character and says so when asked
    /// for a range or a class, rather than translating the first character of
    /// each and dropping the rest.
    fn test_tr_translates_one_character_and_refuses_ranges() -> TestResult {
        test_assert_eq!(filtered("tr a X", "banana\napple\n"), "bXnXnX\nXpple\n");
        test_assert_eq!(filtered("tr , ;", "a,b,c"), "a;b;c");
        // Not a line filter: the stream ends the way it arrived.
        test_assert_eq!(filtered("tr a X", "banana"), "bXnXnX");
        for line in ["tr", "tr a", "tr a-z A-Z", "tr a X Y", "tr ab X"] {
            test_assert!(
                filtered(line, FRUIT).starts_with("tr: usage:"),
                "tr must refuse anything that is not two single characters"
            );
        }
        TestResult::Pass
    }

    /// A flag letter no filter accepts is refused rather than ignored: a
    /// dropped flag answers a different question than the one asked. Bundled
    /// flags are accepted, and a bare `-` is text, not a flag.
    fn test_unknown_filter_flag_is_refused() -> TestResult {
        for line in [
            "wc -z",
            "sort -x",
            "uniq -q",
            "head -k 2",
            "grep -z text",
            "sort -ruz",
        ] {
            let out = filtered(line, FRUIT);
            test_assert!(
                out.contains("unknown option") && !out.contains("banana"),
                "an unknown flag must be refused, not ignored"
            );
        }
        // A stray non-flag argument is refused too — these read stdin only.
        test_assert!(filtered("wc file.txt", FRUIT).contains("unexpected argument"));
        test_assert!(filtered("sort file.txt", FRUIT).contains("unexpected argument"));
        test_assert!(filtered("uniq file.txt", FRUIT).contains("unexpected argument"));
        // A lone '-' is a pattern, not a flag.
        test_assert_eq!(filtered("grep -", "a-b\ncd\n"), "a-b\n");
        TestResult::Pass
    }

    /// The handbook names every filter and both metacharacters, and each
    /// filter has a page. A feature no `help` mentions cannot be found.
    fn test_handbook_documents_the_filters_and_pipeline_syntax() -> TestResult {
        let book = help::handbook();
        for token in [
            "grep", "wc", "head", "tail", "sort", "uniq", "cat", "tr", "a | b", "< file",
            "> file", ">> file",
        ] {
            test_assert!(book.contains(token), "the handbook must name every filter");
        }
        for topic in [
            "grep", "wc", "head", "tail", "sort", "uniq", "cat", "tr", "pipes", "redirection",
        ] {
            test_assert!(
                !help::help_for(topic).starts_with("No help available"),
                "every filter needs a help page"
            );
        }
        test_assert!(help::help_for("nosuchcommand").starts_with("No help available"));
        TestResult::Pass
    }

    /// A store holding `pairs`, in the order given.
    fn vars_with(pairs: &[(&str, &str)]) -> Vars {
        let mut vars = Vars::new();
        for (name, value) in pairs {
            let _ = set_var(&mut vars, name, value);
        }
        vars
    }

    /// `NAME=value` is an assignment; everything else is the command it looks
    /// like. `a-b=c` cannot become a variable, because no `$a-b` could ever
    /// name it back.
    fn test_assignment_statements_are_recognized_and_others_fall_through() -> TestResult {
        test_assert_eq!(parse_assignment("A=1"), Some(("A", "1")));
        test_assert_eq!(parse_assignment("_x9=v"), Some(("_x9", "v")));
        test_assert_eq!(parse_assignment("A="), Some(("A", "")));
        // The first '=' ends the name, so the value may hold more of them.
        test_assert_eq!(parse_assignment("A=b=c"), Some(("A", "b=c")));
        // The statement is recognized before the line is split, so a value
        // may hold what a pipeline would otherwise split on.
        test_assert_eq!(parse_assignment("V=a|b"), Some(("V", "a|b")));
        for line in [
            "a-b=c", "1a=c", "=c", "look", "set k v", "A =1", "A B=c", "", "grep a=b c",
        ] {
            test_assert!(
                parse_assignment(line).is_none(),
                "only NAME=value is an assignment"
            );
        }
        TestResult::Pass
    }

    /// `$NAME` and `${NAME}` are replaced by the value; an unset name is
    /// replaced by nothing; a `$` no name follows is the character itself.
    fn test_variables_expand_in_arguments() -> TestResult {
        let vars = vars_with(&[("F", "notes"), ("EMPTY", "")]);
        test_assert_eq!(
            expand_vars("peek $F.txt", &vars).unwrap_or_default(),
            "peek notes.txt"
        );
        test_assert_eq!(
            expand_vars("peek ${F}.txt", &vars).unwrap_or_default(),
            "peek notes.txt"
        );
        // Braces are what let a name touch name-shaped text after it.
        test_assert_eq!(expand_vars("$F_v2", &vars).unwrap_or_default(), "");
        test_assert_eq!(expand_vars("${F}_v2", &vars).unwrap_or_default(), "notes_v2");
        // Unset and set-but-empty both leave the text around them alone.
        test_assert_eq!(expand_vars("a${NOPE}b", &vars).unwrap_or_default(), "ab");
        test_assert_eq!(expand_vars("a$EMPTY.b", &vars).unwrap_or_default(), "a.b");
        test_assert_eq!(expand_vars("$F $F", &vars).unwrap_or_default(), "notes notes");
        test_assert_eq!(
            expand_vars("look | grep -i seal", &vars).unwrap_or_default(),
            "look | grep -i seal"
        );
        // A '$' that names nothing is a literal '$'. This shell has no exit
        // status, so `$?` stays the text it looks like: nothing is invented.
        for text in ["$?", "$5", "cost $", "$ F", "$-", "$$"] {
            test_assert_eq!(expand_vars(text, &vars).unwrap_or_default(), text);
        }
        TestResult::Pass
    }

    /// An unclosed or unnamed `${` is an error, and so is an expansion that
    /// leaves nothing where a command or a file name has to be.
    fn test_malformed_expansion_and_empty_result_are_refused() -> TestResult {
        let vars = vars_with(&[("A", "1")]);
        for text in ["peek ${A", "${", "${}", "${a-b}", "${1a}", "a ${A} ${B"] {
            test_assert!(
                expand_vars(text, &vars).is_err(),
                "an unclosed or unnamed '${' must be refused"
            );
        }
        test_assert_eq!(expand_vars("${A}", &vars).unwrap_or_default(), "1");
        // `peek > $OUT` with OUT unset must not write a file called '$OUT',
        // and `$CMD` alone must not report an unknown command named nothing.
        test_assert!(expand_required("$NOPE", &vars).is_err());
        test_assert!(expand_required("${NOPE}", &vars).is_err());
        test_assert!(expand_required("", &vars).is_err());
        test_assert_eq!(expand_required("${A}", &vars).unwrap_or_default(), "1");
        TestResult::Pass
    }

    /// Expansion is one substitution deep: what a value expands to is copied
    /// out and never scanned again, so no arrangement of variables can make it
    /// run twice over the same text.
    fn test_expansion_is_one_pass_and_terminates() -> TestResult {
        // `D` holds a lone '$', so `X` ends up holding the two characters
        // `$X` — a variable that names itself.
        let mut vars = Vars::new();
        let _ = set_var(&mut vars, "D", "$");
        let x = expand_vars("${D}X", &vars).unwrap_or_default();
        test_assert_eq!(x, "$X");
        let _ = set_var(&mut vars, "X", &x);
        test_assert_eq!(expand_vars("$X", &vars).unwrap_or_default(), "$X");
        test_assert_eq!(expand_vars("[$X]", &vars).unwrap_or_default(), "[$X]");

        // Two variables naming each other. Each right-hand side is expanded
        // once, when the assignment is made, so neither holds a live
        // reference to the other and the pair cannot chase itself.
        let mut ab = Vars::new();
        let b_value = expand_vars("$B", &ab).unwrap_or_default();
        let _ = set_var(&mut ab, "A", &b_value);
        let a_value = expand_vars("$A", &ab).unwrap_or_default();
        let _ = set_var(&mut ab, "B", &a_value);
        test_assert_eq!(list_vars(&ab, false), "A=\nB=\n");
        test_assert_eq!(expand_vars("$A$B", &ab).unwrap_or_default(), "");
        TestResult::Pass
    }

    /// The line is split into stages first and expanded second, so a value
    /// holding `|`, `<` or `>` reaches the command as text. Expanding first
    /// would make every value a possible operator, and with no quoting there
    /// would be no way to ask for the literal.
    fn test_expansion_happens_after_the_pipeline_split() -> TestResult {
        let vars = vars_with(&[("V", "a|b"), ("R", ">stolen.txt")]);
        let stages = match parse_line("peek f.txt | grep $V") {
            Ok(s) => s,
            Err(_) => return TestResult::Fail("the line must split into two stages"),
        };
        test_assert_eq!(stages.len(), 2);
        let second = expand_vars(stages[1].cmd, &vars).unwrap_or_default();
        test_assert_eq!(second, "grep a|b");
        // The `|` inside the value is the pattern, not a third stage.
        test_assert_eq!(filtered(&second, "a|b\nax\n"), "a|b\n");

        let stage = match parse_stage("grep $R") {
            Ok(s) => s,
            Err(_) => return TestResult::Fail("a stage naming a variable must parse"),
        };
        test_assert!(
            stage.output.is_none(),
            "a value must not be able to become a redirection"
        );
        test_assert_eq!(
            expand_vars(stage.cmd, &vars).unwrap_or_default(),
            "grep >stolen.txt"
        );
        TestResult::Pass
    }

    /// A value may not contain whitespace. With no quoting, `A=1 look` cannot
    /// be told from a value with a space in it, and the two readings do
    /// opposite things, so the assignment is refused and nothing is stored.
    fn test_a_value_may_not_hold_a_space() -> TestResult {
        let mut vars = Vars::new();
        for value in ["1 look", "a b", "x\ty", "trailing "] {
            test_assert!(
                set_var(&mut vars, "A", value).is_err(),
                "a value with whitespace must be refused"
            );
        }
        test_assert_eq!(list_vars(&vars, false), "(none)");
        test_assert!(set_var(&mut vars, "A", "1").is_ok());
        test_assert_eq!(list_vars(&vars, false), "A=1\n");
        // The name is checked on the way in as well.
        for name in ["a-b", "", "1a", "a b"] {
            test_assert!(
                set_var(&mut vars, name, "1").is_err(),
                "only a variable name may be assigned to"
            );
        }
        test_assert_eq!(list_vars(&vars, false), "A=1\n");
        TestResult::Pass
    }

    /// `export` marks, `unset` forgets, `env` lists the marked set and `set`
    /// lists all of it, both in name order.
    fn test_export_marks_unset_removes_and_lists_stay_sorted() -> TestResult {
        let mut vars = Vars::new();
        test_assert_eq!(list_vars(&vars, false), "(none)");
        test_assert_eq!(list_vars(&vars, true), "(none)");
        for (name, value) in [("b", "2"), ("A", "1"), ("C", "3")] {
            test_assert!(set_var(&mut vars, name, value).is_ok());
        }
        // Name order, not the order they were set in.
        test_assert_eq!(list_vars(&vars, false), "A=1\nC=3\nb=2\n");
        // Nothing is exported until `export` says so.
        test_assert_eq!(list_vars(&vars, true), "(none)");
        test_assert_eq!(cmd_export(&mut vars, "C"), "");
        test_assert_eq!(cmd_export(&mut vars, "A"), "");
        test_assert_eq!(list_vars(&vars, true), "A=1\nC=3\n");
        // `export NAME=value` assigns and marks in one statement.
        test_assert_eq!(cmd_export(&mut vars, "D=4"), "");
        test_assert_eq!(list_vars(&vars, true), "A=1\nC=3\nD=4\n");
        // Exporting a name that is not set yet creates it empty.
        test_assert_eq!(cmd_export(&mut vars, "E"), "");
        test_assert_eq!(list_vars(&vars, true), "A=1\nC=3\nD=4\nE=\n");
        // Reassigning keeps the mark.
        test_assert!(set_var(&mut vars, "A", "9").is_ok());
        test_assert_eq!(list_vars(&vars, true), "A=9\nC=3\nD=4\nE=\n");
        // `unset` takes the name and its mark together: setting it again
        // starts unexported.
        test_assert_eq!(cmd_unset(&mut vars, "C"), "");
        test_assert!(set_var(&mut vars, "C", "3").is_ok());
        test_assert_eq!(list_vars(&vars, true), "A=9\nD=4\nE=\n");
        test_assert_eq!(list_vars(&vars, false), "A=9\nC=3\nD=4\nE=\nb=2\n");
        // Unsetting something never set is not an error.
        test_assert_eq!(cmd_unset(&mut vars, "Z"), "");
        // A malformed argument is refused with a message and changes nothing.
        for bad in ["", "a-b", "A B", "1a", "A=1 B=2"] {
            test_assert!(
                !cmd_export(&mut vars, bad).is_empty(),
                "export must refuse a bad argument"
            );
            test_assert!(
                !cmd_unset(&mut vars, bad).is_empty(),
                "unset must refuse a bad argument"
            );
        }
        test_assert_eq!(list_vars(&vars, false), "A=9\nC=3\nD=4\nE=\nb=2\n");
        TestResult::Pass
    }

    /// The store is capped by count and by total bytes, and says so rather
    /// than evicting: a variable that quietly vanished would make every later
    /// `$NAME` expand to empty, which reads as a wrong answer.
    fn test_the_variable_store_is_capped() -> TestResult {
        let mut vars = Vars::new();
        for i in 0..MAX_VARS {
            test_assert!(
                set_var(&mut vars, &format!("v{}", i), "x").is_ok(),
                "a variable under the count cap must be accepted"
            );
        }
        test_assert_eq!(vars.len(), MAX_VARS);
        test_assert!(
            set_var(&mut vars, "one_too_many", "x").is_err(),
            "past the count cap must be refused"
        );
        test_assert!(!vars.contains_key("one_too_many"));
        // A name already in the store is not a new one, so it still assigns.
        test_assert!(set_var(&mut vars, "v0", "y").is_ok());

        let mut big = Vars::new();
        test_assert!(
            set_var(&mut big, "a", &"y".repeat(MAX_VAR_BYTES)).is_err(),
            "past the byte cap must be refused"
        );
        test_assert_eq!(list_vars(&big, false), "(none)");
        test_assert!(
            set_var(&mut big, "a", &"y".repeat(MAX_VAR_BYTES - 1)).is_ok(),
            "exactly the byte cap must be allowed"
        );
        test_assert!(
            set_var(&mut big, "b", "z").is_err(),
            "one byte over the cap must be refused"
        );
        // Reassigning releases the old value's bytes rather than adding to
        // them, or a store could fill up without ever growing.
        test_assert!(set_var(&mut big, "a", "small").is_ok());
        test_assert!(set_var(&mut big, "b", "z").is_ok());
        TestResult::Pass
    }

    /// A `#` opens a comment where a word opens: at the start of the line, or
    /// after whitespace. Anywhere else it is one of the argument's characters,
    /// which is what keeps `A=v#1` the value it looks like — this shell has no
    /// quoting, so a rule that cut at every `#` would silently shorten values
    /// with no way to ask for the literal.
    fn test_comment_starts_only_at_a_word_boundary() -> TestResult {
        test_assert_eq!(strip_comment("echo hi # note"), "echo hi");
        test_assert_eq!(strip_comment("echo hi #note"), "echo hi");
        test_assert_eq!(strip_comment("# whole line"), "");
        test_assert_eq!(strip_comment("   # indented"), "");
        test_assert_eq!(strip_comment("#"), "");
        // Blank lines come back empty, which is how a script skips them.
        test_assert_eq!(strip_comment(""), "");
        test_assert_eq!(strip_comment("  \t "), "");
        // Not at a word boundary: the '#' stays in the argument.
        test_assert_eq!(strip_comment("A=v#1"), "A=v#1");
        test_assert_eq!(strip_comment("write tag.txt c#3"), "write tag.txt c#3");
        test_assert_eq!(strip_comment("grep a#b | wc -l"), "grep a#b | wc -l");
        // The stated cost of the boundary: a '#' that starts a word is a
        // comment even when it was meant as a pattern. `grep $H` with H=#
        // is the way to ask for the literal.
        test_assert_eq!(strip_comment("grep #tag"), "grep");
        // An assignment keeps its value when a comment follows it.
        test_assert_eq!(
            parse_assignment(strip_comment("A=1 # note")),
            Some(("A", "1"))
        );
        // The whole line is trimmed, so an indented script line is the
        // command it reads as rather than an unknown one.
        test_assert_eq!(strip_comment("   look   "), "look");
        TestResult::Pass
    }

    /// Comments are cut before the line is split into stages, and variables
    /// are expanded after: a `#` typed on the line ends it, and a `#` that
    /// arrives out of a value is text the command receives.
    fn test_comments_are_cut_before_the_split_and_expansion_after() -> TestResult {
        let line = strip_comment("look | grep a # | grep b");
        test_assert_eq!(line, "look | grep a");
        let stages = match parse_line(line) {
            Ok(s) => s,
            Err(_) => return TestResult::Fail("the surviving line must parse"),
        };
        test_assert_eq!(stages.len(), 2);
        test_assert_eq!(stages[1].cmd, "grep a");

        // Expansion is later, so nothing a value holds can comment out a line.
        let vars = vars_with(&[("H", "#")]);
        let text = strip_comment("echo a $H b");
        test_assert_eq!(text, "echo a $H b");
        let expanded = expand_vars(text, &vars).unwrap_or_default();
        test_assert_eq!(expanded, "echo a # b");
        test_assert_eq!(filtered(&expanded, ""), "a # b\n");
        TestResult::Pass
    }

    /// `echo` prints its argument text and a newline, ignoring stdin, so a
    /// script has a way to say what it is doing.
    fn test_echo_prints_its_argument() -> TestResult {
        test_assert_eq!(filtered("echo hello there", ""), "hello there\n");
        test_assert_eq!(filtered("echo", ""), "\n");
        test_assert_eq!(filtered("echo  spaced  out", ""), " spaced  out\n");
        // It reads no stdin: what it prints is its argument, not the stream.
        test_assert_eq!(filtered("echo a b", FRUIT), "a b\n");
        test_assert_eq!(filtered("wc -w", &filtered("echo a b c", "")), "3\n");
        TestResult::Pass
    }

    /// A script is a file from the filesystem, so its size is bounded: past
    /// either cap it is refused with a message rather than run in part.
    fn test_script_bounds_refuse_an_over_long_file() -> TestResult {
        // The numbers the handbook prints, pinned to the ones the code uses.
        test_assert_eq!(MAX_SCRIPT_BYTES, 65536);
        test_assert_eq!(MAX_SCRIPT_LINES, 1024);
        test_assert_eq!(script_lines("a\nb\n").unwrap_or_default().len(), 2);
        test_assert_eq!(script_lines("").unwrap_or_default().len(), 0);

        let at_bytes = "x".repeat(MAX_SCRIPT_BYTES);
        test_assert!(
            script_lines(&at_bytes).is_ok(),
            "exactly the byte cap must be allowed"
        );
        let over_bytes = "x".repeat(MAX_SCRIPT_BYTES + 1);
        test_assert!(
            script_lines(&over_bytes).is_err(),
            "one byte over the cap must be refused"
        );

        let at_lines = "x\n".repeat(MAX_SCRIPT_LINES);
        test_assert_eq!(
            script_lines(&at_lines).unwrap_or_default().len(),
            MAX_SCRIPT_LINES
        );
        let over_lines = "x\n".repeat(MAX_SCRIPT_LINES + 1);
        test_assert!(
            script_lines(&over_lines).is_err(),
            "one line over the cap must be refused"
        );
        TestResult::Pass
    }

    /// A script that sources itself terminates at the depth bound, and the
    /// bound is what stops it: one level under it still runs.
    fn test_a_script_that_sources_itself_terminates() -> TestResult {
        // The number the handbook prints, pinned to the one the code uses.
        test_assert_eq!(MAX_SOURCE_DEPTH, 8);
        test_assert!(
            run_script("echo hi", MAX_SOURCE_DEPTH, |_| Ok(String::new())).is_ok(),
            "the last allowed level must still run"
        );
        test_assert!(
            run_script("echo hi", MAX_SOURCE_DEPTH + 1, |_| Ok(String::new())).is_err(),
            "one level past the bound must be refused"
        );

        // The real shape: every line of the script sources the script again,
        // which is what `Shell::cmd_source` does one frame lower.
        fn again(depth: usize, runs: &mut usize) -> Result<String, String> {
            run_script("source loop.sh", depth, |_| {
                *runs += 1;
                again(depth + 1, runs)
            })
        }
        let mut runs = 0usize;
        let out = again(1, &mut runs);
        test_assert!(out.is_err(), "a self-sourcing script must stop");
        test_assert_eq!(runs, MAX_SOURCE_DEPTH);
        TestResult::Pass
    }

    /// A line the shell could not run at all stops the script; a command that
    /// ran and printed an error does not, because this shell cannot tell that
    /// output apart from any other. Line numbers count every line of the file.
    fn test_a_line_that_cannot_run_stops_the_script() -> TestResult {
        let mut ran: Vec<String> = Vec::new();
        let stopped = run_script("one\ntwo\nthree", 1, |line| {
            ran.push(String::from(line));
            if line == "two" {
                Err(String::from("seal: syntax error"))
            } else {
                Ok(String::from(line))
            }
        });
        test_assert_eq!(ran.len(), 2);
        match stopped {
            Ok(_) => return TestResult::Fail("a line that cannot run must stop the script"),
            Err(e) => {
                // What ran before it is still shown, with the message after.
                test_assert_eq!(e, "one\nseal: source: line 2: seal: syntax error");
            }
        }

        // Blank and comment lines are handed on like any other, so the number
        // reported is the line's number in the file.
        let numbered = run_script("a\n\n# c\nboom", 1, |line| {
            if line == "boom" {
                Err(String::from("seal: syntax error"))
            } else {
                Ok(String::new())
            }
        });
        match numbered {
            Ok(_) => return TestResult::Fail("the failing line must stop the script"),
            Err(e) => test_assert!(e.contains("line 4"), "the failing line must be named"),
        }

        // Output text that reads like an error is still output: every line
        // runs, and each one's output is terminated.
        let all = run_script("a\nb", 1, |line| Ok(format!("{}: not found", line)));
        test_assert_eq!(
            all.unwrap_or_default(),
            "a: not found\nb: not found\n"
        );
        // A line that prints nothing adds nothing, not a blank line.
        let quiet = run_script("a\nb", 1, |_| Ok(String::new()));
        test_assert_eq!(quiet.unwrap_or_default(), "");
        // Output that already ends in a newline is not given a second one.
        let ended = run_script("a", 1, |_| Ok(String::from("x\n")));
        test_assert_eq!(ended.unwrap_or_default(), "x\n");
        TestResult::Pass
    }

    /// The handbook names `source`, `echo` and the comment rule, and each
    /// token belongs to exactly one line of it — a token that matches two
    /// lines proves nothing about the line it was meant to check.
    fn test_handbook_documents_source_and_echo() -> TestResult {
        let book = help::handbook();
        for token in ["source <file>", "echo <text>", "# comment"] {
            let lines = book.lines().filter(|l| l.contains(token)).count();
            test_assert_eq!(lines, 1);
        }
        for topic in ["source", ".", "echo"] {
            test_assert!(
                !help::help_for(topic).starts_with("No help available"),
                "source, '.' and echo each need a help page"
            );
        }
        // The page carries the three bounds an operator hits.
        let page = help::help_for("source");
        test_assert!(
            page.contains("8") && page.contains("1024") && page.contains("65536"),
            "the script bounds belong in the handbook"
        );
        TestResult::Pass
    }

    /// The handbook names the syntax, the three commands and both caps. A
    /// feature no `help` mentions cannot be found.
    fn test_handbook_documents_variables() -> TestResult {
        let book = help::handbook();
        // Each token is the distinctive part of one handbook line. A bare
        // `export` would be satisfied by the word `exported` on the `env`
        // line, and a bare `env` by `environment`, so both carry the space
        // that only the command itself is followed by.
        for token in [
            "NAME=value",
            "$NAME",
            "${NAME}",
            "export NAME",
            "unset NAME",
            "env ",
        ] {
            test_assert!(
                book.contains(token),
                "the handbook must document the variable syntax"
            );
        }
        for topic in ["variables", "export", "unset", "env", "set"] {
            test_assert!(
                !help::help_for(topic).starts_with("No help available"),
                "every variable command needs a help page"
            );
        }
        let page = help::help_for("variables");
        test_assert!(
            page.contains("64") && page.contains("4096"),
            "the caps belong in the handbook"
        );
        // `set` gained a second meaning, and a page that documents only the
        // settings half hides it.
        test_assert!(
            help::help_for("set").contains("variable"),
            "the 'set' page must say that a bare 'set' lists the variables"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "shell::comment_starts_only_at_a_word_boundary",
            test_comment_starts_only_at_a_word_boundary,
        );
        crate::testing::register_test(
            "shell::comments_are_cut_before_the_split_and_expansion_after",
            test_comments_are_cut_before_the_split_and_expansion_after,
        );
        crate::testing::register_test("shell::echo_prints_its_argument", test_echo_prints_its_argument);
        crate::testing::register_test(
            "shell::script_bounds_refuse_an_over_long_file",
            test_script_bounds_refuse_an_over_long_file,
        );
        crate::testing::register_test(
            "shell::a_script_that_sources_itself_terminates",
            test_a_script_that_sources_itself_terminates,
        );
        crate::testing::register_test(
            "shell::a_line_that_cannot_run_stops_the_script",
            test_a_line_that_cannot_run_stops_the_script,
        );
        crate::testing::register_test(
            "shell::handbook_documents_source_and_echo",
            test_handbook_documents_source_and_echo,
        );
        crate::testing::register_test(
            "shell::assignment_statements_are_recognized_and_others_fall_through",
            test_assignment_statements_are_recognized_and_others_fall_through,
        );
        crate::testing::register_test(
            "shell::variables_expand_in_arguments",
            test_variables_expand_in_arguments,
        );
        crate::testing::register_test(
            "shell::malformed_expansion_and_empty_result_are_refused",
            test_malformed_expansion_and_empty_result_are_refused,
        );
        crate::testing::register_test(
            "shell::expansion_is_one_pass_and_terminates",
            test_expansion_is_one_pass_and_terminates,
        );
        crate::testing::register_test(
            "shell::expansion_happens_after_the_pipeline_split",
            test_expansion_happens_after_the_pipeline_split,
        );
        crate::testing::register_test(
            "shell::a_value_may_not_hold_a_space",
            test_a_value_may_not_hold_a_space,
        );
        crate::testing::register_test(
            "shell::export_marks_unset_removes_and_lists_stay_sorted",
            test_export_marks_unset_removes_and_lists_stay_sorted,
        );
        crate::testing::register_test(
            "shell::the_variable_store_is_capped",
            test_the_variable_store_is_capped,
        );
        crate::testing::register_test(
            "shell::handbook_documents_variables",
            test_handbook_documents_variables,
        );
        crate::testing::register_test(
            "shell::filter_table_routes_each_name",
            test_filter_table_routes_each_name,
        );
        crate::testing::register_test(
            "shell::wc_counts_lines_words_and_bytes",
            test_wc_counts_lines_words_and_bytes,
        );
        crate::testing::register_test(
            "shell::head_and_tail_take_opposite_ends",
            test_head_and_tail_take_opposite_ends,
        );
        crate::testing::register_test(
            "shell::head_and_tail_refuse_a_bad_count",
            test_head_and_tail_refuse_a_bad_count,
        );
        crate::testing::register_test(
            "shell::sort_orders_dedups_and_reverses",
            test_sort_orders_dedups_and_reverses,
        );
        crate::testing::register_test(
            "shell::uniq_collapses_only_adjacent_duplicates",
            test_uniq_collapses_only_adjacent_duplicates,
        );
        crate::testing::register_test(
            "shell::grep_flags_invert_fold_number_and_count",
            test_grep_flags_invert_fold_number_and_count,
        );
        crate::testing::register_test(
            "shell::tr_translates_one_character_and_refuses_ranges",
            test_tr_translates_one_character_and_refuses_ranges,
        );
        crate::testing::register_test(
            "shell::unknown_filter_flag_is_refused",
            test_unknown_filter_flag_is_refused,
        );
        crate::testing::register_test(
            "shell::handbook_documents_the_filters_and_pipeline_syntax",
            test_handbook_documents_the_filters_and_pipeline_syntax,
        );
        crate::testing::register_test(
            "shell::pipeline_forwards_each_stage_output_as_the_next_stdin",
            test_pipeline_forwards_each_stage_output_as_the_next_stdin,
        );
        crate::testing::register_test(
            "shell::three_stage_pipeline_composes",
            test_three_stage_pipeline_composes,
        );
        crate::testing::register_test(
            "shell::stage_with_no_output_feeds_empty_stdin",
            test_stage_with_no_output_feeds_empty_stdin,
        );
        crate::testing::register_test(
            "shell::pipeline_stage_over_capacity_is_refused",
            test_pipeline_stage_over_capacity_is_refused,
        );
        crate::testing::register_test(
            "shell::stage_parse_separates_command_from_redirection",
            test_stage_parse_separates_command_from_redirection,
        );
        crate::testing::register_test(
            "shell::truncate_replaces_and_append_extends",
            test_truncate_replaces_and_append_extends,
        );
        crate::testing::register_test(
            "shell::line_splits_into_stages",
            test_line_splits_into_stages,
        );
        crate::testing::register_test(
            "shell::malformed_stage_is_a_syntax_error",
            test_malformed_stage_is_a_syntax_error,
        );
        crate::testing::register_test(
            "shell::plain_name_resolves_against_cwd",
            test_plain_name_resolves_against_cwd,
        );
        crate::testing::register_test(
            "shell::empty_arg_is_the_cwd_without_trailing_slash",
            test_empty_arg_is_the_cwd_without_trailing_slash,
        );
        crate::testing::register_test(
            "shell::non_canonical_arguments_are_refused",
            test_non_canonical_arguments_are_refused,
        );
        crate::testing::register_test(
            "shell::root_secret_denied_for_unprivileged_shell",
            test_root_secret_denied_for_unprivileged_shell,
        );
        crate::testing::register_test(
            "shell::traversal_out_of_allowed_directory_denied",
            test_traversal_out_of_allowed_directory_denied,
        );
        crate::testing::register_test(
            "shell::allowed_paths_and_root_are_not_broken",
            test_allowed_paths_and_root_are_not_broken,
        );
        crate::testing::register_test(
            "shell::sibling_prefix_is_not_denied",
            test_sibling_prefix_is_not_denied,
        );
    }
}
