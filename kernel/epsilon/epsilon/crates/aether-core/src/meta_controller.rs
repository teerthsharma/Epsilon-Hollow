// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! no_std meta-controller policy over fixed state vectors and slices.

use libm::{exp, sqrt, tanh};

/// Number of actions in the meta-controller policy.
pub const ACTION_COUNT: usize = 4;
/// Maximum number of tools considered by the bounded no_std registry.
pub const MAX_TOOL_COUNT: usize = 16;

const MIN_TEMPERATURE: f64 = 1e-3;
const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Errors returned by slice-oriented meta-controller helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaError {
    /// State length does not match the policy dimension.
    StateDimensionMismatch,
    /// Weight matrix does not have `state_len * ACTION_COUNT` entries.
    WeightDimensionMismatch,
    /// Bias slice does not contain `ACTION_COUNT` entries.
    BiasDimensionMismatch,
    /// Output buffer length is invalid.
    OutputDimensionMismatch,
    /// Tool registry exceeds the bounded controller capacity.
    ToolRegistryTooLarge,
}

/// Result type for fallible meta-controller helpers.
pub type MetaResult<T> = Result<T, MetaError>;

/// Counters exposed by the constitutional safety filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafetyStats {
    /// Total compliance checks performed.
    pub total_checks: u64,
    /// Number of action texts rejected by policy.
    pub violations_blocked: u64,
}

/// no_std text safety filter replacing the legacy host meta-controller guard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstitutionalSafetyFilter {
    entropy_threshold: f64,
    stats: SafetyStats,
}

impl ConstitutionalSafetyFilter {
    /// Construct a safety filter with a normalized entropy threshold.
    pub const fn new(entropy_threshold: f64) -> Self {
        Self {
            entropy_threshold,
            stats: SafetyStats {
                total_checks: 0,
                violations_blocked: 0,
            },
        }
    }

    /// Check a text action against hard constitutional denial patterns.
    pub fn check_compliance_text<const D: usize>(
        &mut self,
        state: &[f64; D],
        action_text: &str,
    ) -> bool {
        self.stats.total_checks = self.stats.total_checks.saturating_add(1);
        let lower = action_text.as_bytes();
        if contains_ascii_case_insensitive(lower, b"rm -rf")
            || contains_ascii_case_insensitive(lower, b"format")
            || contains_ascii_case_insensitive(lower, b"drop table")
            || contains_ascii_case_insensitive(lower, b"exec(")
            || contains_ascii_case_insensitive(lower, b"eval(")
            || contains_ascii_case_insensitive(lower, b"os.system")
            || contains_ascii_case_insensitive(lower, b"subprocess.call")
            || contains_ascii_case_insensitive(lower, b"__import__")
            || contains_ascii_case_insensitive(lower, b"shutil.rmtree")
        {
            self.stats.violations_blocked = self.stats.violations_blocked.saturating_add(1);
            return false;
        }

        let entropy = normalized_abs_entropy(state);
        if entropy < self.entropy_threshold {
            self.stats.violations_blocked = self.stats.violations_blocked.saturating_add(1);
            return false;
        }
        true
    }

    /// Return accumulated safety counters.
    pub const fn stats(&self) -> SafetyStats {
        self.stats
    }

    /// Return the minimum normalized entropy required for compliant actions.
    pub const fn entropy_threshold(&self) -> f64 {
        self.entropy_threshold
    }
}

/// Typed actions produced by the meta-controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Use an available execution tool.
    Execute,
    /// Continue reasoning internally.
    Reason,
    /// Explore the memory frontier.
    Explore,
    /// Refuse the action.
    Refusal,
}

impl Action {
    /// Return the stable action index used by policy vectors.
    pub const fn index(self) -> usize {
        match self {
            Self::Execute => 0,
            Self::Reason => 1,
            Self::Explore => 2,
            Self::Refusal => 3,
        }
    }

    /// Convert a policy index into an action.
    pub const fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Execute,
            1 => Self::Reason,
            2 => Self::Explore,
            _ => Self::Refusal,
        }
    }
}

/// Structured reason for the final decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    /// The sampled policy action was used directly.
    PolicySelected,
    /// Execution was sampled but no tool was available.
    NoToolAvailable,
    /// The caller-provided safety gate rejected the sampled action.
    SafetyViolation,
    /// Execution was sampled but every candidate tool was filtered out.
    NoCompliantTool,
}

/// Structured target selected by the final decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionTarget {
    /// No external target.
    None,
    /// Internal reasoning model.
    InternalModel,
    /// Memory exploration frontier.
    MemoryFrontier,
    /// Tool slot selected by index.
    Tool(usize),
}

/// Output of a meta-controller decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    /// Final action after safety and availability fallbacks.
    pub action: Action,
    /// Raw action sampled from the policy before fallbacks.
    pub sampled_action: Action,
    /// Softmax probabilities in `Action` index order.
    pub probabilities: [f64; ACTION_COUNT],
    /// Decision target, if the action has one.
    pub target: DecisionTarget,
    /// Explanation code for the final action.
    pub reason: DecisionReason,
    /// Selected tool index for `Action::Execute`.
    pub tool_index: Option<usize>,
}

/// Static metadata for a tool the meta-controller may execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolSpec<'a> {
    /// Stable tool name used by the execution host.
    pub name: &'a str,
    /// Short text checked by the safety filter before execution.
    pub action_text: &'a str,
    /// Opaque payload kind understood by the host-side executor.
    pub payload_kind: &'a str,
    /// Whether the tool is expected to avoid external mutation.
    pub read_only: bool,
    /// Small relative execution cost used as a tie-breaker.
    pub estimated_cost: u8,
}

impl<'a> ToolSpec<'a> {
    /// Construct static tool metadata.
    pub const fn new(
        name: &'a str,
        action_text: &'a str,
        payload_kind: &'a str,
        read_only: bool,
        estimated_cost: u8,
    ) -> Self {
        Self {
            name,
            action_text,
            payload_kind,
            read_only,
            estimated_cost,
        }
    }
}

/// Bounded borrowed tool registry for no_std execution routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolRegistry<'a> {
    tools: &'a [ToolSpec<'a>],
}

impl<'a> ToolRegistry<'a> {
    /// Construct a bounded borrowed registry.
    pub const fn new(tools: &'a [ToolSpec<'a>]) -> MetaResult<Self> {
        if tools.len() > MAX_TOOL_COUNT {
            Err(MetaError::ToolRegistryTooLarge)
        } else {
            Ok(Self { tools })
        }
    }

    /// Return the number of registered tools.
    pub const fn len(&self) -> usize {
        self.tools.len()
    }

    /// Return true when the registry contains no tools.
    pub const fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Return the tool at a registry index.
    pub fn get(&self, index: usize) -> Option<&'a ToolSpec<'a>> {
        self.tools.get(index)
    }

    /// Return all tools in registration order.
    pub const fn tools(&self) -> &'a [ToolSpec<'a>] {
        self.tools
    }
}

/// Execution payload metadata selected for a host tool invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionPayloadMetadata<'a> {
    /// Stable tool name selected by policy.
    pub tool_name: &'a str,
    /// Opaque payload kind understood by the host-side executor.
    pub payload_kind: &'a str,
    /// Registry index of the selected tool.
    pub tool_index: usize,
    /// Whether the selected tool is expected to avoid external mutation.
    pub read_only: bool,
    /// Small relative execution cost used during selection.
    pub estimated_cost: u8,
}

/// Tool-aware decision preserving the legacy `Decision` payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToolDecision<'a> {
    /// Base action decision, including probabilities and fallback reason.
    pub decision: Decision,
    /// Selected execution metadata when `decision.action == Action::Execute`.
    pub payload: Option<ExecutionPayloadMetadata<'a>>,
}

/// Borrowed inputs for a meta-controller decision.
#[derive(Debug, Clone, Copy)]
pub struct MetaInput<'a> {
    /// Latent state slice copied into the fixed policy state.
    pub latent: &'a [f64],
    /// Number of context entries available to the controller.
    pub context_len: usize,
    /// Monotonic step value used as a small control signal.
    pub step: f64,
    /// Number of execution tools available to the caller.
    pub tool_count: usize,
    /// Caller-provided safety result for the sampled decision.
    pub safety_compliant: bool,
}

/// Deterministic xorshift64 generator used by the policy sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Construct a generator, replacing zero with a fixed non-zero seed.
    pub const fn new(seed: u64) -> Self {
        let state = if seed == 0 { DEFAULT_SEED } else { seed };
        Self { state }
    }

    /// Return the internal generator state.
    pub const fn state(&self) -> u64 {
        self.state
    }

    /// Generate the next `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a floating-point value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    fn next_signed_f64(&mut self) -> f64 {
        (self.next_f64() * 2.0) - 1.0
    }
}

/// Deterministic linear softmax policy for meta-level routing.
#[derive(Debug, Clone)]
pub struct MetaController<const D: usize> {
    weights: [[f64; ACTION_COUNT]; D],
    bias: [f64; ACTION_COUNT],
    temperature: f64,
    rng: XorShift64,
}

impl<const D: usize> MetaController<D> {
    /// Construct a controller with deterministic xorshift-initialized weights.
    pub fn new(seed: u64, temperature: f64) -> Self {
        let mut rng = XorShift64::new(seed);
        let scale = sqrt(2.0 / ((D + ACTION_COUNT) as f64));
        let mut weights = [[0.0; ACTION_COUNT]; D];

        for row in &mut weights {
            for value in row {
                *value = rng.next_signed_f64() * scale;
            }
        }

        let mut bias = [0.0; ACTION_COUNT];
        bias[Action::Execute.index()] = 0.20;
        bias[Action::Reason.index()] = 0.35;
        bias[Action::Refusal.index()] = -0.70;

        Self {
            weights,
            bias,
            temperature: sanitize_temperature(temperature),
            rng,
        }
    }

    /// Return the current temperature.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Return the deterministic RNG state.
    pub fn rng_state(&self) -> u64 {
        self.rng.state()
    }

    /// Return immutable policy weights.
    pub fn weights(&self) -> &[[f64; ACTION_COUNT]; D] {
        &self.weights
    }

    /// Return mutable policy weights for training updates.
    pub fn weights_mut(&mut self) -> &mut [[f64; ACTION_COUNT]; D] {
        &mut self.weights
    }

    /// Return immutable policy bias values.
    pub fn bias(&self) -> &[f64; ACTION_COUNT] {
        &self.bias
    }

    /// Return mutable policy bias values for training updates.
    pub fn bias_mut(&mut self) -> &mut [f64; ACTION_COUNT] {
        &mut self.bias
    }

    /// Set all policy weights to zero.
    pub fn clear_weights(&mut self) {
        for row in &mut self.weights {
            row.fill(0.0);
        }
    }

    /// Set one weight value by row and action.
    pub fn set_weight(&mut self, row: usize, action: Action, value: f64) -> MetaResult<()> {
        if row >= D {
            return Err(MetaError::StateDimensionMismatch);
        }
        self.weights[row][action.index()] = value;
        Ok(())
    }

    /// Set one action bias value.
    pub fn set_bias(&mut self, action: Action, value: f64) {
        self.bias[action.index()] = value;
    }

    /// Extract a fixed policy state from a latent slice and control signals.
    pub fn extract_state(&self, latent: &[f64], context_len: usize, step: f64) -> [f64; D] {
        let mut state = [0.0; D];
        let copy_len = D.min(latent.len());
        state[..copy_len].copy_from_slice(&latent[..copy_len]);

        let state_norm = vector_norm(&state);
        if state_norm > 1e-12 {
            for value in &mut state {
                *value /= state_norm;
            }
        }

        if D > 0 {
            state[0] += context_len.min(20) as f64 / 20.0;
        }
        if D > 1 {
            state[1] += tanh(step / 1000.0);
        }

        state
    }

    /// Compute policy probabilities from a fixed-size state vector.
    pub fn policy_probabilities(
        &self,
        state: &[f64; D],
        tool_count: usize,
        context_empty: bool,
    ) -> [f64; ACTION_COUNT] {
        let logits = self.policy_logits(state, tool_count, context_empty);
        self.softmax(logits)
    }

    /// Compute policy probabilities from a state slice.
    pub fn policy_probabilities_from_slice(
        &self,
        state: &[f64],
        tool_count: usize,
        context_empty: bool,
    ) -> MetaResult<[f64; ACTION_COUNT]> {
        if state.len() != D {
            return Err(MetaError::StateDimensionMismatch);
        }

        let mut logits = self.bias;
        for (i, state_value) in state.iter().enumerate() {
            for (action, logit) in logits.iter_mut().enumerate().take(ACTION_COUNT) {
                *logit += *state_value * self.weights[i][action];
            }
        }
        adjust_logits(&mut logits, tool_count, context_empty);
        Ok(self.softmax(logits))
    }

    /// Compute stable softmax probabilities for action logits.
    pub fn softmax(&self, logits: [f64; ACTION_COUNT]) -> [f64; ACTION_COUNT] {
        softmax_fixed(logits, self.temperature)
    }

    /// Produce a typed decision from borrowed latent-state inputs.
    pub fn decide(&mut self, input: MetaInput<'_>) -> Decision {
        let state = self.extract_state(input.latent, input.context_len, input.step);
        self.decide_from_state(
            &state,
            input.tool_count,
            input.context_len == 0,
            input.safety_compliant,
        )
    }

    /// Produce a typed decision from an already extracted state vector.
    pub fn decide_from_state(
        &mut self,
        state: &[f64; D],
        tool_count: usize,
        context_empty: bool,
        safety_compliant: bool,
    ) -> Decision {
        let probabilities = self.policy_probabilities(state, tool_count, context_empty);
        let sampled_action = Action::from_index(sample_index(&mut self.rng, &probabilities));

        if !safety_compliant {
            return Decision {
                action: Action::Refusal,
                sampled_action,
                probabilities,
                target: DecisionTarget::None,
                reason: DecisionReason::SafetyViolation,
                tool_index: None,
            };
        }

        match sampled_action {
            Action::Execute if tool_count > 0 => Decision {
                action: Action::Execute,
                sampled_action,
                probabilities,
                target: DecisionTarget::Tool(0),
                reason: DecisionReason::PolicySelected,
                tool_index: Some(0),
            },
            Action::Execute => Decision {
                action: Action::Reason,
                sampled_action,
                probabilities,
                target: DecisionTarget::InternalModel,
                reason: DecisionReason::NoToolAvailable,
                tool_index: None,
            },
            Action::Explore => Decision {
                action: Action::Explore,
                sampled_action,
                probabilities,
                target: DecisionTarget::MemoryFrontier,
                reason: DecisionReason::PolicySelected,
                tool_index: None,
            },
            Action::Refusal => Decision {
                action: Action::Refusal,
                sampled_action,
                probabilities,
                target: DecisionTarget::None,
                reason: DecisionReason::PolicySelected,
                tool_index: None,
            },
            Action::Reason => Decision {
                action: Action::Reason,
                sampled_action,
                probabilities,
                target: DecisionTarget::InternalModel,
                reason: DecisionReason::PolicySelected,
                tool_index: None,
            },
        }
    }

    /// Produce a tool-aware decision from borrowed latent-state inputs.
    pub fn decide_with_tools<'a>(
        &mut self,
        input: MetaInput<'_>,
        registry: ToolRegistry<'a>,
    ) -> ToolDecision<'a> {
        let state = self.extract_state(input.latent, input.context_len, input.step);
        self.decide_from_state_with_tools(
            &state,
            input.context_len == 0,
            input.safety_compliant,
            registry,
        )
    }

    /// Produce a tool-aware decision from an extracted state vector.
    pub fn decide_from_state_with_tools<'a>(
        &mut self,
        state: &[f64; D],
        context_empty: bool,
        safety_compliant: bool,
        registry: ToolRegistry<'a>,
    ) -> ToolDecision<'a> {
        let tool_count = registry.len();
        let probabilities = self.policy_probabilities(state, tool_count, context_empty);
        let sampled_action = Action::from_index(sample_index(&mut self.rng, &probabilities));

        if !safety_compliant {
            return ToolDecision {
                decision: Decision {
                    action: Action::Refusal,
                    sampled_action,
                    probabilities,
                    target: DecisionTarget::None,
                    reason: DecisionReason::SafetyViolation,
                    tool_index: None,
                },
                payload: None,
            };
        }

        match sampled_action {
            Action::Execute => self.execute_tool_decision(state, registry, probabilities),
            Action::Explore => ToolDecision {
                decision: Decision {
                    action: Action::Explore,
                    sampled_action,
                    probabilities,
                    target: DecisionTarget::MemoryFrontier,
                    reason: DecisionReason::PolicySelected,
                    tool_index: None,
                },
                payload: None,
            },
            Action::Refusal => ToolDecision {
                decision: Decision {
                    action: Action::Refusal,
                    sampled_action,
                    probabilities,
                    target: DecisionTarget::None,
                    reason: DecisionReason::PolicySelected,
                    tool_index: None,
                },
                payload: None,
            },
            Action::Reason => ToolDecision {
                decision: Decision {
                    action: Action::Reason,
                    sampled_action,
                    probabilities,
                    target: DecisionTarget::InternalModel,
                    reason: DecisionReason::PolicySelected,
                    tool_index: None,
                },
                payload: None,
            },
        }
    }

    fn policy_logits(
        &self,
        state: &[f64; D],
        tool_count: usize,
        context_empty: bool,
    ) -> [f64; ACTION_COUNT] {
        let mut logits = self.bias;
        for (i, state_value) in state.iter().enumerate() {
            for (action, logit) in logits.iter_mut().enumerate().take(ACTION_COUNT) {
                *logit += *state_value * self.weights[i][action];
            }
        }
        adjust_logits(&mut logits, tool_count, context_empty);
        logits
    }

    fn execute_tool_decision<'a>(
        &mut self,
        state: &[f64; D],
        registry: ToolRegistry<'a>,
        probabilities: [f64; ACTION_COUNT],
    ) -> ToolDecision<'a> {
        match select_tool(state, registry) {
            Some((tool_index, tool)) => ToolDecision {
                decision: Decision {
                    action: Action::Execute,
                    sampled_action: Action::Execute,
                    probabilities,
                    target: DecisionTarget::Tool(tool_index),
                    reason: DecisionReason::PolicySelected,
                    tool_index: Some(tool_index),
                },
                payload: Some(ExecutionPayloadMetadata {
                    tool_name: tool.name,
                    payload_kind: tool.payload_kind,
                    tool_index,
                    read_only: tool.read_only,
                    estimated_cost: tool.estimated_cost,
                }),
            },
            None if registry.is_empty() => ToolDecision {
                decision: Decision {
                    action: Action::Reason,
                    sampled_action: Action::Execute,
                    probabilities,
                    target: DecisionTarget::InternalModel,
                    reason: DecisionReason::NoToolAvailable,
                    tool_index: None,
                },
                payload: None,
            },
            None => ToolDecision {
                decision: Decision {
                    action: Action::Refusal,
                    sampled_action: Action::Execute,
                    probabilities,
                    target: DecisionTarget::None,
                    reason: DecisionReason::NoCompliantTool,
                    tool_index: None,
                },
                payload: None,
            },
        }
    }
}

impl<const D: usize> Default for MetaController<D> {
    fn default() -> Self {
        Self::new(42, 0.8)
    }
}

/// Extract a normalized state into caller-provided storage.
pub fn extract_state_into(
    latent: &[f64],
    context_len: usize,
    step: f64,
    out: &mut [f64],
) -> MetaResult<()> {
    out.fill(0.0);
    let copy_len = out.len().min(latent.len());
    out[..copy_len].copy_from_slice(&latent[..copy_len]);

    let state_norm = vector_norm(out);
    if state_norm > 1e-12 {
        for value in out.iter_mut() {
            *value /= state_norm;
        }
    }

    if !out.is_empty() {
        out[0] += context_len.min(20) as f64 / 20.0;
    }
    if out.len() > 1 {
        out[1] += tanh(step / 1000.0);
    }

    Ok(())
}

/// Compute action probabilities from row-major slice policy parts.
pub fn policy_probabilities_from_parts(
    state: &[f64],
    weights_row_major: &[f64],
    bias: &[f64],
    temperature: f64,
    tool_count: usize,
    context_empty: bool,
) -> MetaResult<[f64; ACTION_COUNT]> {
    if weights_row_major.len() != state.len() * ACTION_COUNT {
        return Err(MetaError::WeightDimensionMismatch);
    }
    if bias.len() != ACTION_COUNT {
        return Err(MetaError::BiasDimensionMismatch);
    }

    let mut logits = [0.0; ACTION_COUNT];
    logits.copy_from_slice(bias);

    for (row, state_value) in state.iter().enumerate() {
        let offset = row * ACTION_COUNT;
        for action in 0..ACTION_COUNT {
            logits[action] += *state_value * weights_row_major[offset + action];
        }
    }

    adjust_logits(&mut logits, tool_count, context_empty);
    Ok(softmax_fixed(logits, temperature))
}

/// Compute a stable softmax over arbitrary slices.
pub fn softmax_slice_into(logits: &[f64], temperature: f64, out: &mut [f64]) -> MetaResult<()> {
    if logits.is_empty() || logits.len() != out.len() {
        return Err(MetaError::OutputDimensionMismatch);
    }

    let temperature = sanitize_temperature(temperature);
    let mut max_logit = f64::NEG_INFINITY;
    for &logit in logits {
        let z = logit / temperature;
        if z > max_logit {
            max_logit = z;
        }
    }

    let mut total = 0.0;
    for (idx, value) in out.iter_mut().enumerate() {
        *value = exp((logits[idx] / temperature) - max_logit);
        total += *value;
    }

    if total <= 0.0 {
        let uniform = 1.0 / out.len() as f64;
        out.fill(uniform);
        return Ok(());
    }

    for value in out {
        *value /= total;
    }
    Ok(())
}

fn adjust_logits(logits: &mut [f64; ACTION_COUNT], tool_count: usize, context_empty: bool) {
    if tool_count == 0 {
        logits[Action::Execute.index()] -= 1.5;
    }
    if context_empty {
        logits[Action::Explore.index()] += 0.2;
    }
}

fn sample_index(rng: &mut XorShift64, probabilities: &[f64; ACTION_COUNT]) -> usize {
    let threshold = rng.next_f64();
    let mut cumulative = 0.0;
    for (idx, probability) in probabilities.iter().enumerate() {
        cumulative += *probability;
        if threshold <= cumulative {
            return idx;
        }
    }
    ACTION_COUNT - 1
}

fn select_tool<'a, const D: usize>(
    state: &[f64; D],
    registry: ToolRegistry<'a>,
) -> Option<(usize, &'a ToolSpec<'a>)> {
    let mut best: Option<(usize, &ToolSpec<'a>, u64)> = None;
    for (idx, tool) in registry.tools().iter().enumerate() {
        if !tool_is_compliant(tool) {
            continue;
        }

        let score = tool_score(state, tool, idx);
        match best {
            Some((_, best_tool, best_score))
                if score < best_score
                    || (score == best_score && tool.estimated_cost >= best_tool.estimated_cost) => {
            }
            _ => best = Some((idx, tool, score)),
        }
    }
    best.map(|(idx, tool, _)| (idx, tool))
}

fn tool_is_compliant(tool: &ToolSpec<'_>) -> bool {
    !tool.name.is_empty()
        && !tool.payload_kind.is_empty()
        && !contains_ascii_case_insensitive(tool.action_text.as_bytes(), b"rm -rf")
        && !contains_ascii_case_insensitive(tool.action_text.as_bytes(), b"format")
        && !contains_ascii_case_insensitive(tool.action_text.as_bytes(), b"drop table")
        && !contains_ascii_case_insensitive(tool.action_text.as_bytes(), b"exec(")
        && !contains_ascii_case_insensitive(tool.action_text.as_bytes(), b"eval(")
        && !contains_ascii_case_insensitive(tool.action_text.as_bytes(), b"os.system")
        && !contains_ascii_case_insensitive(tool.action_text.as_bytes(), b"subprocess.call")
        && !contains_ascii_case_insensitive(tool.action_text.as_bytes(), b"__import__")
        && !contains_ascii_case_insensitive(tool.action_text.as_bytes(), b"shutil.rmtree")
}

fn tool_score<const D: usize>(state: &[f64; D], tool: &ToolSpec<'_>, index: usize) -> u64 {
    let mut hash = fnv1a(tool.name.as_bytes()) ^ fnv1a(tool.payload_kind.as_bytes());
    hash ^= (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    hash ^= (tool.estimated_cost as u64) << 48;

    for (idx, value) in state.iter().enumerate() {
        let bucket = if value.is_finite() {
            libm::floor(libm::fabs(*value) * 1024.0) as u64
        } else {
            0
        };
        hash ^= bucket
            .wrapping_add(idx as u64)
            .wrapping_mul(0x1000_0000_01B3);
    }

    if tool.read_only {
        hash & !(1u64 << 63)
    } else {
        hash | (1u64 << 63)
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01B3);
    }
    hash
}

fn softmax_fixed(logits: [f64; ACTION_COUNT], temperature: f64) -> [f64; ACTION_COUNT] {
    let mut out = [0.0; ACTION_COUNT];
    let _ = softmax_slice_into(&logits, temperature, &mut out);
    out
}

fn sanitize_temperature(temperature: f64) -> f64 {
    if temperature.is_finite() && temperature >= MIN_TEMPERATURE {
        temperature
    } else {
        MIN_TEMPERATURE
    }
}

fn normalized_abs_entropy<const D: usize>(state: &[f64; D]) -> f64 {
    if D <= 1 {
        return 0.0;
    }
    let mut total = 0.0;
    for value in state {
        total += libm::fabs(*value);
    }
    if total <= 1e-30 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for value in state {
        let p = libm::fabs(*value) / total;
        if p > 1e-30 {
            entropy -= p * (libm::log(p) / libm::log(2.0));
        }
    }
    let max_entropy = libm::log(D as f64) / libm::log(2.0);
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}

fn contains_ascii_case_insensitive(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }

    let mut i = 0usize;
    while i + needle.len() <= haystack.len() {
        let mut matched = true;
        let mut j = 0usize;
        while j < needle.len() {
            if to_ascii_lower(haystack[i + j]) != to_ascii_lower(needle[j]) {
                matched = false;
                break;
            }
            j += 1;
        }
        if matched {
            return true;
        }
        i += 1;
    }
    false
}

const fn to_ascii_lower(byte: u8) -> u8 {
    if byte >= b'A' && byte <= b'Z' {
        byte + 32
    } else {
        byte
    }
}

fn vector_norm(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    for value in values {
        sum += value * value;
    }
    sqrt(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) {
        assert!(
            (a - b).abs() <= tolerance,
            "expected {a} to be within {tolerance} of {b}"
        );
    }

    #[test]
    fn softmax_is_normalized_and_stable() {
        let controller = MetaController::<4>::new(7, 0.8);
        let probs = controller.softmax([1000.0, 999.0, 998.0, 997.0]);
        let total: f64 = probs.iter().sum();

        close(total, 1.0, 1e-12);
        assert!(probs[Action::Execute.index()] > probs[Action::Refusal.index()]);
    }

    #[test]
    fn state_extraction_normalizes_latent_and_adds_control_signals() {
        let controller = MetaController::<4>::new(7, 0.8);
        let state = controller.extract_state(&[3.0, 4.0, 12.0, 0.0, 99.0], 10, 1000.0);

        close(state[0], 3.0 / 13.0 + 0.5, 1e-12);
        close(state[1], 4.0 / 13.0 + libm::tanh(1.0), 1e-12);
        close(state[2], 12.0 / 13.0, 1e-12);
        close(state[3], 0.0, 1e-12);
    }

    #[test]
    fn policy_downweights_execution_without_tools_and_empty_context_explores() {
        let mut controller = MetaController::<2>::new(11, 0.8);
        controller.clear_weights();
        controller.set_bias(Action::Execute, 0.2);
        controller.set_bias(Action::Reason, 0.35);
        controller.set_bias(Action::Explore, 0.0);
        controller.set_bias(Action::Refusal, -0.7);

        let with_tools = controller.policy_probabilities(&[1.0, 0.0], 1, false);
        let without_tools = controller.policy_probabilities(&[1.0, 0.0], 0, true);

        assert!(without_tools[Action::Execute.index()] < with_tools[Action::Execute.index()]);
        assert!(without_tools[Action::Explore.index()] > with_tools[Action::Explore.index()]);
    }

    #[test]
    fn deterministic_rng_replays_decisions_for_same_seed() {
        let mut a = MetaController::<3>::new(42, 0.8);
        let mut b = MetaController::<3>::new(42, 0.8);
        let input = MetaInput {
            latent: &[0.25, -0.5, 0.75],
            context_len: 0,
            step: 7.0,
            tool_count: 2,
            safety_compliant: true,
        };

        for _ in 0..8 {
            assert_eq!(a.decide(input), b.decide(input));
        }
    }

    #[test]
    fn safety_violation_forces_refusal_but_preserves_policy_probabilities() {
        let mut controller = MetaController::<2>::new(99, 0.8);
        let decision = controller.decide(MetaInput {
            latent: &[1.0, 0.0],
            context_len: 1,
            step: 0.0,
            tool_count: 1,
            safety_compliant: false,
        });

        assert_eq!(decision.action, Action::Refusal);
        assert_eq!(decision.reason, DecisionReason::SafetyViolation);
        close(decision.probabilities.iter().sum::<f64>(), 1.0, 1e-12);
    }

    #[test]
    fn bounded_registry_rejects_more_than_max_tools() {
        const TOOL: ToolSpec<'static> =
            ToolSpec::new("reason", "summarize context", "text", true, 1);
        let tools = [TOOL; MAX_TOOL_COUNT + 1];

        assert_eq!(
            ToolRegistry::new(&tools),
            Err(MetaError::ToolRegistryTooLarge)
        );
    }

    #[test]
    fn tool_aware_execution_selects_name_and_payload_metadata() {
        let mut controller = MetaController::<3>::new(2, MIN_TEMPERATURE);
        controller.clear_weights();
        controller.set_bias(Action::Execute, 10.0);
        controller.set_bias(Action::Reason, -10.0);
        controller.set_bias(Action::Explore, -10.0);
        controller.set_bias(Action::Refusal, -10.0);

        let tools = [
            ToolSpec::new(
                "shell.read",
                "inspect directory",
                "filesystem.read",
                true,
                3,
            ),
            ToolSpec::new(
                "memory.scan",
                "scan memory frontier",
                "memory.query",
                true,
                1,
            ),
        ];
        let registry = ToolRegistry::new(&tools).unwrap();
        let decision =
            controller.decide_from_state_with_tools(&[0.25, -0.75, 0.5], false, true, registry);
        let payload = decision
            .payload
            .expect("execution should carry payload metadata");

        assert_eq!(decision.decision.action, Action::Execute);
        assert_eq!(
            decision.decision.target,
            DecisionTarget::Tool(payload.tool_index)
        );
        assert_eq!(decision.decision.tool_index, Some(payload.tool_index));
        assert_eq!(payload.tool_name, tools[payload.tool_index].name);
        assert_eq!(payload.payload_kind, tools[payload.tool_index].payload_kind);
        assert!(payload.read_only);
        assert!(payload.tool_name == "shell.read" || payload.tool_name == "memory.scan");
    }

    #[test]
    fn unsafe_tool_text_is_filtered_before_execution() {
        let mut controller = MetaController::<2>::new(3, MIN_TEMPERATURE);
        controller.clear_weights();
        controller.set_bias(Action::Execute, 10.0);
        controller.set_bias(Action::Reason, -10.0);
        controller.set_bias(Action::Explore, -10.0);
        controller.set_bias(Action::Refusal, -10.0);

        let tools = [
            ToolSpec::new("danger", "rm -rf /tmp/cache", "shell.command", false, 1),
            ToolSpec::new("lookup", "inspect index", "search.query", true, 2),
        ];
        let decision = controller.decide_from_state_with_tools(
            &[1.0, 0.0],
            false,
            true,
            ToolRegistry::new(&tools).unwrap(),
        );

        assert_eq!(decision.decision.action, Action::Execute);
        assert_eq!(decision.payload.unwrap().tool_name, "lookup");
    }

    #[test]
    fn all_filtered_tools_refuse_instead_of_accepting_any_action() {
        let mut controller = MetaController::<2>::new(4, MIN_TEMPERATURE);
        controller.clear_weights();
        controller.set_bias(Action::Execute, 10.0);
        controller.set_bias(Action::Reason, -10.0);
        controller.set_bias(Action::Explore, -10.0);
        controller.set_bias(Action::Refusal, -10.0);

        let tools = [
            ToolSpec::new("danger-a", "eval(user_code)", "python.eval", false, 1),
            ToolSpec::new("danger-b", "drop table users", "sql.write", false, 1),
        ];
        let decision = controller.decide_from_state_with_tools(
            &[1.0, 0.0],
            false,
            true,
            ToolRegistry::new(&tools).unwrap(),
        );

        assert_eq!(decision.decision.action, Action::Refusal);
        assert_eq!(decision.decision.reason, DecisionReason::NoCompliantTool);
        assert_eq!(decision.payload, None);
    }

    #[test]
    fn legacy_tool_count_decision_still_targets_first_tool_slot() {
        let mut controller = MetaController::<2>::new(5, MIN_TEMPERATURE);
        controller.clear_weights();
        controller.set_bias(Action::Execute, 10.0);
        controller.set_bias(Action::Reason, -10.0);
        controller.set_bias(Action::Explore, -10.0);
        controller.set_bias(Action::Refusal, -10.0);

        let decision = controller.decide_from_state(&[1.0, 0.0], 2, false, true);

        assert_eq!(decision.action, Action::Execute);
        assert_eq!(decision.target, DecisionTarget::Tool(0));
        assert_eq!(decision.tool_index, Some(0));
    }
}
