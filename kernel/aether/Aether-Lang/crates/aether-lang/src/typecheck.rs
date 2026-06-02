// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

// ═══════════════════════════════════════════════════════════════════════════════
//! Hindley-Milner Type System for AEGIS
// ═══════════════════════════════════════════════════════════════════════════════
//!
//! Provides type inference and unification so the bootstrap compiler can
//! reason about types before execution.
//!
//! Core components:
//! - `Type`: AST type representations
//! - `TypeEnv`: Variable-to-type mapping
//! - `unify`: Robinson unification with occurs-check
//! - `infer_expr` / `typecheck_program`: Top-level inference
// ═══════════════════════════════════════════════════════════════════════════════

extern crate alloc;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::ast::*;

// ═══════════════════════════════════════════════════════════════════════════════
// Errors
// ═══════════════════════════════════════════════════════════════════════════════

/// Type inference errors
#[derive(Debug, Clone, PartialEq)]
pub enum TypeError {
    UnboundVariable(String),
    UnificationFail(Type, Type),
    OccursCheck(String, Type),
    UnsupportedExpression(String),
}

// ═══════════════════════════════════════════════════════════════════════════════
// Type Variable Generation
// ═══════════════════════════════════════════════════════════════════════════════

/// Generates fresh type variables (`t0`, `t1`, ...)
pub struct TypeVarGen(usize);

impl TypeVarGen {
    pub fn new() -> Self {
        Self(0)
    }
    pub fn fresh(&mut self) -> Type {
        let id = self.0;
        self.0 += 1;
        Type::Var(format!("t{}", id))
    }
}

impl Default for TypeVarGen {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Substitution & Unification
// ═══════════════════════════════════════════════════════════════════════════════

/// Substitution mapping type variable names to resolved types
pub type Subst = BTreeMap<String, Type>;

/// Apply a substitution to a type, recursively resolving variables
pub fn apply_subst(ty: &Type, subst: &Subst) -> Type {
    match ty {
        Type::Var(name) => {
            if let Some(t) = subst.get(name) {
                apply_subst(t, subst)
            } else {
                ty.clone()
            }
        }
        Type::List(inner) => Type::List(Box::new(apply_subst(inner, subst))),
        Type::Fun(params, ret) => Type::Fun(
            params.iter().map(|p| apply_subst(p, subst)).collect(),
            Box::new(apply_subst(ret, subst)),
        ),
        _ => ty.clone(),
    }
}

/// Unify two types under the current substitution (Robinson unification)
pub fn unify(t1: &Type, t2: &Type, subst: &mut Subst) -> Result<(), TypeError> {
    let t1 = apply_subst(t1, subst);
    let t2 = apply_subst(t2, subst);

    if t1 == t2 {
        return Ok(());
    }

    match (&t1, &t2) {
        (Type::Var(name), _) => {
            if occurs_check(name, &t2) {
                return Err(TypeError::OccursCheck(name.clone(), t2.clone()));
            }
            subst.insert(name.clone(), t2.clone());
            Ok(())
        }
        (_, Type::Var(name)) => {
            if occurs_check(name, &t1) {
                return Err(TypeError::OccursCheck(name.clone(), t1.clone()));
            }
            subst.insert(name.clone(), t1.clone());
            Ok(())
        }
        (Type::List(a), Type::List(b)) => unify(a, b, subst),
        (Type::Fun(p1, r1), Type::Fun(p2, r2)) => {
            if p1.len() != p2.len() {
                return Err(TypeError::UnificationFail(t1.clone(), t2.clone()));
            }
            for (a, b) in p1.iter().zip(p2.iter()) {
                unify(a, b, subst)?;
            }
            unify(r1, r2, subst)
        }
        _ => Err(TypeError::UnificationFail(t1.clone(), t2.clone())),
    }
}

/// Occurs check: prevents infinite types like t = list(t)
fn occurs_check(name: &str, ty: &Type) -> bool {
    match ty {
        Type::Var(n) => n == name,
        Type::List(inner) => occurs_check(name, inner),
        Type::Fun(params, ret) => {
            params.iter().any(|p| occurs_check(name, p)) || occurs_check(name, ret)
        }
        _ => false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Type Environment
// ═══════════════════════════════════════════════════════════════════════════════

/// Type environment mapping identifiers to their types
#[derive(Clone)]
pub struct TypeEnv {
    vars: BTreeMap<String, Type>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            vars: BTreeMap::new(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&Type> {
        self.vars.get(name)
    }

    pub fn insert(&mut self, name: String, ty: Type) {
        self.vars.insert(name, ty);
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Expression Inference
// ═══════════════════════════════════════════════════════════════════════════════

/// Infer the type of an expression
pub fn infer_expr(
    expr: &Expr,
    env: &mut TypeEnv,
    gen: &mut TypeVarGen,
    subst: &mut Subst,
) -> Result<Type, TypeError> {
    match &expr.node {
        ExprKind::Literal(lit) => match lit {
            Literal::Num(_) => Ok(Type::Float),
            Literal::Bool(_) => Ok(Type::Bool),
            Literal::Str(_) => Ok(Type::Str),
        },
        ExprKind::Ident(name) => {
            if let Some(ty) = env.get(name) {
                Ok(ty.clone())
            } else {
                Err(TypeError::UnboundVariable(name.clone()))
            }
        }
        ExprKind::BinaryOp(left, op, right) => {
            let t1 = infer_expr(left, env, gen, subst)?;
            let t2 = infer_expr(right, env, gen, subst)?;
            match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                    unify(&t1, &Type::Float, subst)?;
                    unify(&t2, &Type::Float, subst)?;
                    Ok(Type::Float)
                }
                BinaryOp::Eq
                | BinaryOp::Neq
                | BinaryOp::Lt
                | BinaryOp::Gt
                | BinaryOp::Le
                | BinaryOp::Ge => {
                    unify(&t1, &t2, subst)?;
                    Ok(Type::Bool)
                }
                BinaryOp::And | BinaryOp::Or => {
                    unify(&t1, &Type::Bool, subst)?;
                    unify(&t2, &Type::Bool, subst)?;
                    Ok(Type::Bool)
                }
                BinaryOp::Shl | BinaryOp::Shr => {
                    unify(&t1, &Type::Float, subst)?;
                    unify(&t2, &Type::Float, subst)?;
                    Ok(Type::Float)
                }
            }
        }
        ExprKind::UnaryOp(op, expr) => {
            let t = infer_expr(expr, env, gen, subst)?;
            match op {
                UnaryOp::Neg => {
                    unify(&t, &Type::Float, subst)?;
                    Ok(Type::Float)
                }
                UnaryOp::Not => {
                    unify(&t, &Type::Bool, subst)?;
                    Ok(Type::Bool)
                }
            }
        }
        ExprKind::Let { name, ty, value } => {
            let inferred = infer_expr(value, env, gen, subst)?;
            let final_ty = if let Some(ann) = ty {
                unify(&inferred, ann, subst)?;
                apply_subst(ann, subst)
            } else {
                inferred
            };
            env.insert(name.clone(), final_ty.clone());
            Ok(final_ty)
        }
        ExprKind::List(elements) => {
            if elements.is_empty() {
                Ok(Type::List(Box::new(gen.fresh())))
            } else {
                let elem_ty = infer_expr(&elements[0], env, gen, subst)?;
                for el in elements.iter().skip(1) {
                    let t = infer_expr(el, env, gen, subst)?;
                    unify(&elem_ty, &t, subst)?;
                }
                Ok(Type::List(Box::new(apply_subst(&elem_ty, subst))))
            }
        }
        ExprKind::Call { name, args } => {
            let arg_types: Result<Vec<Type>, TypeError> = args
                .iter()
                .map(|arg| match arg {
                    CallArg::Positional(expr) => infer_expr(expr, env, gen, subst),
                    CallArg::Named { value, .. } => infer_expr(value, env, gen, subst),
                })
                .collect();
            let arg_types = arg_types?;

            let ret_ty = gen.fresh();
            let func_ty = Type::Fun(arg_types, Box::new(ret_ty.clone()));

            if let Some(known_ty) = env.get(name) {
                unify(known_ty, &func_ty, subst)?;
            } else {
                // Unknown function: assume it exists with the inferred signature
                env.insert(name.clone(), func_ty);
            }
            Ok(apply_subst(&ret_ty, subst))
        }
        ExprKind::FieldAccess { .. } => Ok(gen.fresh()),
        ExprKind::MethodCall { .. } => Ok(gen.fresh()),
        ExprKind::Index { .. } => Ok(gen.fresh()),
        ExprKind::Config(_) => Ok(Type::Unit),
        ExprKind::New { .. } => Ok(gen.fresh()),
        ExprKind::Range(_) => Ok(Type::Unit),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Statement & Program Inference
// ═══════════════════════════════════════════════════════════════════════════════

/// Typecheck an entire program, returning the type of the last statement
pub fn typecheck_program(program: &Program) -> Result<Type, TypeError> {
    let mut env = TypeEnv::new();
    let mut gen = TypeVarGen::new();
    let mut subst = Subst::new();

    // Pre-populate with known native function signatures
    env.insert(String::from("print"), Type::Fun(vec![Type::Str], Box::new(Type::Unit)));
    env.insert(
        String::from("sin"),
        Type::Fun(vec![Type::Float], Box::new(Type::Float)),
    );
    env.insert(
        String::from("cos"),
        Type::Fun(vec![Type::Float], Box::new(Type::Float)),
    );
    env.insert(
        String::from("sqrt"),
        Type::Fun(vec![Type::Float], Box::new(Type::Float)),
    );
    env.insert(
        String::from("exp"),
        Type::Fun(vec![Type::Float], Box::new(Type::Float)),
    );
    env.insert(
        String::from("range"),
        Type::Fun(
            vec![Type::Float, Type::Float],
            Box::new(Type::List(Box::new(Type::Float))),
        ),
    );

    let mut last_ty = Type::Unit;

    for stmt in &program.statements {
        last_ty = infer_stmt(stmt, &mut env, &mut gen, &mut subst)?;
    }

    Ok(apply_subst(&last_ty, &subst))
}

/// Infer the type produced by a statement
fn infer_stmt(
    stmt: &Statement,
    env: &mut TypeEnv,
    gen: &mut TypeVarGen,
    subst: &mut Subst,
) -> Result<Type, TypeError> {
    match &stmt.node {
        StmtKind::Var(decl) => {
            let inferred = infer_expr(&decl.value, env, gen, subst)?;
            let final_ty = if let Some(hint) = &decl.type_hint {
                let hint_ty = ident_to_type(hint);
                unify(&inferred, &hint_ty, subst)?;
                apply_subst(&hint_ty, subst)
            } else {
                inferred
            };
            env.insert(decl.name.clone(), final_ty);
            Ok(Type::Unit)
        }
        StmtKind::Expr(expr) => infer_expr(expr, env, gen, subst),
        StmtKind::Fn(decl) => {
            let mut fn_env = env.clone();
            let mut param_types = Vec::new();
            for param in &decl.params {
                let pty = gen.fresh();
                param_types.push(pty.clone());
                fn_env.insert(param.clone(), pty);
            }
            let body_ty = infer_block(&decl.body, &mut fn_env, gen, subst)?;
            let func_ty = Type::Fun(param_types, Box::new(body_ty));
            env.insert(decl.name.clone(), func_ty);
            Ok(Type::Unit)
        }
        StmtKind::If(stmt) => {
            let cond_ty = infer_expr(&stmt.condition, env, gen, subst)?;
            unify(&cond_ty, &Type::Bool, subst)?;
            infer_block(&stmt.then_branch, env, gen, subst)?;
            if let Some(else_branch) = &stmt.else_branch {
                infer_block(else_branch, env, gen, subst)?;
            }
            Ok(Type::Unit)
        }
        StmtKind::While(stmt) => {
            let cond_ty = infer_expr(&stmt.condition, env, gen, subst)?;
            unify(&cond_ty, &Type::Bool, subst)?;
            infer_block(&stmt.body, env, gen, subst)?;
            Ok(Type::Unit)
        }
        StmtKind::For(stmt) => {
            let iter_ty = infer_expr(&stmt.iterable, env, gen, subst)?;
            match &iter_ty {
                Type::List(elem_ty) => {
                    let mut loop_env = env.clone();
                    loop_env.insert(stmt.iterator.clone(), *elem_ty.clone());
                    infer_block(&stmt.body, &mut loop_env, gen, subst)?;
                }
                _ => {
                    let elem_ty = gen.fresh();
                    unify(&iter_ty, &Type::List(Box::new(elem_ty.clone())), subst)?;
                    let mut loop_env = env.clone();
                    loop_env.insert(stmt.iterator.clone(), elem_ty);
                    infer_block(&stmt.body, &mut loop_env, gen, subst)?;
                }
            }
            Ok(Type::Unit)
        }
        StmtKind::Return(stmt) => {
            if let Some(expr) = &stmt.value {
                infer_expr(expr, env, gen, subst)
            } else {
                Ok(Type::Unit)
            }
        }
        _ => Ok(Type::Unit),
    }
}

/// Infer the type of a block (statements in a nested scope)
fn infer_block(
    block: &Block,
    env: &mut TypeEnv,
    gen: &mut TypeVarGen,
    subst: &mut Subst,
) -> Result<Type, TypeError> {
    let mut last_ty = Type::Unit;
    let mut block_env = env.clone();
    for stmt in &block.statements {
        last_ty = infer_stmt(stmt, &mut block_env, gen, subst)?;
    }
    Ok(last_ty)
}

/// Convert a textual type hint into a structured `Type`
fn ident_to_type(s: &str) -> Type {
    match s {
        "int" => Type::Int,
        "float" => Type::Float,
        "bool" => Type::Bool,
        "str" | "string" => Type::Str,
        "unit" => Type::Unit,
        _ => Type::Var(s.to_string()),
    }
}
