# Syntax

Aether syntax is statement-oriented. The current language surface supports
manifold declarations, block declarations, variable bindings, control flow,
functions, classes, imports, render statements, and regression statements.

## Minimal Script

```aegis
manifold M = embed(data, dim=3, tau=5)
block B = M.cluster(0:64)
center = B.center

render M {
  color: by_density,
  highlight: B
}
```

## Declarations

```aegis
manifold M = embed(data, dim=3, tau=5)
block B = M[0:64]
value = 42
```

The parser also accepts typed-looking declarations such as:

```aegis
centroid C = B.center
radius R = B.spread
```

These are parsed as language-level declarations. Whether a type has semantic
meaning depends on the interpreter and type-checking path used by the caller.

## Regression

```aegis
regress {
  model: "polynomial",
  degree: 3,
  escalate: true,
  until: convergence(1e-6)
}
```

Supported model names in docs are `linear`, `polynomial`, `rbf`, `gp`, and
`geodesic`. The current claim is that scripts can express these options. Domain
fitness and speed require separate evidence.

## Control Flow

The parser and interpreter contain tests for loops, breaks, continues,
functions, recursion, closures, arrays, dictionaries, and class/object behavior.
Those constructs are part of the active language surface when the
`aether-lang` crate is built and tested.

## Grammar Sketch

```ebnf
program       = { statement } ;
statement     = manifold_decl | block_decl | var_decl | regress_stmt
              | render_stmt | control_flow | function_decl | class_decl ;
manifold_decl = "manifold" IDENT "=" expr ;
block_decl    = "block" IDENT "=" expr ;
regress_stmt  = "regress" config_block ;
render_stmt   = "render" IDENT [ config_block ] ;
config_block  = "{" { config_pair } "}" ;
config_pair   = IDENT ":" expr [ "," ] ;
expr          = primary { "." IDENT [ call_args ] } ;
```

The implementation source is authoritative when this sketch and parser behavior
diverge.
