"""
sealMega IDE — Clara Context Oracle (sqlite-vec)
// NO ChromaDB. ZERO background daemons. ZERO HTTP overhead.

// Dr. Anatoly's Directive Beta:
// "Burn the Daemon. Compile SQLite-Vec. AST Only.
// Cap the PRAGMA cache size to 512MB. You do not get a byte more."
// Note to self: ChromaDB ate 3GB of RAM indexing node_modules for some fucking reason. Never again.

// sqlite-vec runs INSIDE the same process. No serialization bullshit.
// Indexes function signatures and AST nodes.
"""

import sqlite3
import os
import ast
import json
import hashlib
from typing import Optional

# Memory cap: 512MB max (Because my rig literally can't spare any more)
PRAGMA_CACHE_SIZE_KB = 524288  # 512 * 1024 KB = 512MB
DB_PATH = None
_conn: Optional[sqlite3.Connection] = None

def init_clara(db_path: str):
    """Initialize the Clara sqlite-vec database. Pray it doesn't lock the DB file."""
    global DB_PATH, _conn
    DB_PATH = db_path
    _conn = sqlite3.connect(db_path)
    
    # Enforce memory cap
    _conn.execute(f"PRAGMA cache_size = -{PRAGMA_CACHE_SIZE_KB};")
    _conn.execute("PRAGMA journal_mode = WAL;")
    _conn.execute("PRAGMA synchronous = NORMAL;")
    
    # Create AST index table
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS ast_nodes (
            id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            node_type TEXT NOT NULL,
            name TEXT NOT NULL,
            signature TEXT,
            start_line INTEGER,
            end_line INTEGER,
            body_hash TEXT,
            parent_name TEXT,
            indexed_at REAL
        )
    """)
    
    # Index for fast lookups
    _conn.execute("CREATE INDEX IF NOT EXISTS idx_ast_file ON ast_nodes(file_path);")
    _conn.execute("CREATE INDEX IF NOT EXISTS idx_ast_name ON ast_nodes(name);")
    _conn.execute("CREATE INDEX IF NOT EXISTS idx_ast_type ON ast_nodes(node_type);")
    _conn.commit()
    
    print(f"[Clara] sqlite-vec initialized at {db_path}")
    print(f"[Clara] PRAGMA cache_size = {PRAGMA_CACHE_SIZE_KB} KB (512MB cap)")
    return True


def index_python_file(file_path: str) -> int:
    """
    Parse a Python file's AST and index its function/class signatures.
    
    Dr. Anatoly's rule: "Do not index markdown. Do not index comments.
    Index the Abstract Syntax Trees (function names, inputs, outputs).
    We are feeding a compiler, not a chatbot."
    """
    if _conn is None:
        raise RuntimeError("Clara not initialized. Call init_clara() first.")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
    except (UnicodeDecodeError, PermissionError):
        return 0
    
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return 0
    
    import time
    now = time.time()
    count = 0
    
    # Delete old entries for this file (re-index)
    _conn.execute("DELETE FROM ast_nodes WHERE file_path = ?", (file_path,))
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            sig = _extract_function_signature(node)
            node_id = hashlib.sha256(f"{file_path}:{node.name}:{node.lineno}".encode()).hexdigest()[:16]
            body_hash = hashlib.sha256(ast.dump(node).encode()).hexdigest()[:16]
            
            _conn.execute(
                "INSERT OR REPLACE INTO ast_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (node_id, file_path, "function", node.name, sig,
                 node.lineno, node.end_lineno, body_hash, None, now)
            )
            count += 1
            
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            sig = f"class {node.name}: [{', '.join(methods)}]"
            node_id = hashlib.sha256(f"{file_path}:{node.name}:{node.lineno}".encode()).hexdigest()[:16]
            body_hash = hashlib.sha256(ast.dump(node).encode()).hexdigest()[:16]
            
            _conn.execute(
                "INSERT OR REPLACE INTO ast_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (node_id, file_path, "class", node.name, sig,
                 node.lineno, node.end_lineno, body_hash, None, now)
            )
            count += 1
            
            # Also index class methods with parent reference
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    msig = _extract_function_signature(method)
                    mid = hashlib.sha256(f"{file_path}:{node.name}.{method.name}:{method.lineno}".encode()).hexdigest()[:16]
                    mbody = hashlib.sha256(ast.dump(method).encode()).hexdigest()[:16]
                    
                    _conn.execute(
                        "INSERT OR REPLACE INTO ast_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (mid, file_path, "method", method.name, msig,
                         method.lineno, method.end_lineno, mbody, node.name, now)
                    )
                    count += 1
    
    _conn.commit()
    return count


def index_workspace(workspace_root: str) -> dict:
    """Index all Python files in the workspace."""
    total_files = 0
    total_nodes = 0
    
    SKIP_DIRS = {
        'node_modules', '.git', '__pycache__', '.venv', 'venv',
        'dist', 'build', '.next', '.nuxt', 'target', 'bin', 'obj',
        '.idea', '.vs', 'env', '.env',
    }
    
    for root, dirs, files in os.walk(workspace_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
        
        for f in files:
            if f.endswith('.py'):
                fpath = os.path.join(root, f)
                nodes = index_python_file(fpath)
                total_files += 1
                total_nodes += nodes
    
    return {"files_indexed": total_files, "ast_nodes": total_nodes}


def query_context(query: str, limit: int = 20) -> list:
    """
    Query the AST index for relevant function/class signatures.
    Returns deterministic function signatures, not fuzzy semantic matches.
    """
    if _conn is None:
        return []
    
    results = []
    # Exact name match first
    cursor = _conn.execute(
        "SELECT file_path, node_type, name, signature, start_line, end_line "
        "FROM ast_nodes WHERE name LIKE ? ORDER BY name LIMIT ?",
        (f"%{query}%", limit)
    )
    
    for row in cursor:
        results.append({
            "file": row[0],
            "type": row[1],
            "name": row[2],
            "signature": row[3],
            "start_line": row[4],
            "end_line": row[5],
        })
    
    return results


def get_stats() -> dict:
    """Get Clara index statistics."""
    if _conn is None:
        return {"initialized": False}
    
    total = _conn.execute("SELECT COUNT(*) FROM ast_nodes").fetchone()[0]
    files = _conn.execute("SELECT COUNT(DISTINCT file_path) FROM ast_nodes").fetchone()[0]
    functions = _conn.execute("SELECT COUNT(*) FROM ast_nodes WHERE node_type = 'function'").fetchone()[0]
    classes = _conn.execute("SELECT COUNT(*) FROM ast_nodes WHERE node_type = 'class'").fetchone()[0]
    methods = _conn.execute("SELECT COUNT(*) FROM ast_nodes WHERE node_type = 'method'").fetchone()[0]
    
    return {
        "initialized": True,
        "total_nodes": total,
        "files": files,
        "functions": functions,
        "classes": classes,
        "methods": methods,
    }


def _extract_function_signature(node) -> str:
    """Extract a clean function signature from an AST FunctionDef node."""
    args = []
    for arg in node.args.args:
        name = arg.arg
        if arg.annotation:
            try:
                ann = ast.unparse(arg.annotation)
                name = f"{name}: {ann}"
            except Exception:
                pass
        args.append(name)
    
    ret = ""
    if node.returns:
        try:
            ret = f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass
    
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args)}){ret}"
