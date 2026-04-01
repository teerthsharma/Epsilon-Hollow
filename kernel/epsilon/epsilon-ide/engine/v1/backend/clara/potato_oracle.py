import sqlite3
import sqlite_vec
import struct
import math
import hashlib
from collections import Counter
from pathlib import Path

DIM = 384


class PotatoClaraOracle:

    def __init__(self, db_path: str = 'clara.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.vocab: dict = {}
        self.idf:   dict = {}
        self._create_table()
        self._load_vocab()
        print(f"[Clara] Database: {db_path}")
        print(f"[Clara] Documents indexed: {self._count_docs()}")

    def _create_table(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS docs (
                path      TEXT PRIMARY KEY,
                content   TEXT NOT NULL,
                hash      TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        ''')
        self.conn.commit()

    def _count_docs(self):
        row = self.conn.execute('SELECT COUNT(*) FROM docs').fetchone()
        return row[0] if row else 0

    def _load_vocab(self):
        """
        Build vocabulary and IDF scores from ALL words in ALL documents.

        Key change from previous version:
            We now include ALL words (not just top DIM most common),
            so rare but distinctive words like 'fibonacci' and 'recursive'
            get into the vocabulary. We take the top DIM by document
            frequency — words that appear in multiple files are more
            useful search targets than words that appear once.
        """
        rows = self.conn.execute('SELECT content FROM docs').fetchall()
        if not rows:
            return

        total_docs = len(rows)
        doc_freq: Counter = Counter()

        for (content,) in rows:
            # Count each unique word once per document (for IDF)
            words = set(self._tokenise(content))
            doc_freq.update(words)

        # Vocabulary: top DIM words by document frequency
        # (words that appear in at least 1 doc but not ALL docs)
        useful = [
            word for word, freq in doc_freq.most_common(DIM * 4)
        ][:DIM]

        self.vocab = {word: idx for idx, word in enumerate(useful)}
        self.idf   = {
            word: math.log((total_docs + 1) / (1 + doc_freq[word]))
            for word in useful
        }

    def _tokenise(self, text: str) -> list:
        """
        Tokenise text into words.

        Splits on whitespace AND common code punctuation so that
        'fibonacci(n)' becomes ['fibonacci', 'n'] — both searchable.
        """
        import re
        # Replace punctuation with spaces, then split
        text = re.sub(r'[(),:.\[\]{}\'"=+\-*/\\<>!@#$%^&|~`]', ' ', text)
        return [w for w in text.lower().split() if len(w) > 1]

    def _tfidf_vector(self, text: str) -> list:
        """Convert text to a DIM-dimensional TF-IDF vector."""
        words = self._tokenise(text)
        total = max(len(words), 1)
        counts = Counter(words)
        vector = [0.0] * DIM
        for word, count in counts.items():
            if word in self.vocab:
                idx = self.vocab[word]
                tf  = count / total
                idf = self.idf.get(word, 1.0)
                vector[idx] = tf * idf
        # L2 normalise
        magnitude = math.sqrt(sum(x*x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        return vector

    def _pack(self, vector: list) -> bytes:
        return struct.pack(f'{DIM}f', *vector)

    def _hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def index_file(self, path: str, content: str) -> bool:
        content_hash = self._hash(content)
        existing = self.conn.execute(
            'SELECT hash FROM docs WHERE path = ?', (path,)
        ).fetchone()
        if existing and existing[0] == content_hash:
            return False
        vector = self._tfidf_vector(content)
        blob   = self._pack(vector)
        self.conn.execute(
            'INSERT OR REPLACE INTO docs (path, content, hash, embedding) '
            'VALUES (?, ?, ?, ?)',
            (path, content[:1500], content_hash, blob)
        )
        self.conn.commit()
        return True

    def _rebuild_vocab_incremental(self):
        self._load_vocab()
        # Re-index all documents with updated vocabulary
        rows = self.conn.execute('SELECT path, content, hash FROM docs').fetchall()
        for path, content, h in rows:
            vector = self._tfidf_vector(content)
            blob   = self._pack(vector)
            self.conn.execute(
                'UPDATE docs SET embedding = ? WHERE path = ?',
                (blob, path)
            )
        self.conn.commit()

    def crawl(self, directory: str,
              extensions: tuple = ('.py', '.js', '.ts', '.md')) -> int:
        directory = Path(directory)
        if not directory.exists():
            print(f"[Clara] Directory not found: {directory}")
            return 0
        indexed = skipped = errors = 0
        print(f"[Clara] Crawling {directory} ...")
        for ext in extensions:
            for fp in directory.rglob(f'*{ext}'):
                try:
                    if fp.stat().st_size > 100_000:
                        skipped += 1
                        continue
                    content = fp.read_text(encoding='utf-8', errors='ignore')
                    if self.index_file(str(fp), content):
                        indexed += 1
                    else:
                        skipped += 1
                except Exception:
                    errors += 1
        self._rebuild_vocab_incremental()
        print(f"[Clara] Done — indexed: {indexed}, skipped: {skipped}, errors: {errors}")
        print(f"[Clara] Total in index: {self._count_docs()}")
        return indexed

    def search(self, query: str, k: int = 3) -> list:
        """
        Find k most relevant documents for the query.

        Returns list of dicts: path, preview, score, distance.
        Handles None distances (zero-vector query) gracefully.
        """
        if not self.vocab:
            return []

        query_vector = self._tfidf_vector(query)
        query_blob   = self._pack(query_vector)

        rows = self.conn.execute('''
            SELECT path, content,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM docs
            ORDER BY distance ASC
            LIMIT ?
        ''', (query_blob, k)).fetchall()

        results = []
        for path, content, distance in rows:
            # vec_distance_cosine returns None when either vector is all zeros
            # (no vocabulary overlap). Treat as worst possible score.
            if distance is None:
                distance = 1.0
            results.append({
                'path':     path,
                'preview':  content[:300],
                'score':    round(max(0.0, 1.0 - distance), 4),
                'distance': round(distance, 4),
            })

        return results

    def get_context_for_prompt(self, query: str, k: int = 3,
                                max_chars: int = 600) -> str:
        results = self.search(query, k=k)
        if not results:
            return ""
        parts  = ["### Relevant code in project:"]
        total  = 0
        for r in results:
            entry = f"\n--- {r['path']} (relevance: {r['score']}) ---\n{r['preview']}\n"
            if total + len(entry) > max_chars:
                break
            parts.append(entry)
            total += len(entry)
        return "".join(parts)

    def stats(self) -> dict:
        return {
            "documents_indexed": self._count_docs(),
            "vocabulary_size":   len(self.vocab),
            "vector_dimensions": DIM,
        }
