# Cryptographic Audit — Seal OS

**Version:** 0.4.7.5 (`kernel/seal-os/Cargo.toml:12`)
**Scope:** in-kernel TLS 1.3 client, hardware entropy driver, and package
signature verification.
**First issued:** 2026-06-01 — Agent 28-32 (Security Hardening & Documentation)
**Revised:** 2026-08-11 against commit `2b113d7`.

Every claim below carries a `file:line` under `kernel/seal-os/src/` unless a
different root is named. Line numbers are as of commit `2b113d7`; the file and
function names are the stable part of a citation, the number is a starting
point.

**This document quotes no Rust source.** The 2026-06-01 revision reproduced
`format::verify_signature` as the live package verifier. Commit `d86e1c2`
deleted that function, and the preimage the quote showed was already not the one
the installer used at the time it was written. A quotation outlives the code it
copies; a citation does not. Behaviour is described in prose against a location.

**What changed in this revision.** §3 is rewritten from
`pkg/mod.rs::signature_preimage` and `pkg/mod.rs::verify_package_signature`. §1
was stale in the opposite direction: the TLS client gained ephemeral X25519, an
X.509 chain validator and an RFC 5869 HMAC-SHA256 key schedule after the
original text was written, so its findings on HKDF and on the absence of a
certificate path no longer describe the code. §4 is re-derived from §1–§3.
Claims this revision could not establish from the source are listed in §6 rather
than left in place reading as verified.

---

## 1. TLS 1.3 Client

Source: `drivers/net/tls.rs`, `drivers/net/tls_socket.rs`,
`drivers/net/ecdhe.rs`, `drivers/net/x509.rs`, `drivers/net/certs.rs`.

The client is real cryptography, not a stub, and deliberately not a complete
TLS 1.3 implementation. The default path is ephemeral X25519 with X.509 chain
validation; PSK-only is a fallback taken when the server offers no key share
and a PSK has been configured (`tls.rs:216-237`). A server that offers neither
is rejected (`tls.rs:232-236`).

### 1.1 Cipher Suite and Record Layer

- **AES-128-GCM** through the `aes-gcm` crate (`Cargo.toml:29`), invoked at
  `tls.rs:292-298` (encrypt) and `tls.rs:317-322` (decrypt). The 16-byte tag is
  appended to the ciphertext (`tls.rs:298`) and split off before decryption
  (`tls.rs:315`).
- `TLS_AES_128_GCM_SHA256` (0x1301) is the only suite offered
  (`tls.rs:144-146`). The suite the server echoes is not compared against it —
  nothing reads the ServerHello `cipher_suite` field (`tls.rs:201-257`).
- **Additional authenticated data is empty** in both directions
  (`tls.rs:296`, `tls.rs:321`). RFC 8446 §5.2 binds the record header into the
  AEAD as `additional_data`. This deviates, and it means the record's declared
  type and length are not authenticated.
- **No record size limit is enforced.** `wrap_record` casts the payload length
  to `u16` without a check (`tls.rs:381`), so a plaintext of 65,520 bytes or
  more produces a record whose length field has wrapped. On receive,
  `parse_record` bounds the payload by the buffer it was given
  (`tls.rs:396-399`) but does not apply the RFC 8446 §5.1 limit of 2^14 + 256.
- TLS 1.3 inner plaintext framing (trailing content-type byte, padding) is not
  implemented: `encrypt` seals the application payload as-is (`tls.rs:294`) and
  `decrypt` returns it as-is (`tls.rs:319-324`).

### 1.2 Nonce Construction

`make_nonce` (`tls.rs:350-358`) copies the 96-bit IV derived at handshake time
and XORs the 64-bit record sequence number into its last eight bytes,
least-significant byte at index 11. That matches RFC 8446 §5.3. The write
counter increments on each successful encrypt (`tls.rs:299`) and the read
counter on each successful decrypt (`tls.rs:323`).

Nonce uniqueness therefore holds within one session for as long as the
`(key, iv)` pair is instantiated once. Two properties bound that:

- Traffic keys are derived from the X25519 shared secret plus both randoms
  (`tls.rs:244-253`), so distinct handshakes give distinct keys — provided
  `client_random` is unique, which is `getrandom` (`tls.rs:128`, `tls.rs:626`).
- On the PSK-only path the shared secret is all-zero (`tls.rs:230`), so the
  traffic keys are a function of the PSK and the two randoms alone. A repeated
  `(client_random, server_random)` pair under the same PSK repeats the keystream.
- There is **no KeyUpdate** (RFC 8446 §4.6.3): no such message is built or
  parsed anywhere in `tls.rs`, so keys are never rotated within a connection.

### 1.3 Key Derivation

**The RFC 5869 finding in the 2026-06-01 revision no longer applies.** The
construction is now HMAC-based:

- `hmac_sha256` (`tls.rs:535-560`) is RFC 2104 HMAC with SHA-256, block size 64
  (`tls.rs:533`), including the long-key hashing case (`tls.rs:537-540`).
- `hkdf_extract` is `HMAC(salt, ikm)` (`tls.rs:562-564`).
- `hkdf_expand` is the RFC 5869 `T(n)` chain with the counter appended and the
  255-block cap honoured (`tls.rs:566-589`).
- `hkdf_expand_label` builds the RFC 8446 §7.1 `HkdfLabel` with the `tls13 `
  prefix (`tls.rs:600-610`); `derive_secret` uses the empty-transcript hash
  (`tls.rs:592-598`).

Known-answer tests exist for HMAC (RFC 4231 case 2, `tls.rs:713-723`), HKDF
(RFC 5869 A.1, `tls.rs:725-748`) and X25519 (RFC 7748 §6.1,
`ecdhe.rs:116-147`), and are registered with the in-kernel runner
(`tls.rs:988-1026`, `testing/runner.rs:61`). See §6 for what "registered" does
and does not establish here.

**The key schedule still deviates from RFC 8446 in one load-bearing way.**
Handshake traffic secrets are derived over `client_random` and `server_random`
as context (`tls.rs:247-248`), not over a running transcript hash. Nothing in
the module maintains a transcript. Consequences, stated plainly:

- The derived keys do not interoperate with a stock TLS 1.3 server.
- Nothing binds the derived keys to the ClientHello or ServerHello contents, so
  a modification to either — including the `supported_versions` and
  `cipher_suite` fields — leaves no cryptographic trace.

### 1.4 Handshake Sequence and What It Authenticates

1. `build_client_hello` (`tls.rs:126-194`) draws `client_random` from hardware
   entropy and generates an X25519 key pair, failing closed if entropy is
   unavailable (`tls.rs:128-130`). It offers `supported_versions` TLS 1.3
   (`tls.rs:154-158`), `psk_key_exchange_modes = psk_dhe_ke` (`tls.rs:160-165`),
   `supported_groups = x25519` (`tls.rs:167-171`) and a real 32-byte key share
   (`tls.rs:173-180`).
2. `handle_server_hello` (`tls.rs:201-257`) extracts the server key share with
   fully bounds-checked offsets (`tls.rs:414-453`), runs the X25519 agreement,
   drops the ephemeral private key (`tls.rs:240`), and derives the traffic keys.
   A peer key share that is a small-order point yields an all-zero secret and is
   rejected (`ecdhe.rs:69-80`).
3. `handle_certificate` (`tls.rs:264-284`) parses an RFC 8446 §4.4.2 Certificate
   message and validates the chain (§1.5). It records the outcome on the session;
   the caller decides what to do with it.

**Absent from the handshake:** EncryptedExtensions, CertificateVerify, Finished,
and any post-handshake message. No such message is constructed or parsed in
`tls.rs`. Two consequences, and they are the most important sentences in this
section:

- **Chain validation does not authenticate the peer.** A certificate chain is
  public data. Without CertificateVerify there is no proof that the peer holds
  the leaf's private key, so any party able to replay a chain that validates
  against the embedded anchor passes the check at `tls_socket.rs:118-123`. The
  check establishes that a valid chain was presented, not who presented it.
- **There is no handshake integrity check.** Without Finished, a modified
  ServerHello is not detected; the handshake simply proceeds with whatever keys
  the modified message produced.

The peer's Certificate message is also read as plaintext handshake
(`tls_socket.rs:103-113`), where RFC 8446 encrypts it under the handshake
traffic keys.

### 1.5 Certificate Validation

`x509.rs` is Ed25519-only (OID 1.3.101.112); RSA and ECDSA are rejected with
`UnsupportedAlgorithm` at `x509.rs:322`, `x509.rs:348` and `x509.rs:375` rather
than accepted unverified. `verify_chain` (`x509.rs:593-625`) does, in order:

| Check | Location |
|---|---|
| Peer-supplied self-signed root at the tail is discarded | `x509.rs:594-599` |
| Chain length ≤ `MAX_CHAIN_LEN` = 4 | `x509.rs:265`, `x509.rs:603-605` |
| Every certificate inside its validity window | `x509.rs:496-503`, `x509.rs:607-609` |
| Issuer DN of child equals subject DN of parent | `x509.rs:520-522` |
| Issuer carries BasicConstraints CA:TRUE | `x509.rs:523-525` |
| Issuer KeyUsage, when present, includes `keyCertSign` | `x509.rs:526-528` |
| `pathLenConstraint` respected | `x509.rs:529-533` |
| Ed25519 signature over the child's TBS bytes verifies (`verify_strict`) | `x509.rs:534`, `x509.rs:577-582` |
| Top of chain issued by an embedded trust anchor | `x509.rs:615-624` |
| Unrecognised **critical** extension is a hard reject | `x509.rs:448` |

The trust store is one certificate compiled into the image
(`x509.rs:268` → `certs.rs:15`). There is no way to add, remove or override an
anchor at runtime: `TRUST_ANCHORS` is a `static` with no mutator.

**Not checked, and absent rather than weak:**

- **Hostname binding.** `Certificate::matches_dns` exists (`x509.rs:507-509`)
  and has no caller outside the module's own tests (`x509.rs:656-658`). Nothing
  in `tls.rs` or `tls_socket.rs` compares the leaf's SubjectAltName against the
  peer being contacted, and `TlsSocket::connect` takes an `IpAddr` rather than a
  name (`tls_socket.rs:53`), so no name is available to compare against. Any
  certificate that chains to the embedded anchor is accepted for any destination.
- **Revocation.** No CRL, no OCSP, no expiry-independent invalidation. No code
  in the tree reads a revocation list.
- **Clock trust.** The validity window is compared against
  `rtc::seconds_since_epoch` (`tls.rs:271` → `drivers/rtc.rs:224`), a CMOS RTC
  read with no lower bound or sanity floor. A clock reading far in the past
  makes an expired certificate acceptable, and one far in the future rejects
  everything.

### 1.6 Enforcement Points

`TlsSocket` requires peer authentication by default (`require_peer_auth: true`,
`tls_socket.rs:30`), and the requirement is enforced only when the handshake
settled on ECDHE (`tls_socket.rs:118-123`). Specifically:

- ECDHE + no valid chain within `CERT_WAIT_TICKS` = 1000 ticks
  (`tls_socket.rs:11`) → the connection fails (`tls_socket.rs:122`).
- A Certificate message that arrives and fails validation fails the connection
  immediately (`tls_socket.rs:107-112`).
- PSK-only → no certificate is required, by construction of the condition at
  `tls_socket.rs:119`.
- `set_require_peer_auth(false)` (`tls_socket.rs:43-45`) disables the check. Its
  own doc comment states the consequence. No caller in the tree sets it.

### 1.7 Key Lifecycle

- Session keys live in fixed arrays on `TlsSession` (`tls.rs:58-61`).
- **No zeroization anywhere in the crate.** There is no `Drop` implementation on
  `TlsSession`, and `zeroize` appears in neither the source tree nor
  `Cargo.toml`. The ephemeral X25519 private key is dropped after the agreement
  (`tls.rs:240`) but not wiped; the PSK persists in `TlsSession` for the life of
  the session (`tls.rs:50`).
- No key update, no rekey, no session resumption, no ticket handling.

---

## 2. Entropy

Source: `drivers/entropy.rs`.

### 2.1 Hardware Probing

`init` (`entropy.rs:13-22`) probes CPUID and records two flags: RDRAND from
CPUID.1:ECX[30] (`entropy.rs:15`) and RDSEED from CPUID.7:EBX[18]
(`entropy.rs:19`). It is called from `lib.rs:317` and again from
`security/kaslr.rs:109`.

### 2.2 Fill Path

`getrandom` (`entropy.rs:80-104`) returns `false` immediately when neither flag
is set (`entropy.rs:83-85`), then fills the buffer eight bytes at a time from
`rdseed_u64().or_else(rdrand_u64)` (`entropy.rs:89`), with a short final block
handled at `entropy.rs:95-102`. Each primitive retries ten times on a clear
carry flag and then reports failure (`entropy.rs:31-47`, `entropy.rs:57-73`).
Any failure aborts the fill and returns `false`; no partial buffer is reported
as success.

Callers and their behaviour on `false`:

| Caller | Location | Behaviour |
|---|---|---|
| TLS `client_random` | `drivers/net/tls.rs:624-631` | `Err("entropy unavailable")` |
| X25519 key generation | `drivers/net/ecdhe.rs:41-45` | `Err(EntropyUnavailable)` |
| DNS transaction IDs | `net/dns.rs:77` | falls back to §2.3 PRNG |
| UDP ephemeral ports | `net/udp.rs:224` | falls back to §2.3 PRNG |
| `getrandom` syscall | `syscall/table.rs:1545` | reports failure to the caller |
| Installer | `apps/installer.rs:798` | reports failure to the caller |

### 2.3 Software Fallback

`fallback_random_u64` (`entropy.rs:142-157`) is a xorshift64 seeded once from
RDTSC on first use. Its own doc comment (`entropy.rs:121-141`) states the
ceiling without softening it: xorshift64 is linear and invertible, so observing
a few dozen outputs recovers the state, and there is no reseeding. It is used
only for DNS transaction IDs and UDP ephemeral ports, and only when no hardware
source exists. **It is not a CSPRNG and no key material is derived from it** —
no caller in `drivers/net/` or `pkg/` reaches for it.

### 2.4 Other Randomness Consumers

- **KASLR** draws directly from `rdseed_u64`/`rdrand_u64` with a
  stuck-generator negative control (two draws must differ) and fails closed
  otherwise (`security/kaslr.rs:108-127`).
- **User-space ASLR** uses a separate xorshift64 seeded from RDTSC plus CPUID
  leaf 0 (`security/aslr.rs:36-43`, `security/aslr.rs:46`), not `getrandom`.
  Leaf 0 is a constant on a given machine, so the seed's entropy is the TSC
  reading alone.

### 2.5 Limitations

- No software entropy pool (Fortuna, ChaCha20 DRBG); every call reaches
  hardware.
- No mixing of independent sources; RDSEED and RDRAND are tried in sequence,
  never combined.
- On virtualised or broken hardware, ten retries may still fail; the fail-closed
  behaviour above is then the whole story.
- No health tests on the hardware source beyond the KASLR stuck-generator check.

---

## 3. Package Signature Verification

Source: `pkg/format.rs` (parsing), `pkg/mod.rs` (signing and verification),
`pkg/channel.rs` (remote channel).

### 3.1 `.eph` Format

`parse_eph` (`format.rs:60-137`) reads:

| Offset | Size | Content | Location |
|---|---|---|---|
| 0 | 4 | Magic `"EPH\0"` | `format.rs:61-63` |
| 4 | 4 | Manifest length, big-endian `u32` | `format.rs:64` |
| 8 | N | Manifest bytes, UTF-8 key-value lines | `format.rs:69`, `format.rs:139-171` |
| 8+N | 64 | Ed25519 signature | `format.rs:74-76` |
| … | 2 | Per file: path length, big-endian `u16` | `format.rs:89` |
| … | P | Per file: path bytes, UTF-8 | `format.rs:97-100` |
| … | 4 | Per file: data length, big-endian `u32` | `format.rs:102-107` |
| … | D | Per file: data | `format.rs:112` |
| end-4 | 4 | Trailer `"END\0"` | `format.rs:80-84` |

Properties the parser enforces, each with the check that does it:

- Every attacker-controlled length is bounds-checked against the buffer before
  it is used to slice: manifest plus signature plus trailer (`format.rs:66`),
  path plus the following length field (`format.rs:91`), and file data
  (`format.rs:109`).
- The trailer is mandatory and terminal: the loop must have matched `END\0`
  *and* consumed the whole buffer (`format.rs:128-130`). Running out of bytes
  without a match is a rejection, not an acceptance.
- Path bytes and manifest bytes must be valid UTF-8, refused rather than
  lossily repaired (`format.rs:97-100`, `format.rs:144-147`). The manifest case
  matters to the signature: the preimage is built from parsed fields, so a lossy
  decode would let two different wire manifests collapse to one preimage.
- Per-file SHA-256 is computed from the data at parse time (`format.rs:115-117`).
  The format carries no declared per-file digest, so there is nothing to compare
  against and `EphError::HashMismatch` (`format.rs:56`) is never constructed by
  any code in the tree. Integrity of file data comes from the signature over the
  computed digest (§3.2), not from a field in the package.

The manifest grammar (`format.rs:152-163`) reads `name=`, `version=`,
`description=` and repeated `dep=` lines, trimming surrounding quotes. Unknown
lines are ignored and a repeated key overwrites. `name` and `version` must be
non-empty (`format.rs:164-166`). `carrier` and `voronoi_cell` are hardcoded
(`format.rs:167`) and never read off the wire.

### 3.2 Verification

`verify_package_signature` (`pkg/mod.rs:324-333`) loads the caller's key with
`VerifyingKey::from_bytes` and calls `verify_strict` over
`signature_preimage(pkg)` (`pkg/mod.rs:297-318`). `verify_strict` is
`ed25519-dalek` 2.1 (`Cargo.toml:28`), which rejects small-order public keys and
non-canonical encodings. Any failure — bad key, bad signature — returns `false`.

The preimage is domain-separated and length-prefixed. Each field is preceded by
its own big-endian `u32` length, and each repeated section by its count:

```text
"EPHSIG1\0"                                  pkg/mod.rs:42
u32(len name)          name                  pkg/mod.rs:305
u32(len version)       version               pkg/mod.rs:306
u32(len description)   description           pkg/mod.rs:307
u32(count deps)                              pkg/mod.rs:308
  per dependency:  u32(len) bytes            pkg/mod.rs:309-311
u32(count files)                             pkg/mod.rs:312
  per file:        u32(len path) path        pkg/mod.rs:314
                   sha256(data) [32 bytes]   pkg/mod.rs:315
```

What that buys, stated no more strongly than the code earns it:

- **Every manifest field the wire format lets a package vary is covered**: name,
  version, description, and the full dependency list. The dependency list is
  what `install_bytes` hands to the resolver (`pkg/mod.rs:429-441`), so it must
  be signed; under the deleted scheme it was not.
- **Framing is injective.** Because every variable-length field carries its
  length, two distinct field tuples cannot produce the same byte string. The
  deleted scheme concatenated raw, so `name="ab" version="c"` and
  `name="a" version="bc"` were indistinguishable and one signature covered both.
- **File contents are covered** through the SHA-256 the parser computes over the
  data (`format.rs:115-117`), and file paths through their own length-prefixed
  entry.
- **`carrier` and `voronoi_cell` are deliberately outside.** `parse_manifest`
  hardcodes them (`format.rs:167`) and never reads them from the package, so
  they carry no attacker-controlled bits. The doc comment at `pkg/mod.rs:293-296`
  records that they must be added the moment the parser starts reading them.

One precise limit on the claim: **the signature covers the parsed fields, not
the manifest bytes.** Two wire manifests that differ only in ignored lines,
duplicate keys, or quoting (`format.rs:152-163`) parse to identical fields and
therefore share one signature. Nothing downstream reads the raw manifest bytes,
so this is currently inert — but it is a property of the encoding, not an
accident that can be relied on if a future field is read straight from the wire.

No property beyond origin authentication is implemented: there is **no
timestamp, no nonce, no version floor and no expiry inside the package
signature**, so a signed `.eph` is valid forever and a replayed old package is
indistinguishable from a fresh one at this layer. Rollback resistance exists
only at the channel layer (§3.4).

### 3.3 Where Verification Is Applied, and Where It Is Skipped

`install_bytes` takes `public_key: Option<&[u8; 32]>` (`pkg/mod.rs:407-411`) and
performs the signature check **only when the caller passes `Some`**
(`pkg/mod.rs:414-418`). This is the single most important operational fact in
this section:

| Call site | Key | Signature checked |
|---|---|---|
| Shell `install <file>.eph` | `None` | **no** — `apps/shell.rs::cmd_install`, `apps/shell.rs:1623` |
| Bundle fixture install | `None` | **no** — `bundle/mod.rs:380` |
| Registry download `install <name>` | `Some(SEAL_PKG_PUBLIC_KEY)` | yes — `pkg/mod.rs:476` |
| Release channel | `Some(package_key)` | yes — `pkg/channel.rs:276` |
| Boot proof | `Some(proof key)` | yes — `pkg/mod.rs:80` |

A locally supplied `.eph` installed from the shell is therefore **not
signature-checked at all**. What still applies to it is path containment:
`install_path_ok` (`pkg/mod.rs:350-368`) requires every declared path to sit
inside `/packages` or the bundle store (`pkg/mod.rs:52`) at a component
boundary, with no empty, `.` or `..` component, and every path is checked before
any file is written (`pkg/mod.rs:422-426`), so a package naming one bad path
installs none of its files.

A regression gate exists over the shape of this code, though not over this
document: `kernel/seal-mkimage/src/main.rs:8299-8313` fails the build if
`pkg/format.rs` reintroduces `fn verify_signature`, and
`main.rs:8339-8347` requires `pkg/mod.rs` to still contain
`verify_package_signature(&pkg, key)`.

### 3.4 Release Channel

`pkg/channel.rs` applies three checks before the package signature, refusing on
the first failure:

1. Ed25519 over the `EPHIDX2` index body (`channel.rs:282-296`), again
   `verify_strict`.
2. Monotonic `index_version`: an index at or below the last accepted version is
   refused as a rollback (`channel.rs:239-245`). The floor is a struct field
   initialised to 0 (`channel.rs:215`) — **it is in memory only and does not
   survive a reboot**, so the rollback window reopens on every boot.
3. Declared byte count and SHA-256 of the fetched package against the signed
   index entry (`channel.rs:265-274`), then `install_bytes` with the channel's
   package key (`channel.rs:276`).

### 3.5 Trust Model and Keys

- The verifying key is supplied by the caller in every path; there is no PKI, no
  certificate chain, no revocation and no rotation mechanism for package keys.
  No code implements any of the three.
- **Four Ed25519 private signing keys are compiled into the kernel image**:
  `pkg/mod.rs:34` (packages), `pkg/channel.rs:51` (channel index),
  `bundle/mod.rs:45` (bundle index), `atlas/mod.rs:44` (charts). They exist so
  the boot proofs can sign the fixtures they then verify. Anyone with the image
  has them, so a signature made under any of them proves only that the signer
  had the image.
- The key used by the network registry path, `SEAL_PKG_PUBLIC_KEY`
  (`pkg/mod.rs:26-29`), has **no matching signing key anywhere in the tree**;
  every signer in `pkg/` uses `PROOF_PKG_SIGNING_KEY` (`pkg/mod.rs:34`) or the
  channel key. It is a well-formed curve point, so it loads and verification
  simply always fails. Its provenance is undocumented; its first 16 bytes
  coincide with those of the public key derived from the all-zero seed, which is
  the shape of a hand-edited placeholder. Treat `install <name>` over the
  registry as untested rather than as a working trust path.

---

## 4. Known Limitations

Re-derived from §1–§3 at commit `2b113d7`. Where the 2026-06-01 row no longer
described the code, the row is replaced rather than annotated.

| # | Limitation | Evidence | Risk |
|---|---|---|---|
| L1 | **No peer authentication despite chain validation.** No CertificateVerify message is parsed, so possession of the leaf's private key is never proven. | `tls.rs:264-284`, `tls_socket.rs:103-123` | Any party replaying a chain that validates to the embedded anchor completes the handshake. |
| L2 | **No handshake integrity.** No Finished message and no transcript hash; traffic secrets are derived over the two randoms. | `tls.rs:247-248` | A modified ClientHello or ServerHello leaves no cryptographic trace; no downgrade protection. |
| L3 | **No hostname binding.** `matches_dns` has no production caller and `connect` takes an `IpAddr`. | `x509.rs:507-509`, `tls_socket.rs:53` | A certificate valid for one name is accepted for every destination. |
| L4 | **Empty AEAD additional data.** RFC 8446 §5.2 binds the record header. | `tls.rs:296`, `tls.rs:321` | Record type and length are unauthenticated; also breaks interoperability. |
| L5 | **No record size limit**, and the record length is a `u16` cast. | `tls.rs:381`, `tls.rs:391-404` | Payloads ≥ 65,520 bytes emit a record with a wrapped length. |
| L6 | **No secure zeroization** of key material anywhere in the crate. | `tls.rs:50`, `tls.rs:58-61`; no `zeroize` in `Cargo.toml` | Keys and PSKs linger in freed memory. |
| L7 | **No key update or resumption** in TLS. | absent from `tls.rs` | A long-lived connection never rotates keys. |
| L8 | **Single embedded trust anchor, no revocation, RTC-dependent expiry.** | `x509.rs:268`, `certs.rs:15`, `drivers/rtc.rs:224` | Anchor compromise is unrecoverable without a rebuild; a wrong clock breaks validity in either direction. |
| L9 | **No entropy pool**; `getrandom` reaches hardware on every call, with no mixing and no health tests beyond KASLR's. | `entropy.rs:80-104`, `security/kaslr.rs:108-127` | Single point of failure; per-call cost. |
| L10 | **Non-crypto fallback PRNG** for DNS IDs and ephemeral ports on hardware with neither RDRAND nor RDSEED. | `entropy.rs:142-157` | State-recoverable after a few dozen observations; documented at `entropy.rs:121-141`. |
| L11 | **Local `.eph` installs are unsigned.** The shell passes `None`. | `apps/shell.rs:1623`, `pkg/mod.rs:414-418` | Only path containment stands between a local package and the filesystem. |
| L12 | **Package signatures carry no freshness.** No timestamp, nonce or version floor in the preimage. | `pkg/mod.rs:297-318` | An old signed package replays forever; the channel's rollback floor resets each boot (`channel.rs:215`). |
| L13 | **Private signing keys ship in the image**, and the registry public key has no known private counterpart. | `pkg/mod.rs:34`, `pkg/channel.rs:51`, `bundle/mod.rs:45`, `atlas/mod.rs:44`, `pkg/mod.rs:26-29` | Fixture signatures prove nothing about origin; the registry path cannot succeed. |
| L14 | **PSK-only fallback has no forward secrecy** and requires no certificate. | `tls.rs:228-231`, `tls_socket.rs:118-123` | PSK compromise reveals all traffic on that path. |

Resolved since 2026-06-01, recorded so the history is not lost: the KDF is now
RFC 5869 HMAC-SHA256 (§1.3), ephemeral X25519 is the default key exchange
(§1.4), an X.509 chain validator exists (§1.5), and the package signature
covers every manifest field under a domain-separated, length-prefixed preimage
(§3.2).

---

## 5. Audit Checklist for Future Crypto Changes

A checklist of questions to answer before merging, not a record of answers. No
box below is asserted as ticked by this revision.

- [ ] **KDF compliance:** does the KDF follow RFC 5869 or an approved NIST standard?
- [ ] **Test vectors:** are there known-answer tests, and are they *registered* with `testing/runner.rs` as well as written?
- [ ] **Nonce uniqueness:** can you show the same `(key, nonce)` pair is never reused, including across the PSK path?
- [ ] **Transcript binding:** does the handshake authenticate the messages that produced the keys?
- [ ] **Peer authentication:** does anything prove the peer holds the private key of the certificate it presented?
- [ ] **Name binding:** is the certificate checked against the destination that was actually contacted?
- [ ] **Forward secrecy:** is the ephemeral private key discarded after use?
- [ ] **Secure zeroization:** are key buffers cleared on drop and on error paths?
- [ ] **Side-channel resistance:** is the implementation constant-time with respect to secret data?
- [ ] **Entropy quality:** is the source seeded from ≥128 bits of hardware entropy, and does it fail closed?
- [ ] **Fail-closed behaviour:** on verification or entropy failure, does the system refuse rather than fall back?
- [ ] **Signature coverage:** does the preimage cover every attacker-controllable field, with injective framing?
- [ ] **Optional verification:** does any caller pass "no key" and thereby skip the check?
- [ ] **Fuzzing:** have the record, certificate and package parsers been fuzzed?
- [ ] **Dependency audit:** are `aes-gcm`, `sha2`, `ed25519-dalek` and `x25519-dalek` (`Cargo.toml:27-30`) pinned to audited versions?

---

## 6. Not Verified in This Revision

Listed so that no sentence above is read as carrying evidence it does not have.

- **No kernel was booted.** The in-kernel tests cited in §1.3 are present in the
  source and registered at `testing/runner.rs:61` and `testing/runner.rs:46-48`,
  which establishes that a `test-mode` QEMU boot would run them. Their current
  pass/fail status was not observed here. `seal-os` is outside the workspace and
  its `#[cfg(test)]` host tests do not run under a plain `cargo test`.
- **Dependency crates were not audited.** The constant-time and side-channel
  properties of `aes-gcm`, `sha2`, `ed25519-dalek` and `x25519-dalek` are taken
  on trust from RustCrypto. No `cargo audit` run backs §5's last item.
- **Provenance of `SEAL_PKG_PUBLIC_KEY`** (`pkg/mod.rs:26-29`) is unknown. The
  statement in §3.5 is that no signer in this tree matches it, which was checked
  by enumerating every `SigningKey::from_bytes` call site. Whether someone
  outside the tree holds the private key cannot be determined from the source.
- **The certificate fixtures in `certs.rs` were not re-derived.** They are
  asserted by `tls.rs:644-704` to parse and chain, which this revision did not
  execute.
- **No claim is made about assembly-level behaviour** of the RDRAND/RDSEED
  inline asm (`entropy.rs:35-41`, `entropy.rs:61-67`) beyond reading it.

---

*This document is a living audit. Update it after every crypto-related PR, and
prefer a `file:line` citation to a quotation — the 2026-06-01 revision quoted a
function that was deleted underneath it and the quote survived for two months.*
