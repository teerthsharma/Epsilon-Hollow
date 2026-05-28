# User Copy Safety

Seal ABI copies between userspace and kernel only after validating the whole
byte span. Address-range checks alone are not enough.

## Required Invariant

Before copying from userspace:

- pointer is non-null when length is nonzero
- range does not overflow
- full range is below `USER_SPACE_LIMIT`
- every page in the range is mapped
- every page is user-accessible

Before copying to userspace:

- all copy-from checks pass
- every page in the range is writable

Cross-page spans must be validated page by page.

## Current Audit Gap

`kernel/seal-os/src/security/smap_smep.rs` currently validates the address range
but still needs page-table walk validation and a page-fault recovery path before
untrusted user pointers become a stable public ABI surface.

Closure tests:

- null pointer rejected
- kernel pointer rejected
- unmapped pointer rejected
- read-only output span rejected
- cross-page overflow rejected
- valid mapped user span accepted

