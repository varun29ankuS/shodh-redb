# Fuzz regression corpus

Inputs that once crashed or hung a fuzz target, kept so they are replayed
forever. Two purposes:

1. **Regression coverage.** A fixed bug stays fixed. Fuzzing is probabilistic --
   rediscovering a given input is not guaranteed, so a fix verified only by "the
   fuzzer stopped complaining" is not verified at all.
2. **Diagnosis.** The `Fuzz Regressions` workflow replays these under
   AddressSanitizer, which prints a stack trace for a hang or crash. That is the
   only way to get one here: cargo-fuzz cannot link on Windows MSVC, because
   `__start___sancov_cntrs` and friends are ELF section-boundary symbols that
   the COFF linker does not synthesise. Fuzzing is Linux/macOS only.

## Layout

    fuzz/regressions/<target-name>/<artifact-file>

The file name is cargo-fuzz's own (`crash-<sha1>`, `timeout-<sha1>`), so it can
be traced back to the run that produced it. Contents are the raw input bytes.

## Adding one

When CI reports a fuzz failure, download the artifact from the run (PR CI and
the nightly both upload them now) and drop it in the matching directory. If the
artifact was lost, the input can be reconstructed from the `Base64:` line that
libFuzzer prints in the log.

## Current entries

| target | input | status |
|---|---|---|
| `fuzz_redb` | `timeout-310af3f4...` (2 bytes, `f5 f5`) | **OPEN** -- hangs indefinitely, 1201s observed |
| `fuzz_db_image` | `crash-0e68d370...` (4 bytes, `ff 09 e1 0e`) | fixed -- reversed slice range in `LeafMutator::insert` |
