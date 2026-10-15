# P3o-M rechunk attempt freeze

The first P3o-M rechunk command incorrectly used compact level numbering while
the source power-sweep directories use canonical indices (`level_04`,
`level_06`, `level_09`). It failed before opening a source HDF5 file.

The only side effect was an empty target directory containing an empty
`level_01__jnr_25dB` folder. It contains no MAT file, no data cache, and no
identity record. The directory is preserved as a failed execution artifact and
is excluded from the subsequent correct rechunk target, which uses the suffix
`_rechunked_v2` and canonical level indices.

No training, development evaluation, test generation, test opening, or locked
test access occurred in this attempt.
