# H1t receiver identity mapping correction

The formal H1t source-generation receipt labels the range
`160000001--160008064` as a receiver range. This is a naming error in the
receipt, not a source-generation or pairing error: it is the `sample_seed`
range. The official generator derives receiver-noise identity using
`mod(sample_seed + 2*2246822519, 2^32)` for role code 2. Consequently the
observed and expected H1t `receiver_noise_seed` range is
`358677743--358685806`, with 8,064 unique consecutive values.

The source-audit program now checks the correct receiver range and records
both ranges separately. The original formal receipt remains immutable; this
correction must accompany it. No I/Q, training, calibration, threshold
selection or test set was accessed before this discrepancy was found.
