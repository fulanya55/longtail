# Route-A mixed-long-tail feasibility audit (2026-09-01)

This audit tests whether label imbalance and combination imbalance can be
simultaneously imposed at fixed training budget on the current six-class source.
It is development-only; no test or locked-test file was opened and E5 was not
modified.

## Data geometry

The 30 dB fit split contains all 63 non-empty six-label combinations, with
approximately 50--82 samples per combination. Labels are highly coupled with
the combination identities. A fixed 1600-sample budget therefore constrains the
possible label marginal counts once a geometric combination profile is fixed.

## Joint feasibility result

Hard combination profiles with approximate CF=2, 4, or 8 and a simultaneous
label target IF=2 were tested using integer programming. With tight combination
count tolerances, no feasible solution was found. Relaxing the combination
constraints eventually yields solutions only by allowing the effective CF to
collapse to a very different value (for example, effective CF close to 80--200
when the label IF is forced to 2).

Conversely, enforcing a strict CF profile keeps label marginals near balanced
(effective IF about 1.1--1.3) rather than IF=2.

## Consequence for the protocol

The current source cannot support a clean, independent fixed-budget
`mixed-long-tail(IF=2, CF=4/8)` condition without changing the generated
combination pool or greatly increasing the available samples per combination.
The old P3 sequential thinning suffered from exactly this problem: its nominal
IF=5 became effective IF≈1.4--1.5 while CF became 12--23.

Therefore:

1. label-only and combo-only experiments remain the valid causal comparisons;
2. an emergent mixed manifest may be retained as a diagnostic, but its achieved
   IF/CF must be reported rather than its requested values;
3. mixed-long-tail promotion is suspended until a jointly generated source or
   a larger combination pool makes the target region feasible.

This is a data-protocol limitation, not evidence that mixed long-tail learning
is intrinsically ineffective.

