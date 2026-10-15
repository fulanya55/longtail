# Route A H1b: full five-background staggered-PRI VFT attribution

Status: preregistered before source generation. H1b is the single permitted
follow-up to passing H1a; it remains a physical attribution screen rather than
a six-label classifier experiment.

## Hypothesis and source

With the H1a score, control and compute budget fixed, true staggered pulse
timestamps preserve VFT-specific evidence against all five non-VFT 0716
background types. On each new 640-pair source identity, both arms share
receiver noise and FMB, SawFM, SinFM, MFT and ISRJ components at 20 dB each.
The present arm alone adds an 8 dB VFT component. FMB/SawFM/SinFM formulas and
their published parameter ranges are reproduced from the read-only official
generator; the nonuniform pulse-time extension is local and therefore the
source is never described as official uniform-PRI 0716 data.

## Frozen analysis and gates

Candidate and control are unchanged from H1a: normalized LFM-delay coherence,
full fixed `1000:250:7000 Hz` grid, true timestamps versus fixed permutation
`[0,5,2,7,4,1,6,3]`, delay samples `[100,500]`, no learned parameters, no
calibration and no threshold. The 5,000-draw paired bootstrap is fixed.

Passing requires all of:

1. candidate present/absent AUROC >= `0.58`;
2. candidate minus control AUROC >= `+0.04` and lower 95% CI > 0;
3. candidate minus control AUPR >= `+0.025` and lower 95% CI > 0;
4. paired present > absent fraction >= `0.56`;
5. source pairing, historical/H0q/H1a identity isolation and test-v3 seal all
   pass.

The lower absolute threshold acknowledges the harder all-background setting;
the matched-control effect and its positive lower CI remain the specificity
criterion. Failure permanently closes this fixed full-grid statistic for the
all-background extension: no changes to JNR, source composition, frequency
grid, delay ROI, normalization, permutation, threshold or fusion are allowed
on these identities.

## Evidence boundary

H1b success would support only a physical, composition-matched VFT observable
under a changed multi-pulse protocol. It does not by itself authorize a neural
architecture claim, original 0716 comparison, test-v3 access, or a final
six-label paper conclusion.
