# Route A U2 paired statistical audit plan

Use only frozen development-v3 and identity-isolated development-v7 summary artifacts. Compare
U2 with U0 using five paired training seeds on each dataset. Primary metrics are macro-AUPR,
macro-F1, target mean weak recall, and target mean weak-vs-negative AUROC. Report paired bootstrap
CI from the source summaries, exact one-sided Wilcoxon p, Holm adjustment within the four-metric
family, paired Cohen dz, and direction consistency. Also report MFT/VFT/ISRJ AUPR, weak recall,
weak AUROC, strong recall, and FPR as secondary outcomes. No test-v3 access is allowed.
