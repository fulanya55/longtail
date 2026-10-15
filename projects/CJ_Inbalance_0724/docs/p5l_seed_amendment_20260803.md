# P5l pre-generation seed amendment

The unexecuted P5l draft named 5.15B--5.19B generator seeds.  Those values exceed the unsigned 32-bit domain commonly accepted by MATLAB's reproducible RNG interfaces and could be invalid or reduced modulo the RNG domain.  Before any P5l directory, data file, model inference, test data, test opening, or result was created, the draft was amended.

The development identities are now 4180M, 4190M, and 4210M; reserved confirmation identities are 4220M and 4230M.  A repository-wide literal search found no prior references to these values.  This amendment changes only unexecuted RNG identifiers: the model, frozen parents, data roles, full-pool equality rule, fixed 126-example timing unit, gates, and stop rule are unchanged.
