function results = generate_p3om_sparse_peak_ordinal_data(executeGeneration, baseSeed, roleName)
%GENERATE_P3OM_SPARSE_PEAK_ORDINAL_DATA Create one fresh P3o-M role on D:.
% P3o-M deliberately reuses the audited five-regime P3o waveform contract;
% only the MFT count readout is changed.  The wrapper keeps the D: output
% and I/O guard in generate_p3o_low_power_attribution_data authoritative.
    if nargin < 1
        executeGeneration = false;
    end
    if nargin < 2
        baseSeed = 1570000000;
    end
    if nargin < 3
        roleName = 'train';
    end
    if ~(strcmp(roleName, 'train') || strcmp(roleName, 'development'))
        error('P3o-M roleName must be train or development.');
    end
    prefix = sprintf('p3om_sparse_peak_%s_seed%d', roleName, baseSeed);
    results = generate_p3o_low_power_attribution_data(logical(executeGeneration), ...
        baseSeed, prefix, 33, 'D:\\CJ0724_generated_data');
end
