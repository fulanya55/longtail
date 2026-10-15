classdef TestLegacyCompatibility < matlab.unittest.TestCase
%TESTLEGACYCOMPATIBILITY 验证历史干扰函数的接口兼容与可观测真值语义。
%   测试只使用短波形和临时目录，不写入或覆盖正式数据集。

    methods (TestClassSetup)
        function addRepositoryPaths(testCase)
            testsDir = fileparts(mfilename('fullpath'));
            appRoot = fileparts(testsDir);
            repoRoot = fileparts(appRoot);
            originalPath = path;
            addpath(repoRoot, fullfile(repoRoot, 'Jam9'), appRoot, ...
                fullfile(appRoot, 'core'), fullfile(appRoot, 'generators'));
            testCase.addTeardown(@() path(originalPath));
        end
    end

    methods (Test)
        function truthInitializerUsesColumnNaNPadding(testCase)
            truth = legacy_initialize_truth(3, {'alpha', 'beta'});

            testCase.verifyEqual(fieldnames(truth), {'alpha'; 'beta'});
            testCase.verifySize(truth.alpha, [3, 1]);
            testCase.verifyTrue(all(isnan(truth.alpha)));
            testCase.verifyTrue(all(isnan(truth.beta)));
        end

        function zeroBoundaryDelayNeverWrapsTailToHead(testCase)
            signal = [1, 2, 3, 4, 5];

            shifted = legacy_zero_boundary_delay(signal, 2);

            testCase.verifyEqual(shifted, [0, 0, 1, 2, 3]);
            testCase.verifyEqual(legacy_zero_boundary_delay(signal, 5), ...
                zeros(1, 5));
        end

        function realizedNoiseSupportIsSymmetricAndQuantized(testCase)
            [maskA, bandwidthA, infoA] = ...
                legacy_realized_noise_support(16, 16, 4.1);
            [maskB, bandwidthB, infoB] = ...
                legacy_realized_noise_support(16, 16, 4.9);

            testCase.verifyEqual(maskA, maskB);
            testCase.verifyEqual(bandwidthA, 4);
            testCase.verifyEqual(bandwidthB, 4);
            testCase.verifyEqual(infoA.positive_bin_count, 2);
            testCase.verifyEqual(infoB.support_bin_count, 5);
            testCase.verifyEqual(maskA(2:3), true(1, 2));
            testCase.verifyEqual(maskA(end-1:end), true(1, 2));
        end

        function realizedNoiseSupportRejectsDcOnly(testCase)
            testCase.verifyError(@() ...
                legacy_realized_noise_support(16, 16, 1), ...
                'LegacyJamming:InvalidPhysicalParameter');
        end

        function fmbAmaKeepOldWaveformOutputAndReturnTruth(testCase)
            fs = 1e6;
            sampleCount = 4096;
            reference = complex(ones(sampleCount, 1));

            rng(21);
            oldFmb = Data_gen_NoiseFM_Block(reference, 1, 30, 30, ...
                2, 2, 10e3, 0, fs, sampleCount, 2, 1, 0, 0, 0);
            rng(21);
            [newFmb, fmbTruth] = Data_gen_NoiseFM_Block( ...
                reference, 1, 30, 30, 2, 2, 10e3, 0, fs, ...
                sampleCount, 2, 1, 0, 0, 0);

            testCase.verifyEqual(oldFmb, newFmb);
            testCase.verifySize(newFmb, [2, sampleCount]);
            testCase.verifyEqual(fmbTruth.bandwidth_factor, ...
                fmbTruth.interference_bandwidth_Hz / 10e3, ...
                'AbsTol', 1e-12);
            testCase.verifyEqual(fmbTruth.center_frequency_Hz, ...
                zeros(2, 1));
            testCase.verifyEqual(fmbTruth.JNR_dB, repmat(30, 2, 1));

            rng(22);
            oldAma = Data_gen_NoiseAM_Spot(reference, 1, 30, 30, ...
                2, 2, 10e3, 0, fs, sampleCount, 2, 1, 0, 0, 0);
            rng(22);
            [newAma, amaTruth] = Data_gen_NoiseAM_Spot( ...
                reference, 1, 30, 30, 2, 2, 10e3, 0, fs, ...
                sampleCount, 2, 1, 0, 0, 0);

            testCase.verifyEqual(oldAma, newAma);
            testCase.verifySize(newAma, [2, sampleCount]);
            testCase.verifyEqual(amaTruth.modulation_index, ...
                repmat(1 / 3, 2, 1));
            testCase.verifyEqual(amaTruth.bandwidth_factor, ...
                amaTruth.interference_bandwidth_Hz / 10e3, ...
                'AbsTol', 1e-12);
            testCase.verifyEqual(amaTruth.JNR_dB, repmat(30, 2, 1));
        end

        function legacyTruthRejectsRepeatedCopies(testCase)
            fs = 1e6;
            reference = complex(ones(4096, 1));

            testCase.verifyError(@() request_fmb_truth(reference, fs), ...
                'LegacyJamming:TruthRequiresSingleRepetition');
            oneOutput = Data_gen_NoiseFM_Block(reference, 1, 30, 30, ...
                2, 2, 10e3, 0, fs, numel(reference), 1, 2, 0, 0, 0);
            testCase.verifySize(oneOutput, [1, numel(reference)]);
        end

        function uniqueDelaySamplerSortsObservableTotals(testCase)
            rng(31);

            selection = legacy_sample_unique_delays( ...
                [1, 6], [2, 2], 3, 1, 16);

            testCase.verifyEqual(selection.total_samples, ...
                sort(selection.total_samples));
            testCase.verifyEqual(numel(unique(selection.total_samples)), 3);
            testCase.verifyEqual(selection.arrival_samples, 2);
            testCase.verifyEqual(selection.total_samples, ...
                selection.relative_samples + selection.arrival_samples);
        end

        function uniqueDelaySamplerRejectsInsufficientSlots(testCase)
            testCase.verifyError(@() legacy_sample_unique_delays( ...
                [1, 2], [0, 0], 3, 1, 16), ...
                'LegacyJamming:InsufficientDelaySlots');
        end

        function mftCountDelayAndAmplitudeTruthAreObservable(testCase)
            fs = 1e6;
            reference = complex([ones(16, 1); zeros(240, 1)]);

            rng(32);
            oldSignal = Data_gen_Multi_fake_target(reference, 1, fs, ...
                3, 5, 10e-6, 40e-6, 0.5, 1.5, ...
                20, 1, 0, 5e-6, 5e-6);
            rng(32);
            [newSignal, truth] = Data_gen_Multi_fake_target( ...
                reference, 1, fs, 3, 5, 10e-6, 40e-6, 0.5, 1.5, ...
                20, 1, 0, 5e-6, 5e-6);

            testCase.verifyEqual(oldSignal, newSignal);
            testCase.verifySize(newSignal, [20, numel(reference)]);
            testCase.verifyGreaterThanOrEqual(truth.num_targets, 3);
            testCase.verifyLessThanOrEqual(truth.num_targets, 5);
            for rowIndex = 1:20
                targetCount = truth.num_targets(rowIndex);
                delays = truth.delay_s(rowIndex, 1:targetCount);
                amplitudes = truth.amplitude(rowIndex, 1:targetCount);
                testCase.verifyEqual(delays, sort(delays));
                testCase.verifyEqual(numel(unique(delays)), targetCount);
                testCase.verifyTrue(all(isfinite(amplitudes)));
                testCase.verifyEqual(truth.mean_delay_s(rowIndex), ...
                    mean(delays), 'AbsTol', 1e-15);
                if targetCount > 1
                    testCase.verifyEqual( ...
                        truth.mean_spacing_s(rowIndex), mean(diff(delays)), ...
                        'AbsTol', 1e-15);
                end
                testCase.verifyTrue(isnan(truth.JNR_dB(rowIndex)));
                testCase.verifyTrue(isnan(truth.arrival_delay_s(rowIndex)));
                testCase.verifyTrue(all(isnan( ...
                    truth.delay_s(rowIndex, targetCount + 1:end))));
                testCase.verifyTrue(all(isnan( ...
                    truth.amplitude(rowIndex, targetCount + 1:end))));
            end
        end

        function mftCommonDelayUsesSecondsWithoutExtraScaling(testCase)
            fs = 1e6;
            reference = complex([ones(16, 1); zeros(240, 1)]);

            [~, truth] = Data_gen_Multi_fake_target(reference, 1, fs, ...
                1, 1, 20e-6, 20e-6, 2, 2, 2, 1, 0, 5e-6, 5e-6);

            testCase.verifyEqual(truth.delay_s(:, 1), ...
                repmat(25e-6, 2, 1), 'AbsTol', 1e-15);
        end

        function rangeTruthUsesFinalTotalDelay(testCase)
            fs = 1e6;
            reference = complex([ones(16, 1); zeros(240, 1)]);

            rng(33);
            oldSignal = Data_gen_Range_fake_target(reference, 1, fs, ...
                20e-6, 20e-6, 2, 2, 2, 1, 0, 5e-6, 5e-6);
            rng(33);
            [newSignal, truth] = Data_gen_Range_fake_target( ...
                reference, 1, fs, 20e-6, 20e-6, 2, 2, ...
                2, 1, 0, 5e-6, 5e-6);

            testCase.verifyEqual(oldSignal, newSignal);
            testCase.verifyEqual(truth.delay_s(:, 1), ...
                repmat(25e-6, 2, 1), 'AbsTol', 1e-15);
            testCase.verifyEqual(truth.range_m(:, 1), ...
                3e8 * truth.delay_s(:, 1) / 2, 'AbsTol', 1e-12);
            testCase.verifyTrue(all(isnan(truth.JNR_dB)));
            testCase.verifyTrue(all(isnan(truth.arrival_delay_s)));
            testCase.verifyTrue(all(isnan(truth.delay_s(:, 2:end)), 'all'));
            testCase.verifyTrue(all(isnan(truth.range_m(:, 2:end)), 'all'));
            testCase.verifyTrue(all(isnan(truth.amplitude(:, 2:end)), 'all'));
        end

        function tractionUsesAbsoluteRangeAndReturnsStageTruth(testCase)
            fs = 1e6;

            rng(41);
            [oldSignal, oldCompressed] = Data_gen_Tuoyin( ...
                20e3, 16e-6, 0, fs, 128e-6, 900, 900, ...
                2, 2, 30, 30, 1, 4);
            rng(41);
            [newSignal, newCompressed, truth] = Data_gen_Tuoyin( ...
                20e3, 16e-6, 0, fs, 128e-6, 900, 900, ...
                2, 2, 30, 30, 1, 4);

            testCase.verifyEqual(oldSignal, newSignal);
            testCase.verifyEqual(oldCompressed, newCompressed);
            testCase.verifySize(newSignal, [4, 128]);
            testCase.verifySize(newCompressed, [4, 128]);
            testCase.verifyEqual(truth.stage_time_s, [0; 1; 4; 9]);
            testCase.verifyEqual(truth.initial_range_m, repmat(900, 4, 1));
            testCase.verifyEqual(truth.velocity_mps, repmat(30, 4, 1));
            testCase.verifyEqual(truth.nominal_amplitude, repmat(2, 4, 1));
            testCase.verifyEqual(truth.effective_amplitude, repmat(2, 4, 1));
            testCase.verifyEqual(truth.range_change_per_stage_m, ...
                30 * truth.stage_time_s);
            testCase.verifyEqual(truth.delay_change_per_stage_s, ...
                2 * truth.range_change_per_stage_m / 3e8);
            testCase.verifyEqual(truth.current_range_m, ...
                truth.initial_range_m + truth.range_change_per_stage_m);
            testCase.verifyEqual(truth.current_delay_s, ...
                round(2 * truth.current_range_m / 3e8 * fs) / fs);

            firstNonzero = arrayfun(@(rowIndex) ...
                find(abs(newSignal(rowIndex, :)) > 0, 1), (1:4).');
            testCase.verifyGreaterThan(firstNonzero(1), 1);
            testCase.verifyGreaterThanOrEqual(diff(firstNonzero), 0);
        end

        function tractionRejectsIncompleteFourStageGroups(testCase)
            testCase.verifyError(@() Data_gen_Tuoyin( ...
                20e3, 16e-6, 0, 1e6, 128e-6, 900, 900, ...
                2, 2, 30, 30, 1, 3), ...
                'LegacyJamming:InvalidPhysicalParameter');
        end

        function legacyBundleUsesExplicitSourceSampleIndex(testCase)
            label = logical([1, 0; 1, 1]);
            codes = {'FMB', 'MFT'};
            sourceSampleIndex = [2; 1];
            atomicTruth = struct();
            atomicTruth.FMB = make_fmb_truth([10; 20]);
            atomicTruth.MFT = make_mft_truth([1; 2]);
            metadata = make_legacy_metadata('fixture.mat');

            bundle = build_legacy_truth_bundle(label, codes, ...
                atomicTruth, sourceSampleIndex, metadata);

            fmbJnr = strcmp( ...
                bundle.parameter_schema.field_name, 'FMB.JNR_dB');
            mftCount = strcmp( ...
                bundle.parameter_schema.field_name, 'MFT.num_targets');
            testCase.verifyEqual(bundle.parameter_values(:, fmbJnr), ...
                [20; 10]);
            testCase.verifyTrue(bundle.parameter_mask(1, fmbJnr));
            testCase.verifyFalse(bundle.parameter_mask(1, mftCount));
            testCase.verifyEqual(bundle.parameter_values(2, mftCount), 1);
            testCase.verifyEqual( ...
                bundle.source_sample_index, sourceSampleIndex);
            testCase.verifySize(bundle.parameter_values, [2, 79]);
            testCase.verifyEqual( ...
                bundle.metadata.parameter_registry_version, 'v1');
            testCase.verifyEqual( ...
                bundle.metadata.scientific_semantics_version, ...
                current_scientific_semantics_version());
        end

        function unmigratedActiveTypeRemainsMasked(testCase)
            label = logical([1, 1]);
            codes = {'FMB', 'SawFM'};
            atomicTruth = struct('FMB', make_fmb_truth(30));

            bundle = build_legacy_truth_bundle(label, codes, ...
                atomicTruth, 1, make_legacy_metadata('fixture.mat'));

            sawColumns = strcmp( ...
                bundle.parameter_schema.owner_code, 'SawFM');
            testCase.verifyTrue(all(isnan( ...
                bundle.parameter_values(1, sawColumns))));
            testCase.verifyFalse(any( ...
                bundle.parameter_mask(1, sawColumns)));
            sawCoverage = strcmp(bundle.truth_coverage.code, 'SawFM');
            testCase.verifyFalse(bundle.truth_coverage.migrated(sawCoverage));
        end

        function invalidSourceIndexIsRejected(testCase)
            atomicTruth = struct('FMB', make_fmb_truth(30));

            testCase.verifyError(@() build_legacy_truth_bundle( ...
                true, {'FMB'}, atomicTruth, 2, ...
                make_legacy_metadata('fixture.mat')), ...
                'LegacyJamming:InvalidTruthShape');
        end

        function legacySidecarRoundTripsTopLevelVariables(testCase)
            bundle = build_legacy_truth_bundle(true, {'FMB'}, ...
                struct('FMB', make_fmb_truth(30)), 1, ...
                make_legacy_metadata('Jam4_fixture.mat'));
            tempDir = tempname;
            mkdir(tempDir);
            cleanup = onCleanup(@() rmdir(tempDir, 's'));
            mainPath = fullfile(tempDir, 'Jam4_fixture.mat');
            data = 1;
            save(mainPath, 'data');

            truthPath = save_legacy_truth_sidecar( ...
                mainPath, bundle, false);
            loaded = load(truthPath);

            testCase.verifyEqual(truthPath, ...
                fullfile(tempDir, 'Jam4_fixture_truth.mat'));
            testCase.verifyTrue(isfield(loaded, 'parameter_values'));
            testCase.verifyTrue(isfield(loaded, 'parameter_mask'));
            testCase.verifyTrue(isfield(loaded, 'parameter_schema'));
            testCase.verifyTrue(isfield(loaded, 'atomic_truth'));
            testCase.verifyTrue(isfield(loaded, 'metadata'));
            testCase.verifySize(loaded.parameter_values, [1, 79]);
        end

        function legacyEntrypointsDeclareTruthBundleAndSidecar(testCase)
            repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
            entryPaths = { ...
                fullfile(repoRoot, '组合干扰生成', 'Gen_com.m'), ...
                fullfile(repoRoot, '组合干扰生成', 'SP_gen_comjam.m'), ...
                fullfile(repoRoot, 'Gen_openjamset.m')};

            for pathIndex = 1:numel(entryPaths)
                source = fileread(entryPaths{pathIndex});
                testCase.verifyNotEmpty(strfind(source, ...
                    'build_legacy_truth_bundle'));
                testCase.verifyNotEmpty(strfind(source, ...
                    'save_legacy_truth_sidecar'));
                testCase.verifyNotEmpty(strfind(source, ...
                    'source_sample_index'));
                testCase.verifyNotEmpty(strfind(source, 'truth_FMB'));
                testCase.verifyNotEmpty(strfind(source, 'truth_MFT'));
                testCase.verifyNotEmpty(strfind(source, ...
                    'clear pathCleanup'));
            end
        end

        function legacyClosedSetEntrypointsDeclareExplicitClassMaps(testCase)
            repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
            genSource = regexprep(fileread(fullfile(repoRoot, ...
                '组合干扰生成', 'Gen_com.m')), '\s+', '');
            spSource = regexprep(fileread(fullfile(repoRoot, ...
                '组合干扰生成', 'SP_gen_comjam.m')), '\s+', '');

            testCase.verifyNotEmpty(strfind(genSource, ...
                "{'FMB','SawFM','MFT','ISRJ'}"));
            testCase.verifyNotEmpty(strfind(spSource, ...
                "{'FMB','SawFM','SinFM','SN','MFT','VFT','ISRJ'}"));
        end

        function openSetEntrypointDeclaresFiveExplicitClassMaps(testCase)
            repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
            source = regexprep(fileread( ...
                fullfile(repoRoot, 'Gen_openjamset.m')), '\s+', '');
            expectedMaps = { ...
                "{'AMA','SawFM','MFT','ISRJ'}", ...
                "{'FMB','SawFM','MFT','ISRJ'}", ...
                "{'FMB','AMA','MFT','ISRJ'}", ...
                "{'FMB','AMA','SawFM','ISRJ'}", ...
                "{'FMB','AMA','SawFM','MFT'}"};

            for mapIndex = 1:numel(expectedMaps)
                testCase.verifyNotEmpty(strfind(source, ...
                    expectedMaps{mapIndex}));
            end
        end
    end
end

function request_fmb_truth(reference, fs)
%REQUEST_FMB_TRUTH 触发历史真值的重复次数守卫。

    [~, ~] = Data_gen_NoiseFM_Block(reference, 1, 30, 30, ...
        2, 2, 10e3, 0, fs, numel(reference), 1, 2, 0, 0, 0);
end

function truth = make_fmb_truth(jnrValues)
%MAKE_FMB_TRUTH 构造逐行 FMB 原子真值夹具。

    jnrValues = jnrValues(:);
    sampleCount = numel(jnrValues);
    truth = struct( ...
        'JNR_dB', jnrValues, ...
        'bandwidth_factor', repmat(2, sampleCount, 1), ...
        'interference_bandwidth_Hz', repmat(10e6, sampleCount, 1), ...
        'center_frequency_Hz', zeros(sampleCount, 1), ...
        'arrival_delay_s', zeros(sampleCount, 1), ...
        'requested_bandwidth_factor', repmat(2.1, sampleCount, 1));
end

function truth = make_mft_truth(targetCounts)
%MAKE_MFT_TRUTH 构造固定容量 8 的 MFT 原子真值夹具。

    targetCounts = targetCounts(:);
    sampleCount = numel(targetCounts);
    truth = legacy_initialize_truth(sampleCount, {'JNR_dB', ...
        'num_targets', 'mean_delay_s', 'mean_spacing_s', ...
        'arrival_delay_s'});
    truth.delay_s = nan(sampleCount, 8);
    truth.amplitude = nan(sampleCount, 8);
    for rowIndex = 1:sampleCount
        targetCount = targetCounts(rowIndex);
        delays_s = (1:targetCount) * 1e-6;
        truth.num_targets(rowIndex) = targetCount;
        truth.mean_delay_s(rowIndex) = mean(delays_s);
        if targetCount > 1
            truth.mean_spacing_s(rowIndex) = mean(diff(delays_s));
        end
        truth.delay_s(rowIndex, 1:targetCount) = delays_s;
        truth.amplitude(rowIndex, 1:targetCount) = 1:targetCount;
    end
end

function metadata = make_legacy_metadata(mainFile)
%MAKE_LEGACY_METADATA 构造历史旁车测试所需的最小元数据。

    metadata = struct( ...
        'legacy_entrypoint', 'unit-test', ...
        'legacy_main_file', mainFile, ...
        'nRepetition', 1);
end
