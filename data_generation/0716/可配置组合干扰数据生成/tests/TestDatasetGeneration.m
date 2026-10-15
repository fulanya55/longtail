classdef TestDatasetGeneration < matlab.unittest.TestCase
%TESTDATASETGENERATION 验证组合样本合成、功率语义和参数监督编码。
%   本测试类覆盖 Task 8 的样本级接口。所有测试均使用显式样本种子，
%   并把 manifest 中的 component_target_power_dB 解释为相对功率偏置，
%   而不是绝对目标功率。

    methods (Test)
        function compositionCreatesStableLabelsAndMasks(testCase)
            % enabled/known 即使按用户自定义顺序填写，输出仍按注册表顺序。
            cfg = create_default_config();
            cfg.types.enabled = {'SinFM', 'ISRJ', 'FMB'};
            cfg.types.known = {'SinFM', 'FMB'};
            cfg.types.unknown = {'ISRJ'};
            cfg = normalize_and_validate_config(cfg, 'openset');
            registry = build_jamming_registry(cfg);
            schema = build_parameter_schema(registry);
            row = make_manifest_row(cfg, 9001, [1, 8], ...
                {'FMB', 'ISRJ'}, [0, -6], [false, true]);

            sample = compose_jamming_sample(cfg, registry, schema, row);

            testCase.verifyEqual(sample.label_all, ...
                logical([1, 0, 0, 0, 0, 0, 0, 1, 0, 0]));
            testCase.verifyEqual(sample.label_selected, logical([1, 0, 1]));
            testCase.verifyEqual(sample.known_label, logical([1, 0]));
            testCase.verifyTrue(sample.unknown_present);
            testCase.verifyEqual(sample.unknown_component_count, 1);
            testCase.verifySize(sample.parameter_values, [1, height(schema)]);
            testCase.verifySize(sample.parameter_mask, [1, height(schema)]);
            testCase.verifyTrue(all(sample.parameter_mask( ...
                strcmp(schema.owner_code, 'ISRJ'))));
            testCase.verifySize(sample.iq_complex, ...
                [1, cfg.derived.sample_length]);
        end

        function nonUnitNoiseUsesEffectiveJnrAndOffset(testCase)
            % 绝对目标功率 = 噪声功率dB + 采样JNR + manifest相对偏置。
            cfg = single_type_config('FMB');
            cfg.radar.noise_power = 4;
            cfg.params.common.JNR_dB = [10, 10];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            schema = build_parameter_schema(registry);
            row = make_manifest_row(cfg, 9002, 1, {'FMB'}, -3, true);

            sample = compose_jamming_sample(cfg, registry, schema, row);

            expectedEffectiveJnr_dB = 7;
            expectedPower_dB = 10 * log10(cfg.radar.noise_power) + ...
                expectedEffectiveJnr_dB;
            jnrColumn = strcmp(schema.field_name, 'FMB.JNR_dB');
            testCase.verifyEqual(sample.component_nominal_JNR_dB, 10, ...
                'AbsTol', 1e-12);
            testCase.verifyEqual(sample.component_effective_JNR_dB, ...
                expectedEffectiveJnr_dB, 'AbsTol', 1e-12);
            testCase.verifyEqual(sample.component_target_power_dB, ...
                expectedPower_dB, 'AbsTol', 1e-10);
            testCase.verifyEqual(sample.component_measured_power_dB, ...
                expectedPower_dB, 'AbsTol', 1e-10);
            testCase.verifyEqual(sample.parameter_values(jnrColumn), ...
                expectedEffectiveJnr_dB, 'AbsTol', 1e-12);
        end

        function relativePowerUsesFinalAbsoluteTargets(testCase)
            % 最终相对功率必须由绝对目标功率计算，不能直接复制 manifest 偏置。
            cfg = create_default_config();
            cfg.types.enabled = {'FMB', 'AMA'};
            cfg.types.known = cfg.types.enabled;
            cfg.types.unknown = {};
            cfg.dataset.max_components = 2;
            cfg.params.common.JNR_dB = [5, 25];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            schema = build_parameter_schema(registry);
            row = make_manifest_row(cfg, 9003, [1, 2], ...
                {'FMB', 'AMA'}, [0, 0], [false, false]);

            sample = compose_jamming_sample(cfg, registry, schema, row);

            expectedRelative_dB = sample.component_target_power_dB - ...
                max(sample.component_target_power_dB);
            testCase.verifyEqual(sample.relative_power_dB, ...
                expectedRelative_dB, 'AbsTol', 1e-12);
            testCase.verifyGreaterThan(abs(diff( ...
                sample.component_nominal_JNR_dB)), 1e-9);
            testCase.verifyGreaterThan(max(abs(sample.relative_power_dB)), 1e-9);
        end

        function tractionOffsetUpdatesOnlyObservableAmplitude(testCase)
            % Traction 不做独立JNR归一化；相对功率偏置并入最终幅度真值。
            cfg = single_type_config('Traction');
            cfg.params.Traction.amplitude = [2, 2];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            schema = build_parameter_schema(registry);
            rowZero = make_manifest_row(cfg, 9004, 10, ...
                {'Traction'}, 0, false);
            rowWeak = make_manifest_row(cfg, 9004, 10, ...
                {'Traction'}, -12, true);

            sampleZero = compose_jamming_sample( ...
                cfg, registry, schema, rowZero);
            sampleWeak = compose_jamming_sample( ...
                cfg, registry, schema, rowWeak);

            amplitudeColumn = strcmp(schema.field_name, 'Traction.amplitude');
            expectedGain = 10 ^ (-12 / 20);
            testCase.verifyTrue(isnan(sampleWeak.component_nominal_JNR_dB));
            testCase.verifyTrue(isfinite(sampleWeak.component_effective_JNR_dB));
            testCase.verifyEqual(sampleWeak.component_nominal_amplitude, 2, ...
                'AbsTol', 1e-12);
            testCase.verifyEqual(sampleWeak.component_effective_amplitude, ...
                2 * expectedGain, 'AbsTol', 1e-12);
            testCase.verifyEqual(sampleWeak.parameter_values(amplitudeColumn), ...
                2 * expectedGain, 'AbsTol', 1e-12);
            testCase.verifyEqual(sampleWeak.component_target_power_dB - ...
                sampleZero.component_target_power_dB, -12, 'AbsTol', 1e-10);
            testCase.verifyEqual(sampleWeak.component_measured_power_dB - ...
                sampleZero.component_measured_power_dB, -12, 'AbsTol', 1e-10);
        end

        function sameRowIsReproducibleAndDoesNotChangeGlobalRng(testCase)
            cfg = single_type_config('SinFM');
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            schema = build_parameter_schema(registry);
            row = make_manifest_row(cfg, 9005, 4, {'SinFM'}, 0, false);
            rng(12345, 'twister');
            beforeState = rng;

            first = compose_jamming_sample(cfg, registry, schema, row);
            afterState = rng;
            second = compose_jamming_sample(cfg, registry, schema, row);

            testCase.verifyEqual(afterState, beforeState);
            testCase.verifyEqual(first.iq_complex, second.iq_complex);
            testCase.verifyEqual(first.parameter_values, ...
                second.parameter_values);
            testCase.verifyEqual(first.component_seeds, second.component_seeds);
            testCase.verifyEqual(first.receiver_noise_seed, ...
                second.receiver_noise_seed);
        end

        function typeAndNoiseSeedsStayStableWhenCombinationGrows(testCase)
            % 子流按样本、角色和类型ID派生，新增其他类型不能改变已有类型。
            cfg = create_default_config();
            cfg.types.enabled = {'FMB', 'AMA'};
            cfg.types.known = cfg.types.enabled;
            cfg.types.unknown = {};
            cfg.dataset.max_components = 2;
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            schema = build_parameter_schema(registry);
            rowOne = make_manifest_row(cfg, 9006, 1, {'FMB'}, 0, false);
            rowTwo = make_manifest_row(cfg, 9006, [1, 2], ...
                {'FMB', 'AMA'}, [0, 0], [false, false]);

            one = compose_jamming_sample(cfg, registry, schema, rowOne);
            two = compose_jamming_sample(cfg, registry, schema, rowTwo);

            testCase.verifyEqual(one.component_seeds(1), two.component_seeds(1));
            testCase.verifyEqual(one.receiver_noise_seed, two.receiver_noise_seed);
            testCase.verifyEqual(one.component_nominal_JNR_dB(1), ...
                two.component_nominal_JNR_dB(1), 'AbsTol', 0);
        end

        function mftUnobservableArrivalAndPaddingRemainMasked(testCase)
            cfg = single_type_config('MFT');
            cfg.params.MFT.num_targets = [3, 3];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            schema = build_parameter_schema(registry);
            row = make_manifest_row(cfg, 9007, 6, {'MFT'}, 0, false);

            sample = compose_jamming_sample(cfg, registry, schema, row);

            arrivalColumn = strcmp(schema.field_name, 'MFT.arrival_delay_s');
            activeDelayColumns = ismember(schema.field_name, ...
                {'MFT.delay_01_s', 'MFT.delay_02_s', 'MFT.delay_03_s'});
            paddedDelayColumn = strcmp(schema.field_name, 'MFT.delay_04_s');
            testCase.verifyTrue(all(sample.parameter_mask(activeDelayColumns)));
            testCase.verifyFalse(sample.parameter_mask(paddedDelayColumn));
            testCase.verifyTrue(isnan(sample.parameter_values(paddedDelayColumn)));
            testCase.verifyFalse(sample.parameter_mask(arrivalColumn));
            testCase.verifyTrue(isnan(sample.parameter_values(arrivalColumn)));
        end

        function optionalTargetIsAddedOnceWithZeroBoundaryDelay(testCase)
            cfgOff = single_type_config('FMB');
            cfgOff.radar.add_target = false;
            cfgOff.radar.target_range_m = 9000;
            cfgOff.radar.target_SNR_dB = 8;
            cfgOff = normalize_and_validate_config(cfgOff, 'balanced');
            cfgOn = cfgOff;
            cfgOn.radar.add_target = true;
            registry = build_jamming_registry(cfgOff);
            schema = build_parameter_schema(registry);
            row = make_manifest_row(cfgOff, 9008, 1, {'FMB'}, 0, false);

            withoutTarget = compose_jamming_sample( ...
                cfgOff, registry, schema, row);
            withTarget = compose_jamming_sample( ...
                cfgOn, registry, schema, row);

            reference = build_reference_signal(cfgOn);
            delaySamples = round(2 * cfgOn.radar.target_range_m / ...
                cfgOn.radar.c_mps * cfgOn.radar.fs_Hz);
            delayedTarget = zero_boundary_shift(reference.train, delaySamples);
            targetPower_dB = 10 * log10(cfgOn.radar.noise_power) + ...
                cfgOn.radar.target_SNR_dB;
            expectedTarget = calibrate_component_power( ...
                delayedTarget, targetPower_dB);
            actualTarget = withTarget.iq_complex - withoutTarget.iq_complex;

            testCase.verifyTrue(withTarget.target_present);
            testCase.verifyEqual(actualTarget, expectedTarget, 'AbsTol', 1e-10);
            testCase.verifyEqual(withTarget.target_target_power_dB, ...
                targetPower_dB, 'AbsTol', 1e-10);
            testCase.verifyEqual(withTarget.target_measured_power_dB, ...
                targetPower_dB, 'AbsTol', 1e-10);
        end

        function targetOutsideUnambiguousPriIsRejected(testCase)
            cfg = single_type_config('FMB');
            cfg.radar.add_target = true;
            cfg.radar.target_range_m = cfg.radar.c_mps * cfg.radar.pri_s / 2;
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            schema = build_parameter_schema(registry);
            row = make_manifest_row(cfg, 9009, 1, {'FMB'}, 0, false);

            testCase.verifyError(@() compose_jamming_sample( ...
                cfg, registry, schema, row), ...
                'JammingComposition:InvalidTargetDelay');
        end

        function compositionAddsExactlyOneReceiverNoiseRealization(testCase)
            cfg = single_type_config('FMB');
            cfg.params.common.JNR_dB = [12, 12];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            schema = build_parameter_schema(registry);
            row = make_manifest_row(cfg, 9010, 1, {'FMB'}, 0, false);

            sample = compose_jamming_sample(cfg, registry, schema, row);

            reference = build_reference_signal(cfg);
            componentStream = RandStream('mt19937ar', ...
                'Seed', sample.component_seeds(1));
            raw = generate_single_jamming('FMB', cfg, registry, ...
                componentStream, reference);
            scaled = calibrate_component_power( ...
                raw, sample.component_target_power_dB(1));
            noiseStream = RandStream('mt19937ar', ...
                'Seed', sample.receiver_noise_seed);
            expectedNoise = sqrt(cfg.radar.noise_power / 2) * ( ...
                randn(noiseStream, 1, cfg.derived.sample_length) + ...
                1i * randn(noiseStream, 1, cfg.derived.sample_length));

            testCase.verifyEqual(sample.iq_complex - scaled, ...
                expectedNoise, 'AbsTol', 1e-10);
        end

        function inconsistentManifestRowIsRejected(testCase)
            cfg = create_default_config();
            cfg.types.enabled = {'FMB', 'AMA'};
            cfg.types.known = cfg.types.enabled;
            cfg.types.unknown = {};
            cfg.dataset.max_components = 2;
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            schema = build_parameter_schema(registry);
            row = make_manifest_row(cfg, 9011, 1, {'AMA'}, 0, false);

            testCase.verifyError(@() compose_jamming_sample( ...
                cfg, registry, schema, row), ...
                'JammingComposition:InvalidManifestRow');
        end

        function nonIntegerManifestTypeIdUsesStableError(testCase)
            % 非整数 ID 不得泄漏为 MATLAB 下标异常，必须使用统一错误标识。
            cfg = single_type_config('FMB');
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            schema = build_parameter_schema(registry);
            row = make_manifest_row(cfg, 9012, 1.5, {'FMB'}, 0, false);

            testCase.verifyError(@() compose_jamming_sample( ...
                cfg, registry, schema, row), ...
                'JammingComposition:InvalidManifestRow');
        end

        function smallBundleRoundTripsThroughMatFile(testCase)
            % 保存函数必须把训练常用字段作为 MAT 顶层变量写出。
            [cfg, registry, schema, manifest, bundle] = ...
                create_small_bundle_fixture();
            tempDir = tempname;
            mkdir(tempDir);
            cleanup = onCleanup(@() rmdir(tempDir, 's'));
            outputPath = fullfile(tempDir, 'bundle.mat');

            testCase.verifyWarningFree(@() validate_generated_dataset( ...
                bundle, cfg, registry, schema, 'balanced'));
            save_dataset_bundle(outputPath, bundle, cfg, registry, schema);
            loaded = load(outputPath);

            testCase.verifyTrue(isfile(outputPath));
            testCase.verifyEqual(loaded.data_legacy, ...
                [real(loaded.iq_complex), imag(loaded.iq_complex)]);
            testCase.verifySize(loaded.label_all, [height(manifest), 10]);
            testCase.verifySize(loaded.parameter_values, ...
                size(loaded.parameter_mask));
            testCase.verifyTrue(isfield(loaded, 'metadata'));
            testCase.verifyTrue(isfield(loaded, 'config'));
            testCase.verifyTrue(isfield(loaded, 'parameter_schema'));
        end

        function aggregationRetainsManifestAndVariableMetadata(testCase)
            [cfg, registry, schema, manifest, bundle] = ...
                create_small_bundle_fixture();

            testCase.verifyEqual(bundle.manifest, manifest);
            testCase.verifySize(bundle.iq_complex, ...
                [height(manifest), cfg.derived.sample_length]);
            testCase.verifySize(bundle.label_selected, ...
                [height(manifest), numel(cfg.types.enabled)]);
            testCase.verifySize(bundle.parameter_values, ...
                [height(manifest), height(schema)]);
            testCase.verifyEqual(cellfun(@numel, ...
                bundle.component_target_power_dB), manifest.combination_size);
            testCase.verifyEqual(cellfun(@numel, bundle.component_seeds), ...
                manifest.combination_size);
            testCase.verifyEqual(bundle.metadata.matlab_version, version);
            testCase.verifyEqual({bundle.metadata.registry_snapshot.code}, ...
                {registry.code});
        end

        function validationRejectsTamperedMeasuredPower(testCase)
            [cfg, registry, schema, ~, bundle] = ...
                create_small_bundle_fixture();
            bundle.component_measured_power_dB{1}(1) = ...
                bundle.component_measured_power_dB{1}(1) + 1;

            testCase.verifyError(@() validate_generated_dataset( ...
                bundle, cfg, registry, schema, 'balanced'), ...
                'JammingDataset:PowerMismatch');
        end

        function validationUsesConfiguredPowerTolerance(testCase)
            % 小于 cfg.output.power_tolerance_dB 的数值误差应被接受。
            [cfg, registry, schema, ~, bundle] = ...
                create_small_bundle_fixture();
            bundle.component_measured_power_dB{1}(1) = ...
                bundle.component_measured_power_dB{1}(1) + ...
                cfg.output.power_tolerance_dB / 2;

            testCase.verifyWarningFree(@() validate_generated_dataset( ...
                bundle, cfg, registry, schema, 'balanced'));
        end

        function statisticsReportContainsRequiredAxes(testCase)
            [~, registry, ~, ~, bundle] = ...
                create_small_bundle_fixture(); %#ok<ASGLU>

            printed = evalc('stats = print_dataset_statistics(bundle, registry);');

            testCase.verifyTrue(contains(printed, '分量边际计数'));
            testCase.verifyTrue(contains(printed, '组合阶数计数'));
            testCase.verifyTrue(contains(printed, '弱分量总数'));
            testCase.verifyTrue(isfield(stats, 'component_counts'));
            testCase.verifyTrue(isfield(stats, 'cardinality_counts'));
            testCase.verifyTrue(isfield(stats, 'pattern_counts'));
            testCase.verifyTrue(isfield(stats, 'relative_power_quantiles_dB'));
            testCase.verifyTrue(isfield(stats, 'open_set_split_counts'));
        end

        function validationRejectsPerSampleRowCountMismatch(testCase)
            [cfg, registry, schema, ~, bundle] = ...
                create_small_bundle_fixture();
            bundle.unknown_present = bundle.unknown_present(1:end - 1);

            testCase.verifyError(@() validate_generated_dataset( ...
                bundle, cfg, registry, schema, 'balanced'), ...
                'JammingDataset:InvalidStructure');
        end

        function closedsetSixClassConfigMatchesApprovedParameters(testCase)
            % 六类专用配置必须逐项固化 0716 实验参数，避免正式仿真时
            % 仍依赖调用者手工覆盖默认十类配置。
            cfg = create_closedset_6class_config();

            testCase.verifyEqual(cfg.seed, 20260716);
            testCase.verifyEqual(cfg.output_dir, ...
                'G:\论文相关\干扰数据\复合干扰数据0716');
            testCase.verifyTrue(cfg.output.save_v73);
            testCase.verifyTrue(cfg.output.save_legacy);
            testCase.verifyEqual(cfg.radar.fs_Hz, 10e6);
            testCase.verifyEqual(cfg.radar.pulse_width_s, 100e-6);
            testCase.verifyEqual(cfg.radar.pri_s, 300e-6);
            testCase.verifyEqual(cfg.radar.bandwidth_Hz, 5e6);
            testCase.verifyEqual(cfg.radar.carrier_Hz, 10e9);
            testCase.verifyEqual(cfg.radar.num_pulses, 1);
            testCase.verifyFalse(cfg.radar.add_target);
            testCase.verifyEqual(cfg.radar.noise_power, 1);
            expectedTypes = {'FMB', 'SawFM', 'SinFM', 'MFT', ...
                'VFT', 'ISRJ'};
            testCase.verifyEqual(cfg.types.enabled, expectedTypes);
            testCase.verifyEqual(cfg.types.known, expectedTypes);
            testCase.verifyEmpty(cfg.types.unknown);
            testCase.verifyEqual(cfg.dataset.combination_mode, 'all');
            testCase.verifyEqual(cfg.dataset.min_components, 1);
            testCase.verifyEqual(cfg.dataset.max_components, 6);
            testCase.verifyEqual(cfg.dataset.samples_per_pattern, 400);
            testCase.verifyEqual(cfg.params.common.JNR_dB, [10, 30]);
            testCase.verifyEqual(cfg.params.common.arrival_delay_s, ...
                [10e-6, 50e-6]);
            testCase.verifyEqual(cfg.params.FMB.bandwidth_factor, [1.5, 3]);
            testCase.verifyEqual(cfg.params.FMB.center_frequency_Hz, [0, 0]);
            testCase.verifyEqual(cfg.params.SawFM.sweep_bandwidth_Hz, ...
                [2e6, 5e6]);
            testCase.verifyEqual(cfg.params.SawFM.sweep_period_s, ...
                [20e-6, 80e-6]);
            testCase.verifyEqual(cfg.params.SinFM.center_frequency_Hz, ...
                [0.5e6, 1.5e6]);
            testCase.verifyEqual(cfg.params.SinFM.frequency_deviation_Hz, ...
                [1e6, 3e6]);
            testCase.verifyEqual(cfg.params.SinFM.modulation_frequency_Hz, ...
                [10e3, 20e3]);
            testCase.verifyEqual(cfg.params.MFT.num_targets, [3, 8]);
            testCase.verifyEqual(cfg.params.MFT.target_delay_s, ...
                [30e-6, 50e-6]);
            testCase.verifyEqual(cfg.params.MFT.target_relative_amplitude, ...
                [0.5, 1.5]);
            testCase.verifyEqual(cfg.params.VFT.doppler_Hz, [1e3, 7e3]);
            testCase.verifyEqual(cfg.params.ISRJ.sample_width_s, ...
                [5e-6, 20e-6]);
            testCase.verifyEqual(cfg.params.ISRJ.num_replicas, [1, 5]);
            testCase.verifyEqual(cfg.params.ISRJ.mode, 'direct');

            normalized = normalize_and_validate_config(cfg, 'balanced');
            testCase.verifyEqual(normalized.derived.sample_length, 3000);
        end

        function powerSweepConfigLocksTypeJnrAndOutputLevel(testCase)
            % 九挡配置必须只改变 MFT/VFT/ISRJ 的当前挡位，并为每挡
            % 分配唯一目录；六类、物理参数和配对基础种子保持不变。
            standardLevels = 10:5:50;
            outputDirs = cell(size(standardLevels));
            for levelIndex = 1:numel(standardLevels)
                level_dB = standardLevels(levelIndex);
                cfg = create_power_sweep_6class_config(level_dB);
                profile = cfg.experiment.power_sweep;
                outputDirs{levelIndex} = cfg.output_dir;

                testCase.verifyEqual(cfg.seed, 20260716);
                testCase.verifyEqual(cfg.params.common.JNR_dB, [30, 30]);
                testCase.verifyEqual(cfg.params.FMB.bandwidth_factor, [1.5, 3]);
                testCase.verifyEqual(profile.fixed_codes, ...
                    {'FMB', 'SawFM', 'SinFM'});
                testCase.verifyEqual(profile.swept_codes, ...
                    {'MFT', 'VFT', 'ISRJ'});
                testCase.verifyEqual(profile.fixed_JNR_dB, 30);
                testCase.verifyEqual(profile.sweep_JNR_dB, level_dB);
                testCase.verifyEqual(profile.standard_levels_dB, standardLevels);
                testCase.verifyEqual(profile.level_index, levelIndex);
                testCase.verifyTrue(profile.paired_seeds);
                testCase.verifyEqual(cfg.output_dir, fullfile( ...
                    'G:\论文相关\干扰数据\复合干扰数据0716', ...
                    'power_sweep_9level_v1', sprintf( ...
                    'level_%02d__jnr_%ddB', levelIndex, level_dB)));

                normalized = normalize_and_validate_config(cfg, 'balanced');
                testCase.verifyEqual(normalized.derived.sample_length, 3000);
            end
            testCase.verifyEqual(numel(unique(outputDirs)), 9);

            testCase.verifyError(@() create_power_sweep_6class_config(12), ...
                'JammingPowerSweep:InvalidLevel');
            testCase.verifyError(@() create_power_sweep_6class_config([10, 15]), ...
                'JammingPowerSweep:InvalidLevel');
            testCase.verifyError(@() create_power_sweep_6class_config("10"), ...
                'JammingPowerSweep:InvalidLevel');
        end

        function powerSweepManifestAppliesPairedTypeProfile(testCase)
            % 三个挡位必须复用相同组合和样本种子，仅功率偏置与弱分量
            % 标记随挡位变化；普通闭集配置经过包装器后必须保持原样。
            levels_dB = [10, 30, 50];
            manifests = cell(size(levels_dB));
            for index = 1:numel(levels_dB)
                cfg = create_power_sweep_6class_config(levels_dB(index));
                cfg = normalize_and_validate_config(cfg, 'balanced');
                registry = build_jamming_registry(cfg);
                manifests{index} = build_closedset_manifest( ...
                    cfg, registry, 'train');
            end

            reference = manifests{1};
            for index = 2:numel(manifests)
                testCase.verifyEqual(manifests{index}.sample_seed, ...
                    reference.sample_seed);
                testCase.verifyEqual(manifests{index}.combination_id, ...
                    reference.combination_id);
                testCase.verifyEqual(manifests{index}.active_type_codes, ...
                    reference.active_type_codes);
            end

            sixCode = 'FMB+SawFM+SinFM+MFT+VFT+ISRJ';
            sixIndex = find(strcmp(reference.combination_id, sixCode), 1);
            testCase.assertNotEmpty(sixIndex);
            expectedOffsets = { ...
                [0, 0, 0, -20, -20, -20], ...
                zeros(1, 6), ...
                [0, 0, 0, 20, 20, 20]};
            expectedWeak = { ...
                logical([0, 0, 0, 1, 1, 1]), ...
                false(1, 6), ...
                logical([1, 1, 1, 0, 0, 0])};
            for index = 1:numel(manifests)
                manifest = manifests{index};
                testCase.verifyEqual( ...
                    manifest.component_target_power_dB{sixIndex}, ...
                    expectedOffsets{index});
                testCase.verifyEqual( ...
                    manifest.weak_component_mask{sixIndex}, ...
                    expectedWeak{index});
                testCase.verifyTrue(all(cellfun(@(flags) ...
                    isequal(flags, logical([false, false, false])), ...
                    manifest.imbalance_flags)));
            end

            mftSingleton = find(strcmp(reference.combination_id, 'MFT'), 1);
            testCase.verifyEqual( ...
                manifests{1}.component_target_power_dB{mftSingleton}, -20);
            testCase.verifyFalse( ...
                manifests{1}.weak_component_mask{mftSingleton});

            ordinaryCfg = create_closedset_6class_config();
            ordinaryCfg = normalize_and_validate_config(ordinaryCfg, 'balanced');
            ordinaryRegistry = build_jamming_registry(ordinaryCfg);
            wrapped = build_closedset_manifest( ...
                ordinaryCfg, ordinaryRegistry, 'test');
            direct = build_balanced_manifest( ...
                ordinaryCfg, ordinaryRegistry, 'test');
            testCase.verifyEqual(wrapped, direct);
        end

        function closedsetDryRunBuildsTwoCompleteIndependentManifests(testCase)
            % executeGeneration=false 只完成配置、模式和清单预检；不得创建
            % 或改写训练/测试 MAT 文件。
            expectedTrainPath = fullfile( ...
                'G:\论文相关\干扰数据\复合干扰数据0716', ...
                'closedset_6class_train.mat');
            expectedTestPath = fullfile( ...
                'G:\论文相关\干扰数据\复合干扰数据0716', ...
                'closedset_6class_test.mat');
            beforeTrain = file_signature(expectedTrainPath);
            beforeTest = file_signature(expectedTestPath);

            result = generate_closedset_6class_train_test(false);

            testCase.verifyEqual(result.train_path, expectedTrainPath);
            testCase.verifyEqual(result.test_path, expectedTestPath);
            testCase.verifyFalse(result.executed);
            testCase.verifyFalse( ...
                result.splits.train.verification.performed);
            testCase.verifyFalse( ...
                result.splits.test.verification.performed);
            testCase.verifyEqual(height(result.train_manifest), 25200);
            testCase.verifyEqual(height(result.test_manifest), 25200);
            testCase.verifyEqual(numel(unique( ...
                result.train_manifest.combination_id)), 63);
            testCase.verifyEqual(numel(unique( ...
                result.test_manifest.combination_id)), 63);
            trainCounts = groupcounts(categorical( ...
                result.train_manifest.combination_id));
            testCounts = groupcounts(categorical( ...
                result.test_manifest.combination_id));
            testCase.verifyEqual(trainCounts, repmat(400, 63, 1));
            testCase.verifyEqual(testCounts, repmat(400, 63, 1));
            testCase.verifyEqual(unique(result.train_manifest.split), ...
                {'train'});
            testCase.verifyEqual(unique(result.test_manifest.split), ...
                {'test'});
            testCase.verifyEmpty(intersect( ...
                result.train_manifest.sample_seed, ...
                result.test_manifest.sample_seed));
            testCase.verifyEqual(file_signature(expectedTrainPath), ...
                beforeTrain);
            testCase.verifyEqual(file_signature(expectedTestPath), ...
                beforeTest);
        end

        function savedClosedsetReadbackDetectsDimensionCorruption(testCase)
            % 回载校验必须核对完整期望 manifest、配置和监督矩阵尺寸；
            % 配置、种子顺序或维度被改坏时均须使用稳定错误标识。
            tempDir = tempname;
            mkdir(tempDir);
            cleanup = onCleanup(@() rmdir(tempDir, 's'));
            outputPath = fullfile(tempDir, 'closedset_split.mat');
            [cfg, registry, schema] = ...
                create_small_closedset_runner_fixture(tempDir);
            manifest = build_balanced_manifest(cfg, registry, 'train');
            bundle = generate_dataset_from_manifest( ...
                cfg, registry, schema, manifest);
            save_dataset_bundle( ...
                outputPath, bundle, cfg, registry, schema);
            clear bundle

            verification = verify_saved_closedset_split( ...
                outputPath, manifest, cfg, schema);

            testCase.verifyTrue(verification.performed);
            testCase.verifyEqual(verification.sample_count, 3);
            testCase.verifyEqual(verification.sample_length, ...
                cfg.derived.sample_length);
            testCase.verifyEqual(verification.selected_label_count, 2);
            testCase.verifyEqual(verification.pattern_count, 3);
            testCase.verifyEqual(verification.samples_per_pattern, 1);

            badManifest = manifest;
            badManifest.sample_seed(1) = badManifest.sample_seed(1) + 1;
            testCase.verifyError(@() verify_saved_closedset_split( ...
                outputPath, badManifest, cfg, schema), ...
                'JammingDataset:SavedFileVerificationFailed');

            badCfg = cfg;
            badCfg.seed = badCfg.seed + 1;
            testCase.verifyError(@() verify_saved_closedset_split( ...
                outputPath, manifest, badCfg, schema), ...
                'JammingDataset:SavedFileVerificationFailed');

            label_selected = false(2, 2);
            save(outputPath, 'label_selected', '-append');
            testCase.verifyError(@() verify_saved_closedset_split( ...
                outputPath, manifest, cfg, schema), ...
                'JammingDataset:SavedFileVerificationFailed');
        end

        function powerSweepReadbackDetectsPowerTruthCorruption(testCase)
            % 功率扫描 final 的回载不能只核对尺寸和manifest；有效JNR、
            % 目标/实测功率及参数矩阵任一被篡改都必须被拒绝。
            tempDir = tempname;
            mkdir(tempDir);
            cleanup = onCleanup(@() rmdir(tempDir, 's'));
            pristinePath = fullfile(tempDir, 'pristine.mat');
            [cfg, registry, schema, manifest, bundle] = ...
                create_small_power_sweep_fixture(tempDir, 10);
            save_dataset_bundle( ...
                pristinePath, bundle, cfg, registry, schema);
            clear bundle

            valid = verify_saved_closedset_split( ...
                pristinePath, manifest, cfg, schema);
            testCase.verifyTrue(valid.performed);
            testCase.verifyTrue(valid.power_truth_equal);

            corruptionNames = {'effective', 'target', 'measured', 'parameter'};
            for corruptionIndex = 1:numel(corruptionNames)
                corruptPath = fullfile(tempDir, sprintf( ...
                    'corrupt_%s.mat', corruptionNames{corruptionIndex}));
                copyfile(pristinePath, corruptPath);
                saved = load(corruptPath, ...
                    'component_effective_JNR_dB', ...
                    'component_target_power_dB', ...
                    'component_measured_power_dB', 'parameter_values');
                switch corruptionNames{corruptionIndex}
                    case 'effective'
                        component_effective_JNR_dB = ...
                            saved.component_effective_JNR_dB;
                        component_effective_JNR_dB{1}(1) = ...
                            component_effective_JNR_dB{1}(1) + 1;
                        save(corruptPath, 'component_effective_JNR_dB', ...
                            '-append');
                    case 'target'
                        component_target_power_dB = ...
                            saved.component_target_power_dB;
                        component_target_power_dB{1}(1) = ...
                            component_target_power_dB{1}(1) + 1;
                        save(corruptPath, 'component_target_power_dB', ...
                            '-append');
                    case 'measured'
                        component_measured_power_dB = ...
                            saved.component_measured_power_dB;
                        component_measured_power_dB{1}(1) = ...
                            component_measured_power_dB{1}(1) + 1;
                        save(corruptPath, 'component_measured_power_dB', ...
                            '-append');
                    case 'parameter'
                        parameter_values = saved.parameter_values;
                        firstCode = manifest.active_type_codes{1}{1};
                        jnrColumn = strcmp(schema.field_name, ...
                            [firstCode '.JNR_dB']);
                        parameter_values(1, jnrColumn) = ...
                            parameter_values(1, jnrColumn) + 1;
                        save(corruptPath, 'parameter_values', '-append');
                end
                testCase.verifyError(@() verify_saved_closedset_split( ...
                    corruptPath, manifest, cfg, schema), ...
                'JammingDataset:SavedFileVerificationFailed');
            end
        end

        function powerSweepPublicDryRunIsPairedAndWriteFree(testCase)
            % 公共入口的挡位子集仍必须保持配对种子；干运行不得创建根CSV、
            % 挡位目录或MAT文件，并应返回轻量摘要而不是18张完整manifest。
            sweepRoot = fullfile( ...
                'G:\论文相关\干扰数据\复合干扰数据0716', ...
                'power_sweep_9level_v1');
            csvPath = fullfile(sweepRoot, 'power_sweep_manifest.csv');
            beforeCsv = file_signature(csvPath);

            result = generate_closedset_6class_power_sweep( ...
                false, [10, 50], 'both', 'resume');

            testCase.verifyFalse(result.executed);
            testCase.verifyEqual(result.levels_dB, [10, 50]);
            testCase.verifyTrue(result.paired_seed_verified);
            testCase.verifySize(result.levels, [2, 1]);
            testCase.verifyEqual(result.levels(1).train_status, 'dry_run');
            testCase.verifyEqual(result.levels(2).test_status, 'dry_run');
            testCase.verifyEqual( ...
                result.levels(1).train_seed_signature, ...
                result.levels(2).train_seed_signature);
            testCase.verifyEqual( ...
                result.levels(1).test_seed_signature, ...
                result.levels(2).test_seed_signature);
            testCase.verifyNotEqual(result.levels(1).output_dir, ...
                result.levels(2).output_dir);
            testCase.verifyFalse(isfield(result.levels, 'train_manifest'));
            testCase.verifyEqual(file_signature(csvPath), beforeCsv);

            testCase.verifyError(@() ...
                generate_closedset_6class_power_sweep( ...
                false, 12, 'both', 'resume'), ...
                'JammingPowerSweep:InvalidLevels');
            testCase.verifyError(@() ...
                generate_closedset_6class_power_sweep( ...
                false, [10, 10], 'both', 'resume'), ...
                'JammingPowerSweep:InvalidLevels');
        end

        function smallPowerSweepRunnerGeneratesAtomicPairedLevels(testCase)
            % 小型两挡正式运行验证公共执行内核、状态CSV、原子提交、增强
            % 回载和跨挡配对种子，不在正式0716目录写测试文件。
            tempDir = tempname;
            mkdir(tempDir);
            cleanup = onCleanup(@() rmdir(tempDir, 's'));
            configs = create_small_power_sweep_configs(tempDir, [10, 50]);

            result = run_power_sweep_levels( ...
                configs, true, 'both', 'overwrite', true);

            testCase.verifyTrue(result.executed);
            testCase.verifyTrue(result.paired_seed_verified);
            statusPath = fullfile(tempDir, 'power_sweep_manifest.csv');
            testCase.verifyTrue(isfile(statusPath));
            % Windows路径中的反斜杠数量多于逗号时，自动分隔符检测可能
            % 错把反斜杠当分隔符；状态CSV的契约明确使用英文逗号。
            statusTable = readtable(statusPath, 'Delimiter', ',');
            testCase.verifyEqual( ...
                statusTable.expected_samples_per_split, repmat(63, 2, 1));
            testCase.verifyEqual( ...
                statusTable.expected_pattern_count, repmat(63, 2, 1));
            testCase.verifyEqual( ...
                statusTable.samples_per_pattern, ones(2, 1));
            for levelIndex = 1:numel(result.levels)
                level = result.levels(levelIndex);
                testCase.verifyEqual(level.train_status, 'generated');
                testCase.verifyEqual(level.test_status, 'generated');
                testCase.verifyTrue(level.train_verification.power_truth_equal);
                testCase.verifyTrue(level.test_verification.power_truth_equal);
                testCase.verifyTrue(isfile(level.train_path));
                testCase.verifyTrue(isfile(level.test_path));
                testCase.verifyFalse(isfile(fullfile(level.output_dir, ...
                    'closedset_6class_train.partial.mat')));
                testCase.verifyFalse(isfile(fullfile(level.output_dir, ...
                    'closedset_6class_test.partial.mat')));
            end

            firstTrain = load(result.levels(1).train_path, 'manifest');
            secondTrain = load(result.levels(2).train_path, 'manifest');
            firstTest = load(result.levels(1).test_path, 'manifest');
            secondTest = load(result.levels(2).test_path, 'manifest');
            testCase.verifyEqual(firstTrain.manifest.sample_seed, ...
                secondTrain.manifest.sample_seed);
            testCase.verifyEqual(firstTest.manifest.sample_seed, ...
                secondTest.manifest.sample_seed);
            testCase.verifyEmpty(intersect( ...
                firstTrain.manifest.sample_seed, ...
                firstTest.manifest.sample_seed));
        end

        function closedsetRunnerResumesVerifiedFinalAndErrorRejects(testCase)
            % resume 只能复用通过增强回载的最终文件，不得重新保存；error
            % 策略只要看到最终文件就必须拒绝。
            tempDir = tempname;
            mkdir(tempDir);
            cleanup = onCleanup(@() rmdir(tempDir, 's'));
            [cfg, registry, schema] = ...
                create_small_closedset_runner_fixture(tempDir);

            first = run_closedset_splits( ...
                cfg, registry, schema, true, 'train', 'overwrite');
            finalPath = fullfile(tempDir, 'closedset_6class_train.mat');
            partialPath = fullfile( ...
                tempDir, 'closedset_6class_train.partial.mat');
            beforeResume = file_signature(finalPath);

            testCase.verifyTrue(first.splits.train.generated);
            testCase.verifyFalse(first.splits.train.reused);
            testCase.verifyTrue(first.splits.train.verification.performed);
            testCase.verifyTrue(isfile(finalPath));
            testCase.verifyFalse(isfile(partialPath));
            testCase.verifyFalse(isfield(first, 'train_manifest'));

            resumed = run_closedset_splits( ...
                cfg, registry, schema, true, 'train', 'resume');
            testCase.verifyTrue(resumed.splits.train.reused);
            testCase.verifyTrue(resumed.splits.train.skipped);
            testCase.verifyFalse(resumed.splits.train.generated);
            testCase.verifyTrue( ...
                resumed.splits.train.verification.performed);
            testCase.verifyEqual(file_signature(finalPath), beforeResume);

            testCase.verifyError(@() run_closedset_splits( ...
                cfg, registry, schema, true, 'train', 'error'), ...
                'JammingDataset:OutputAlreadyExists');
        end

        function closedsetRunnerSelectsOnlyTestAndCleansStalePartial(testCase)
            % test 选择不得构建或生成 train；启动所选划分时只清理同名
            % stale partial，不能影响无关文件。
            tempDir = tempname;
            mkdir(tempDir);
            cleanup = onCleanup(@() rmdir(tempDir, 's'));
            [cfg, registry, schema] = ...
                create_small_closedset_runner_fixture(tempDir);
            stalePartial = fullfile( ...
                tempDir, 'closedset_6class_test.partial.mat');
            unrelated = fullfile(tempDir, 'keep_me.partial.mat');
            fclose(fopen(stalePartial, 'w'));
            fclose(fopen(unrelated, 'w'));

            dry = run_closedset_splits( ...
                cfg, registry, schema, false, 'test', 'resume');
            testCase.verifyFalse(dry.splits.train.requested);
            testCase.verifyTrue(dry.splits.test.requested);
            testCase.verifyFalse(isfield(dry, 'train_manifest'));
            testCase.verifyTrue(isfield(dry, 'test_manifest'));
            testCase.verifyTrue(isfile(stalePartial));

            actual = run_closedset_splits( ...
                cfg, registry, schema, true, 'test', 'resume');
            testCase.verifyEqual(actual.split_selection, 'test');
            testCase.verifyTrue(actual.splits.test.generated);
            testCase.verifyFalse(isfile(fullfile( ...
                tempDir, 'closedset_6class_train.mat')));
            testCase.verifyTrue(isfile(fullfile( ...
                tempDir, 'closedset_6class_test.mat')));
            testCase.verifyFalse(isfile(stalePartial));
            testCase.verifyTrue(isfile(unrelated));
        end

        function closedsetEntryNormalizesSelectionAndExistingPolicy(testCase)
            % 公共入口新增参数必须向后兼容，并对非法策略提供稳定错误。
            testOnly = generate_closedset_6class_train_test( ...
                false, "test", "resume");
            testCase.verifyEqual(testOnly.split_selection, 'test');
            testCase.verifyEqual(testOnly.existing_policy, 'resume');
            testCase.verifyFalse(testOnly.splits.train.requested);
            testCase.verifyFalse(isfield(testOnly, 'train_manifest'));
            testCase.verifyTrue(isfield(testOnly, 'test_manifest'));

            testCase.verifyError(@() generate_closedset_6class_train_test( ...
                false, 'invalid', 'resume'), ...
                'JammingDataset:InvalidSplitSelection');
            testCase.verifyError(@() generate_closedset_6class_train_test( ...
                false, 'both', 'invalid'), ...
                'JammingDataset:InvalidExistingPolicy');
        end

        function publicEntriesCreateExpectedFiles(testCase)
            % 同一配置可分别进入普通、不均衡和开放集三个公共入口。
            tempDir = tempname;
            mkdir(tempDir);
            cleanup = onCleanup(@() rmdir(tempDir, 's'));
            cfg = create_default_config();
            cfg.output_dir = tempDir;
            cfg.output.save_v73 = false;
            cfg.output.save_legacy = false;
            cfg.types.enabled = {'FMB', 'AMA', 'SinFM', 'SN'};
            cfg.types.known = {'FMB', 'AMA'};
            cfg.types.unknown = {'SinFM', 'SN'};
            cfg.dataset.num_samples = 4;
            cfg.dataset.min_components = 1;
            cfg.dataset.max_components = 3;
            cfg.imbalance.component.enable = true;
            cfg.imbalance.component.weights = [4, 2, 1, 1];
            cfg.imbalance.combination.enable = true;
            cfg.imbalance.combination.cardinality_weights = [4, 2, 1];
            cfg.imbalance.power.enable = true;
            cfg.imbalance.power.relative_power_dB = [-12, -12];
            cfg.openset.train_known = 2;
            cfg.openset.calibration_known = 1;
            cfg.openset.validation_mixed = 4;
            cfg.openset.test_mixed = 4;

            balanced = generate_compound_dataset(cfg);
            imbalanced = generate_imbalanced_compound_dataset(cfg);
            openset = generate_openset_compound_dataset(cfg);

            testCase.verifyTrue(isfile(balanced.data_path));
            testCase.verifyTrue(isfile(imbalanced.data_path));
            testCase.verifyTrue(all(structfun(@isfile, openset.data_paths)));
            testCase.verifyEqual(sort(fieldnames(openset.data_paths)), ...
                sort({'train_known'; 'calibration_known'; ...
                'validation_mixed'; 'test_mixed'}));
            saved = load(balanced.data_path, 'label_all', 'metadata');
            testCase.verifySize(saved.label_all, [cfg.dataset.num_samples, 10]);
            testCase.verifyEqual(saved.metadata.config.types.enabled, ...
                cfg.types.enabled);
            testCase.verifyEqual(imbalanced.statistics.weak_component_count, ...
                sum(cellfun(@nnz, imbalanced.bundle.weak_component_mask)));
        end
    end
end

function cfg = single_type_config(code)
%SINGLE_TYPE_CONFIG 构造只启用一个已知类型的最小配置。

    cfg = create_default_config();
    cfg.types.enabled = {code};
    cfg.types.known = {code};
    cfg.types.unknown = {};
    cfg.dataset.min_components = 1;
    cfg.dataset.max_components = 1;
end

function row = make_manifest_row(cfg, sampleSeed, ids, codes, offsets, weak)
%MAKE_MANIFEST_ROW 构造与三个 manifest builder 完全同列的一行测试数据。

    ids = reshape(double(ids), 1, []);
    codes = reshape(codes, 1, []);
    offsets = reshape(double(offsets), 1, []);
    weak = reshape(logical(weak), 1, []);
    unknownCount = sum(ismember(codes, cfg.types.unknown));
    unknownPresent = unknownCount > 0;
    flags = logical([false, false, any(offsets ~= 0)]);
    row = table(1, {'test'}, double(sampleSeed), {ids}, {codes}, ...
        {strjoin(codes, '+')}, numel(ids), {offsets}, {weak}, 1, 1, ...
        {flags}, unknownPresent, unknownCount, 'VariableNames', { ...
        'sample_id', 'split', 'sample_seed', 'active_type_ids', ...
        'active_type_codes', 'combination_id', 'combination_size', ...
        'component_target_power_dB', 'weak_component_mask', ...
        'component_sampling_weight', 'pattern_sampling_weight', ...
        'imbalance_flags', 'unknown_present', 'unknown_component_count'});
end

function shifted = zero_boundary_shift(signal, delaySamples)
%ZERO_BOUNDARY_SHIFT 在测试中重构非循环整数样点延时。

    shifted = complex(zeros(size(signal)));
    if delaySamples < numel(signal)
        shifted(delaySamples + 1:end) = signal(1:end - delaySamples);
    end
end

function signature = file_signature(pathValue)
%FILE_SIGNATURE 记录文件是否存在、字节数和修改时间，用于验证干运行无写入。

    if ~isfile(pathValue)
        signature = struct('exists', false, 'bytes', 0, 'datenum', 0);
        return;
    end
    info = dir(pathValue);
    signature = struct('exists', true, 'bytes', info.bytes, ...
        'datenum', info.datenum);
end

function [cfg, registry, schema, manifest, bundle] = ...
        create_small_bundle_fixture()
%CREATE_SMALL_BUNDLE_FIXTURE 生成含单分量和双分量模式的小型普通数据集。

    cfg = create_default_config();
    cfg.types.enabled = {'FMB', 'AMA'};
    cfg.types.known = cfg.types.enabled;
    cfg.types.unknown = {};
    cfg.dataset.min_components = 1;
    cfg.dataset.max_components = 2;
    cfg.dataset.combination_mode = 'all';
    cfg.dataset.samples_per_pattern = 1;
    cfg.output.save_v73 = false;
    cfg.output.save_legacy = true;
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    manifest = build_balanced_manifest(cfg, registry, 'train');
    bundle = generate_dataset_from_manifest(cfg, registry, schema, manifest);
end

function [cfg, registry, schema] = ...
        create_small_closedset_runner_fixture(outputDir)
%CREATE_SMALL_CLOSEDSET_RUNNER_FIXTURE 创建三条样本的原子保存测试配置。

    cfg = create_default_config();
    cfg.output_dir = outputDir;
    cfg.output.save_v73 = true;
    cfg.output.save_legacy = true;
    cfg.radar.num_pulses = 1;
    cfg.types.enabled = {'FMB', 'AMA'};
    cfg.types.known = cfg.types.enabled;
    cfg.types.unknown = {};
    cfg.dataset.min_components = 1;
    cfg.dataset.max_components = 2;
    cfg.dataset.combination_mode = 'all';
    cfg.dataset.samples_per_pattern = 1;
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
end

function [cfg, registry, schema, manifest, bundle] = ...
        create_small_power_sweep_fixture(outputDir, level_dB)
%CREATE_SMALL_POWER_SWEEP_FIXTURE 构造63条九挡功率真值回载测试数据。

    cfg = create_power_sweep_6class_config(level_dB);
    cfg.output_dir = outputDir;
    cfg.output.save_v73 = true;
    cfg.output.save_legacy = false;
    cfg.dataset.samples_per_pattern = 1;
    cfg.dataset.num_samples = 63;
    cfg = normalize_and_validate_config(cfg, 'balanced');
    registry = build_jamming_registry(cfg);
    schema = build_parameter_schema(registry);
    manifest = build_closedset_manifest(cfg, registry, 'train');
    bundle = generate_dataset_from_manifest( ...
        cfg, registry, schema, manifest);
    validate_generated_dataset(bundle, cfg, registry, schema, 'balanced');
end

function configs = create_small_power_sweep_configs(outputRoot, levels_dB)
%CREATE_SMALL_POWER_SWEEP_CONFIGS 创建用于执行内核测试的两挡小配置。

    configs = cell(numel(levels_dB), 1);
    for levelIndex = 1:numel(levels_dB)
        cfg = create_power_sweep_6class_config(levels_dB(levelIndex));
        cfg.output_dir = fullfile(outputRoot, sprintf( ...
            'level_%02d__jnr_%ddB', levelIndex, levels_dB(levelIndex)));
        cfg.output.save_v73 = true;
        cfg.output.save_legacy = false;
        cfg.dataset.samples_per_pattern = 1;
        cfg.dataset.num_samples = 63;
        configs{levelIndex} = cfg;
    end
end
