classdef TestConfigAndRegistry < matlab.unittest.TestCase
    %TESTCONFIGANDREGISTRY 验证 v1/v2 配置与干扰注册表的公共契约。
    %   测试覆盖版本选择、类型顺序、唯一标识，以及每类参数元数据和
    %   生成器句柄是否完整，防止后续数据生成流程读取到不一致的配置。

    methods (Test)
        function defaultConfigAndRegistryUseV2(testCase)
            % 验证无参默认切换到十三类 v2，并采用 295 维参数模式。
            cfg = create_default_config();
            registry = build_jamming_registry();
            schema = build_parameter_schema(registry);
            expectedCodes = {'FMB', 'AMA', 'SawFM', 'SinFM', 'SN', ...
                'MFT', 'VFT', 'ISRJ', 'RangeFakeTarget', 'Traction', ...
                'CombSpectrum', 'ClutterPulse', 'SpectralDispersion'};
            expectedKnown = {'FMB', 'AMA', 'SawFM', 'MFT', 'ISRJ', ...
                'CombSpectrum', 'ClutterPulse', 'SpectralDispersion'};

            testCase.verifyEqual(cfg.registry.version, 2);
            testCase.verifyEqual({registry.code}, expectedCodes);
            testCase.verifyEqual([registry.id], 1:13);
            testCase.verifyEqual(height(schema), 295);
            testCase.verifyEqual(cfg.types.enabled, expectedCodes);
            testCase.verifyEqual(cfg.types.known, expectedKnown);
            testCase.verifyEqual(cfg.types.unknown, ...
                setdiff(expectedCodes, expectedKnown, 'stable'));
            testCase.verifyEqual( ...
                numel(cfg.imbalance.component.weights), 13);
        end

        function explicitV1PreservesLegacyShape(testCase)
            % 验证显式 v1 不增加 registry 字段，并保持十类、79维和十权重。
            cfg = create_default_config(1);
            registry = build_jamming_registry(cfg);
            schema = build_parameter_schema(registry);

            testCase.verifyFalse(isfield(cfg, 'registry'));
            testCase.verifyNumElements(registry, 10);
            testCase.verifyEqual(height(schema), 79);
            testCase.verifyEqual( ...
                numel(cfg.imbalance.component.weights), 10);
            testCase.verifyFalse(isfield(cfg.params, 'CombSpectrum'));
            testCase.verifyFalse(isfield(cfg.params, 'ClutterPulse'));
            testCase.verifyFalse(isfield(cfg.params, 'SpectralDispersion'));
        end

        function configWithoutRegistrySelectsV1(testCase)
            % 验证缺少版本字段的旧配置始终按 v1 注册表解释。
            cfg = rmfield(create_default_config(2), 'registry');
            cfg.types.enabled = cfg.types.enabled(1:10);
            cfg.types.known = {'FMB', 'AMA', 'SawFM', 'MFT', 'ISRJ'};
            cfg.types.unknown = setdiff( ...
                cfg.types.enabled, cfg.types.known, 'stable');
            registry = build_jamming_registry(cfg);

            testCase.verifyNumElements(registry, 10);
            testCase.verifyEqual({registry.code}, cfg.types.enabled);
        end

        function closedsetWrappersStayV1(testCase)
            % 验证六类及其四/五类派生配置继续使用旧注册表和旧参数维度。
            configs = {create_closedset_6class_config(), ...
                create_closedset_4class_config(), ...
                create_closedset_5class_config()};
            for index = 1:numel(configs)
                cfg = configs{index};
                registry = build_jamming_registry(cfg);
                testCase.verifyFalse(isfield(cfg, 'registry'));
                testCase.verifyNumElements(registry, 10);
                testCase.verifyEqual( ...
                    height(build_parameter_schema(registry)), 79);
            end
        end

        function defaultV2NormalizesNewTypeParameters(testCase)
            % 验证十三类默认配置可直接通过平衡场景规范化和物理边界检查。
            cfg = normalize_and_validate_config( ...
                create_default_config(2), 'balanced');
            testCase.verifyEqual(cfg.registry.version, 2);
            testCase.verifyEqual(cfg.params.CombSpectrum.max_teeth, 8);
            testCase.verifyEqual( ...
                cfg.params.ClutterPulse.max_total_pulses, 32);
            testCase.verifyEqual( ...
                cfg.params.SpectralDispersion.num_subpulses, [2, 8]);
        end

        function disabledNewTypeParametersDoNotBlockValidation(testCase)
            % 验证未启用新增类型的无效参数块不会阻塞旧十类 v2 子集配置。
            cfg = create_default_config(2);
            cfg.types.enabled = cfg.types.enabled(1:10);
            cfg.types.known = {'FMB', 'AMA', 'SawFM', 'MFT', 'ISRJ'};
            cfg.types.unknown = setdiff( ...
                cfg.types.enabled, cfg.types.known, 'stable');
            cfg.params.CombSpectrum.max_teeth = 99;
            cfg.params.ClutterPulse.mode = 'unsupported';
            cfg.params.SpectralDispersion.num_subpulses = [1, 99];

            actual = normalize_and_validate_config(cfg, 'balanced');
            testCase.verifyEqual(actual.types.enabled, cfg.types.enabled);
        end

        function invalidNewTypeParametersUseStableErrors(testCase)
            % 验证三类新增参数的整数、模式、支撑和时长错误标识稳定。
            invalid = cell(1, 8);
            identifiers = cell(1, 8);

            invalid{1} = create_default_config(2);
            invalid{1}.params.CombSpectrum.max_teeth = 9;
            identifiers{1} = 'JammingConfig:InvalidIntegerRange';
            invalid{2} = create_default_config(2);
            invalid{2}.params.CombSpectrum.amplitude_mode = 'unsupported';
            identifiers{2} = 'JammingConfig:UnsupportedMode';
            invalid{3} = create_default_config(2);
            invalid{3}.radar.num_pulses = 5;
            identifiers{3} = 'JammingConfig:InvalidPhysicalParameter';
            invalid{4} = create_default_config(2);
            invalid{4}.params.ClutterPulse.frequency_offset_Hz = [4e6, 4e6];
            identifiers{4} = 'JammingConfig:InvalidPhysicalParameter';
            invalid{5} = create_default_config(2);
            invalid{5}.params.SpectralDispersion.num_subpulses = [1, 1];
            identifiers{5} = 'JammingConfig:InvalidIntegerRange';
            invalid{6} = create_default_config(2);
            invalid{6}.params.SpectralDispersion.max_subpulses = 9;
            identifiers{6} = 'JammingConfig:InvalidIntegerRange';
            invalid{7} = create_default_config(2);
            invalid{7}.radar.pulse_width_s = 0.8e-6;
            identifiers{7} = 'JammingConfig:InvalidPhysicalParameter';
            invalid{8} = create_default_config(2);
            invalid{8}.params.common.arrival_delay_s = [250e-6, 250e-6];
            identifiers{8} = 'JammingConfig:InvalidPhysicalParameter';

            for index = 1:numel(invalid)
                testCase.verifyError(@() normalize_and_validate_config( ...
                    invalid{index}, 'balanced'), identifiers{index});
            end
        end

        function invalidRegistryVersionIsRejectedBeforeRegistryBuild(testCase)
            % 验证配置层对非法版本给出自身稳定标识，而不是泄漏注册表异常。
            cfg = create_default_config(2);
            cfg.registry.version = 3;
            testCase.verifyError(@() normalize_and_validate_config( ...
                cfg, 'balanced'), ...
                'JammingConfig:InvalidRegistryVersion');
        end

        function legacyV1RegistryContainsTenUniqueTypes(testCase)
            % 验证显式 v1 配置和注册表仍包含顺序固定的旧十类干扰。
            cfg = create_default_config(1);
            registry = build_jamming_registry(cfg);

            expectedCodes = {'FMB', 'AMA', 'SawFM', 'SinFM', 'SN', ...
                'MFT', 'VFT', 'ISRJ', 'RangeFakeTarget', 'Traction'};
            expectedKnown = {'FMB', 'AMA', 'SawFM', 'MFT', 'ISRJ'};
            expectedUnknown = setdiff(expectedCodes, expectedKnown, 'stable');
            expectedGenerators = {'generate_fmb', 'generate_ama', ...
                'generate_sawfm', 'generate_sinfm', 'generate_smart_noise', ...
                'generate_mft', 'generate_vft', 'generate_isrj', ...
                'generate_range_fake_target', 'generate_traction'};
            actualCodes = {registry.code};
            actualGenerators = cellfun(@func2str, {registry.generator}, ...
                'UniformOutput', false);
            configuredTypes = [cfg.types.known, cfg.types.unknown];

            testCase.verifyEqual(actualCodes, expectedCodes);
            testCase.verifyEqual([registry.id], 1:10);
            testCase.verifyEqual(numel(unique(actualCodes)), 10);
            testCase.verifyEqual(cfg.radar.num_pulses, 4);
            testCase.verifyEqual(cfg.types.enabled, expectedCodes);
            testCase.verifyEqual(cfg.types.known, expectedKnown);
            testCase.verifyEqual(cfg.types.unknown, expectedUnknown);
            testCase.verifyEmpty(intersect(cfg.types.known, ...
                cfg.types.unknown, 'stable'));
            testCase.verifyEmpty(setdiff(expectedCodes, configuredTypes, 'stable'));
            testCase.verifyEmpty(setdiff(configuredTypes, expectedCodes, 'stable'));
            testCase.verifyEqual(numel(unique(configuredTypes)), ...
                numel(expectedCodes));
            testCase.verifyEqual(actualGenerators, expectedGenerators);
        end

        function eachRegistryEntryDeclaresParametersAndGenerator(testCase)
            % 验证当前默认 v2 的十三个注册项均声明完整、精确的元数据。
            registry = build_jamming_registry();

            delayNames8 = arrayfun(@(index) sprintf('delay_%02d_s', index), ...
                1:8, 'UniformOutput', false);
            amplitudeNames8 = arrayfun( ...
                @(index) sprintf('amplitude_%02d', index), ...
                1:8, 'UniformOutput', false);
            delayNames4 = arrayfun(@(index) sprintf('delay_%02d_s', index), ...
                1:4, 'UniformOutput', false);
            rangeNames4 = arrayfun(@(index) sprintf('range_%02d_m', index), ...
                1:4, 'UniformOutput', false);
            amplitudeNames4 = arrayfun( ...
                @(index) sprintf('amplitude_%02d', index), ...
                1:4, 'UniformOutput', false);
            frequencyNames8 = arrayfun( ...
                @(index) sprintf('frequency_%02d_Hz', index), ...
                1:8, 'UniformOutput', false);
            phaseNames8 = arrayfun( ...
                @(index) sprintf('phase_%02d_rad', index), ...
                1:8, 'UniformOutput', false);
            startNames32 = arrayfun( ...
                @(index) sprintf('start_%02d_s', index), ...
                1:32, 'UniformOutput', false);
            widthNames32 = arrayfun( ...
                @(index) sprintf('width_%02d_s', index), ...
                1:32, 'UniformOutput', false);
            amplitudeNames32 = arrayfun( ...
                @(index) sprintf('amplitude_%02d', index), ...
                1:32, 'UniformOutput', false);
            frequencyNames32 = arrayfun( ...
                @(index) sprintf('frequency_%02d_Hz', index), ...
                1:32, 'UniformOutput', false);
            phaseNames32 = arrayfun( ...
                @(index) sprintf('phase_%02d_rad', index), ...
                1:32, 'UniformOutput', false);
            subpulseStartNames8 = arrayfun( ...
                @(index) sprintf('subpulse_start_%02d_s', index), ...
                1:8, 'UniformOutput', false);

            expectedNames = cell(1, 13);
            expectedUnits = cell(1, 13);
            expectedNames{1} = {'JNR_dB', 'bandwidth_factor', ...
                'interference_bandwidth_Hz', 'center_frequency_Hz', ...
                'arrival_delay_s'};
            expectedUnits{1} = {'dB', '1', 'Hz', 'Hz', 's'};
            expectedNames{2} = {'JNR_dB', 'bandwidth_factor', ...
                'interference_bandwidth_Hz', 'center_frequency_Hz', ...
                'modulation_index', 'arrival_delay_s'};
            expectedUnits{2} = {'dB', '1', 'Hz', 'Hz', '1', 's'};
            expectedNames{3} = {'JNR_dB', 'sweep_bandwidth_Hz', ...
                'sweep_slope_Hz_per_s', 'sweep_period_s', 'arrival_delay_s'};
            expectedUnits{3} = {'dB', 'Hz', 'Hz/s', 's', 's'};
            expectedNames{4} = {'JNR_dB', 'center_frequency_Hz', ...
                'frequency_deviation_Hz', 'modulation_frequency_Hz', ...
                'modulation_period_s', 'arrival_delay_s'};
            expectedUnits{4} = {'dB', 'Hz', 'Hz', 'Hz', 's', 's'};
            expectedNames{5} = {'JNR_dB', 'sample_width_s', ...
                'num_replicas', 'sample_period_s', 'arrival_delay_s'};
            expectedUnits{5} = {'dB', 's', '1', 's', 's'};
            expectedNames{6} = [{'JNR_dB', 'num_targets', 'mean_delay_s', ...
                'mean_spacing_s'}, delayNames8, amplitudeNames8, ...
                {'arrival_delay_s'}];
            expectedUnits{6} = [{'dB', '1', 's', 's'}, ...
                repmat({'s'}, 1, 8), repmat({'1'}, 1, 8), {'s'}];
            expectedNames{7} = {'JNR_dB', 'doppler_Hz', ...
                'velocity_mps', 'arrival_delay_s'};
            expectedUnits{7} = {'dB', 'Hz', 'm/s', 's'};
            expectedNames{8} = {'JNR_dB', 'sample_width_s', ...
                'num_replicas', 'sample_period_s', 'arrival_delay_s', ...
                'mode_code'};
            expectedUnits{8} = {'dB', 's', '1', 's', 's', '1'};
            expectedNames{9} = [{'JNR_dB', 'num_targets', 'mean_delay_s'}, ...
                delayNames4, rangeNames4, amplitudeNames4, ...
                {'arrival_delay_s'}];
            expectedUnits{9} = [{'dB', '1', 's'}, ...
                repmat({'s'}, 1, 4), repmat({'m'}, 1, 4), ...
                repmat({'1'}, 1, 4), {'s'}];
            expectedNames{10} = {'initial_range_m', 'velocity_mps', ...
                'amplitude', 'range_change_per_pulse_m', ...
                'delay_change_per_pulse_s'};
            expectedUnits{10} = {'m', 'm/s', '1', 'm', 's'};
            expectedNames{11} = [{'JNR_dB', 'num_teeth', 'spacing_Hz', ...
                'center_offset_Hz', 'amplitude_mode_code', ...
                'arrival_delay_s'}, frequencyNames8, amplitudeNames8, ...
                phaseNames8];
            expectedUnits{11} = [{'dB', '1', 'Hz', 'Hz', '1', 's'}, ...
                repmat({'Hz'}, 1, 8), repmat({'1'}, 1, 8), ...
                repmat({'rad'}, 1, 8)];
            expectedNames{12} = [{'JNR_dB', 'mode_code', ...
                'total_pulse_count', 'mean_width_s', 'duty_cycle', ...
                'mean_amplitude', 'minimum_frequency_Hz', ...
                'maximum_frequency_Hz', 'arrival_delay_s'}, ...
                startNames32, widthNames32, amplitudeNames32, ...
                frequencyNames32, phaseNames32];
            expectedUnits{12} = [{'dB', '1', '1', 's', '1', '1', ...
                'Hz', 'Hz', 's'}, repmat({'s'}, 1, 32), ...
                repmat({'s'}, 1, 32), repmat({'1'}, 1, 32), ...
                repmat({'Hz'}, 1, 32), repmat({'rad'}, 1, 32)];
            expectedNames{13} = [{'JNR_dB', 'num_subpulses', ...
                'subpulse_width_s', 'chirp_rate_multiplier', ...
                'subpulse_chirp_rate_Hz_per_s', ...
                'subpulse_bandwidth_Hz', 'output_duration_s', ...
                'arrival_delay_s', 'initial_phase_rad'}, ...
                subpulseStartNames8];
            expectedUnits{13} = [{'dB', '1', 's', '1', 'Hz/s', 'Hz', ...
                's', 's', 'rad'}, repmat({'s'}, 1, 8)];

            for index = 1:numel(registry)
                entry = registry(index);
                testCase.verifyNotEmpty(entry.chinese_name);
                testCase.verifyNotEmpty(entry.parameter_names);
                testCase.verifyNotEmpty(entry.parameter_units);
                testCase.verifyEqual(numel(entry.parameter_names), ...
                    numel(entry.parameter_units));
                testCase.verifyClass(entry.generator, 'function_handle');
                testCase.verifyEqual(entry.parameter_names, ...
                    expectedNames{index});
                testCase.verifyEqual(entry.parameter_units, ...
                    expectedUnits{index});
                testCase.verifyEqual(numel(unique(entry.parameter_names)), ...
                    numel(entry.parameter_names));
            end
        end

        function registryAcceptsOptionalConfigArgument(testCase)
            % 验证注册表兼容无参调用和正式的配置入参调用。
            cfg = create_default_config();

            withoutConfig = build_jamming_registry();
            withConfig = build_jamming_registry(cfg);

            testCase.verifyEqual(withConfig, withoutConfig);
        end

        function normalizesDerivedRadarFieldsAndTypeLists(testCase)
            % 验证派生量写入 derived，且不会污染原始 radar 参数结构。
            cfg = create_default_config();
            cfg.types.enabled = string(cfg.types.enabled);
            cfg.types.known = string(cfg.types.known);
            cfg.types.unknown = string(cfg.types.unknown);

            actual = normalize_and_validate_config(cfg, "balanced");
            expectedBytes = cfg.dataset.num_samples * ...
                round(cfg.radar.pri_s * cfg.radar.fs_Hz) * ...
                cfg.radar.num_pulses * (16 + 8);

            testCase.verifyEqual(actual.types.enabled, cellstr(cfg.types.enabled));
            testCase.verifyEqual(actual.types.known, cellstr(cfg.types.known));
            testCase.verifyEqual(actual.types.unknown, cellstr(cfg.types.unknown));
            testCase.verifyEqual(actual.derived.samples_per_pri, 3000);
            testCase.verifyEqual(actual.derived.samples_per_pulse, 1000);
            testCase.verifyEqual(actual.derived.sample_length, 12000);
            testCase.verifyEqual(actual.derived.chirp_rate_Hz_per_s, 5e10);
            testCase.verifyEqual(actual.derived.estimated_output_bytes, ...
                expectedBytes);
            derivedNames = {'samples_per_pri', 'samples_per_pulse', ...
                'sample_length', 'chirp_rate_Hz_per_s'};
            testCase.verifyFalse(any(isfield(actual.radar, derivedNames)));
        end

        function integerInputsUseDoubleCapacityArithmetic(testCase)
            % 验证整数类输入先转 double，避免脉冲数和字节估算发生饱和。
            cfg = create_default_config();
            cfg.radar.num_pulses = uint8(4);
            cfg.dataset.num_samples = uint32(1e6);
            cfg.output.max_estimated_bytes = 3e11;

            actual = normalize_and_validate_config(cfg, 'balanced');

            testCase.verifyClass(actual.radar.num_pulses, 'double');
            testCase.verifyClass(actual.dataset.num_samples, 'double');
            testCase.verifyClass(actual.derived.sample_length, 'double');
            testCase.verifyEqual(actual.derived.sample_length, 12000);
            testCase.verifyClass( ...
                actual.derived.estimated_output_bytes, 'double');
            testCase.verifyEqual(actual.derived.estimated_output_bytes, ...
                288000000000);
        end

        function largeIntegerSampleCountExceedsDefaultLimit(testCase)
            % 验证 uint32 大样本数按真实 double 字节数触发默认容量上限。
            cfg = create_default_config();
            cfg.dataset.num_samples = uint32(1e6);

            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'balanced'), ...
                'JammingConfig:EstimatedSizeExceeded');
        end

        function rejectsCapacityScalarsAboveFlintmax(testCase)
            % 验证雷达、数据集、开放集和输出容量标量不超过 flintmax。
            tooLarge = uint64(flintmax) + uint64(1);

            cfg = create_default_config();
            cfg.radar.num_pulses = tooLarge;
            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'balanced'), ...
                'JammingConfig:InvalidRadar');

            cfg = create_default_config();
            cfg.dataset.num_samples = tooLarge;
            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'balanced'), ...
                'JammingConfig:InvalidDatasetCount');

            cfg = create_default_config();
            cfg.openset.train_known = tooLarge;
            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'openset'), ...
                'JammingConfig:InvalidOpenSetSplit');

            cfg = create_default_config();
            cfg.output.max_estimated_bytes = tooLarge;
            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'balanced'), ...
                'JammingConfig:EstimatedSizeExceeded');
        end

        function rejectsInvalidBooleanFields(testCase)
            % 验证布尔字段拒绝向量、文本、缺失数值和非 0/1 数值。
            invalidConfigs = cell(1, 6);
            invalidConfigs{1} = create_default_config();
            invalidConfigs{1}.output.save_v73 = [0, 1];
            invalidConfigs{2} = create_default_config();
            invalidConfigs{2}.output.save_legacy = 'false';
            invalidConfigs{3} = create_default_config();
            invalidConfigs{3}.radar.add_target = "false";
            invalidConfigs{4} = create_default_config();
            invalidConfigs{4}.imbalance.component.enable = NaN;
            invalidConfigs{5} = create_default_config();
            invalidConfigs{5}.imbalance.combination.enable = 2;
            invalidConfigs{6} = create_default_config();
            invalidConfigs{6}.imbalance.power.enable = [true, false];

            for index = 1:numel(invalidConfigs)
                testCase.verifyError(@() normalize_and_validate_config( ...
                    invalidConfigs{index}, 'balanced'), ...
                    'JammingConfig:InvalidBoolean');
            end
        end

        function normalizesNumericBooleanFields(testCase)
            % 验证数值 0/1 开关统一转换为 logical false/true。
            cfg = create_default_config();
            cfg.output.save_v73 = 0;
            cfg.output.save_legacy = 1;
            cfg.radar.add_target = 0;
            cfg.imbalance.component.enable = 1;
            cfg.imbalance.combination.enable = 0;
            cfg.imbalance.power.enable = 1;

            actual = normalize_and_validate_config(cfg, 'balanced');

            booleanValues = [actual.output.save_v73, ...
                actual.output.save_legacy, actual.radar.add_target, ...
                actual.imbalance.component.enable, ...
                actual.imbalance.combination.enable, ...
                actual.imbalance.power.enable];
            testCase.verifyClass(booleanValues, 'logical');
            testCase.verifyEqual(booleanValues, ...
                [false, true, false, true, false, true]);
        end

        function normalizesWeightAndRatioVectorsToDoubleRows(testCase)
            % 验证默认 v2 的十三维分量权重等向量统一为 double 行向量。
            cfg = create_default_config();
            cfg.imbalance.component.enable = true;
            cfg.imbalance.component.weights = single(ones(13, 1));
            cfg.imbalance.combination.enable = true;
            cfg.imbalance.combination.cardinality_weights = ...
                uint8([1; 1; 1]);
            cfg.openset.mixed_ratios = single([0.25; 0.25; 0.35; 0.15]);

            actual = normalize_and_validate_config(cfg, 'openset');

            testCase.verifyClass(actual.imbalance.component.weights, ...
                'double');
            testCase.verifySize(actual.imbalance.component.weights, [1, 13]);
            testCase.verifyClass( ...
                actual.imbalance.combination.cardinality_weights, 'double');
            testCase.verifySize( ...
                actual.imbalance.combination.cardinality_weights, [1, 3]);
            testCase.verifyClass(actual.openset.mixed_ratios, 'double');
            testCase.verifySize(actual.openset.mixed_ratios, [1, 4]);
        end

        function rejectsUnknownScenario(testCase)
            % 验证场景名称仅接受 balanced、imbalanced 和 openset。
            cfg = create_default_config();

            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'unsupported'), ...
                'JammingConfig:UnknownScenario');
        end

        function rejectsUnknownTypeInEveryTypeList(testCase)
            % 验证 enabled、known、unknown 中的未知类型均使用稳定错误标识。
            fields = {'enabled', 'known', 'unknown'};
            for index = 1:numel(fields)
                cfg = create_default_config();
                cfg.types.(fields{index}){end + 1} = 'NotAType';
                testCase.verifyError( ...
                    @() normalize_and_validate_config(cfg, 'balanced'), ...
                    'JammingConfig:UnknownType');
            end
        end

        function rejectsDuplicateTypeInEveryTypeList(testCase)
            % 验证 enabled、known、unknown 中的重复类型均被拒绝。
            fields = {'enabled', 'known', 'unknown'};
            for index = 1:numel(fields)
                cfg = create_default_config();
                cfg.types.(fields{index}){end + 1} = ...
                    cfg.types.(fields{index}){1};
                testCase.verifyError( ...
                    @() normalize_and_validate_config(cfg, 'balanced'), ...
                    'JammingConfig:DuplicateType');
            end
        end

        function rejectsKnownOrUnknownOutsideEnabled(testCase)
            % 验证 known 和 unknown 必须是 enabled 的子集。
            cfg = create_default_config();
            cfg.types.enabled = setdiff(cfg.types.enabled, {'FMB'}, 'stable');

            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'balanced'), ...
                'JammingConfig:UnknownType');
        end

        function rejectsOverlappingKnownAndUnknown(testCase)
            % 验证开放集的已知类和未知类不能包含同一干扰类型。
            cfg = create_default_config();
            cfg.types.known = {'FMB', 'ISRJ'};
            cfg.types.unknown = {'ISRJ', 'VFT'};

            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'openset'), ...
                'JammingConfig:KnownUnknownOverlap');
        end

        function openSetRequiresKnownTypes(testCase)
            % 验证开放集场景至少声明一个已知类。
            cfg = create_default_config();
            cfg.types.known = {};

            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'openset'), ...
                'JammingConfig:OpenSetRequiresKnown');
        end

        function openSetRequiresUnknownTypes(testCase)
            % 验证开放集场景至少声明一个未知类。
            cfg = create_default_config();
            cfg.types.unknown = {};

            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'openset'), ...
                'JammingConfig:OpenSetRequiresUnknown');
        end

        function rejectsInvalidRadarValues(testCase)
            % 验证雷达正值约束、脉宽约束和整数脉冲数共用稳定标识。
            invalidConfigs = cell(1, 3);
            invalidConfigs{1} = create_default_config();
            invalidConfigs{1}.radar.fs_Hz = 0;
            invalidConfigs{2} = create_default_config();
            invalidConfigs{2}.radar.pulse_width_s = ...
                2 * invalidConfigs{2}.radar.pri_s;
            invalidConfigs{3} = create_default_config();
            invalidConfigs{3}.radar.num_pulses = 1.5;

            for index = 1:numel(invalidConfigs)
                testCase.verifyError(@() normalize_and_validate_config( ...
                    invalidConfigs{index}, 'balanced'), ...
                    'JammingConfig:InvalidRadar');
            end
        end

        function rejectsRoundedZeroDerivedSampleCounts(testCase)
            % 验证采样率过低导致 PRI 或脉冲样点数舍入为零时拒绝配置。
            invalidConfigs = cell(1, 2);
            invalidConfigs{1} = create_default_config();
            invalidConfigs{1}.radar.fs_Hz = 1;
            invalidConfigs{2} = create_default_config();
            invalidConfigs{2}.radar.fs_Hz = 2000;

            for index = 1:numel(invalidConfigs)
                testCase.verifyError(@() normalize_and_validate_config( ...
                    invalidConfigs{index}, 'balanced'), ...
                    'JammingConfig:InvalidRadar');
            end
        end

        function rejectsDescendingOrNonfiniteParameterRange(testCase)
            % 验证 params 内下界大于上界或含非有限值的范围均被拒绝。
            cfg = create_default_config();
            cfg.params.FMB.bandwidth_factor = [5, 2];
            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'balanced'), ...
                'JammingConfig:InvalidParameterRange');

            cfg = create_default_config();
            cfg.params.common.JNR_dB = [20, Inf];
            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'balanced'), ...
                'JammingConfig:InvalidParameterRange');
        end

        function rejectsInvalidIntegerRanges(testCase)
            % 验证目标数、副本数和最大目标数必须为合法正整数。
            invalidConfigs = cell(1, 4);
            invalidConfigs{1} = create_default_config();
            invalidConfigs{1}.params.MFT.num_targets = [0, 3];
            invalidConfigs{2} = create_default_config();
            invalidConfigs{2}.params.RangeFakeTarget.num_targets = [1, 2.5];
            invalidConfigs{3} = create_default_config();
            invalidConfigs{3}.params.SN.num_replicas = [1, 1.5];
            invalidConfigs{4} = create_default_config();
            invalidConfigs{4}.params.MFT.max_targets = 7.5;

            for index = 1:numel(invalidConfigs)
                testCase.verifyError(@() normalize_and_validate_config( ...
                    invalidConfigs{index}, 'balanced'), ...
                    'JammingConfig:InvalidIntegerRange');
            end
        end

        function rejectsMaxTargetsBelowConfiguredUpperBound(testCase)
            % 验证 MFT 和距离假目标的最大容量覆盖 num_targets 上界。
            cfg = create_default_config();
            cfg.params.MFT.max_targets = 7;

            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'balanced'), ...
                'JammingConfig:InvalidIntegerRange');
        end

        function tractionRequiresMultiplePulsesWhenEnabled(testCase)
            % 验证启用拖引干扰时至少需要两个脉冲描述脉间变化。
            cfg = create_default_config();
            cfg.radar.num_pulses = 1;

            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'balanced'), ...
                'JammingConfig:TractionRequiresMultiplePulses');
        end

        function tractionTrajectoryMustKeepPulseInsidePri(testCase)
            % 验证最坏拖引轨迹的传播时延与脉宽之和不能超过 PRI。
            cfg = create_default_config();
            nearWrapDelay_s = cfg.radar.pri_s - ...
                cfg.radar.pulse_width_s / 2;
            nearWrapRange_m = nearWrapDelay_s * ...
                cfg.radar.c_mps / 2;
            cfg.params.Traction.initial_range_m = ...
                [nearWrapRange_m, nearWrapRange_m];
            cfg.params.Traction.velocity_mps = [0, 0];

            testCase.verifyError(@() normalize_and_validate_config( ...
                cfg, 'balanced'), ...
                'JammingConfig:InvalidPhysicalParameter');
        end

        function enabledReplicaRangesMustFitRecord(testCase)
            % 验证启用 SN/ISRJ 时基段和全部副本必须完整位于记录内。
            codes = {'SN', 'ISRJ'};
            for codeIndex = 1:numel(codes)
                cfg = create_default_config();
                code = codes{codeIndex};
                cfg.radar.pri_s = 1e-6;
                cfg.radar.pulse_width_s = 0.5e-6;
                cfg.radar.num_pulses = 1;
                cfg.types.enabled = {code};
                cfg.types.known = {code};
                cfg.types.unknown = {};
                cfg.dataset.min_components = 1;
                cfg.dataset.max_components = 1;
                cfg.params.common.arrival_delay_s = [0, 0];
                cfg.params.(codes{codeIndex}).sample_width_s = ...
                    [3 / cfg.radar.fs_Hz, 3 / cfg.radar.fs_Hz];
                cfg.params.(codes{codeIndex}).num_replicas = [3, 3];

                testCase.verifyError(@() normalize_and_validate_config( ...
                    cfg, 'balanced'), ...
                    'JammingConfig:InvalidPhysicalParameter');

                cfg.params.(codes{codeIndex}).sample_width_s = ...
                    [2 / cfg.radar.fs_Hz, 2 / cfg.radar.fs_Hz];
                actual = normalize_and_validate_config(cfg, 'balanced');
                testCase.verifyEqual(actual.derived.sample_length, 10);
            end
        end

        function enabledTargetAmplitudesMustBeStrictlyPositive(testCase)
            % 验证启用 MFT/RangeFakeTarget 时相对幅度范围严格为正。
            owners = {'MFT', 'RangeFakeTarget'};
            invalidRanges = {[0, 1], [-1, -0.5]};
            for ownerIndex = 1:numel(owners)
                for rangeIndex = 1:numel(invalidRanges)
                    cfg = create_default_config();
                    cfg.params.(owners{ownerIndex}). ...
                        target_relative_amplitude = ...
                        invalidRanges{rangeIndex};
                    testCase.verifyError( ...
                        @() normalize_and_validate_config(cfg, 'balanced'), ...
                        'JammingConfig:InvalidPhysicalParameter');
                end
            end
        end

        function targetDelayAndArrivalMustKeepPulseInsidePri(testCase)
            % 验证假目标最坏总时延加脉宽不得越过当前 PRI 边界。
            codes = {'MFT', 'RangeFakeTarget'};
            for codeIndex = 1:numel(codes)
                cfg = create_default_config();
                cfg.types.enabled = codes(codeIndex);
                cfg.types.known = codes(codeIndex);
                cfg.types.unknown = {};
                cfg.dataset.min_components = 1;
                cfg.dataset.max_components = 1;
                cfg.params.common.arrival_delay_s = [50e-6, 50e-6];
                cfg.params.(codes{codeIndex}).target_delay_s = ...
                    [151e-6, 151e-6];

                testCase.verifyError(@() normalize_and_validate_config( ...
                    cfg, 'balanced'), ...
                    'JammingConfig:InvalidPhysicalParameter');
            end
        end

        function targetDelaySlotsMustCoverMaximumTargetCount(testCase)
            % 验证量化后的可选时延槽数必须容纳配置目标数上界。
            cases = {'MFT', 4; 'RangeFakeTarget', 3};
            for caseIndex = 1:size(cases, 1)
                code = cases{caseIndex, 1};
                cfg = create_default_config();
                cfg.types.enabled = {code};
                cfg.types.known = {code};
                cfg.types.unknown = {};
                cfg.dataset.min_components = 1;
                cfg.dataset.max_components = 1;
                cfg.params.common.arrival_delay_s = [0, 0];
                cfg.params.(code).num_targets = ...
                    repmat(cases{caseIndex, 2}, 1, 2);
                cfg.params.(code).target_delay_s = ...
                    [0, 1 / cfg.radar.fs_Hz];

                testCase.verifyError(@() normalize_and_validate_config( ...
                    cfg, 'balanced'), ...
                    'JammingConfig:InvalidPhysicalParameter');
            end
        end

        function disabledTypePhysicalValuesDoNotBlockEnabledSubset(testCase)
            % 验证禁用 SawFM 的零扫频周期不阻断仅启用 FMB 的配置。
            cfg = create_default_config();
            cfg.types.enabled = {'FMB'};
            cfg.types.known = {'FMB'};
            cfg.types.unknown = {};
            cfg.dataset.min_components = 1;
            cfg.dataset.max_components = 1;
            cfg.params.SawFM.sweep_period_s = [0, 0];

            actual = normalize_and_validate_config(cfg, 'balanced');

            testCase.verifyEqual(actual.types.enabled, {'FMB'});
        end

        function vftRangeMustIntersectReferenceAwareFrequencyDomain(testCase)
            % 验证 VFT 多普勒必须扣除参考 LFM 已占用的离散频率支持。
            cfg = create_default_config();
            cfg.params.VFT.doppler_Hz = [30e3, 30e3];

            testCase.verifyError(@() normalize_and_validate_config( ...
                cfg, 'balanced'), ...
                'JammingConfig:InvalidPhysicalParameter');
        end

        function fmbAmaCenterRangesMustStayStrictlyInsideGuard(testCase)
            % 验证 FMB/AMA 中心频率范围任一端触及 guard 都在配置层拒绝。
            codes = {'FMB', 'AMA'};
            for codeIndex = 1:numel(codes)
                cfg = create_default_config();
                guard_Hz = cfg.radar.fs_Hz / 2 - max( ...
                    eps(cfg.radar.fs_Hz / 2), cfg.radar.fs_Hz * 1e-12);
                cfg.params.(codes{codeIndex}).center_frequency_Hz = ...
                    [0, guard_Hz];

                testCase.verifyError(@() normalize_and_validate_config( ...
                    cfg, 'balanced'), ...
                    'JammingConfig:InvalidPhysicalParameter');
            end
        end

        function fmbAmaRequireNonzeroConjugateFftSupport(testCase)
            % 验证最小请求带宽在最不利中心频率处至少覆盖一对非零
            % 共轭 FFT 频点；只剩 DC 的配置必须在随机采样前被拒绝。
            codes = {'FMB', 'AMA'};
            for codeIndex = 1:numel(codes)
                cfg = create_default_config();
                cfg.params.(codes{codeIndex}).bandwidth_factor = [1e-5, 1e-5];

                testCase.verifyError(@() normalize_and_validate_config( ...
                    cfg, 'balanced'), ...
                    'JammingConfig:InvalidPhysicalParameter');
            end
        end

        function fmbAmaCheckWorstCenterForDiscreteNoiseSupport(testCase)
            % 验证中心频率范围靠近 Nyquist 时使用最不利端点预检；即使
            % 请求带宽很大，裁剪后不足一对非零频点也必须拒绝。
            codes = {'FMB', 'AMA'};
            for codeIndex = 1:numel(codes)
                cfg = create_default_config();
                sampleLength = round(cfg.radar.pri_s * ...
                    cfg.radar.fs_Hz) * cfg.radar.num_pulses;
                frequencyStep_Hz = cfg.radar.fs_Hz / sampleLength;
                guard_Hz = cfg.radar.fs_Hz / 2 - max( ...
                    eps(cfg.radar.fs_Hz / 2), cfg.radar.fs_Hz * 1e-12);
                worstCenter_Hz = guard_Hz - 0.75 * frequencyStep_Hz;
                cfg.params.(codes{codeIndex}).center_frequency_Hz = ...
                    [0, worstCenter_Hz];
                cfg.params.(codes{codeIndex}).bandwidth_factor = [2, 2];

                testCase.verifyError(@() normalize_and_validate_config( ...
                    cfg, 'balanced'), ...
                    'JammingConfig:InvalidPhysicalParameter');
            end
        end

        function rejectsNonphysicalJammingParameterRanges(testCase)
            % 验证时间、带宽因子、频偏和拖引参数满足基本物理定义域。
            invalidConfigs = cell(1, 16);
            invalidConfigs{1} = create_default_config();
            invalidConfigs{1}.params.common.arrival_delay_s = [-1e-6, 0];
            invalidConfigs{2} = create_default_config();
            invalidConfigs{2}.params.common.arrival_delay_s = ...
                [0, invalidConfigs{2}.radar.pri_s];
            invalidConfigs{3} = create_default_config();
            invalidConfigs{3}.params.SawFM.sweep_period_s = [0, 1e-6];
            invalidConfigs{4} = create_default_config();
            invalidConfigs{4}.params.SinFM.modulation_frequency_Hz = [0, 1];
            invalidConfigs{5} = create_default_config();
            invalidConfigs{5}.params.SinFM.frequency_deviation_Hz = [-1, 0];
            invalidConfigs{6} = create_default_config();
            invalidConfigs{6}.params.SN.sample_width_s = [0, 1e-6];
            invalidConfigs{7} = create_default_config();
            invalidConfigs{7}.params.ISRJ.sample_width_s = [0, 1e-6];
            invalidConfigs{8} = create_default_config();
            invalidConfigs{8}.params.MFT.target_delay_s = [-1e-6, 0];
            invalidConfigs{9} = create_default_config();
            invalidConfigs{9}.params.MFT.target_delay_s = ...
                [0, invalidConfigs{9}.radar.pri_s];
            invalidConfigs{10} = create_default_config();
            invalidConfigs{10}.params.RangeFakeTarget.target_delay_s = ...
                [-1e-6, 0];
            invalidConfigs{11} = create_default_config();
            invalidConfigs{11}.params.RangeFakeTarget.target_delay_s = ...
                [0, invalidConfigs{11}.radar.pri_s];
            invalidConfigs{12} = create_default_config();
            invalidConfigs{12}.params.FMB.bandwidth_factor = [0, 1];
            invalidConfigs{13} = create_default_config();
            invalidConfigs{13}.params.AMA.bandwidth_factor = [0, 1];
            invalidConfigs{14} = create_default_config();
            invalidConfigs{14}.params.Traction.initial_range_m = [-1, 0];
            invalidConfigs{15} = create_default_config();
            invalidConfigs{15}.params.Traction.amplitude = [0, 1];
            invalidConfigs{16} = create_default_config();
            invalidConfigs{16}.params.Traction.amplitude = [-1, -0.5];

            for index = 1:numel(invalidConfigs)
                testCase.verifyError(@() normalize_and_validate_config( ...
                    invalidConfigs{index}, 'balanced'), ...
                    'JammingConfig:InvalidPhysicalParameter');
            end
        end

        function targetTruthCapacitiesAreFixedByRegistrySchema(testCase)
            % 验证 MFT/RangeFakeTarget 容量固定为 8/4，范围上界不可越界。
            invalidConfigs = cell(1, 6);
            invalidConfigs{1} = create_default_config();
            invalidConfigs{1}.params.MFT.max_targets = 7;
            invalidConfigs{2} = create_default_config();
            invalidConfigs{2}.params.MFT.max_targets = 9;
            invalidConfigs{3} = create_default_config();
            invalidConfigs{3}.params.MFT.num_targets = [1, 9];
            invalidConfigs{4} = create_default_config();
            invalidConfigs{4}.params.RangeFakeTarget.max_targets = 3;
            invalidConfigs{5} = create_default_config();
            invalidConfigs{5}.params.RangeFakeTarget.max_targets = 5;
            invalidConfigs{6} = create_default_config();
            invalidConfigs{6}.params.RangeFakeTarget.num_targets = [1, 5];

            for index = 1:numel(invalidConfigs)
                testCase.verifyError(@() normalize_and_validate_config( ...
                    invalidConfigs{index}, 'balanced'), ...
                    'JammingConfig:InvalidIntegerRange');
            end
        end

        function rejectsInvalidDatasetCounts(testCase)
            % 验证数据集样本数和每模式样本数必须为正整数。
            cfg = create_default_config();
            cfg.dataset.num_samples = 1.5;
            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'balanced'), ...
                'JammingConfig:InvalidDatasetCount');

            cfg = create_default_config();
            cfg.dataset.samples_per_pattern = 0;
            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'balanced'), ...
                'JammingConfig:InvalidDatasetCount');
        end

        function rejectsInvalidComponentCounts(testCase)
            % 验证分量数为正整数、上下界有序且不超过启用类型数。
            invalidConfigs = cell(1, 3);
            invalidConfigs{1} = create_default_config();
            invalidConfigs{1}.dataset.min_components = 0;
            invalidConfigs{2} = create_default_config();
            invalidConfigs{2}.dataset.min_components = 3;
            invalidConfigs{2}.dataset.max_components = 2;
            invalidConfigs{3} = create_default_config();
            invalidConfigs{3}.dataset.max_components = 14;

            for index = 1:numel(invalidConfigs)
                testCase.verifyError(@() normalize_and_validate_config( ...
                    invalidConfigs{index}, 'balanced'), ...
                    'JammingConfig:InvalidComponentCount');
            end
        end

        function rejectsInvalidComponentWeights(testCase)
            % 验证分量权重不能为负、全零或与启用类型数量不匹配。
            invalidWeights = {[-1, ones(1, 12)], zeros(1, 13), ones(1, 12)};
            for index = 1:numel(invalidWeights)
                cfg = create_default_config();
                cfg.imbalance.component.enable = true;
                cfg.imbalance.component.weights = invalidWeights{index};
                testCase.verifyError( ...
                    @() normalize_and_validate_config(cfg, 'imbalanced'), ...
                    'JammingConfig:InvalidImbalanceWeights');
            end
        end

        function rejectsInvalidCardinalityWeights(testCase)
            % 验证组合基数权重覆盖最大分量数且使用部分非负、非全零。
            invalidWeights = {[1, 1], [1, -1, 1], [0, 0, 0]};
            for index = 1:numel(invalidWeights)
                cfg = create_default_config();
                cfg.imbalance.combination.enable = true;
                cfg.imbalance.combination.cardinality_weights = ...
                    invalidWeights{index};
                testCase.verifyError( ...
                    @() normalize_and_validate_config(cfg, 'imbalanced'), ...
                    'JammingConfig:InvalidImbalanceWeights');
            end
        end

        function cardinalityPositiveWeightMustBeAllowed(testCase)
            % 验证正基数权重必须落在 min_components:max_components 区间。
            cfg = create_default_config();
            cfg.dataset.min_components = 2;
            cfg.dataset.max_components = 3;
            cfg.imbalance.combination.enable = true;
            cfg.imbalance.combination.cardinality_weights = [1, 0, 0];

            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'imbalanced'), ...
                'JammingConfig:InvalidImbalanceWeights');
        end

        function rejectsInvalidPatternWeights(testCase)
            % 验证模式权重表列完整，且权重有限、非负并至少有一个正值。
            invalidTables = {table(1, 'VariableNames', {'weight'}), ...
                table({'FMB+AMA'}', -1, ...
                'VariableNames', {'pattern_codes', 'weight'}), ...
                table({'FMB+AMA'; 'FMB+SN'}, [0; 0], ...
                'VariableNames', {'pattern_codes', 'weight'})};
            for index = 1:numel(invalidTables)
                cfg = create_default_config();
                cfg.imbalance.combination.enable = true;
                cfg.imbalance.combination.pattern_weights = ...
                    invalidTables{index};
                testCase.verifyError( ...
                    @() normalize_and_validate_config(cfg, 'imbalanced'), ...
                    'JammingConfig:InvalidImbalanceWeights');
            end
        end

        function rejectsInvalidRelativePowerRange(testCase)
            % 验证相对功率范围也遵循有限且下界不大于上界的约束。
            cfg = create_default_config();
            cfg.imbalance.power.enable = true;
            cfg.imbalance.power.relative_power_dB = [-6, -20];

            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'imbalanced'), ...
                'JammingConfig:InvalidParameterRange');
        end

        function rejectsInvalidOpenSetRatios(testCase)
            % 验证开放集四类混合比例非负、有限且总和严格归一。
            invalidRatios = {[0.2, 0.2, 0.2, 0.2], ...
                [0.25, -0.25, 0.5, 0.5], [0.5, 0.5, 0]};
            for index = 1:numel(invalidRatios)
                cfg = create_default_config();
                cfg.openset.mixed_ratios = invalidRatios{index};
                testCase.verifyError( ...
                    @() normalize_and_validate_config(cfg, 'openset'), ...
                    'JammingConfig:InvalidOpenSetRatios');
            end
        end

        function rejectsComplexOpenSetRatios(testCase)
            % 验证总和为一的复数 mixed_ratios 仍属于非法开放集比例。
            cfg = create_default_config();
            cfg.openset.mixed_ratios = ...
                [0.25 + 1i, 0.25 - 1i, 0.25, 0.25];

            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'openset'), ...
                'JammingConfig:InvalidOpenSetRatios');
        end

        function rejectsInvalidOpenSetSplitCounts(testCase)
            % 验证训练和校准数量为正整数，验证和测试数量为非负整数。
            invalidConfigs = cell(1, 2);
            invalidConfigs{1} = create_default_config();
            invalidConfigs{1}.openset.train_known = 0;
            invalidConfigs{2} = create_default_config();
            invalidConfigs{2}.openset.validation_mixed = -1;

            for index = 1:numel(invalidConfigs)
                testCase.verifyError(@() normalize_and_validate_config( ...
                    invalidConfigs{index}, 'openset'), ...
                    'JammingConfig:InvalidOpenSetSplit');
            end
        end

        function openSetEstimateUsesAllFourSplits(testCase)
            % 验证开放集输出估算使用四个划分数量之和。
            cfg = create_default_config();
            actual = normalize_and_validate_config(cfg, 'openset');
            totalSamples = cfg.openset.train_known + ...
                cfg.openset.calibration_known + ...
                cfg.openset.validation_mixed + cfg.openset.test_mixed;
            expectedBytes = totalSamples * actual.derived.sample_length * 24;

            testCase.verifyEqual(actual.derived.estimated_output_bytes, ...
                expectedBytes);
        end

        function rejectsEstimatedOutputAboveConfiguredLimit(testCase)
            % 验证复数数据和 legacy 数据合计估算超过上限时立即拒绝。
            cfg = create_default_config();
            cfg.output.max_estimated_bytes = 1;

            testCase.verifyError( ...
                @() normalize_and_validate_config(cfg, 'balanced'), ...
                'JammingConfig:EstimatedSizeExceeded');
        end

        function disabledComponentWeightsFollowEnabledSubset(testCase)
            % 验证关闭分量不平衡时，权重按三类 enabled 确定性归一为全一。
            cfg = create_default_config();
            cfg.types.enabled = {'FMB', 'ISRJ', 'VFT'};
            cfg.types.known = cfg.types.enabled;
            cfg.types.unknown = {};

            actual = normalize_and_validate_config(cfg, 'balanced');

            testCase.verifyEqual(actual.imbalance.component.weights, ...
                ones(1, 3));
        end

        function openSetAllowsFiveEnabledTypesWithDefaultWeights(testCase)
            % 验证开放集选择五类时，关闭的默认十维权重不会阻塞配置。
            cfg = create_default_config();
            cfg.types.enabled = {'FMB', 'AMA', 'SawFM', 'ISRJ', 'VFT'};
            cfg.types.known = {'FMB', 'AMA', 'SawFM'};
            cfg.types.unknown = {'ISRJ', 'VFT'};

            actual = normalize_and_validate_config(cfg, 'openset');

            testCase.verifyEqual(actual.imbalance.component.weights, ...
                ones(1, 5));
        end

        function disabledCombinationWeightsDoNotBlockConfiguration(testCase)
            % 验证关闭组合不平衡时，不校验未使用的基数和模式权重。
            cfg = create_default_config();
            cfg.imbalance.combination.enable = false;
            cfg.imbalance.combination.cardinality_weights = -1;
            cfg.imbalance.combination.pattern_weights = 42;

            actual = normalize_and_validate_config(cfg, 'balanced');

            testCase.verifyEqual(actual.imbalance.combination.enable, false);
        end

        function disabledPowerImbalanceDoesNotBlockConfiguration(testCase)
            % 验证关闭功率不平衡时，不校验未使用的相对功率范围。
            cfg = create_default_config();
            cfg.imbalance.power.enable = false;
            cfg.imbalance.power.relative_power_dB = 'unused';

            actual = normalize_and_validate_config(cfg, 'balanced');

            testCase.verifyEqual(actual.imbalance.power.enable, false);
        end

        function rejectsScalarColumnOrTextParameterRange(testCase)
            % 验证普通参数范围必须是有限数值 1×2 行向量。
            invalidValues = {2, [2; 5], 'invalid'};
            for index = 1:numel(invalidValues)
                cfg = create_default_config();
                cfg.params.FMB.bandwidth_factor = invalidValues{index};
                testCase.verifyError( ...
                    @() normalize_and_validate_config(cfg, 'balanced'), ...
                    'JammingConfig:InvalidParameterRange');
            end
        end

        function rejectsInvalidTargetRadarFields(testCase)
            % 验证目标距离和目标信噪比的类型和值域。
            invalidConfigs = cell(1, 2);
            invalidConfigs{1} = create_default_config();
            invalidConfigs{1}.radar.add_target = true;
            invalidConfigs{1}.radar.target_range_m = -1;
            invalidConfigs{2} = create_default_config();
            invalidConfigs{2}.radar.add_target = true;
            invalidConfigs{2}.radar.target_SNR_dB = NaN;

            for index = 1:numel(invalidConfigs)
                testCase.verifyError(@() normalize_and_validate_config( ...
                    invalidConfigs{index}, 'balanced'), ...
                    'JammingConfig:InvalidRadar');
            end
        end

        function rejectsInvalidPatternCodeContents(testCase)
            % 验证模式文本中的类型、重复分量、阶数和重复组合均合法。
            invalidTables = { ...
                table({'FMB+NotAType'}', 1, ...
                'VariableNames', {'pattern_codes', 'weight'}), ...
                table({'FMB+FMB'}', 1, ...
                'VariableNames', {'pattern_codes', 'weight'}), ...
                table({'FMB+ISRJ'; 'ISRJ+FMB'}, [1; 1], ...
                'VariableNames', {'pattern_codes', 'weight'}), ...
                table({'FMB+AMA+SN+MFT'}', 1, ...
                'VariableNames', {'pattern_codes', 'weight'}), ...
                table(42, 1, ...
                'VariableNames', {'pattern_codes', 'weight'})};
            for index = 1:numel(invalidTables)
                cfg = create_default_config();
                cfg.imbalance.combination.enable = true;
                cfg.imbalance.combination.pattern_weights = ...
                    invalidTables{index};
                testCase.verifyError( ...
                    @() normalize_and_validate_config(cfg, 'imbalanced'), ...
                    'JammingConfig:InvalidImbalanceWeights');
            end
        end

        function acceptsValidPatternWeightTable(testCase)
            % 验证合法文本组合和正权重表可以完成配置标准化。
            cfg = create_default_config();
            cfg.imbalance.combination.enable = true;
            cfg.imbalance.combination.pattern_weights = table( ...
                {'FMB+ISRJ'; 'AMA+SN+VFT'}, [1; 2], ...
                'VariableNames', {'pattern_codes', 'weight'});

            actual = normalize_and_validate_config(cfg, 'imbalanced');

            testCase.verifyEqual(height( ...
                actual.imbalance.combination.pattern_weights), 2);
        end

        function schemaColumnsAreUniqueAndOwned(testCase)
            % 验证默认 v2 的 295 列模式连续、唯一且归属明确。
            cfg = create_default_config();
            normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry();
            schema = build_parameter_schema(registry);

            testCase.verifyEqual(schema.Properties.VariableNames, ...
                {'column_index', 'field_name', 'owner_code', 'unit'});
            testCase.verifyEqual(schema.column_index, (1:height(schema))');
            testCase.verifyEqual(numel(unique(schema.field_name)), ...
                height(schema));
            testCase.verifyTrue(all(ismember(schema.owner_code, ...
                {registry.code}')));
            testCase.verifyTrue(all(~cellfun(@isempty, schema.unit)));
            testCase.verifyEqual(schema.field_name{1}, 'FMB.JNR_dB');
            testCase.verifyEqual(height(schema), 295);
            testCase.verifyEqual(schema.field_name{end}, ...
                'SpectralDispersion.subpulse_start_08_s');
        end

        function schemaRejectsDuplicateFullFieldName(testCase)
            % 验证同一 owner_code 下出现重复参数全名时拒绝构建模式。
            registry = build_jamming_registry();
            registry(1).parameter_names{end + 1} = 'JNR_dB';
            registry(1).parameter_units{end + 1} = 'dB';

            testCase.verifyError(@() build_parameter_schema(registry), ...
                'JammingSchema:DuplicateField');
        end

        function schemaRejectsMalformedEntryMetadata(testCase)
            % 验证元数据长度、参数文本和 owner code 畸形时使用统一标识。
            invalidRegistries = cell(1, 3);
            invalidRegistries{1} = build_jamming_registry();
            invalidRegistries{1}(1).parameter_units(end) = [];
            invalidRegistries{2} = build_jamming_registry();
            invalidRegistries{2}(1).parameter_names{1} = 42;
            invalidRegistries{3} = build_jamming_registry();
            invalidRegistries{3}(1).code = '';

            for index = 1:numel(invalidRegistries)
                testCase.verifyError( ...
                    @() build_parameter_schema(invalidRegistries{index}), ...
                    'JammingSchema:InvalidRegistryEntry');
            end
        end

        function schemaRejectsEmptyOrNonstructRegistry(testCase)
            % 验证 schema 入口拒绝空注册表和非结构体输入。
            testCase.verifyError(@() build_parameter_schema([]), ...
                'JammingSchema:InvalidRegistryEntry');
            testCase.verifyError(@() build_parameter_schema(42), ...
                'JammingSchema:InvalidRegistryEntry');
        end
    end
end
