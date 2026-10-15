classdef TestSingleJammingGenerators < matlab.unittest.TestCase
    %TESTSINGLEJAMMINGGENERATORS 验证单干扰生成阶段的公共信号工具。
    %   测试覆盖参考 LFM 脉冲及其 PRI 展平规则、配置边界校验、分量功率
    %   标定精度、波形相位保持、输入形状与数值类型转换。

    methods (Test)
        function referenceHasConfiguredFlattenedLength(testCase)
            % 验证参考脉冲与脉冲串均为按配置长度展平的 double 行向量。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');

            reference = build_reference_signal(cfg);

            testCase.verifyEqual(fieldnames(reference), ...
                {'pulse'; 'train'; 'time_pri_s'; 'time_train_s'});
            testCase.verifySize(reference.pulse, ...
                [1, cfg.derived.samples_per_pri]);
            testCase.verifySize(reference.train, ...
                [1, cfg.derived.sample_length]);
            testCase.verifySize(reference.time_pri_s, size(reference.pulse));
            testCase.verifySize(reference.time_train_s, size(reference.train));
            testCase.verifyClass(reference.pulse, 'double');
            testCase.verifyClass(reference.train, 'double');
            testCase.verifyClass(reference.time_pri_s, 'double');
            testCase.verifyClass(reference.time_train_s, 'double');
            referenceFields = fieldnames(reference);
            for fieldIndex = 1:numel(referenceFields)
                values = reference.(referenceFields{fieldIndex});
                testCase.verifyTrue(all(isfinite(real(values))));
                testCase.verifyTrue(all(isfinite(imag(values))));
            end
        end

        function referencePulseContainsLfmThenZeros(testCase)
            % 验证 PRI 前段为单位幅度 LFM，后段严格补零且逐 PRI 重复。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            reference = build_reference_signal(cfg);
            pulseLength = cfg.derived.samples_per_pulse;
            priLength = cfg.derived.samples_per_pri;

            testCase.verifyEqual(abs(reference.pulse(1:pulseLength)), ...
                ones(1, pulseLength), 'AbsTol', 10 * eps);
            testCase.verifyEqual(reference.pulse(pulseLength + 1:end), ...
                zeros(1, priLength - pulseLength));
            for pulseIndex = 1:cfg.radar.num_pulses
                blockStart = (pulseIndex - 1) * priLength + 1;
                blockStop = pulseIndex * priLength;
                testCase.verifyEqual( ...
                    reference.train(blockStart:blockStop), reference.pulse);
            end
        end

        function referenceUsesConfiguredChirpRate(testCase)
            % 验证有效脉冲严格采用配置派生调频斜率生成二次相位。
            cfg = create_default_config();
            cfg.radar.bandwidth_Hz = 2.5e6;
            cfg = normalize_and_validate_config(cfg, 'balanced');
            pulseLength = cfg.derived.samples_per_pulse;
            time = (0:pulseLength - 1) / cfg.radar.fs_Hz;
            expected = exp(1i * pi * ...
                cfg.derived.chirp_rate_Hz_per_s * time .^ 2);

            reference = build_reference_signal(cfg);

            testCase.verifyEqual(reference.pulse(1:pulseLength), ...
                expected, 'AbsTol', 1e-12);
            testCase.verifyEqual(reference.time_pri_s, ...
                (0:cfg.derived.samples_per_pri - 1) / cfg.radar.fs_Hz);
            testCase.verifyEqual(reference.time_train_s, ...
                (0:cfg.derived.sample_length - 1) / cfg.radar.fs_Hz);
        end

        function centeredReferencePreservesEnvelopeAndRecentersSupport(testCase)
            cfg = normalize_and_validate_config(create_default_config(), 'balanced');
            reference = build_reference_signal(cfg);
            centered = build_centered_reference_signal(cfg, reference);
            active = 1:cfg.derived.samples_per_pulse;

            testCase.verifySize(centered.pulse, size(reference.pulse));
            testCase.verifySize(centered.train, size(reference.train));
            testCase.verifyEqual(abs(centered.pulse(active)), ...
                abs(reference.pulse(active)), 'AbsTol', 64 * eps);
            testCase.verifyEqual(mean(centered.support_Hz), 0, ...
                'AbsTol', cfg.radar.fs_Hz * 64 * eps);
            testCase.verifyLessThan(centered.support_Hz(2), cfg.radar.fs_Hz / 2);
            testCase.verifyGreaterThan(centered.support_Hz(1), -cfg.radar.fs_Hz / 2);
        end

        function centeredReferenceRepeatsExactlyAcrossPris(testCase)
            cfg = normalize_and_validate_config(create_default_config(), 'balanced');
            reference = build_reference_signal(cfg);
            centered = build_centered_reference_signal(cfg, reference);
            blocks = reshape(centered.train, cfg.derived.samples_per_pri, []).';
            for pulseIndex = 1:cfg.radar.num_pulses
                testCase.verifyEqual(blocks(pulseIndex, :), centered.pulse);
            end
        end

        function combSpectrumEqualModeHasExactTeethAndPadding(testCase)
            % 验证等幅模式的三齿频率、单位幅度和固定八槽 NaN 填充。
            cfg = TestSingleJammingGenerators.createCombSpectrumFixture( ...
                'equal', [3, 3]);
            cfg.params.CombSpectrum.spacing_Hz = [0.4e6, 0.4e6];
            cfg.params.CombSpectrum.center_offset_Hz = [0, 0];
            reference = build_reference_signal(cfg);

            [signal, truth] = generate_comb_spectrum(cfg, ...
                RandStream('mt19937ar', 'Seed', 31), reference);

            testCase.verifySize(signal, [1, cfg.derived.sample_length]);
            testCase.verifyEqual(truth.num_teeth, 3);
            testCase.verifyEqual([truth.frequency_01_Hz, ...
                truth.frequency_02_Hz, truth.frequency_03_Hz], ...
                [-0.4e6, 0, 0.4e6], 'AbsTol', 1e-9);
            testCase.verifyEqual([truth.amplitude_01, ...
                truth.amplitude_02, truth.amplitude_03], [1, 1, 1]);
            testCase.verifyTrue(isnan(truth.frequency_04_Hz));
            testCase.verifyTrue(isnan(truth.phase_08_rad));
        end

        function combSpectrumRandomModeIsSeedReproducible(testCase)
            % 验证随机幅度与相位只消耗传入流，固定种子精确复现波形和真值。
            cfg = TestSingleJammingGenerators.createCombSpectrumFixture( ...
                'random', [8, 8]);
            reference = build_reference_signal(cfg);

            [signalA, truthA] = generate_comb_spectrum(cfg, ...
                RandStream('mt19937ar', 'Seed', 32), reference);
            [signalB, truthB] = generate_comb_spectrum(cfg, ...
                RandStream('mt19937ar', 'Seed', 32), reference);

            testCase.verifyEqual(signalA, signalB);
            testCase.verifyEqual(truthA, truthB);
            amplitudes = arrayfun(@(index) truthA.(sprintf( ...
                'amplitude_%02d', index)), 1:8);
            testCase.verifyGreaterThanOrEqual(amplitudes, 0.5);
            testCase.verifyLessThanOrEqual(amplitudes, 1.5);
        end

        function combSpectrumDechirpedPeaksMatchSavedSpacing(testCase)
            % 验证对零中心参考解调后，主峰频率与保存齿频在一个 FFT 栅格内一致。
            cfg = TestSingleJammingGenerators.createCombSpectrumFixture( ...
                'equal', [3, 3]);
            cfg.params.CombSpectrum.spacing_Hz = [0.4e6, 0.4e6];
            cfg.params.CombSpectrum.center_offset_Hz = [0, 0];
            reference = build_reference_signal(cfg);
            centered = build_centered_reference_signal(cfg, reference);

            [signal, truth] = generate_comb_spectrum(cfg, ...
                RandStream('mt19937ar', 'Seed', 33), reference);
            count = cfg.derived.samples_per_pulse;
            dechirped = signal(1:count) .* conj(centered.pulse(1:count));
            spectrum = abs(fftshift(fft(dechirped)));
            frequency_Hz = (-floor(count / 2):ceil(count / 2) - 1) * ...
                (cfg.radar.fs_Hz / count);
            [~, peakIndices] = maxk(spectrum, truth.num_teeth);
            measured_Hz = sort(frequency_Hz(peakIndices));
            saved_Hz = sort([truth.frequency_01_Hz, ...
                truth.frequency_02_Hz, truth.frequency_03_Hz]);

            testCase.verifyEqual(measured_Hz, saved_Hz, ...
                'AbsTol', cfg.radar.fs_Hz / count);
            testCase.verifyEqual(diff(measured_Hz), ...
                repmat(truth.spacing_Hz, 1, 2), ...
                'AbsTol', cfg.radar.fs_Hz / count);
        end

        function combSpectrumArrivalDelayUsesZeroBoundaries(testCase)
            % 验证公共时延只做零填充平移，不把记录尾部循环复制到开头。
            base = TestSingleJammingGenerators.createCombSpectrumFixture( ...
                'equal', [3, 3]);
            delayed = base;
            delaySamples = 7;
            delayed.params.common.arrival_delay_s = repmat( ...
                delaySamples / delayed.radar.fs_Hz, 1, 2);
            reference = build_reference_signal(base);

            signalBase = generate_comb_spectrum(base, ...
                RandStream('mt19937ar', 'Seed', 34), reference);
            signalDelayed = generate_comb_spectrum(delayed, ...
                RandStream('mt19937ar', 'Seed', 34), reference);

            testCase.verifyEqual(signalDelayed(1:delaySamples), ...
                zeros(1, delaySamples));
            testCase.verifyFalse(isreal(signalDelayed));
            testCase.verifyEqual(signalDelayed(delaySamples + 1:end), ...
                signalBase(1:end - delaySamples), 'AbsTol', 1e-12);
        end

        function combSpectrumRejectsUnsupportedOrAliasedRequests(testCase)
            % 验证非法模式和无可行 Nyquist 支撑的齿域均使用稳定错误标识。
            unsupported = ...
                TestSingleJammingGenerators.createCombSpectrumFixture( ...
                'unsupported', [3, 3]);
            reference = build_reference_signal(unsupported);
            testCase.verifyError(@() generate_comb_spectrum(unsupported, ...
                RandStream('mt19937ar', 'Seed', 35), reference), ...
                'JammingGenerator:UnsupportedMode');

            aliased = TestSingleJammingGenerators.createCombSpectrumFixture( ...
                'equal', [8, 8]);
            aliased.params.CombSpectrum.spacing_Hz = [0.6e6, 0.6e6];
            aliased.params.CombSpectrum.center_offset_Hz = [4e6, 4e6];
            testCase.verifyError(@() generate_comb_spectrum(aliased, ...
                RandStream('mt19937ar', 'Seed', 36), reference), ...
                'JammingGenerator:InvalidPhysicalParameter');
        end

        function clutterPulseRepeatedModeCopiesLayoutAcrossPris(testCase)
            % 验证重复模式在各 PRI 复用相同脉内布局和全部脉冲参数。
            cfg = TestSingleJammingGenerators.createClutterPulseFixture( ...
                'repeated', [3, 3]);
            reference = build_reference_signal(cfg);
            [~, truth] = generate_clutter_pulse(cfg, ...
                RandStream('mt19937ar', 'Seed', 41), reference);

            testCase.verifyEqual(truth.mode_code, 1);
            testCase.verifyEqual(truth.total_pulse_count, 12);
            for priIndex = 2:cfg.radar.num_pulses
                offset = (priIndex - 1) * 3;
                for localIndex = 1:3
                    current = offset + localIndex;
                    testCase.verifyEqual( ...
                        TestSingleJammingGenerators.readClutterTruth( ...
                        truth, 'width', current, '_s'), ...
                        TestSingleJammingGenerators.readClutterTruth( ...
                        truth, 'width', localIndex, '_s'));
                    testCase.verifyEqual( ...
                        TestSingleJammingGenerators.readClutterTruth( ...
                        truth, 'amplitude', current, ''), ...
                        TestSingleJammingGenerators.readClutterTruth( ...
                        truth, 'amplitude', localIndex, ''));
                    testCase.verifyEqual( ...
                        TestSingleJammingGenerators.readClutterTruth( ...
                        truth, 'frequency', current, '_Hz'), ...
                        TestSingleJammingGenerators.readClutterTruth( ...
                        truth, 'frequency', localIndex, '_Hz'));
                    testCase.verifyEqual( ...
                        TestSingleJammingGenerators.readClutterTruth( ...
                        truth, 'phase', current, '_rad'), ...
                        TestSingleJammingGenerators.readClutterTruth( ...
                        truth, 'phase', localIndex, '_rad'));
                    timeDifference_s = ...
                        TestSingleJammingGenerators.readClutterTruth( ...
                        truth, 'start', current, '_s') - ...
                        TestSingleJammingGenerators.readClutterTruth( ...
                        truth, 'start', localIndex, '_s');
                    testCase.verifyEqual(timeDifference_s, ...
                        (priIndex - 1) * cfg.radar.pri_s, ...
                        'AbsTol', 1e-12);
                end
            end
            testCase.verifyTrue(isnan(truth.start_13_s));
        end

        function clutterPulseIndependentModeChangesLayouts(testCase)
            % 验证独立模式在固定回归种子下至少产生两个不同的 PRI 布局。
            cfg = TestSingleJammingGenerators.createClutterPulseFixture( ...
                'independent', [3, 3]);
            reference = build_reference_signal(cfg);
            [~, truth] = generate_clutter_pulse(cfg, ...
                RandStream('mt19937ar', 'Seed', 42), reference);
            first = [truth.start_01_s, truth.start_02_s, truth.start_03_s];
            second = [truth.start_04_s, truth.start_05_s, truth.start_06_s] ...
                - cfg.radar.pri_s;
            testCase.verifyNotEqual(first, second);
        end

        function clutterPulseDutyCycleUsesUnionOccupancy(testCase)
            % 验证重叠样点只计一次，并由详细真值独立重建占空比。
            cfg = TestSingleJammingGenerators.createClutterPulseFixture( ...
                'independent', [8, 8]);
            cfg.params.ClutterPulse.pulse_width_s = ...
                repmat(cfg.radar.pulse_width_s, 1, 2);
            reference = build_reference_signal(cfg);
            [~, truth] = generate_clutter_pulse(cfg, ...
                RandStream('mt19937ar', 'Seed', 43), reference);
            occupied = false(1, cfg.derived.sample_length);
            summedWidthSamples = 0;
            for pulseIndex = 1:truth.total_pulse_count
                startSample = round( ...
                    TestSingleJammingGenerators.readClutterTruth( ...
                    truth, 'start', pulseIndex, '_s') * cfg.radar.fs_Hz);
                widthSamples = round( ...
                    TestSingleJammingGenerators.readClutterTruth( ...
                    truth, 'width', pulseIndex, '_s') * cfg.radar.fs_Hz);
                indices = startSample + (1:widthSamples);
                occupied(indices) = true;
                summedWidthSamples = summedWidthSamples + widthSamples;
            end

            testCase.verifyEqual(truth.total_pulse_count, 32);
            testCase.verifyGreaterThan(summedWidthSamples, nnz(occupied));
            testCase.verifyEqual(truth.duty_cycle, ...
                nnz(occupied) / cfg.derived.sample_length, ...
                'AbsTol', 1e-15);
        end

        function clutterPulseIsSeedReproducibleAndFrequencySafe(testCase)
            % 验证固定种子复现，且全部频移后的参考支撑严格位于 Nyquist 内。
            cfg = TestSingleJammingGenerators.createClutterPulseFixture( ...
                'independent', [8, 8]);
            reference = build_reference_signal(cfg);
            centered = build_centered_reference_signal(cfg, reference);
            [signalA, truthA] = generate_clutter_pulse(cfg, ...
                RandStream('mt19937ar', 'Seed', 44), reference);
            [signalB, truthB] = generate_clutter_pulse(cfg, ...
                RandStream('mt19937ar', 'Seed', 44), reference);
            testCase.verifyEqual(signalA, signalB);
            testCase.verifyEqual(truthA, truthB);

            guard_Hz = cfg.radar.fs_Hz / 2 - max( ...
                eps(cfg.radar.fs_Hz / 2), cfg.radar.fs_Hz * 1e-12);
            frequencies_Hz = arrayfun(@(index) ...
                TestSingleJammingGenerators.readClutterTruth( ...
                truthA, 'frequency', index, '_Hz'), 1:32);
            testCase.verifyGreaterThan( ...
                min(frequencies_Hz) + centered.support_Hz(1), -guard_Hz);
            testCase.verifyLessThan( ...
                max(frequencies_Hz) + centered.support_Hz(2), guard_Hz);
        end

        function clutterPulseArrivalDelayHasNoCircularFrontLeakage(testCase)
            % 验证正到达时延产生前导零，记录尾部不会循环泄漏到开头。
            cfg = TestSingleJammingGenerators.createClutterPulseFixture( ...
                'repeated', [3, 3]);
            delaySamples = 7;
            cfg.params.common.arrival_delay_s = repmat( ...
                delaySamples / cfg.radar.fs_Hz, 1, 2);
            reference = build_reference_signal(cfg);
            [signal, truth] = generate_clutter_pulse(cfg, ...
                RandStream('mt19937ar', 'Seed', 45), reference);

            testCase.verifyEqual(signal(1:delaySamples), ...
                zeros(1, delaySamples));
            testCase.verifyFalse(isreal(signal));
            testCase.verifyGreaterThanOrEqual(truth.start_01_s, ...
                delaySamples / cfg.radar.fs_Hz);
        end

        function clutterPulseRejectsModeCapacityAndFrequencyErrors(testCase)
            % 验证模式、32 槽容量和频率支撑越界均使用稳定错误标识。
            unsupported = ...
                TestSingleJammingGenerators.createClutterPulseFixture( ...
                'unsupported', [3, 3]);
            reference = build_reference_signal(unsupported);
            testCase.verifyError(@() generate_clutter_pulse(unsupported, ...
                RandStream('mt19937ar', 'Seed', 46), reference), ...
                'JammingGenerator:UnsupportedMode');

            capacity = TestSingleJammingGenerators.createClutterPulseFixture( ...
                'independent', [8, 8]);
            capacity.radar.num_pulses = 5;
            capacity.derived.sample_length = ...
                capacity.derived.samples_per_pri * 5;
            testCase.verifyError(@() generate_clutter_pulse(capacity, ...
                RandStream('mt19937ar', 'Seed', 47), reference), ...
                'JammingGenerator:InvalidPhysicalParameter');

            aliased = TestSingleJammingGenerators.createClutterPulseFixture( ...
                'independent', [3, 3]);
            aliased.params.ClutterPulse.frequency_offset_Hz = [4e6, 4e6];
            testCase.verifyError(@() generate_clutter_pulse(aliased, ...
                RandStream('mt19937ar', 'Seed', 48), reference), ...
                'JammingGenerator:InvalidPhysicalParameter');
        end

        function smspUsesIntegerSubpulseCountAndOriginalBandwidth(testCase)
            % 验证 M=8 时子脉宽、总时长和原始带宽真值均按标准模型保存。
            cfg = TestSingleJammingGenerators.createSmspFixture([8, 8]);
            reference = build_reference_signal(cfg);
            [signal, truth] = generate_spectral_dispersion(cfg, ...
                RandStream('mt19937ar', 'Seed', 51), reference);

            expectedSamples = round(cfg.derived.samples_per_pulse / 8);
            expectedWidth_s = expectedSamples / cfg.radar.fs_Hz;
            testCase.verifyEqual(truth.num_subpulses, 8);
            testCase.verifyEqual(truth.subpulse_width_s, ...
                expectedWidth_s, 'AbsTol', 1e-15);
            testCase.verifyEqual(truth.chirp_rate_multiplier, ...
                cfg.radar.pulse_width_s / expectedWidth_s, ...
                'AbsTol', 1e-12);
            testCase.verifyEqual(truth.subpulse_bandwidth_Hz, ...
                cfg.radar.bandwidth_Hz);
            testCase.verifyEqual(truth.output_duration_s, ...
                8 * expectedWidth_s, 'AbsTol', 1e-15);
            testCase.verifySize(signal, [1, cfg.derived.sample_length]);
        end

        function smspRepeatsIdenticalLocalSubpulses(testCase)
            % 验证 M=2 的局部复波形精确重复，未使用起点槽固定为 NaN。
            cfg = TestSingleJammingGenerators.createSmspFixture([2, 2]);
            reference = build_reference_signal(cfg);
            [signal, truth] = generate_spectral_dispersion(cfg, ...
                RandStream('mt19937ar', 'Seed', 52), reference);
            widthSamples = round( ...
                truth.subpulse_width_s * cfg.radar.fs_Hz);

            testCase.verifyEqual(signal(1:widthSamples), ...
                signal(widthSamples + 1:2 * widthSamples), ...
                'AbsTol', 64 * eps);
            testCase.verifyEqual(truth.subpulse_start_01_s, 0);
            testCase.verifyEqual(truth.subpulse_start_02_s, ...
                widthSamples / cfg.radar.fs_Hz);
            testCase.verifyTrue(isnan(truth.subpulse_start_03_s));
        end

        function smspChirpRateAndFrequencyResetMatchTruth(testCase)
            % 验证每个短 LFM 斜率与真值一致，并在子脉冲边界从高频复位到低频。
            cfg = TestSingleJammingGenerators.createSmspFixture([4, 4]);
            reference = build_reference_signal(cfg);
            [signal, truth] = generate_spectral_dispersion(cfg, ...
                RandStream('mt19937ar', 'Seed', 53), reference);
            widthSamples = round( ...
                truth.subpulse_width_s * cfg.radar.fs_Hz);
            first = signal(1:widthSamples);
            frequency_Hz = angle(first(2:end) .* conj(first(1:end - 1))) * ...
                cfg.radar.fs_Hz / (2 * pi);
            intervalTime_s = ((0:numel(frequency_Hz) - 1) + 0.5) / ...
                cfg.radar.fs_Hz;
            coefficients = polyfit(intervalTime_s, frequency_Hz, 1);

            testCase.verifyEqual(coefficients(1), ...
                truth.subpulse_chirp_rate_Hz_per_s, ...
                'RelTol', 1e-10);
            testCase.verifyGreaterThan(frequency_Hz(end), 0);
            testCase.verifyLessThan(frequency_Hz(1), 0);
            testCase.verifyEqual(signal(1:widthSamples), ...
                signal(widthSamples + 1:2 * widthSamples), ...
                'AbsTol', 64 * eps);
            testCase.verifyLessThanOrEqual(max(abs(frequency_Hz)), ...
                cfg.radar.bandwidth_Hz / 2);
        end

        function smspArrivalDelayUsesZeroBoundaries(testCase)
            % 验证公共时延只产生前导零，并保持每个 PRI 的 SMSP 完整支撑。
            cfg = TestSingleJammingGenerators.createSmspFixture([4, 4]);
            delaySamples = 7;
            cfg.params.common.arrival_delay_s = repmat( ...
                delaySamples / cfg.radar.fs_Hz, 1, 2);
            reference = build_reference_signal(cfg);
            [signal, truth] = generate_spectral_dispersion(cfg, ...
                RandStream('mt19937ar', 'Seed', 54), reference);

            testCase.verifyEqual(signal(1:delaySamples), ...
                zeros(1, delaySamples));
            testCase.verifyFalse(isreal(signal));
            testCase.verifyEqual(truth.arrival_delay_s, ...
                delaySamples / cfg.radar.fs_Hz);
        end

        function smspRejectsCountCapacitySampleAndDelayErrors(testCase)
            % 验证 M、固定容量、最小样点数和 PRI 支撑均使用稳定错误标识。
            referenceCfg = TestSingleJammingGenerators.createSmspFixture([2, 2]);
            reference = build_reference_signal(referenceCfg);

            invalidLow = TestSingleJammingGenerators.createSmspFixture([1, 1]);
            testCase.verifyError(@() generate_spectral_dispersion( ...
                invalidLow, RandStream('mt19937ar', 'Seed', 55), reference), ...
                'JammingGenerator:InvalidPhysicalParameter');
            invalidHigh = TestSingleJammingGenerators.createSmspFixture([9, 9]);
            testCase.verifyError(@() generate_spectral_dispersion( ...
                invalidHigh, RandStream('mt19937ar', 'Seed', 56), reference), ...
                'JammingGenerator:InvalidPhysicalParameter');
            invalidCapacity = ...
                TestSingleJammingGenerators.createSmspFixture([8, 8]);
            invalidCapacity.params.SpectralDispersion.max_subpulses = 9;
            testCase.verifyError(@() generate_spectral_dispersion( ...
                invalidCapacity, RandStream('mt19937ar', 'Seed', 57), ...
                reference), 'JammingGenerator:InvalidPhysicalParameter');

            tooFewSamples = ...
                TestSingleJammingGenerators.createSmspFixture([8, 8]);
            tooFewSamples.derived.samples_per_pulse = 8;
            testCase.verifyError(@() generate_spectral_dispersion( ...
                tooFewSamples, RandStream('mt19937ar', 'Seed', 58), ...
                reference), 'JammingGenerator:InvalidPhysicalParameter');

            tooLate = TestSingleJammingGenerators.createSmspFixture([2, 2]);
            tooLate.params.common.arrival_delay_s = [250e-6, 250e-6];
            testCase.verifyError(@() generate_spectral_dispersion( ...
                tooLate, RandStream('mt19937ar', 'Seed', 59), reference), ...
                'JammingGenerator:InvalidPhysicalParameter');
        end

        function invalidReferenceConfig(testCase)
            % 验证缺字段、非法长度、采样率和调频斜率使用统一错误标识。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            invalidConfigs = cell(1, 18);
            invalidConfigs{1} = rmfield(cfg, 'derived');
            invalidConfigs{2} = cfg;
            invalidConfigs{2}.derived = rmfield( ...
                invalidConfigs{2}.derived, 'samples_per_pri');
            invalidConfigs{3} = cfg;
            invalidConfigs{3}.derived.samples_per_pulse = ...
                cfg.derived.samples_per_pri + 1;
            invalidConfigs{4} = cfg;
            invalidConfigs{4}.derived.samples_per_pri = 0;
            invalidConfigs{5} = cfg;
            invalidConfigs{5}.derived.samples_per_pulse = -1;
            invalidConfigs{6} = cfg;
            invalidConfigs{6}.derived.sample_length = 1.5;
            invalidConfigs{7} = cfg;
            invalidConfigs{7}.derived.samples_per_pri = [1, 2];
            invalidConfigs{8} = cfg;
            invalidConfigs{8}.derived.samples_per_pulse = complex(1, 1);
            invalidConfigs{9} = cfg;
            invalidConfigs{9}.radar.num_pulses = 0;
            invalidConfigs{10} = cfg;
            invalidConfigs{10}.radar.num_pulses = 1.5;
            invalidConfigs{11} = cfg;
            invalidConfigs{11}.radar.fs_Hz = 0;
            invalidConfigs{12} = cfg;
            invalidConfigs{12}.radar.fs_Hz = Inf;
            invalidConfigs{13} = cfg;
            invalidConfigs{13}.radar.fs_Hz = complex(1, 1);
            invalidConfigs{14} = cfg;
            invalidConfigs{14}.derived.chirp_rate_Hz_per_s = 0;
            invalidConfigs{15} = cfg;
            invalidConfigs{15}.derived.chirp_rate_Hz_per_s = NaN;
            invalidConfigs{16} = cfg;
            invalidConfigs{16}.derived.chirp_rate_Hz_per_s = [1, 2];
            invalidConfigs{17} = cfg;
            invalidConfigs{17}.derived.sample_length = ...
                cfg.derived.sample_length + 1;
            invalidConfigs{18} = cfg;
            invalidConfigs{18}.radar.num_pulses = '4';

            for configIndex = 1:numel(invalidConfigs)
                testCase.verifyError(@() build_reference_signal( ...
                    invalidConfigs{configIndex}), ...
                    'JammingReference:InvalidConfig');
            end
        end

        function referenceRejectsUnsupportedTimeAndPhaseDomains(testCase)
            % 验证有限配置若导致时间或二次相位溢出，仍稳定归类为非法配置。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            cfg.radar.num_pulses = 2;
            cfg.derived.samples_per_pri = 4;
            cfg.derived.samples_per_pulse = 2;
            cfg.derived.sample_length = 8;

            overflowingTimeCfg = cfg;
            overflowingTimeCfg.radar.fs_Hz = realmin;
            testCase.verifyError(@() build_reference_signal( ...
                overflowingTimeCfg), 'JammingReference:InvalidConfig');

            overflowingPhaseCfg = cfg;
            overflowingPhaseCfg.radar.fs_Hz = 1;
            overflowingPhaseCfg.derived.chirp_rate_Hz_per_s = realmax;
            testCase.verifyError(@() build_reference_signal( ...
                overflowingPhaseCfg), 'JammingReference:InvalidConfig');
        end

        function referenceAcceptsFiniteExtremePhaseWithStableOrdering(testCase)
            % 验证先计算 t^2 可保留最终有限、但 pi*K 会先溢出的极端相位。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            cfg.radar.fs_Hz = 2;
            cfg.radar.num_pulses = 1;
            cfg.derived.samples_per_pri = 2;
            cfg.derived.samples_per_pulse = 2;
            cfg.derived.sample_length = 2;
            cfg.derived.chirp_rate_Hz_per_s = realmax;
            time = [0, 0.5];
            expected = exp(1i * pi * (realmax * (time .^ 2)));

            reference = build_reference_signal(cfg);

            testCase.verifyEqual(reference.pulse, expected, ...
                'AbsTol', 1e-12);
            testCase.verifyTrue(all(isfinite(real(reference.pulse))));
            testCase.verifyTrue(all(isfinite(imag(reference.pulse))));
            testCase.verifyTrue(all(isfinite(reference.time_train_s)));
        end

        function singleSampleReferenceRemainsComplex(testCase)
            % 验证单有效样点虽数值全实，脉冲与脉冲串仍保留复数 double 契约。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            cfg.radar.fs_Hz = 10;
            cfg.radar.num_pulses = 2;
            cfg.derived.samples_per_pri = 4;
            cfg.derived.samples_per_pulse = 1;
            cfg.derived.sample_length = 8;
            cfg.derived.chirp_rate_Hz_per_s = 1;

            reference = build_reference_signal(cfg);

            testCase.verifyClass(reference.pulse, 'double');
            testCase.verifyClass(reference.train, 'double');
            testCase.verifyFalse(isreal(reference.pulse));
            testCase.verifyFalse(isreal(reference.train));
            testCase.verifyEqual(real(reference.pulse), [1, 0, 0, 0]);
            testCase.verifyEqual(imag(reference.pulse), zeros(1, 4));
            testCase.verifyEqual(real(reference.train), ...
                [1, 0, 0, 0, 1, 0, 0, 0]);
            testCase.verifyEqual(imag(reference.train), zeros(1, 8));
        end

        function referenceRejectsUnsupportedCapacityBeforeAllocation(testCase)
            % 验证缺失/非法容量及明显超限数组在实际分配前统一快速拒绝。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            invalidCapacityConfigs = cell(1, 8);
            invalidCapacityConfigs{1} = rmfield(cfg, 'output');
            invalidCapacityConfigs{2} = cfg;
            invalidCapacityConfigs{2}.output = rmfield( ...
                invalidCapacityConfigs{2}.output, 'max_estimated_bytes');
            invalidCapacityConfigs{3} = cfg;
            invalidCapacityConfigs{3}.output.max_estimated_bytes = NaN;
            invalidCapacityConfigs{4} = cfg;
            invalidCapacityConfigs{4}.output.max_estimated_bytes = Inf;
            invalidCapacityConfigs{5} = cfg;
            invalidCapacityConfigs{5}.output.max_estimated_bytes = -1;
            invalidCapacityConfigs{6} = cfg;
            invalidCapacityConfigs{6}.output.max_estimated_bytes = [1, 2];
            invalidCapacityConfigs{7} = cfg;
            invalidCapacityConfigs{7}.output.max_estimated_bytes = '1024';
            invalidCapacityConfigs{8} = cfg;
            invalidCapacityConfigs{8}.output.max_estimated_bytes = ...
                complex(1024, 1);
            for configIndex = 1:numel(invalidCapacityConfigs)
                testCase.verifyError(@() build_reference_signal( ...
                    invalidCapacityConfigs{configIndex}), ...
                    'JammingReference:InvalidConfig');
            end

            tooSmallCapacityCfg = cfg;
            tooSmallCapacityCfg.output.max_estimated_bytes = 1;
            testCase.verifyError(@() build_reference_signal( ...
                tooSmallCapacityCfg), 'JammingReference:InvalidConfig');

            largeCfg = cfg;
            largeCfg.radar.num_pulses = 1;
            largeCfg.derived.samples_per_pri = 1e5;
            largeCfg.derived.samples_per_pulse = 1;
            largeCfg.derived.sample_length = 1e5;
            largeCfg.output.max_estimated_bytes = 1024;
            testCase.verifyError(@() build_reference_signal(largeCfg), ...
                'JammingReference:InvalidConfig');
        end

        function referenceRejectsPlatformImpossibleArrayBeforeAllocation( ...
                testCase)
            % 验证配置总量虽未超 flintmax，平台不可能分配时仍抛稳定错误标识。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            cfg.radar.fs_Hz = 10e6;
            cfg.radar.num_pulses = 1;
            cfg.derived.samples_per_pri = 1e14;
            cfg.derived.samples_per_pulse = 1;
            cfg.derived.sample_length = 1e14;
            cfg.derived.chirp_rate_Hz_per_s = 1;
            cfg.output.max_estimated_bytes = flintmax;

            testCase.verifyError(@() build_reference_signal(cfg), ...
                'JammingReference:InvalidConfig');
        end

        function referenceRejectsInexactIntegerConfig(testCase)
            % 验证超过 flintmax 的整数采样率和调频斜率不会静默舍入。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            inexactInteger = bitshift(uint64(1), 53) + uint64(1);

            invalidFsCfg = cfg;
            invalidFsCfg.radar.fs_Hz = inexactInteger;
            testCase.verifyError(@() build_reference_signal(invalidFsCfg), ...
                'JammingReference:InvalidConfig');

            invalidChirpCfg = cfg;
            invalidChirpCfg.derived.chirp_rate_Hz_per_s = inexactInteger;
            testCase.verifyError(@() build_reference_signal( ...
                invalidChirpCfg), 'JammingReference:InvalidConfig');
        end

        function powerCalibrationHitsRequestedLevel(testCase)
            % 验证复数分量的平均功率精确标定到请求的 dB 水平。
            x = ones(1, 16) + 1i * ones(1, 16);

            [y, measuredPower_dB] = calibrate_component_power(x, -12);
            actualPower_dB = 10 * log10(mean(abs(y) .^ 2));

            testCase.verifyEqual(actualPower_dB, -12, 'AbsTol', 1e-10);
            testCase.verifyEqual(measuredPower_dB, -12, 'AbsTol', 1e-10);
            testCase.verifySize(y, size(x));
        end

        function calibrationPreservesWaveformDirection(testCase)
            % 验证标定只施加同一正实幅度因子，不改变任一非零样点相位。
            x = [1 + 2i, -2 + 3i, 0, 4 - 1i];

            y = calibrate_component_power(x, -6);
            nonzeroMask = x ~= 0;
            ratios = y(nonzeroMask) ./ x(nonzeroMask);

            testCase.verifyEqual(ratios, ...
                repmat(ratios(1), size(ratios)), 'AbsTol', 10 * eps);
            testCase.verifyEqual(imag(ratios), zeros(size(ratios)), ...
                'AbsTol', 10 * eps);
            testCase.verifyGreaterThan(real(ratios), zeros(size(ratios)));
            testCase.verifyEqual(y(~nonzeroMask), zeros(1, 1));
        end

        function invalidComponentPowerInputs(testCase)
            % 验证非法信号和非法目标功率统一使用分量功率错误标识。
            invalidSignals = {[], eye(2), [1, NaN], [1, Inf], ...
                complex([1, 1], [0, Inf]), zeros(1, 4), 'signal'};
            for signalIndex = 1:numel(invalidSignals)
                testCase.verifyError(@() calibrate_component_power( ...
                    invalidSignals{signalIndex}, -12), ...
                    'JammingGenerator:InvalidComponentPower');
            end

            validSignal = [1, 2, 3];
            invalidTargets = {NaN, Inf, -Inf, complex(1, 1), ...
                [1, 2], '12'};
            for targetIndex = 1:numel(invalidTargets)
                testCase.verifyError(@() calibrate_component_power( ...
                    validSignal, invalidTargets{targetIndex}), ...
                    'JammingGenerator:InvalidComponentPower');
            end

            extremeTargets = {7000, -7000};
            for targetIndex = 1:numel(extremeTargets)
                testCase.verifyError(@() calibrate_component_power( ...
                    validSignal, extremeTargets{targetIndex}), ...
                    'JammingGenerator:InvalidComponentPower');
            end
        end

        function calibrationConvertsInputsAndPreservesShape(testCase)
            % 验证 single/整数输入转 double，并分别保持行向量和列向量形状。
            rowInput = single([1 + 2i, 3 - 4i]);
            columnInput = int16([1; 2; 3]);

            [rowOutput, rowMeasured] = calibrate_component_power( ...
                rowInput, single(-9));
            [columnOutput, columnMeasured] = calibrate_component_power( ...
                columnInput, int16(-6));

            testCase.verifyClass(rowOutput, 'double');
            testCase.verifyClass(columnOutput, 'double');
            testCase.verifyClass(rowMeasured, 'double');
            testCase.verifyClass(columnMeasured, 'double');
            testCase.verifySize(rowOutput, size(rowInput));
            testCase.verifySize(columnOutput, size(columnInput));
            testCase.verifyEqual(rowMeasured, -9, 'AbsTol', 1e-10);
            testCase.verifyEqual(columnMeasured, -6, 'AbsTol', 1e-10);
        end

        function stableCalibrationHandlesExtremeFiniteLevels(testCase)
            % 验证稳定 RMS 路径支持极大/极小输入及平方会溢出的有限目标。
            signals = {1e200, 1e-200, 1e-150, 1e150, [1, 1], 1, 1};
            targets_dB = [0, 0, 3000, -3000, 3080, 4000, -4000];

            for caseIndex = 1:numel(signals)
                [y, measuredPower_dB] = calibrate_component_power( ...
                    signals{caseIndex}, targets_dB(caseIndex));
                maximumMagnitude = max(abs(y));
                normalized = y / maximumMagnitude;
                stableMeasured_dB = 20 * log10(maximumMagnitude) + ...
                    20 * log10(sqrt(mean(abs(normalized) .^ 2)));

                testCase.verifyTrue(all(isfinite(real(y))));
                testCase.verifyTrue(all(isfinite(imag(y))));
                testCase.verifyGreaterThan(maximumMagnitude, 0);
                testCase.verifyEqual(measuredPower_dB, ...
                    targets_dB(caseIndex), 'AbsTol', 1e-10);
                testCase.verifyEqual(stableMeasured_dB, ...
                    targets_dB(caseIndex), 'AbsTol', 1e-10);
            end
        end

        function calibrationHandlesComplexComponentsNearRealmax(testCase)
            % 验证实虚分量均为 realmax 时不先计算溢出的原始复数模。
            x = complex(realmax, realmax);
            expected = complex(1 / sqrt(2), 1 / sqrt(2));

            [y, measuredPower_dB] = calibrate_component_power(x, 0);

            testCase.verifyTrue(isfinite(real(y)));
            testCase.verifyTrue(isfinite(imag(y)));
            testCase.verifyEqual(y, expected, 'AbsTol', 1e-15);
            testCase.verifyEqual(measuredPower_dB, 0, 'AbsTol', 1e-10);
        end

        function calibrationRejectsInexactIntegerSamples(testCase)
            % 验证超过 flintmax 的整数样点在 double 转换前被明确拒绝。
            inexactInteger = bitshift(uint64(1), 53) + uint64(1);

            testCase.verifyError(@() calibrate_component_power( ...
                inexactInteger, 0), ...
                'JammingGenerator:InvalidComponentPower');
        end

        function allThirteenGeneratorsReturnFiniteSignalAndTruth(testCase)
            % 验证默认 v2 十三类 dispatcher 均满足统一信号和真值字段契约。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);

            for typeIndex = 1:numel(registry)
                stream = RandStream('mt19937ar', ...
                    'Seed', 1000 + typeIndex);
                [signal, params] = generate_single_jamming( ...
                    registry(typeIndex).code, cfg, registry, stream);

                testCase.verifySize(signal, ...
                    [1, cfg.derived.sample_length]);
                testCase.verifyClass(signal, 'double');
                testCase.verifyFalse(isreal(signal));
                testCase.verifyTrue(all(isfinite(real(signal))));
                testCase.verifyTrue(all(isfinite(imag(signal))));
                testCase.verifyGreaterThan(mean(abs(signal) .^ 2), 0);
                testCase.verifyTrue(isstruct(params) && isscalar(params));
                expectedNames = registry(typeIndex).parameter_names;
                testCase.verifyTrue(all(isfield(params, expectedNames)));
            end
        end

        function fixedSeedReproducesWaveformAndParameters(testCase)
            % 验证随机噪声类 FMB 与转发类 ISRJ 使用局部流完全可复现。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            codes = {'FMB', 'ISRJ'};

            for codeIndex = 1:numel(codes)
                streamA = RandStream('mt19937ar', 'Seed', 24680);
                streamB = RandStream('mt19937ar', 'Seed', 24680);
                [signalA, paramsA] = generate_single_jamming( ...
                    codes{codeIndex}, cfg, registry, streamA);
                [signalB, paramsB] = generate_single_jamming( ...
                    codes{codeIndex}, cfg, registry, streamB);

                testCase.verifyEqual(signalA, signalB);
                testCase.verifyEqual(paramsA, paramsB);
            end
        end

        function dispatcherDoesNotChangeGlobalRng(testCase)
            % 验证生成器仅消费显式 RandStream，不读写 MATLAB 全局随机状态。
            originalState = rng;
            cleanupGuard = onCleanup(@() rng(originalState));
            rng(314159, 'twister');
            stateBefore = rng;
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            stream = RandStream('mt19937ar', 'Seed', 271828);

            generate_single_jamming('SN', cfg, registry, stream);

            testCase.verifyEqual(rng, stateBefore);
        end

        function dispatcherUsesStableValidationIdentifiers(testCase)
            % 验证未知类型、畸形注册表及非法生成输出使用稳定错误标识。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            malformedCodes = {'NotAType', 42, ["FMB", "AMA"], ...
                string(missing), char(['FMB'; 'AMA'])};
            for codeIndex = 1:numel(malformedCodes)
                stream = RandStream('mt19937ar', 'Seed', codeIndex);
                testCase.verifyError(@() generate_single_jamming( ...
                    malformedCodes{codeIndex}, cfg, registry, stream), ...
                    'JammingGenerator:UnknownType');
            end

            duplicateRegistry = registry;
            duplicateRegistry(2).code = duplicateRegistry(1).code;
            testCase.verifyError(@() generate_single_jamming( ...
                'FMB', cfg, duplicateRegistry, ...
                RandStream('mt19937ar', 'Seed', 1)), ...
                'JammingGenerator:InvalidRegistry');

            invalidHandleRegistry = registry;
            invalidHandleRegistry(1).generator = 42;
            testCase.verifyError(@() generate_single_jamming( ...
                'FMB', cfg, invalidHandleRegistry, ...
                RandStream('mt19937ar', 'Seed', 1)), ...
                'JammingGenerator:InvalidRegistry');

            shortRegistry = registry(1);
            shortRegistry.generator = ...
                @TestSingleJammingGenerators.shortGenerator;
            testCase.verifyError(@() generate_single_jamming( ...
                'FMB', cfg, shortRegistry, ...
                RandStream('mt19937ar', 'Seed', 1)), ...
                'JammingGenerator:InvalidGeneratorOutput');

            missingFieldRegistry = registry(1);
            missingFieldRegistry.generator = ...
                @TestSingleJammingGenerators.missingFieldGenerator;
            testCase.verifyError(@() generate_single_jamming( ...
                'FMB', cfg, missingFieldRegistry, ...
                RandStream('mt19937ar', 'Seed', 1)), ...
                'JammingGenerator:InvalidGeneratorOutput');
        end

        function dispatcherRejectsGeneratorExtraParameterFields(testCase)
            % 验证生成器真值字段集合必须与注册表声明精确一致，不能有额外项。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            fmbRegistry = registry(1);
            fmbRegistry.parameter_names = {'JNR_dB'};
            fmbRegistry.parameter_units = {'dB'};

            testCase.verifyError(@() generate_single_jamming( ...
                'FMB', cfg, fmbRegistry, ...
                RandStream('mt19937ar', 'Seed', 77)), ...
                'JammingGenerator:InvalidGeneratorOutput');
        end

        function commonAndTractionTruthFieldsAreSeparated(testCase)
            % 验证 v1 前九类采样公共参数，Traction 仅返回五个运动真值字段。
            cfg = normalize_and_validate_config( ...
                create_default_config(1), 'balanced');
            registry = build_jamming_registry(cfg);

            for typeIndex = 1:9
                [~, params] = generate_single_jamming( ...
                    registry(typeIndex).code, cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', 500 + typeIndex));
                testCase.verifyGreaterThanOrEqual(params.JNR_dB, ...
                    cfg.params.common.JNR_dB(1));
                testCase.verifyLessThanOrEqual(params.JNR_dB, ...
                    cfg.params.common.JNR_dB(2));
                if ismember(registry(typeIndex).code, ...
                        {'MFT', 'RangeFakeTarget'})
                    % 两类假目标已把公共到达时延吸收到逐目标最终时延中。
                    testCase.verifyTrue(isnan(params.arrival_delay_s));
                else
                    testCase.verifyGreaterThanOrEqual( ...
                        params.arrival_delay_s, ...
                        cfg.params.common.arrival_delay_s(1));
                    testCase.verifyLessThanOrEqual( ...
                        params.arrival_delay_s, ...
                        cfg.params.common.arrival_delay_s(2));
                end
            end

            [~, tractionParams] = generate_single_jamming( ...
                'Traction', cfg, registry, ...
                RandStream('mt19937ar', 'Seed', 510));
            testCase.verifyEqual(fieldnames(tractionParams), ...
                {'initial_range_m'; 'velocity_mps'; 'amplitude'; ...
                'range_change_per_pulse_m'; ...
                'delay_change_per_pulse_s'});
            testCase.verifyFalse(isfield(tractionParams, 'JNR_dB'));
            testCase.verifyFalse(isfield(tractionParams, ...
                'arrival_delay_s'));
        end

        function derivedTruthFieldsFollowPhysicalDefinitions(testCase)
            % 验证斜率、周期、速度及拖引脉间增量严格按定义计算。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            codes = {'SawFM', 'SinFM', 'SN', 'ISRJ', 'VFT', 'Traction'};
            params = cell(size(codes));
            for codeIndex = 1:numel(codes)
                [~, params{codeIndex}] = generate_single_jamming( ...
                    codes{codeIndex}, cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', 600 + codeIndex));
            end

            testCase.verifyEqual(params{1}.sweep_slope_Hz_per_s, ...
                params{1}.sweep_bandwidth_Hz / ...
                params{1}.sweep_period_s, 'RelTol', 10 * eps);
            testCase.verifyEqual(params{2}.modulation_period_s, ...
                1 / params{2}.modulation_frequency_Hz, ...
                'RelTol', 10 * eps);
            testCase.verifyEqual(params{3}.sample_period_s, ...
                (1 + params{3}.num_replicas) * ...
                params{3}.sample_width_s, 'RelTol', 10 * eps);
            testCase.verifyEqual(params{4}.sample_period_s, ...
                (1 + params{4}.num_replicas) * ...
                params{4}.sample_width_s, 'RelTol', 10 * eps);
            testCase.verifyEqual(params{5}.velocity_mps, ...
                params{5}.doppler_Hz * cfg.radar.c_mps / ...
                (2 * cfg.radar.carrier_Hz), 'RelTol', 10 * eps);
            expectedRangeChange = params{6}.velocity_mps * ...
                cfg.radar.pri_s;
            testCase.verifyEqual( ...
                params{6}.range_change_per_pulse_m, ...
                expectedRangeChange, 'RelTol', 10 * eps);
            testCase.verifyEqual( ...
                params{6}.delay_change_per_pulse_s, ...
                2 * expectedRangeChange / cfg.radar.c_mps, ...
                'RelTol', 10 * eps);
        end

        function multiFalseTargetTruthUsesSortedFinitePadding(testCase)
            % 验证 MFT 使用升序时延、有限有效槽和 NaN 容量填充。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            [~, params] = generate_single_jamming( ...
                'MFT', cfg, registry, ...
                RandStream('mt19937ar', 'Seed', 707));
            count = params.num_targets;
            delays = arrayfun(@(index) params.(sprintf( ...
                'delay_%02d_s', index)), 1:8);
            amplitudes = arrayfun(@(index) params.(sprintf( ...
                'amplitude_%02d', index)), 1:8);

            testCase.verifyGreaterThanOrEqual(count, ...
                cfg.params.MFT.num_targets(1));
            testCase.verifyLessThanOrEqual(count, ...
                cfg.params.MFT.num_targets(2));
            testCase.verifyEqual(count, fix(count));
            testCase.verifyTrue(all(isfinite(delays(1:count))));
            testCase.verifyTrue(issorted(delays(1:count)));
            testCase.verifyTrue(all(isfinite(amplitudes(1:count))));
            testCase.verifyGreaterThanOrEqual(delays(1:count), ...
                repmat(cfg.params.MFT.target_delay_s(1) + ...
                cfg.params.common.arrival_delay_s(1) - ...
                1 / cfg.radar.fs_Hz, 1, count));
            testCase.verifyLessThanOrEqual(delays(1:count), ...
                repmat(cfg.params.MFT.target_delay_s(2) + ...
                cfg.params.common.arrival_delay_s(2) + ...
                1 / cfg.radar.fs_Hz, 1, count));
            testCase.verifyGreaterThanOrEqual(amplitudes(1:count), ...
                repmat(cfg.params.MFT.target_relative_amplitude(1), ...
                1, count));
            testCase.verifyLessThanOrEqual(amplitudes(1:count), ...
                repmat(cfg.params.MFT.target_relative_amplitude(2), ...
                1, count));
            testCase.verifyTrue(all(isnan(delays(count + 1:end))));
            testCase.verifyTrue(all(isnan(amplitudes(count + 1:end))));
            testCase.verifyEqual(params.mean_delay_s, ...
                mean(delays(1:count)), 'AbsTol', 10 * eps);
            testCase.verifyEqual(params.mean_spacing_s, ...
                mean(diff(delays(1:count))), 'AbsTol', 10 * eps);
            testCase.verifyTrue(isnan(params.arrival_delay_s));
        end

        function rangeFalseTargetTruthUsesRangeConversionAndPadding(testCase)
            % 验证距离假目标的升序时延、距离换算及 NaN 容量填充。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            [~, params] = generate_single_jamming( ...
                'RangeFakeTarget', cfg, registry, ...
                RandStream('mt19937ar', 'Seed', 808));
            count = params.num_targets;
            delays = arrayfun(@(index) params.(sprintf( ...
                'delay_%02d_s', index)), 1:4);
            ranges = arrayfun(@(index) params.(sprintf( ...
                'range_%02d_m', index)), 1:4);
            amplitudes = arrayfun(@(index) params.(sprintf( ...
                'amplitude_%02d', index)), 1:4);
            targetDelayRange = ...
                cfg.params.RangeFakeTarget.target_delay_s;
            targetAmplitudeRange = ...
                cfg.params.RangeFakeTarget.target_relative_amplitude;

            testCase.verifyGreaterThanOrEqual(count, ...
                cfg.params.RangeFakeTarget.num_targets(1));
            testCase.verifyLessThanOrEqual(count, ...
                cfg.params.RangeFakeTarget.num_targets(2));
            testCase.verifyEqual(count, fix(count));
            testCase.verifyTrue(issorted(delays(1:count)));
            testCase.verifyTrue(all(isfinite(delays(1:count))));
            testCase.verifyTrue(all(isfinite(ranges(1:count))));
            testCase.verifyTrue(all(isfinite(amplitudes(1:count))));
            testCase.verifyGreaterThanOrEqual(delays(1:count), ...
                repmat(targetDelayRange(1) + ...
                cfg.params.common.arrival_delay_s(1) - ...
                1 / cfg.radar.fs_Hz, 1, count));
            testCase.verifyLessThanOrEqual(delays(1:count), ...
                repmat(targetDelayRange(2) + ...
                cfg.params.common.arrival_delay_s(2) + ...
                1 / cfg.radar.fs_Hz, 1, count));
            testCase.verifyGreaterThanOrEqual(amplitudes(1:count), ...
                repmat(targetAmplitudeRange(1), 1, count));
            testCase.verifyLessThanOrEqual(amplitudes(1:count), ...
                repmat(targetAmplitudeRange(2), 1, count));
            testCase.verifyEqual(ranges(1:count), ...
                cfg.radar.c_mps * delays(1:count) / 2, ...
                'RelTol', 10 * eps);
            testCase.verifyTrue(all(isnan(delays(count + 1:end))));
            testCase.verifyTrue(all(isnan(ranges(count + 1:end))));
            testCase.verifyTrue(all(isnan(amplitudes(count + 1:end))));
            testCase.verifyEqual(params.mean_delay_s, ...
                mean(delays(1:count)), 'AbsTol', 10 * eps);
            testCase.verifyTrue(isnan(params.arrival_delay_s));
        end

        function everyTypeSamplesWithinConfiguredRanges(testCase)
            % 验证默认 v2 十三类专属参数均落入标准化配置定义的物理范围。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            truth = struct();
            for typeIndex = 1:numel(registry)
                [~, truth.(registry(typeIndex).code)] = ...
                    generate_single_jamming(registry(typeIndex).code, ...
                    cfg, registry, RandStream('mt19937ar', ...
                    'Seed', 900 + typeIndex));
            end

            testCase.verifyInRange(truth.FMB.center_frequency_Hz, ...
                cfg.params.FMB.center_frequency_Hz);
            nyquistGuard_Hz = cfg.radar.fs_Hz / 2 - max( ...
                eps(cfg.radar.fs_Hz / 2), cfg.radar.fs_Hz * 1e-12);
            testCase.verifyEqual(truth.FMB.bandwidth_factor, ...
                truth.FMB.interference_bandwidth_Hz / ...
                cfg.radar.bandwidth_Hz, 'RelTol', 10 * eps);
            testCase.verifyLessThanOrEqual( ...
                abs(truth.FMB.center_frequency_Hz) + ...
                truth.FMB.interference_bandwidth_Hz / 2, ...
                nyquistGuard_Hz);
            testCase.verifyInRange(truth.AMA.center_frequency_Hz, ...
                cfg.params.AMA.center_frequency_Hz);
            testCase.verifyInRange(truth.AMA.modulation_index, ...
                cfg.params.AMA.modulation_index);
            testCase.verifyEqual(truth.AMA.bandwidth_factor, ...
                truth.AMA.interference_bandwidth_Hz / ...
                cfg.radar.bandwidth_Hz, 'RelTol', 10 * eps);
            testCase.verifyLessThanOrEqual( ...
                abs(truth.AMA.center_frequency_Hz) + ...
                truth.AMA.interference_bandwidth_Hz / 2, ...
                nyquistGuard_Hz);
            testCase.verifyInRange(truth.SawFM.sweep_bandwidth_Hz, ...
                cfg.params.SawFM.sweep_bandwidth_Hz);
            testCase.verifyInRange(truth.SawFM.sweep_period_s, ...
                cfg.params.SawFM.sweep_period_s);
            testCase.verifyInRange(truth.SinFM.center_frequency_Hz, ...
                cfg.params.SinFM.center_frequency_Hz);
            testCase.verifyInRange(truth.SinFM.frequency_deviation_Hz, ...
                cfg.params.SinFM.frequency_deviation_Hz);
            testCase.verifyInRange(truth.SinFM.modulation_frequency_Hz, ...
                cfg.params.SinFM.modulation_frequency_Hz);
            testCase.verifyInRange(truth.SN.sample_width_s, ...
                cfg.params.SN.sample_width_s);
            testCase.verifyInRange(truth.SN.num_replicas, ...
                cfg.params.SN.num_replicas);
            testCase.verifyEqual(truth.SN.num_replicas, ...
                fix(truth.SN.num_replicas));
            testCase.verifyInRange(truth.VFT.doppler_Hz, ...
                cfg.params.VFT.doppler_Hz);
            testCase.verifyInRange(truth.ISRJ.sample_width_s, ...
                cfg.params.ISRJ.sample_width_s);
            testCase.verifyInRange(truth.ISRJ.num_replicas, ...
                cfg.params.ISRJ.num_replicas);
            testCase.verifyEqual(truth.ISRJ.num_replicas, ...
                fix(truth.ISRJ.num_replicas));
            testCase.verifyEqual(truth.ISRJ.mode_code, 0);
            testCase.verifyInRange(truth.Traction.initial_range_m, ...
                cfg.params.Traction.initial_range_m);
            testCase.verifyInRange(truth.Traction.velocity_mps, ...
                cfg.params.Traction.velocity_mps);
            testCase.verifyInRange(truth.Traction.amplitude, ...
                cfg.params.Traction.amplitude);
        end

        function arrivalDelayIsRoundedInSecondsAndAppliedOnce(testCase)
            % 验证到达时延按 round(delay*fs) 仅移位一次，不进行微秒换算。
            cfg = create_default_config();
            exactDelay_s = 3.4 / cfg.radar.fs_Hz;
            doppler_Hz = 5000;
            cfg.params.common.arrival_delay_s = ...
                [exactDelay_s, exactDelay_s];
            cfg.params.common.JNR_dB = [30, 30];
            cfg.params.VFT.doppler_Hz = [doppler_Hz, doppler_Hz];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            reference = build_reference_signal(cfg);
            undelayed = reference.train .* exp(1i * 2 * pi * ...
                doppler_Hz * reference.time_train_s);
            expected = circshift(undelayed, ...
                [0, round(exactDelay_s * cfg.radar.fs_Hz)]);

            [signal, params] = generate_single_jamming( ...
                'VFT', cfg, registry, ...
                RandStream('mt19937ar', 'Seed', 1111));

            testCase.verifyEqual(params.arrival_delay_s, ...
                3 / cfg.radar.fs_Hz);
            testCase.verifyEqual(signal, expected, 'AbsTol', 1e-12);
        end

        function sampledWidthsReportRealizedSampleDurations(testCase)
            % 验证 SN/ISRJ 的宽度和周期真值使用实际整数样点宽度。
            codes = {'SN', 'ISRJ'};
            for codeIndex = 1:numel(codes)
                cfg = create_default_config();
                requestedWidth_s = 5.4 / cfg.radar.fs_Hz;
                cfg.params.common.arrival_delay_s = [0, 0];
                cfg.params.(codes{codeIndex}).sample_width_s = ...
                    [requestedWidth_s, requestedWidth_s];
                cfg.params.(codes{codeIndex}).num_replicas = [2, 2];
                cfg = normalize_and_validate_config(cfg, 'balanced');
                registry = build_jamming_registry(cfg);

                [~, params] = generate_single_jamming( ...
                    codes{codeIndex}, cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', 1120 + codeIndex));

                actualWidth_s = 5 / cfg.radar.fs_Hz;
                testCase.verifyEqual(params.sample_width_s, actualWidth_s);
                testCase.verifyEqual(params.sample_period_s, ...
                    3 * actualWidth_s);
            end
        end

        function quantizedTargetDelaysDriveTruthAndWaveform(testCase)
            % 验证假目标先量化时延，再排序、重构波形并派生距离真值。
            codes = {'MFT', 'RangeFakeTarget'};
            for codeIndex = 1:numel(codes)
                cfg = create_default_config();
                requestedArrival_s = 2.4 / cfg.radar.fs_Hz;
                cfg.params.common.arrival_delay_s = ...
                    [requestedArrival_s, requestedArrival_s];
                requestedDelay_s = 3.4 / cfg.radar.fs_Hz;
                cfg.params.(codes{codeIndex}).num_targets = [1, 1];
                cfg.params.(codes{codeIndex}).target_delay_s = ...
                    [requestedDelay_s, requestedDelay_s];
                cfg.params.(codes{codeIndex}). ...
                    target_relative_amplitude = [1, 1];
                cfg = normalize_and_validate_config(cfg, 'balanced');
                registry = build_jamming_registry(cfg);
                reference = build_reference_signal(cfg);

                [signal, params] = generate_single_jamming( ...
                    codes{codeIndex}, cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', 1130 + codeIndex), ...
                    reference);

                realizedDelay_s = 5 / cfg.radar.fs_Hz;
                expected = circshift(reference.train, [0, 5]);
                testCase.verifyEqual(params.delay_01_s, realizedDelay_s);
                testCase.verifyEqual(params.mean_delay_s, realizedDelay_s);
                testCase.verifyEqual(signal, expected, 'AbsTol', 1e-12);
                testCase.verifyTrue(isnan(params.arrival_delay_s));
                if strcmp(codes{codeIndex}, 'RangeFakeTarget')
                    testCase.verifyEqual(params.range_01_m, ...
                        cfg.radar.c_mps * realizedDelay_s / 2);
                end
            end
        end

        function deterministicGeneratorAddsNoReceiverBackgroundNoise(testCase)
            % 验证固定参数 VFT 仅由物理模型决定，不叠加接收机背景噪声。
            cfg = create_default_config();
            cfg.params.common.arrival_delay_s = [0, 0];
            cfg.params.common.JNR_dB = [25, 25];
            cfg.params.VFT.doppler_Hz = [5000, 5000];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            reference = build_reference_signal(cfg);
            expected = reference.train .* exp(1i * 2 * pi * ...
                5000 * reference.time_train_s);

            signalA = generate_single_jamming('VFT', cfg, registry, ...
                RandStream('mt19937ar', 'Seed', 1));
            signalB = generate_single_jamming('VFT', cfg, registry, ...
                RandStream('mt19937ar', 'Seed', 999));

            testCase.verifyEqual(signalA, expected, 'AbsTol', 1e-12);
            testCase.verifyEqual(signalB, expected, 'AbsTol', 1e-12);
        end

        function isrjRejectsUnsupportedMode(testCase)
            % 验证 ISRJ 非 direct 模式保留明确的模型能力错误标识。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            cfg.params.ISRJ.mode = 'phase_modulated';
            registry = build_jamming_registry(cfg);

            testCase.verifyError(@() generate_single_jamming( ...
                'ISRJ', cfg, registry, ...
                RandStream('mt19937ar', 'Seed', 1212)), ...
                'JammingGenerator:UnsupportedMode');
        end

        function defaultMultiSeedTruthNeverExceedsNyquist(testCase)
            % 验证默认配置多随机种子下各类实际离散频率支持均不越 Nyquist。
            cfg = create_default_config();
            cfg.params.common.arrival_delay_s = [0, 0];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            reference = build_reference_signal(cfg);
            nyquistGuard_Hz = cfg.radar.fs_Hz / 2 - max( ...
                eps(cfg.radar.fs_Hz / 2), cfg.radar.fs_Hz * 1e-12);
            pulseSamples = cfg.derived.samples_per_pulse;
            chirpRate = cfg.derived.chirp_rate_Hz_per_s;
            adjacentIndex = 1:pulseSamples - 1;
            referenceFrequency_Hz = chirpRate * ...
                (2 * adjacentIndex - 1) / (2 * cfg.radar.fs_Hz);
            referenceMinimum_Hz = chirpRate / (2 * cfg.radar.fs_Hz);
            referenceMaximum_Hz = chirpRate * (2 * pulseSamples - 3) / ...
                (2 * cfg.radar.fs_Hz);

            for seed = 1:64
                [~, fmb] = generate_single_jamming('FMB', cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', seed));
                [~, ama] = generate_single_jamming('AMA', cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', 100 + seed));
                [~, saw] = generate_single_jamming('SawFM', cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', 200 + seed));
                [~, sinfm] = generate_single_jamming('SinFM', cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', 300 + seed));
                [vftSignal, vft] = generate_single_jamming( ...
                    'VFT', cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', 400 + seed), reference);
                realizedFrequency_Hz = angle( ...
                    vftSignal(2:pulseSamples) .* ...
                    conj(vftSignal(1:pulseSamples - 1))) * ...
                    cfg.radar.fs_Hz / (2 * pi);
                expectedFrequency_Hz = referenceFrequency_Hz + ...
                    vft.doppler_Hz;

                testCase.verifyGreaterThan(fmb.interference_bandwidth_Hz, 0);
                testCase.verifyLessThanOrEqual( ...
                    abs(fmb.center_frequency_Hz) + ...
                    fmb.interference_bandwidth_Hz / 2, nyquistGuard_Hz);
                testCase.verifyLessThanOrEqual( ...
                    abs(ama.center_frequency_Hz) + ...
                    ama.interference_bandwidth_Hz / 2, nyquistGuard_Hz);
                testCase.verifyLessThanOrEqual( ...
                    saw.sweep_bandwidth_Hz / 2, nyquistGuard_Hz);
                testCase.verifyLessThanOrEqual( ...
                    abs(sinfm.center_frequency_Hz) + ...
                    sinfm.frequency_deviation_Hz, nyquistGuard_Hz);
                testCase.verifyGreaterThanOrEqual(vft.doppler_Hz, ...
                    -nyquistGuard_Hz - referenceMinimum_Hz);
                testCase.verifyLessThanOrEqual(vft.doppler_Hz, ...
                    nyquistGuard_Hz - referenceMaximum_Hz);
                testCase.verifyEqual(realizedFrequency_Hz, ...
                    expectedFrequency_Hz, 'AbsTol', 2e-6);
                testCase.verifyLessThanOrEqual( ...
                    max(abs(realizedFrequency_Hz)), nyquistGuard_Hz);
            end
        end

        function wideConfiguredBandsAreClippedToAliasFreeTruth(testCase)
            % 验证宽配置按采样率及载频剩余支持裁剪实际带宽。
            cfg = create_default_config();
            cfg.params.common.arrival_delay_s = [0, 0];
            cfg.params.FMB.center_frequency_Hz = [4e6, 4e6];
            cfg.params.FMB.bandwidth_factor = [5, 5];
            cfg.params.AMA.center_frequency_Hz = [-3e6, -3e6];
            cfg.params.AMA.bandwidth_factor = [5, 5];
            cfg.params.SawFM.sweep_bandwidth_Hz = [20e6, 20e6];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);

            [fmbSignal, fmb] = generate_single_jamming( ...
                'FMB', cfg, registry, RandStream('mt19937ar', 'Seed', 17));
            [~, ama] = generate_single_jamming( ...
                'AMA', cfg, registry, RandStream('mt19937ar', 'Seed', 18));
            [~, saw] = generate_single_jamming( ...
                'SawFM', cfg, registry, RandStream('mt19937ar', 'Seed', 19));

            nyquistGuard_Hz = cfg.radar.fs_Hz / 2 - max( ...
                eps(cfg.radar.fs_Hz / 2), cfg.radar.fs_Hz * 1e-12);
            testCase.verifyEqual(fmb.bandwidth_factor, ...
                fmb.interference_bandwidth_Hz / ...
                cfg.radar.bandwidth_Hz, 'RelTol', 10 * eps);
            testCase.verifyLessThanOrEqual( ...
                abs(ama.center_frequency_Hz) + ...
                ama.interference_bandwidth_Hz / 2, nyquistGuard_Hz);
            testCase.verifyEqual(saw.sweep_bandwidth_Hz, ...
                2 * nyquistGuard_Hz);

            carrierStep = exp(-1i * 2 * pi * ...
                fmb.center_frequency_Hz / cfg.radar.fs_Hz);
            modulationStep = fmbSignal(2:end) .* ...
                conj(fmbSignal(1:end - 1)) * carrierStep;
            modulationFrequency_Hz = angle(modulationStep) * ...
                cfg.radar.fs_Hz / (2 * pi);
            testCase.verifyLessThanOrEqual(max(abs(modulationFrequency_Hz)), ...
                fmb.interference_bandwidth_Hz / 2 + 1e-5);
            testCase.verifyGreaterThan(max(abs(modulationFrequency_Hz)), ...
                0.95 * fmb.interference_bandwidth_Hz / 2);
        end

        function sinfmSamplesDeviationAfterCenterFrequency(testCase)
            % 验证 SinFM 先采中心频率，再从剩余 Nyquist 支持内采频偏。
            cfg = create_default_config();
            cfg.params.SinFM.center_frequency_Hz = [3.5e6, 4e6];
            cfg.params.SinFM.frequency_deviation_Hz = [1e6, 5e6];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            nyquistGuard_Hz = cfg.radar.fs_Hz / 2 - max( ...
                eps(cfg.radar.fs_Hz / 2), cfg.radar.fs_Hz * 1e-12);

            for seed = 1:24
                [~, params] = generate_single_jamming( ...
                    'SinFM', cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', seed));
                testCase.verifyGreaterThanOrEqual( ...
                    params.frequency_deviation_Hz, ...
                    cfg.params.SinFM.frequency_deviation_Hz(1));
                testCase.verifyLessThanOrEqual( ...
                    abs(params.center_frequency_Hz) + ...
                    params.frequency_deviation_Hz, nyquistGuard_Hz);
            end
        end

        function exactNyquistAndOutOfBandDopplerAreRejected(testCase)
            % 验证 SinFM 不报告 +/-Nyquist，VFT 无合法交集时拒绝生成。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            nyquist_Hz = cfg.radar.fs_Hz / 2;
            invalidSin = cfg;
            invalidSin.params.SinFM.center_frequency_Hz = [0, 0];
            invalidSin.params.SinFM.frequency_deviation_Hz = ...
                [nyquist_Hz, nyquist_Hz];
            invalidVft = cfg;
            invalidVft.params.VFT.doppler_Hz = [30e3, 30e3];

            testCase.verifyError(@() generate_single_jamming( ...
                'SinFM', invalidSin, registry, ...
                RandStream('mt19937ar', 'Seed', 24)), ...
                'JammingGenerator:InvalidPhysicalParameter');
            testCase.verifyError(@() generate_single_jamming( ...
                'VFT', invalidVft, registry, ...
                RandStream('mt19937ar', 'Seed', 25)), ...
                'JammingGenerator:InvalidPhysicalParameter');
        end

        function generatorsDefendCriticalPhysicalBoundaries(testCase)
            % 验证绕过 normalize 后，生成器仍以稳定标识拒绝关键物理越界。
            base = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(base);
            nyquist_Hz = base.radar.fs_Hz / 2;
            invalid = cell(1, 14);

            invalid{1} = struct('code', 'FMB', 'cfg', base);
            invalid{1}.cfg.params.FMB.center_frequency_Hz = ...
                [nyquist_Hz, nyquist_Hz];
            invalid{2} = struct('code', 'FMB', 'cfg', base);
            invalid{2}.cfg.params.FMB.bandwidth_factor = [0, 0];
            invalid{3} = struct('code', 'AMA', 'cfg', base);
            invalid{3}.cfg.params.AMA.center_frequency_Hz = ...
                [nyquist_Hz + 1, nyquist_Hz + 1];
            invalid{4} = struct('code', 'SawFM', 'cfg', base);
            invalid{4}.cfg.params.SawFM.sweep_period_s = [0, 0];
            invalid{5} = struct('code', 'SinFM', 'cfg', base);
            invalid{5}.cfg.params.SinFM.modulation_frequency_Hz = [0, 0];
            invalid{6} = struct('code', 'SinFM', 'cfg', base);
            invalid{6}.cfg.params.SinFM.frequency_deviation_Hz = [-1, -1];
            invalid{7} = struct('code', 'SinFM', 'cfg', base);
            invalid{7}.cfg.params.SinFM.center_frequency_Hz = [4.5e6, 4.5e6];
            invalid{7}.cfg.params.SinFM.frequency_deviation_Hz = [1e6, 2e6];
            invalid{8} = struct('code', 'SN', 'cfg', base);
            invalid{8}.cfg.params.SN.sample_width_s = [0, 0];
            invalid{9} = struct('code', 'ISRJ', 'cfg', base);
            invalid{9}.cfg.params.ISRJ.sample_width_s = [0, 0];
            invalid{10} = struct('code', 'MFT', 'cfg', base);
            invalid{10}.cfg.params.MFT.target_delay_s = ...
                [base.radar.pri_s, base.radar.pri_s];
            invalid{11} = struct('code', 'RangeFakeTarget', 'cfg', base);
            invalid{11}.cfg.params.RangeFakeTarget.target_delay_s = [-1, -1];
            invalid{12} = struct('code', 'Traction', 'cfg', base);
            invalid{12}.cfg.params.Traction.initial_range_m = [-1, -1];
            invalid{13} = struct('code', 'Traction', 'cfg', base);
            invalid{13}.cfg.params.Traction.amplitude = [0, 0];
            invalid{14} = struct('code', 'VFT', 'cfg', base);
            invalid{14}.cfg.params.common.arrival_delay_s = ...
                [base.radar.pri_s, base.radar.pri_s];

            for index = 1:numel(invalid)
                item = invalid{index};
                testCase.verifyError(@() generate_single_jamming( ...
                    item.code, item.cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', index)), ...
                    'JammingGenerator:InvalidPhysicalParameter');
            end
        end

        function fmbAmaRejectRangesTouchingNyquistGuard(testCase)
            % 验证绕过 normalize 时，中心频率范围端点等于 guard 也被拒绝。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            guard_Hz = cfg.radar.fs_Hz / 2 - max( ...
                eps(cfg.radar.fs_Hz / 2), cfg.radar.fs_Hz * 1e-12);
            codes = {'FMB', 'AMA'};

            for codeIndex = 1:numel(codes)
                invalid = cfg;
                invalid.params.(codes{codeIndex}).center_frequency_Hz = ...
                    [0, guard_Hz];
                testCase.verifyError(@() generate_single_jamming( ...
                    codes{codeIndex}, invalid, registry, ...
                    RandStream('mt19937ar', 'Seed', 40 + codeIndex)), ...
                    'JammingGenerator:InvalidPhysicalParameter');
            end
        end

        function generatorsEnforceFixedTargetCapacities(testCase)
            % 验证生成器自身也拒绝 MFT 9 目标和 RangeFakeTarget 5 目标。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            invalidMft = cfg;
            invalidMft.params.MFT.num_targets = [9, 9];
            invalidRange = cfg;
            invalidRange.params.RangeFakeTarget.num_targets = [5, 5];

            testCase.verifyError(@() generate_single_jamming( ...
                'MFT', invalidMft, registry, ...
                RandStream('mt19937ar', 'Seed', 1)), ...
                'JammingGenerator:InvalidPhysicalParameter');
            testCase.verifyError(@() generate_single_jamming( ...
                'RangeFakeTarget', invalidRange, registry, ...
                RandStream('mt19937ar', 'Seed', 2)), ...
                'JammingGenerator:InvalidPhysicalParameter');
        end

        function tractionUsesObservableFractionalPulseDelays(testCase)
            % 验证默认四脉冲下 0.004 样点/脉冲变化不会被整数舍入抹除。
            cfg = create_default_config();
            cfg.params.Traction.initial_range_m = [1000, 1000];
            cfg.params.Traction.velocity_mps = [200, 200];
            cfg.params.Traction.amplitude = [1, 1];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);

            [signal, params] = generate_single_jamming( ...
                'Traction', cfg, registry, ...
                RandStream('mt19937ar', 'Seed', 31));
            blocks = reshape(signal, cfg.derived.samples_per_pri, []).';

            testCase.verifyEqual(params.delay_change_per_pulse_s * ...
                cfg.radar.fs_Hz, 0.004, 'AbsTol', 1e-12);
            for pulseIndex = 2:cfg.radar.num_pulses
                testCase.verifyGreaterThan(max(abs( ...
                    blocks(pulseIndex, :) - blocks(1, :))), 1e-6);
            end
        end

        function tractionIntegerDelayMatchesCircshiftDirection(testCase)
            % 验证零边界线性插值在整数时延且无回卷时与正向 circshift 一致。
            cfg = create_default_config();
            integerDelay = 17;
            range_m = integerDelay * cfg.radar.c_mps / ...
                (2 * cfg.radar.fs_Hz);
            cfg.params.Traction.initial_range_m = [range_m, range_m];
            cfg.params.Traction.velocity_mps = [0, 0];
            cfg.params.Traction.amplitude = [1, 1];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            reference = build_reference_signal(cfg);
            expected = circshift(reference.pulse, [0, integerDelay]);

            signal = generate_single_jamming('Traction', cfg, registry, ...
                RandStream('mt19937ar', 'Seed', 32), reference);
            blocks = reshape(signal, cfg.derived.samples_per_pri, []).';

            for pulseIndex = 1:cfg.radar.num_pulses
                testCase.verifyEqual(blocks(pulseIndex, :), expected, ...
                    'AbsTol', 1e-11);
            end
        end

        function tractionFractionalDelayHasZeroBoundaries(testCase)
            % 验证 500.5 样点分数时延不把脉冲循环泄漏到 PRI 两端。
            cfg = create_default_config();
            delaySamples = 500.5;
            range_m = delaySamples * cfg.radar.c_mps / ...
                (2 * cfg.radar.fs_Hz);
            cfg.params.Traction.initial_range_m = [range_m, range_m];
            cfg.params.Traction.velocity_mps = [0, 0];
            cfg.params.Traction.amplitude = [1, 1];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            signal = generate_single_jamming('Traction', cfg, registry, ...
                RandStream('mt19937ar', 'Seed', 35));
            blocks = reshape(signal, cfg.derived.samples_per_pri, []).';
            lastSupportedZeroBased = ceil(delaySamples + ...
                cfg.derived.samples_per_pulse - 1);
            outsideSupport = [ ...
                blocks(1, 1:floor(delaySamples) + 1), ...
                blocks(1, lastSupportedZeroBased + 2:end)];

            testCase.verifyGreaterThan(max(abs(blocks(1, :))), 0.9);
            testCase.verifyLessThanOrEqual( ...
                sum(abs(outsideSupport) .^ 2), 1e-24);
            testCase.verifyLessThanOrEqual( ...
                max(abs(outsideSupport)), 1e-12);
        end

        function tractionRejectsInvalidPulseTrajectory(testCase)
            % 验证每脉冲距离非负且延迟后的完整脉冲不得越过 PRI。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            negativeLater = cfg;
            negativeLater.params.Traction.initial_range_m = [1, 1];
            negativeLater.params.Traction.velocity_mps = [-1e6, -1e6];
            tooLate = cfg;
            maxRange_m = cfg.radar.c_mps * cfg.radar.pri_s / 2;
            tooLate.params.Traction.initial_range_m = [maxRange_m, maxRange_m];
            nearWrap = cfg;
            nearWrapDelay_s = cfg.radar.pri_s - ...
                cfg.radar.pulse_width_s / 2;
            nearWrapRange_m = nearWrapDelay_s * ...
                cfg.radar.c_mps / 2;
            nearWrap.params.Traction.initial_range_m = ...
                [nearWrapRange_m, nearWrapRange_m];
            nearWrap.params.Traction.velocity_mps = [0, 0];

            testCase.verifyError(@() generate_single_jamming( ...
                'Traction', negativeLater, registry, ...
                RandStream('mt19937ar', 'Seed', 33)), ...
                'JammingGenerator:InvalidPhysicalParameter');
            testCase.verifyError(@() generate_single_jamming( ...
                'Traction', tooLate, registry, ...
                RandStream('mt19937ar', 'Seed', 34)), ...
                'JammingGenerator:InvalidPhysicalParameter');
            testCase.verifyError(@() generate_single_jamming( ...
                'Traction', nearWrap, registry, ...
                RandStream('mt19937ar', 'Seed', 36)), ...
                'JammingGenerator:InvalidPhysicalParameter');
        end

        function dispatcherReusesValidatedReference(testCase)
            % 验证五参数复用 reference，且与四参数路径同 seed 完全一致。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            reference = build_reference_signal(cfg);

            [signalA, paramsA] = generate_single_jamming( ...
                'FMB', cfg, registry, ...
                RandStream('mt19937ar', 'Seed', 41));
            [signalB, paramsB] = generate_single_jamming( ...
                'FMB', cfg, registry, ...
                RandStream('mt19937ar', 'Seed', 41), reference);

            testCase.verifyEqual(signalA, signalB);
            testCase.verifyEqual(paramsA, paramsB);
        end

        function dispatcherRejectsMalformedProvidedReference(testCase)
            % 验证外部 reference 的字段、行形状、有限性和长度均受校验。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            reference = build_reference_signal(cfg);
            invalidReferences = cell(1, 5);
            invalidReferences{1} = rmfield(reference, 'pulse');
            invalidReferences{2} = reference;
            invalidReferences{2}.pulse = invalidReferences{2}.pulse.';
            invalidReferences{3} = reference;
            invalidReferences{3}.train(end) = [];
            invalidReferences{4} = reference;
            invalidReferences{4}.time_pri_s(1) = NaN;
            invalidReferences{5} = reference;
            invalidReferences{5}.time_train_s(1) = 1i;

            for index = 1:numel(invalidReferences)
                testCase.verifyError(@() generate_single_jamming( ...
                    'FMB', cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', index), ...
                    invalidReferences{index}), ...
                    'JammingGenerator:InvalidReference');
            end
        end

        function dispatcherRejectsSemanticallyDifferentReference(testCase)
            % 验证同长度但带宽不同的 reference 不能冒充当前 cfg 的 LFM。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            otherCfg = create_default_config();
            otherCfg.radar.bandwidth_Hz = 4e6;
            otherCfg = normalize_and_validate_config(otherCfg, 'balanced');
            wrongReference = build_reference_signal(otherCfg);

            testCase.verifyError(@() generate_single_jamming( ...
                'FMB', cfg, registry, ...
                RandStream('mt19937ar', 'Seed', 44), wrongReference), ...
                'JammingGenerator:InvalidGeneratorOutput');
        end

        function replicaGeneratorsRejectUnrealizableRequests(testCase)
            % 验证 N=10、宽 3、3 副本时尾部截断请求被拒绝。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            cfg.params.common.arrival_delay_s = [0, 0];
            reference = struct('train', complex([1, 1, 1, zeros(1, 7)]));
            invalidSn = cfg;
            invalidSn.params.SN.sample_width_s = ...
                [3 / cfg.radar.fs_Hz, 3 / cfg.radar.fs_Hz];
            invalidSn.params.SN.num_replicas = [3, 3];
            invalidIsrj = cfg;
            invalidIsrj.params.ISRJ.sample_width_s = ...
                [3 / cfg.radar.fs_Hz, 3 / cfg.radar.fs_Hz];
            invalidIsrj.params.ISRJ.num_replicas = [3, 3];

            testCase.verifyError(@() generate_smart_noise( ...
                invalidSn, RandStream('mt19937ar', 'Seed', 51), reference), ...
                'JammingGenerator:InvalidPhysicalParameter');
            testCase.verifyError(@() generate_isrj( ...
                invalidIsrj, RandStream('mt19937ar', 'Seed', 52), reference), ...
                'JammingGenerator:InvalidPhysicalParameter');
        end

        function replicaGeneratorsKeepEveryFeasibleReplicaComplete(testCase)
            % 验证 N=10、宽 2、3 副本时 8 个样点全部完整且真值已量化。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            cfg.params.common.arrival_delay_s = [0, 0];
            widthSamples = 2;
            numReplicas = 3;
            cfg.params.SN.sample_width_s = ...
                [widthSamples / cfg.radar.fs_Hz, ...
                widthSamples / cfg.radar.fs_Hz];
            cfg.params.SN.num_replicas = [numReplicas, numReplicas];
            cfg.params.ISRJ.sample_width_s = ...
                cfg.params.SN.sample_width_s;
            cfg.params.ISRJ.num_replicas = [numReplicas, numReplicas];
            reference = struct('train', complex([1, 1, zeros(1, 8)]));

            [snSignal, snParams] = generate_smart_noise( ...
                cfg, RandStream('mt19937ar', 'Seed', 53), reference);
            [isrjSignal, isrjParams] = generate_isrj( ...
                cfg, RandStream('mt19937ar', 'Seed', 54), reference);

            testCase.verifyEqual(snParams.sample_width_s, ...
                widthSamples / cfg.radar.fs_Hz);
            testCase.verifyEqual(isrjParams.sample_width_s, ...
                widthSamples / cfg.radar.fs_Hz);
            testCase.verifyEqual(snParams.num_replicas, numReplicas);
            testCase.verifyEqual(isrjParams.num_replicas, numReplicas);
            testCase.verifyEqual(snParams.sample_period_s, ...
                (numReplicas + 1) * widthSamples / cfg.radar.fs_Hz);
            testCase.verifyEqual(isrjParams.sample_period_s, ...
                snParams.sample_period_s);
            for replicaIndex = 0:numReplicas
                segment = replicaIndex * widthSamples + ...
                    (1:widthSamples);
                testCase.verifyEqual(abs(snSignal(segment)), ones(1, 2), ...
                    'AbsTol', 10 * eps);
                testCase.verifyEqual(isrjSignal(segment), ...
                    complex(ones(1, 2)), 'AbsTol', 10 * eps);
            end
            testCase.verifyEqual(snSignal(9:10), zeros(1, 2));
            testCase.verifyEqual(isrjSignal(9:10), zeros(1, 2));
        end

        function targetGeneratorsRejectNonpositiveRelativeAmplitudes(testCase)
            % 验证绕过 normalize 后两类假目标生成器仍拒绝零/负相对幅度。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            invalidMft = cfg;
            invalidMft.params.MFT.target_relative_amplitude = [0, 0];
            invalidRange = cfg;
            invalidRange.params.RangeFakeTarget. ...
                target_relative_amplitude = [-1, -0.5];

            testCase.verifyError(@() generate_single_jamming( ...
                'MFT', invalidMft, registry, ...
                RandStream('mt19937ar', 'Seed', 53)), ...
                'JammingGenerator:InvalidPhysicalParameter');
            testCase.verifyError(@() generate_single_jamming( ...
                'RangeFakeTarget', invalidRange, registry, ...
                RandStream('mt19937ar', 'Seed', 54)), ...
                'JammingGenerator:InvalidPhysicalParameter');
        end

        function fmbClippingReportsOnlyRealizedIdentifiableFactor(testCase)
            % 验证相同中心频率下请求因子 2/5 均饱和时，波形和监督因子相同。
            requestedFactors = [2, 5];
            signals = cell(1, 2);
            truths = cell(1, 2);
            for factorIndex = 1:2
                cfg = create_default_config();
                cfg.params.common.JNR_dB = [30, 30];
                cfg.params.common.arrival_delay_s = [0, 0];
                cfg.params.FMB.center_frequency_Hz = [4e6, 4e6];
                cfg.params.FMB.bandwidth_factor = ...
                    repmat(requestedFactors(factorIndex), 1, 2);
                cfg = normalize_and_validate_config(cfg, 'balanced');
                registry = build_jamming_registry(cfg);
                [signals{factorIndex}, truths{factorIndex}] = ...
                    generate_single_jamming('FMB', cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', 1701));
            end

            testCase.verifyEqual(signals{1}, signals{2}, 'AbsTol', 1e-12);
            testCase.verifyEqual(truths{1}.interference_bandwidth_Hz, ...
                truths{2}.interference_bandwidth_Hz, 'AbsTol', 1e-9);
            testCase.verifyEqual(truths{1}.bandwidth_factor, ...
                truths{2}.bandwidth_factor, 'AbsTol', 1e-12);
            testCase.verifyEqual(truths{1}.bandwidth_factor, ...
                truths{1}.interference_bandwidth_Hz / 5e6, ...
                'RelTol', 10 * eps);
        end

        function amaClippingReportsOnlyRealizedIdentifiableFactor(testCase)
            % 验证 AMA 带宽裁剪后，请求因子不再作为不可辨识监督真值。
            requestedFactors = [2, 5];
            signals = cell(1, 2);
            truths = cell(1, 2);
            for factorIndex = 1:2
                cfg = create_default_config();
                cfg.params.common.JNR_dB = [30, 30];
                cfg.params.common.arrival_delay_s = [0, 0];
                cfg.params.AMA.center_frequency_Hz = [-3e6, -3e6];
                cfg.params.AMA.bandwidth_factor = ...
                    repmat(requestedFactors(factorIndex), 1, 2);
                cfg.params.AMA.modulation_index = [1.5, 1.5];
                cfg = normalize_and_validate_config(cfg, 'balanced');
                registry = build_jamming_registry(cfg);
                [signals{factorIndex}, truths{factorIndex}] = ...
                    generate_single_jamming('AMA', cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', 1702));
            end

            testCase.verifyEqual(signals{1}, signals{2}, 'AbsTol', 1e-12);
            testCase.verifyEqual(truths{1}.interference_bandwidth_Hz, ...
                truths{2}.interference_bandwidth_Hz, 'AbsTol', 1e-9);
            testCase.verifyEqual(truths{1}.bandwidth_factor, ...
                truths{2}.bandwidth_factor, 'AbsTol', 1e-12);
            testCase.verifyEqual(truths{1}.bandwidth_factor, ...
                truths{1}.interference_bandwidth_Hz / 5e6, ...
                'RelTol', 10 * eps);
        end

        function fmbEqualDiscreteMasksShareWaveformAndBandwidthTruth(testCase)
            % 验证两个连续请求带宽落入同一 FFT 掩膜时，FMB 波形与监督
            % 真值都采用同一个离散实现带宽，不能保留不可辨识的小数差异。
            requestedFactors = [0.5, 0.5001];
            signals = cell(1, 2);
            truths = cell(1, 2);
            for factorIndex = 1:2
                cfg = create_default_config();
                cfg.params.common.JNR_dB = [30, 30];
                cfg.params.common.arrival_delay_s = [0, 0];
                cfg.params.FMB.center_frequency_Hz = [0, 0];
                cfg.params.FMB.bandwidth_factor = ...
                    repmat(requestedFactors(factorIndex), 1, 2);
                cfg = normalize_and_validate_config(cfg, 'balanced');
                registry = build_jamming_registry(cfg);
                [signals{factorIndex}, truths{factorIndex}] = ...
                    generate_single_jamming('FMB', cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', 2101));
            end

            expectedBandwidth_Hz = 2.5e6;
            testCase.verifyEqual(signals{1}, signals{2}, 'AbsTol', 1e-12);
            testCase.verifyEqual(truths{1}.interference_bandwidth_Hz, ...
                expectedBandwidth_Hz, 'AbsTol', 1e-9);
            testCase.verifyEqual(truths{2}.interference_bandwidth_Hz, ...
                expectedBandwidth_Hz, 'AbsTol', 1e-9);
            testCase.verifyEqual(truths{1}.bandwidth_factor, ...
                truths{2}.bandwidth_factor, 'AbsTol', 1e-12);
        end

        function amaEqualDiscreteMasksShareWaveformAndBandwidthTruth(testCase)
            % 验证两个连续请求带宽落入同一 FFT 掩膜时，AMA 波形与监督
            % 真值都回写同一个离散实现带宽。
            requestedFactors = [0.5, 0.5001];
            signals = cell(1, 2);
            truths = cell(1, 2);
            for factorIndex = 1:2
                cfg = create_default_config();
                cfg.params.common.JNR_dB = [30, 30];
                cfg.params.common.arrival_delay_s = [0, 0];
                cfg.params.AMA.center_frequency_Hz = [0, 0];
                cfg.params.AMA.bandwidth_factor = ...
                    repmat(requestedFactors(factorIndex), 1, 2);
                cfg.params.AMA.modulation_index = [1.5, 1.5];
                cfg = normalize_and_validate_config(cfg, 'balanced');
                registry = build_jamming_registry(cfg);
                [signals{factorIndex}, truths{factorIndex}] = ...
                    generate_single_jamming('AMA', cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', 2102));
            end

            expectedBandwidth_Hz = 2.5e6;
            testCase.verifyEqual(signals{1}, signals{2}, 'AbsTol', 1e-12);
            testCase.verifyEqual(truths{1}.interference_bandwidth_Hz, ...
                expectedBandwidth_Hz, 'AbsTol', 1e-9);
            testCase.verifyEqual(truths{2}.interference_bandwidth_Hz, ...
                expectedBandwidth_Hz, 'AbsTol', 1e-9);
            testCase.verifyEqual(truths{1}.bandwidth_factor, ...
                truths{2}.bandwidth_factor, 'AbsTol', 1e-12);
        end

        function fmbAmaRejectDcOnlyDiscreteNoiseSupport(testCase)
            % 验证绕过 normalize 后，低通掩膜只包含 DC 时生成器拒绝；
            % 去均值会消除 DC，不能返回零调制却宣称存在正带宽噪声。
            codes = {'FMB', 'AMA'};
            for codeIndex = 1:numel(codes)
                cfg = normalize_and_validate_config( ...
                    create_default_config(), 'balanced');
                cfg.params.(codes{codeIndex}).bandwidth_factor = [1e-5, 1e-5];
                cfg.params.(codes{codeIndex}).center_frequency_Hz = [0, 0];
                registry = build_jamming_registry(cfg);

                testCase.verifyError(@() generate_single_jamming( ...
                    codes{codeIndex}, cfg, registry, ...
                    RandStream('mt19937ar', 'Seed', 2200 + codeIndex)), ...
                    'JammingGenerator:InvalidPhysicalParameter');
            end
        end

        function targetDelaysAreUniqueAndMatchObservableSupport(testCase)
            % 验证离散时延无放回抽样、最终总时延排序及幅度配对均可观测。
            codes = {'MFT', 'RangeFakeTarget'};
            counts = [8, 4];
            for codeIndex = 1:numel(codes)
                code = codes{codeIndex};
                cfg = create_default_config();
                cfg.params.common.arrival_delay_s = ...
                    [7 / cfg.radar.fs_Hz, 7 / cfg.radar.fs_Hz];
                cfg.params.common.JNR_dB = [25, 25];
                cfg.params.(code).num_targets = ...
                    repmat(counts(codeIndex), 1, 2);
                cfg.params.(code).target_delay_s = ...
                    [10 / cfg.radar.fs_Hz, ...
                    (10 + counts(codeIndex)) / cfg.radar.fs_Hz];
                cfg.params.(code).target_relative_amplitude = [0.5, 1.5];
                cfg = normalize_and_validate_config(cfg, 'balanced');
                reference = struct('train', ...
                    complex([1, zeros(1, cfg.derived.sample_length - 1)]));
                registry = build_jamming_registry(cfg);
                registryIndex = find(strcmp({registry.code}, code), 1);

                [signal, params] = feval( ...
                    registry(registryIndex).generator, ...
                    cfg, RandStream('mt19937ar', ...
                    'Seed', 1800 + codeIndex), reference);
                delays_s = arrayfun(@(index) params.(sprintf( ...
                    'delay_%02d_s', index)), 1:counts(codeIndex));
                amplitudes = arrayfun(@(index) params.(sprintf( ...
                    'amplitude_%02d', index)), 1:counts(codeIndex));
                delaySamples = round(delays_s * cfg.radar.fs_Hz);

                testCase.verifyEqual(numel(unique(delaySamples)), ...
                    counts(codeIndex));
                testCase.verifyTrue(issorted(delaySamples));
                testCase.verifyEqual(signal(delaySamples + 1), ...
                    complex(amplitudes), 'AbsTol', 1e-12);
                testCase.verifyEqual(nnz(abs(signal) > 1e-12), ...
                    counts(codeIndex));
                testCase.verifyTrue(isnan(params.arrival_delay_s));
            end
        end

        function equivalentTargetDelayDecompositionsShareTruthAndWaveform(testCase)
            % 验证相对时延与公共到达时延的不同分解不产生不同监督标签。
            codes = {'MFT', 'RangeFakeTarget'};
            for codeIndex = 1:numel(codes)
                code = codes{codeIndex};
                configs = cell(1, 2);
                relativeSamples = [3, 4];
                arrivalSamples = [2, 1];
                for configIndex = 1:2
                    cfg = create_default_config();
                    cfg.params.common.JNR_dB = [20, 20];
                    cfg.params.common.arrival_delay_s = repmat( ...
                        arrivalSamples(configIndex) / cfg.radar.fs_Hz, 1, 2);
                    cfg.params.(code).num_targets = [1, 1];
                    cfg.params.(code).target_delay_s = repmat( ...
                        relativeSamples(configIndex) / cfg.radar.fs_Hz, 1, 2);
                    cfg.params.(code).target_relative_amplitude = [1, 1];
                    configs{configIndex} = normalize_and_validate_config( ...
                        cfg, 'balanced');
                end
                registry = build_jamming_registry(configs{1});
                reference = build_reference_signal(configs{1});

                [signalA, paramsA] = generate_single_jamming( ...
                    code, configs{1}, registry, ...
                    RandStream('mt19937ar', 'Seed', 1900 + codeIndex), ...
                    reference);
                [signalB, paramsB] = generate_single_jamming( ...
                    code, configs{2}, registry, ...
                    RandStream('mt19937ar', 'Seed', 1900 + codeIndex), ...
                    reference);

                testCase.verifyEqual(signalA, signalB, 'AbsTol', 1e-12);
                testCase.verifyEqual(paramsA, paramsB);
                testCase.verifyEqual(paramsA.delay_01_s, ...
                    5 / configs{1}.radar.fs_Hz);
                testCase.verifyTrue(isnan(paramsA.arrival_delay_s));
            end
        end

        function targetGeneratorsRejectInsufficientDiscreteDelaySlots(testCase)
            % 验证绕过配置校验后，离散时延槽少于目标数时生成器稳定拒绝。
            cfg = normalize_and_validate_config( ...
                create_default_config(), 'balanced');
            registry = build_jamming_registry(cfg);
            cases = {'MFT', 4; 'RangeFakeTarget', 3};
            for caseIndex = 1:size(cases, 1)
                invalid = cfg;
                code = cases{caseIndex, 1};
                count = cases{caseIndex, 2};
                invalid.params.(code).num_targets = repmat(count, 1, 2);
                invalid.params.(code).target_delay_s = ...
                    [0, 1 / invalid.radar.fs_Hz];
                testCase.verifyError(@() generate_single_jamming( ...
                    code, invalid, registry, ...
                    RandStream('mt19937ar', 'Seed', 2000 + caseIndex)), ...
                    'JammingGenerator:InvalidPhysicalParameter');
            end
        end
    end

    methods (Access = private)
        function verifyInRange(testCase, value, range)
            %VERIFYINRANGE 验证标量落在闭区间内。
            testCase.verifyGreaterThanOrEqual(value, range(1));
            testCase.verifyLessThanOrEqual(value, range(2));
        end
    end

    methods (Static, Access = private)
        function cfg = createCombSpectrumFixture(mode, numTeeth)
            %CREATECOMBSPECTRUMFIXTURE 构造绕过注册表集成的直接生成器配置。
            cfg = create_default_config();
            cfg.params.common.JNR_dB = [25, 25];
            cfg.params.common.arrival_delay_s = [0, 0];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            cfg.params.CombSpectrum = struct( ...
                'num_teeth', numTeeth, ...
                'spacing_Hz', [0.2e6, 0.6e6], ...
                'center_offset_Hz', [-0.3e6, 0.3e6], ...
                'amplitude', [0.5, 1.5], ...
                'amplitude_mode', mode, ...
                'max_teeth', 8);
        end

        function cfg = createClutterPulseFixture(mode, pulsesPerPri)
            %CREATECLUTTERPULSEFIXTURE 构造绕过注册表集成的杂乱脉冲配置。
            cfg = create_default_config();
            cfg.params.common.JNR_dB = [25, 25];
            cfg.params.common.arrival_delay_s = [0, 0];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            cfg.params.ClutterPulse = struct( ...
                'pulses_per_pri', pulsesPerPri, ...
                'pulse_width_s', [2e-6, 20e-6], ...
                'relative_amplitude', [0.5, 1.5], ...
                'frequency_offset_Hz', [-1e6, 1e6], ...
                'mode', mode, ...
                'max_pulses_per_pri', 8, ...
                'max_total_pulses', 32);
        end

        function value = readClutterTruth(truth, prefix, index, suffix)
            %READCLUTTERTRUTH 按统一编号规则读取杂乱脉冲详细真值字段。
            value = truth.(sprintf('%s_%02d%s', ...
                prefix, index, suffix));
        end

        function cfg = createSmspFixture(numSubpulses)
            %CREATESMSPFIXTURE 构造绕过注册表集成的标准 SMSP 直接配置。
            cfg = create_default_config();
            cfg.params.common.JNR_dB = [25, 25];
            cfg.params.common.arrival_delay_s = [0, 0];
            cfg = normalize_and_validate_config(cfg, 'balanced');
            cfg.params.SpectralDispersion = struct( ...
                'num_subpulses', numSubpulses, ...
                'max_subpulses', 8);
        end

        function [signal, params] = shortGenerator(cfg, ~, ~)
            %SHORTGENERATOR 构造长度错误但字段完整的测试替身。
            signal = complex(ones(1, cfg.derived.sample_length - 1));
            params = struct('JNR_dB', 1, 'bandwidth_factor', 1, ...
                'interference_bandwidth_Hz', 1, ...
                'center_frequency_Hz', 1, 'arrival_delay_s', 0);
        end

        function [signal, params] = missingFieldGenerator(cfg, ~, ~)
            %MISSINGFIELDGENERATOR 构造信号合法但真值缺字段的测试替身。
            signal = complex(ones(1, cfg.derived.sample_length));
            params = struct('JNR_dB', 1);
        end
    end
end
