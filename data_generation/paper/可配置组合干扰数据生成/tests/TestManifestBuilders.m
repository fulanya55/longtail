classdef TestManifestBuilders < matlab.unittest.TestCase
    %TESTMANIFESTBUILDERS 验证普通与多维不均衡 manifest 的公共契约。
    %   测试覆盖随机与全枚举组合、分量/组合/功率三个不均衡轴、四个数据
    %   划分的种子隔离、注册表规范顺序、未知类标记和非法输入错误标识。

    methods (Test)
        function v2BalancedManifestUsesNewTypeIds(testCase)
            % 验证 v2 平衡场景只从三个新增类型的稳定 ID 11--13 中选取。
            cfg = TestManifestBuilders.createSmallV2Fixture();
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);

            manifest = build_balanced_manifest(cfg, registry, 'train');
            active = unique([manifest.active_type_ids{:}]);

            testCase.verifyTrue(all(ismember(active, 11:13)));
        end

        function v2ImbalanceWeightsAcceptThirteenRegistryEntries(testCase)
            % 验证 v2 可同时启用组合、成分和功率不均衡，并使用十三维权重。
            cfg = TestManifestBuilders.createSmallV2Fixture();
            cfg.imbalance.component.enable = true;
            cfg.imbalance.component.weights = [zeros(1, 10), 1, 2, 3];
            cfg.imbalance.combination.enable = true;
            cfg.imbalance.combination.cardinality_weights = [3, 2, 1];
            cfg.imbalance.power.enable = true;
            cfg.imbalance.power.relative_power_dB = [-12, -6];
            cfg = normalize_and_validate_config(cfg, 'imbalanced');
            registry = build_jamming_registry(cfg);

            manifest = build_imbalanced_manifest(cfg, registry, 'train');

            testCase.verifyEqual(height(manifest), cfg.dataset.num_samples);
            testCase.verifyTrue(all(cellfun(@(ids) ...
                all(ismember(ids, 11:13)), manifest.active_type_ids)));
        end

        function v2OpenSetMarksSmspUnknown(testCase)
            % 验证 v2 开放集训练不含 SMSP，而测试混合集包含该未知类型。
            cfg = TestManifestBuilders.createSmallV2Fixture();
            cfg.openset.train_known = 4;
            cfg.openset.calibration_known = 2;
            cfg.openset.validation_mixed = 4;
            cfg.openset.test_mixed = 4;
            cfg = normalize_and_validate_config(cfg, 'openset');
            registry = build_jamming_registry(cfg);

            splits = build_openset_manifest(cfg, registry);
            smspId = find(strcmp({registry.code}, 'SpectralDispersion'));

            testCase.verifyFalse(any(cellfun(@(ids) ...
                ismember(smspId, ids), ...
                splits.train_known.active_type_ids)));
            testCase.verifyTrue(any(cellfun(@(ids) ...
                ismember(smspId, ids), ...
                splits.test_mixed.active_type_ids)));
        end

        function balancedManifestRespectsEnabledTypesAndCardinality(testCase)
            % 验证随机 manifest 仅使用启用类型，并满足样本量和组合阶数。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 30, 1, 2, 'random');
            registry = build_jamming_registry(cfg);

            manifest = build_balanced_manifest(cfg, registry, 'train');

            expectedColumns = {'sample_id', 'split', 'sample_seed', ...
                'active_type_ids', 'active_type_codes', 'combination_id', ...
                'combination_size', 'component_target_power_dB', ...
                'weak_component_mask', 'component_sampling_weight', ...
                'pattern_sampling_weight', 'imbalance_flags', ...
                'unknown_present', 'unknown_component_count'};
            testCase.verifyEqual(manifest.Properties.VariableNames, ...
                expectedColumns);
            testCase.verifyEqual(height(manifest), 30);
            testCase.verifyEqual(manifest.sample_id, (1:30)');
            testCase.verifyTrue(all(manifest.combination_size >= 1));
            testCase.verifyTrue(all(manifest.combination_size <= 2));
            testCase.verifyTrue(all(cellfun(@(ids) ...
                all(ismember(ids, [1, 6, 8])), manifest.active_type_ids)));
            testCase.verifyTrue(all(cellfun(@(ids) ...
                numel(unique(ids)) == numel(ids), ...
                manifest.active_type_ids)));
            testCase.verifyTrue(all(cellfun(@(ids, count) ...
                numel(ids) == count, manifest.active_type_ids, ...
                num2cell(manifest.combination_size))));
            testCase.verifyEqual(numel(unique(manifest.sample_seed)), 30);
            testCase.verifyTrue(all(~manifest.unknown_present));
            testCase.verifyEqual(manifest.unknown_component_count, zeros(30, 1));
        end

        function randomManifestIsReproducible(testCase)
            % 验证同一配置和划分重复构建时，整张表逐项完全一致。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 25, 1, 2, 'random');
            registry = build_jamming_registry(cfg);

            firstManifest = build_balanced_manifest(cfg, registry, "train");
            secondManifest = build_balanced_manifest(cfg, registry, "train");

            testCase.verifyEqual(firstManifest, secondManifest);
        end

        function splitSeedsAreDisjoint(testCase)
            % 验证四个划分使用固定百万级偏移，并且样本种子互不相交。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 4, 1, 2, 'random');
            registry = build_jamming_registry(cfg);
            splitNames = {'train', 'calibration', 'validation', 'test'};
            offsets = [0, 1000000, 2000000, 3000000];
            manifests = cell(size(splitNames));

            for splitIndex = 1:numel(splitNames)
                manifests{splitIndex} = build_balanced_manifest( ...
                    cfg, registry, splitNames{splitIndex});
                expectedSeeds = cfg.seed + offsets(splitIndex) + (1:4)';
                testCase.verifyEqual( ...
                    manifests{splitIndex}.sample_seed, expectedSeeds);
            end

            for leftIndex = 1:numel(splitNames)
                for rightIndex = leftIndex + 1:numel(splitNames)
                    testCase.verifyEmpty(intersect( ...
                        manifests{leftIndex}.sample_seed, ...
                        manifests{rightIndex}.sample_seed));
                end
            end
        end

        function allModeEnumeratesEveryPattern(testCase)
            % 验证全枚举模式覆盖一阶和二阶全部组合，且每种模式重复两次。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 1, 1, 2, 'all');
            cfg.dataset.samples_per_pattern = 2;
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);

            manifest = build_balanced_manifest(cfg, registry, 'train');

            expectedIds = {'FMB'; 'MFT'; 'ISRJ'; 'FMB+MFT'; ...
                'FMB+ISRJ'; 'MFT+ISRJ'};
            testCase.verifyEqual(height(manifest), 12);
            testCase.verifyEqual(unique(manifest.combination_id, 'stable'), ...
                expectedIds);
            for patternIndex = 1:numel(expectedIds)
                testCase.verifyEqual(sum(strcmp( ...
                    manifest.combination_id, expectedIds{patternIndex})), 2);
            end
        end

        function unknownFlagsAreDerived(testCase)
            % 验证未知类存在标志和未知分量数量由活动类型代码直接派生。
            cfg = create_default_config();
            cfg.types.enabled = {'FMB', 'SinFM'};
            cfg.types.known = {'FMB'};
            cfg.types.unknown = {'SinFM'};
            cfg.dataset.num_samples = 1;
            cfg.dataset.min_components = 1;
            cfg.dataset.max_components = 2;
            cfg.dataset.combination_mode = 'all';
            cfg.dataset.samples_per_pattern = 1;
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);

            manifest = build_balanced_manifest(cfg, registry, 'test');

            testCase.verifyEqual(manifest.combination_id, ...
                {'FMB'; 'SinFM'; 'FMB+SinFM'});
            testCase.verifyEqual(manifest.unknown_present, ...
                logical([0; 1; 1]));
            testCase.verifyEqual(manifest.unknown_component_count, [0; 1; 1]);
        end

        function rejectsUnknownSplitAfterNormalization(testCase)
            % 验证即使配置已标准化，未知划分名称仍不能进入 manifest。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 4, 1, 2, 'random');
            registry = build_jamming_registry(cfg);

            testCase.verifyError(@() build_balanced_manifest( ...
                cfg, registry, 'development'), ...
                'JammingManifest:UnknownSplit');
        end

        function rejectsMissingStringSplit(testCase)
            % 验证缺失 string 标量在字符转换前使用统一划分错误拒绝。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 4, 1, 2, 'random');
            registry = build_jamming_registry(cfg);

            testCase.verifyError(@() build_balanced_manifest( ...
                cfg, registry, string(missing)), ...
                'JammingManifest:UnknownSplit');
        end

        function enabledTypesFollowRegistryOrder(testCase)
            % 验证启用类型的输入排列不会改变活动 ID 的注册表规范顺序。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'ISRJ', 'FMB', 'MFT'}, 1, 1, 1, 'all');
            cfg.dataset.samples_per_pattern = 1;
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);

            manifest = build_balanced_manifest(cfg, registry, 'train');

            testCase.verifyEqual(manifest.active_type_ids, ...
                {1; 6; 8});
            testCase.verifyEqual(manifest.combination_id, ...
                {'FMB'; 'MFT'; 'ISRJ'});
        end

        function metadataColumnsUseDocumentedDefaults(testCase)
            % 验证普通组合 manifest 的列类型、单位占位值和默认权重。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT'}, 1, 2, 2, 'all');
            cfg.dataset.samples_per_pattern = 1;
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);

            manifest = build_balanced_manifest(cfg, registry, 'calibration');

            testCase.verifyClass(manifest.sample_id, 'double');
            testCase.verifyClass(manifest.split, 'cell');
            testCase.verifyTrue(iscellstr(manifest.split));
            testCase.verifyEqual(manifest.split, {'calibration'});
            testCase.verifyClass(manifest.sample_seed, 'double');
            testCase.verifyEqual(manifest.active_type_ids, {[1, 6]});
            testCase.verifyEqual(manifest.active_type_codes, ...
                {{'FMB', 'MFT'}});
            testCase.verifyEqual(manifest.component_target_power_dB, ...
                {[0, 0]});
            testCase.verifyEqual(manifest.weak_component_mask, ...
                {logical([0, 0])});
            testCase.verifyEqual(manifest.component_sampling_weight, 1);
            testCase.verifyEqual(manifest.pattern_sampling_weight, 1);
            testCase.verifyEqual(manifest.imbalance_flags, ...
                {logical([0, 0, 0])});
            testCase.verifyClass(manifest.unknown_present, 'logical');
            testCase.verifyClass(manifest.unknown_component_count, 'double');
        end

        function rejectsEnabledTypeMissingFromRegistry(testCase)
            % 验证构建器独立检查 enabled 到注册表的映射完整性。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT'}, 2, 1, 2, 'random');
            cfg.types.enabled = {'FMB', 'NotARegisteredType'};
            registry = build_jamming_registry(cfg);

            testCase.verifyError(@() build_balanced_manifest( ...
                cfg, registry, 'train'), ...
                'JammingManifest:UnknownEnabledType');
        end

        function rejectsBaseSeedAboveUint32Domain(testCase)
            % 验证基础种子超过 uint32 最大值时归类为非法输入域。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT'}, 1, 1, 1, 'random');
            cfg.seed = 2^32;
            registry = build_jamming_registry(cfg);

            testCase.verifyError(@() build_balanced_manifest( ...
                cfg, registry, 'train'), ...
                'JammingManifest:InvalidSeed');
        end

        function independentStreamDoesNotChangeGlobalRng(testCase)
            % 验证随机 manifest 使用独立 RandStream，不推进全局随机状态。
            originalState = rng;
            testCase.addTeardown(@() rng(originalState));
            rng(13579, 'twister');
            stateBefore = rng;
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 8, 1, 2, 'random');
            registry = build_jamming_registry(cfg);

            build_balanced_manifest(cfg, registry, 'validation');

            testCase.verifyEqual(rng, stateBefore);
        end

        function randomModeRejectsInvalidSeedDomain(testCase)
            % 验证 random 模式仅接受 uint32 取值域内的有限非负整数种子。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT'}, 2, 1, 2, 'random');
            registry = build_jamming_registry(cfg);
            invalidSeeds = {'7', -1, 1.5, NaN, Inf, complex(1, 1), 2^32};

            for seedIndex = 1:numel(invalidSeeds)
                invalidCfg = cfg;
                invalidCfg.seed = invalidSeeds{seedIndex};
                testCase.verifyError(@() build_balanced_manifest( ...
                    invalidCfg, registry, 'train'), ...
                    'JammingManifest:InvalidSeed');
            end
        end

        function allModeRejectsInvalidSeedDomain(testCase)
            % 验证 all 模式与 random 模式使用完全相同的基础种子输入域。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT'}, 1, 1, 2, 'all');
            cfg.dataset.samples_per_pattern = 1;
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            invalidSeeds = {-1, 0.25, NaN, 2^32};

            for seedIndex = 1:numel(invalidSeeds)
                invalidCfg = cfg;
                invalidCfg.seed = invalidSeeds{seedIndex};
                testCase.verifyError(@() build_balanced_manifest( ...
                    invalidCfg, registry, 'train'), ...
                    'JammingManifest:InvalidSeed');
            end
        end

        function randomModeHonorsUint32SampleSeedBoundary(testCase)
            % 验证 random 模式最后一个样本种子可等于 uint32 最大值但不能越界。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT'}, 3, 1, 2, 'random');
            registry = build_jamming_registry(cfg);
            maximumSeed = double(intmax('uint32'));
            cfg.seed = maximumSeed - 3000000 - cfg.dataset.num_samples;

            manifest = build_balanced_manifest(cfg, registry, 'test');

            testCase.verifyEqual(manifest.sample_seed(end), maximumSeed);
            cfg.seed = cfg.seed + 1;
            testCase.verifyError(@() build_balanced_manifest( ...
                cfg, registry, 'test'), 'JammingManifest:SeedOverflow');
        end

        function allModeHonorsUint32SampleSeedBoundary(testCase)
            % 验证 all 模式按真实行数检查 uint32 样本种子上界。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT'}, 1, 1, 2, 'all');
            cfg.dataset.samples_per_pattern = 1;
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);
            maximumSeed = double(intmax('uint32'));
            actualSampleCount = 3;
            cfg.seed = maximumSeed - 1000000 - actualSampleCount;

            manifest = build_balanced_manifest(cfg, registry, 'calibration');

            testCase.verifyEqual(manifest.sample_seed(end), maximumSeed);
            cfg.seed = cfg.seed + 1;
            testCase.verifyError(@() build_balanced_manifest( ...
                cfg, registry, 'calibration'), ...
                'JammingManifest:SeedOverflow');
        end

        function randomModeRejectsManifestAboveSplitCapacity(testCase)
            % 验证 random 模式在分配百万级单元数组前拒绝过大划分。
            cfg = TestManifestBuilders.makeLargeCountConfig( ...
                'random', 1000001, 1);
            registry = build_jamming_registry(cfg);

            testCase.verifyError(@() build_balanced_manifest( ...
                cfg, registry, 'train'), ...
                'JammingManifest:SplitTooLarge');
        end

        function allModeRejectsManifestAboveSplitCapacity(testCase)
            % 验证 all 模式按实际枚举行数而非 num_samples 检查划分上限。
            cfg = TestManifestBuilders.makeLargeCountConfig( ...
                'all', 1, 1000001);
            registry = build_jamming_registry(cfg);

            testCase.verifyError(@() build_balanced_manifest( ...
                cfg, registry, 'train'), ...
                'JammingManifest:SplitTooLarge');
        end

        function allModeCapacityUsesActualRowCount(testCase)
            % 验证 all 模式按十二行真实输出重新估算波形容量。
            cfg = create_default_config();
            cfg.types.enabled = {'FMB', 'MFT', 'ISRJ'};
            cfg.types.known = cfg.types.enabled;
            cfg.types.unknown = {};
            cfg.dataset.num_samples = 1;
            cfg.dataset.min_components = 1;
            cfg.dataset.max_components = 2;
            cfg.dataset.combination_mode = 'all';
            cfg.dataset.samples_per_pattern = 2;
            cfg.output.max_estimated_bytes = 1000000;
            cfg = normalize_and_validate_config(cfg, 'balanced');
            registry = build_jamming_registry(cfg);

            testCase.verifyError(@() build_balanced_manifest( ...
                cfg, registry, 'train'), ...
                'JammingConfig:EstimatedSizeExceeded');
        end

        function combinationModeAcceptsCaseInsensitiveText(testCase)
            % 验证 combination_mode 的合法文本统一转换为小写字符模式。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT'}, 3, 1, 2, 'random');
            registry = build_jamming_registry(cfg);
            cfg.dataset.combination_mode = "RANDOM";

            randomManifest = build_balanced_manifest( ...
                cfg, registry, 'train');

            testCase.verifyEqual(height(randomManifest), 3);
            cfg.dataset.combination_mode = 'ALL';
            cfg.dataset.samples_per_pattern = 1;
            allManifest = build_balanced_manifest(cfg, registry, 'train');
            testCase.verifyEqual(height(allManifest), 3);
        end

        function combinationModeRejectsMalformedValues(testCase)
            % 验证空文本、缺失文本、文本数组、cell 和未知模式使用统一错误。
            cfg = TestManifestBuilders.makeBalancedConfig( ...
                {'FMB', 'MFT'}, 3, 1, 2, 'random');
            registry = build_jamming_registry(cfg);
            invalidModes = {'', {'random'}, ["random", "all"], ...
                string(missing), 'weighted'};

            for modeIndex = 1:numel(invalidModes)
                invalidCfg = cfg;
                invalidCfg.dataset.combination_mode = ...
                    invalidModes{modeIndex};
                testCase.verifyError(@() build_balanced_manifest( ...
                    invalidCfg, registry, 'train'), ...
                    'JammingManifest:UnknownCombinationMode');
            end
        end

        function jointImbalanceControlsMarginalsPatternsAndPower(testCase)
            % 验证三个不均衡轴同时控制分量边际、组合阶数和弱分量功率。
            cfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 600, 1, 3, 'random');
            cfg.imbalance.component.enable = true;
            cfg.imbalance.component.weights = [10, 3, 1];
            cfg.imbalance.combination.enable = true;
            cfg.imbalance.combination.cardinality_weights = [10, 4, 1];
            cfg.imbalance.power.enable = true;
            cfg.imbalance.power.weak_component_policy = 'rotate';
            cfg.imbalance.power.relative_power_dB = [-18, -12];
            cfg = normalize_and_validate_config(cfg, 'imbalanced');
            registry = build_jamming_registry(cfg);

            manifest = build_imbalanced_manifest(cfg, registry, 'train');

            allCodes = [manifest.active_type_codes{:}];
            fmbCount = sum(strcmp(allCodes, 'FMB'));
            mftCount = sum(strcmp(allCodes, 'MFT'));
            isrjCount = sum(strcmp(allCodes, 'ISRJ'));
            testCase.verifyGreaterThan(fmbCount, mftCount);
            testCase.verifyGreaterThan(mftCount, isrjCount);
            testCase.verifyGreaterThan( ...
                sum(manifest.combination_size == 1), ...
                sum(manifest.combination_size == 3));
            testCase.verifyTrue(all(cellfun( ...
                @(mask) any(mask), manifest.weak_component_mask)));
            testCase.verifyTrue(all(cellfun( ...
                @(powerDb) min(powerDb) <= -12, ...
                manifest.component_target_power_dB)));
            testCase.verifyEqual(manifest.imbalance_flags, ...
                repmat({logical([1, 1, 1])}, 600, 1));
        end

        function imbalancedManifestIsReproducibleWithoutChangingGlobalRng( ...
                testCase)
            % 验证同配置同划分逐项可复现，且独立流不推进全局随机状态。
            originalState = rng;
            testCase.addTeardown(@() rng(originalState));
            rng(24680, 'twister');
            stateBefore = rng;
            cfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 80, 1, 3, 'random');
            cfg.imbalance.component.enable = true;
            cfg.imbalance.component.weights = [8, 2, 1];
            cfg.imbalance.combination.enable = true;
            cfg.imbalance.combination.cardinality_weights = [6, 3, 1];
            cfg.imbalance.power.enable = true;
            cfg.imbalance.power.weak_component_policy = 'random';
            cfg = normalize_and_validate_config(cfg, 'imbalanced');
            registry = build_jamming_registry(cfg);

            firstManifest = build_imbalanced_manifest( ...
                cfg, registry, 'validation');
            secondManifest = build_imbalanced_manifest( ...
                cfg, registry, 'validation');

            testCase.verifyEqual(firstManifest, secondManifest);
            testCase.verifyEqual(rng, stateBefore);
        end

        function imbalanceAxesOperateIndependently(testCase)
            % 验证每个轴只改变自己的字段，三个轴均关闭时直接退化为普通清单。
            componentCfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 80, 1, 2, 'random');
            componentCfg.imbalance.component.enable = true;
            componentCfg.imbalance.component.weights = [9, 2, 1];
            componentCfg = normalize_and_validate_config( ...
                componentCfg, 'imbalanced');
            registry = build_jamming_registry(componentCfg);
            componentManifest = build_imbalanced_manifest( ...
                componentCfg, registry, 'train');
            testCase.verifyEqual(componentManifest.imbalance_flags, ...
                repmat({logical([1, 0, 0])}, 80, 1));
            testCase.verifyTrue(any( ...
                componentManifest.component_sampling_weight < 1));
            testCase.verifyEqual( ...
                componentManifest.pattern_sampling_weight, ones(80, 1));
            TestManifestBuilders.verifyPowerDefaults( ...
                testCase, componentManifest);

            combinationCfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 80, 1, 2, 'random');
            combinationCfg.imbalance.combination.enable = true;
            combinationCfg.imbalance.combination.cardinality_weights = ...
                [9, 1];
            combinationCfg = normalize_and_validate_config( ...
                combinationCfg, 'imbalanced');
            registry = build_jamming_registry(combinationCfg);
            combinationManifest = build_imbalanced_manifest( ...
                combinationCfg, registry, 'train');
            testCase.verifyEqual(combinationManifest.imbalance_flags, ...
                repmat({logical([0, 1, 0])}, 80, 1));
            testCase.verifyEqual( ...
                combinationManifest.component_sampling_weight, ones(80, 1));
            probabilityDistance = min(abs( ...
                combinationManifest.pattern_sampling_weight - ...
                [0.9, 0.1]), [], 2);
            testCase.verifyLessThanOrEqual( ...
                probabilityDistance, 10 * eps);
            TestManifestBuilders.verifyPowerDefaults( ...
                testCase, combinationManifest);

            powerCfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 80, 1, 2, 'random');
            powerCfg.imbalance.power.enable = true;
            powerCfg.imbalance.power.weak_component_policy = 'rotate';
            powerCfg = normalize_and_validate_config(powerCfg, 'imbalanced');
            registry = build_jamming_registry(powerCfg);
            powerManifest = build_imbalanced_manifest( ...
                powerCfg, registry, 'train');
            testCase.verifyEqual(powerManifest.imbalance_flags, ...
                repmat({logical([0, 0, 1])}, 80, 1));
            testCase.verifyEqual( ...
                powerManifest.component_sampling_weight, ones(80, 1));
            testCase.verifyEqual( ...
                powerManifest.pattern_sampling_weight, ones(80, 1));

            disabledCfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 30, 1, 2, 'random');
            disabledCfg = normalize_and_validate_config( ...
                disabledCfg, 'imbalanced');
            registry = build_jamming_registry(disabledCfg);
            testCase.verifyEqual(build_imbalanced_manifest( ...
                disabledCfg, registry, 'test'), ...
                build_balanced_manifest(disabledCfg, registry, 'test'));
        end

        function powerOnlyAllModePreservesBalancedPatterns(testCase)
            % 验证仅启用功率轴时保留 all 模式的六个普通组合及抽样元数据。
            cfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 1, 1, 2, 'all');
            cfg.dataset.samples_per_pattern = 1;
            cfg.imbalance.power.enable = true;
            cfg.imbalance.power.weak_component_policy = 'rotate';
            cfg = normalize_and_validate_config(cfg, 'imbalanced');
            registry = build_jamming_registry(cfg);
            balancedManifest = build_balanced_manifest( ...
                cfg, registry, 'train');

            powerManifest = build_imbalanced_manifest( ...
                cfg, registry, 'train');

            preservedColumns = {'sample_id', 'split', 'sample_seed', ...
                'active_type_ids', 'active_type_codes', 'combination_id', ...
                'combination_size', 'component_sampling_weight', ...
                'pattern_sampling_weight', 'unknown_present', ...
                'unknown_component_count'};
            testCase.verifyEqual(powerManifest(:, preservedColumns), ...
                balancedManifest(:, preservedColumns));
            expectedPatterns = {'FMB'; 'MFT'; 'ISRJ'; 'FMB+MFT'; ...
                'FMB+ISRJ'; 'MFT+ISRJ'};
            testCase.verifyEqual(powerManifest.combination_id, ...
                expectedPatterns);
            testCase.verifyEqual(cellfun(@(pattern) sum(strcmp( ...
                powerManifest.combination_id, pattern)), ...
                expectedPatterns), ones(6, 1));
        end

        function powerPolicyDoesNotChangeResampledPatterns(testCase)
            % 验证组合重抽流与功率流隔离，切换弱策略不改变组合相关列。
            cfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 120, 1, 3, 'random');
            cfg.imbalance.component.enable = true;
            cfg.imbalance.component.weights = [5, 2, 1];
            cfg.imbalance.combination.enable = true;
            cfg.imbalance.combination.cardinality_weights = [5, 2, 1];
            cfg.imbalance.power.enable = true;
            cfg.imbalance.power.weak_component_policy = 'rotate';
            cfg = normalize_and_validate_config(cfg, 'imbalanced');
            registry = build_jamming_registry(cfg);

            rotateManifest = build_imbalanced_manifest( ...
                cfg, registry, 'validation');
            cfg.imbalance.power.weak_component_policy = 'random';
            randomManifest = build_imbalanced_manifest( ...
                cfg, registry, 'validation');

            excludedColumns = {'component_target_power_dB', ...
                'weak_component_mask'};
            preservedColumns = setdiff( ...
                rotateManifest.Properties.VariableNames, ...
                excludedColumns, 'stable');
            testCase.verifyEqual(rotateManifest(:, preservedColumns), ...
                randomManifest(:, preservedColumns));
        end

        function componentSamplingRejectsTooFewPositiveCandidates(testCase)
            % 验证两个分量的可达阶数不能由唯一正权重候选完成。
            cfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 40, 2, 2, 'random');
            cfg.imbalance.component.enable = true;
            cfg.imbalance.component.weights = [1, 0, 0];
            cfg = normalize_and_validate_config(cfg, 'imbalanced');
            registry = build_jamming_registry(cfg);

            testCase.verifyError(@() build_imbalanced_manifest( ...
                cfg, registry, 'train'), ...
                'JammingManifest:InvalidSamplingWeights');
        end

        function componentSamplingAcceptsEnoughPositiveCandidates(testCase)
            % 验证两个正权重候选可稳定生成二分量组合并排除零权重类型。
            cfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 40, 2, 2, 'random');
            cfg.imbalance.component.enable = true;
            cfg.imbalance.component.weights = [1, 1, 0];
            cfg = normalize_and_validate_config(cfg, 'imbalanced');
            registry = build_jamming_registry(cfg);

            manifest = build_imbalanced_manifest(cfg, registry, 'train');

            testCase.verifyEqual(manifest.combination_id, ...
                repmat({'FMB+MFT'}, 40, 1));
            testCase.verifyEqual(manifest.active_type_ids, ...
                repmat({[1, 6]}, 40, 1));
        end

        function manualPatternWeightsOverrideOtherSampling(testCase)
            % 验证显式模式表覆盖阶数与分量抽样，并记录规范化模式概率。
            cfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 600, 1, 3, 'random');
            cfg.imbalance.component.enable = true;
            cfg.imbalance.component.weights = [100, 1, 1];
            cfg.imbalance.combination.enable = true;
            cfg.imbalance.combination.cardinality_weights = [1, 1, 100];
            cfg.imbalance.combination.pattern_weights = table( ...
                {'FMB+MFT'; 'MFT+ISRJ'}, [9; 1], ...
                'VariableNames', {'pattern_codes', 'weight'});
            cfg = normalize_and_validate_config(cfg, 'imbalanced');
            registry = build_jamming_registry(cfg);

            manifest = build_imbalanced_manifest(cfg, registry, 'train');

            firstCount = sum(strcmp(manifest.combination_id, 'FMB+MFT'));
            secondCount = sum(strcmp(manifest.combination_id, 'MFT+ISRJ'));
            testCase.verifyTrue(all(ismember(manifest.combination_id, ...
                {'FMB+MFT', 'MFT+ISRJ'})));
            testCase.verifyGreaterThan(firstCount, 4 * secondCount);
            expectedPatternWeights = 0.9 * strcmp( ...
                manifest.combination_id, 'FMB+MFT') + 0.1 * strcmp( ...
                manifest.combination_id, 'MFT+ISRJ');
            testCase.verifyEqual(manifest.pattern_sampling_weight, ...
                expectedPatternWeights, 'AbsTol', eps);
            testCase.verifyEqual(manifest.component_sampling_weight, ...
                ones(600, 1));
        end

        function weakComponentPoliciesFollowDocumentedRules(testCase)
            % 验证 random、rotate、fixed 三种弱分量策略及非法策略错误。
            randomCfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 100, 1, 3, 'random');
            randomCfg.imbalance.power.enable = true;
            randomCfg.imbalance.power.weak_component_policy = 'random';
            randomCfg = normalize_and_validate_config( ...
                randomCfg, 'imbalanced');
            registry = build_jamming_registry(randomCfg);
            randomManifest = build_imbalanced_manifest( ...
                randomCfg, registry, 'train');
            testCase.verifyEqual(cellfun(@nnz, ...
                randomManifest.weak_component_mask), ones(100, 1));

            rotateCfg = randomCfg;
            rotateCfg.imbalance.power.weak_component_policy = 'rotate';
            rotateManifest = build_imbalanced_manifest( ...
                rotateCfg, registry, 'train');
            for rowIndex = 1:height(rotateManifest)
                componentCount = rotateManifest.combination_size(rowIndex);
                expectedMask = false(1, componentCount);
                expectedIndex = mod( ...
                    rotateManifest.sample_id(rowIndex) - 1, ...
                    componentCount) + 1;
                expectedMask(expectedIndex) = true;
                testCase.verifyEqual( ...
                    rotateManifest.weak_component_mask{rowIndex}, ...
                    expectedMask);
            end

            fixedCfg = randomCfg;
            fixedCfg.dataset.num_samples = 200;
            fixedCfg.imbalance.power.weak_component_policy = 'fixed';
            fixedCfg.imbalance.power.fixed_component = 'ISRJ';
            fixedManifest = build_imbalanced_manifest( ...
                fixedCfg, registry, 'train');
            containsFixed = false(height(fixedManifest), 1);
            for rowIndex = 1:height(fixedManifest)
                codes = fixedManifest.active_type_codes{rowIndex};
                expectedMask = strcmp(codes, 'ISRJ');
                containsFixed(rowIndex) = any(expectedMask);
                testCase.verifyEqual( ...
                    fixedManifest.weak_component_mask{rowIndex}, ...
                    expectedMask);
            end
            testCase.verifyTrue(any(containsFixed));
            testCase.verifyTrue(any(~containsFixed));

            invalidPolicyCfg = randomCfg;
            invalidPolicyCfg.imbalance.power.weak_component_policy = ...
                'unknown-policy';
            testCase.verifyError(@() build_imbalanced_manifest( ...
                invalidPolicyCfg, registry, 'train'), ...
                'JammingManifest:InvalidWeakComponentPolicy');
            invalidFixedCfg = randomCfg;
            invalidFixedCfg.imbalance.power.weak_component_policy = 'fixed';
            invalidFixedCfg.imbalance.power.fixed_component = 'AMA';
            testCase.verifyError(@() build_imbalanced_manifest( ...
                invalidFixedCfg, registry, 'train'), ...
                'JammingManifest:InvalidWeakComponentPolicy');
        end

        function componentWeightsFollowEnabledInputOrder(testCase)
            % 验证分量权重按 enabled 输入代码绑定，输出仍按注册表 ID 排序。
            cfg = TestManifestBuilders.makeManifestConfig( ...
                {'ISRJ', 'FMB', 'MFT'}, 600, 1, 3, 'random');
            cfg.imbalance.component.enable = true;
            cfg.imbalance.component.weights = [12, 1, 1];
            cfg = normalize_and_validate_config(cfg, 'imbalanced');
            registry = build_jamming_registry(cfg);

            manifest = build_imbalanced_manifest(cfg, registry, 'train');

            allCodes = [manifest.active_type_codes{:}];
            testCase.verifyGreaterThan(sum(strcmp(allCodes, 'ISRJ')), ...
                sum(strcmp(allCodes, 'FMB')));
            testCase.verifyGreaterThan(sum(strcmp(allCodes, 'ISRJ')), ...
                sum(strcmp(allCodes, 'MFT')));
            registryIds = double([registry.id]);
            registryCodes = {registry.code};
            for rowIndex = 1:height(manifest)
                ids = manifest.active_type_ids{rowIndex};
                codes = manifest.active_type_codes{rowIndex};
                testCase.verifyEqual(ids, sort(ids));
                [found, locations] = ismember(ids, registryIds);
                testCase.verifyTrue(all(found));
                testCase.verifyEqual(codes, registryCodes(locations));
            end
        end

        function targetPowersOnlyAttenuateWeakComponents(testCase)
            % 验证弱分量功率落在 dB 范围内，非弱分量和关闭轴时严格为零。
            cfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 120, 1, 3, 'random');
            cfg.imbalance.power.enable = true;
            cfg.imbalance.power.weak_component_policy = 'rotate';
            cfg.imbalance.power.relative_power_dB = [-17, -5];
            cfg = normalize_and_validate_config(cfg, 'imbalanced');
            registry = build_jamming_registry(cfg);

            manifest = build_imbalanced_manifest(cfg, registry, 'train');

            for rowIndex = 1:height(manifest)
                powers = manifest.component_target_power_dB{rowIndex};
                weakMask = manifest.weak_component_mask{rowIndex};
                testCase.verifyTrue(all(powers(~weakMask) == 0));
                testCase.verifyGreaterThanOrEqual( ...
                    powers(weakMask), -17);
                testCase.verifyLessThanOrEqual(powers(weakMask), -5);
            end

            cfg.imbalance.power.enable = false;
            disabledManifest = build_imbalanced_manifest( ...
                cfg, registry, 'train');
            TestManifestBuilders.verifyPowerDefaults( ...
                testCase, disabledManifest);
        end

        function imbalancedManifestInheritsBalancedSafetyChecks(testCase)
            % 验证划分、种子、真实容量和百万行限制先由普通骨架安全拒绝。
            cfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT'}, 4, 1, 2, 'random');
            cfg.imbalance.component.enable = true;
            cfg.imbalance.component.weights = [2, 1];
            cfg = normalize_and_validate_config(cfg, 'imbalanced');
            registry = build_jamming_registry(cfg);
            testCase.verifyError(@() build_imbalanced_manifest( ...
                cfg, registry, 'development'), ...
                'JammingManifest:UnknownSplit');
            cfg.seed = 2^32;
            testCase.verifyError(@() build_imbalanced_manifest( ...
                cfg, registry, 'train'), 'JammingManifest:InvalidSeed');

            largeCfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB'}, 1000001, 1, 1, 'random');
            largeCfg.output.max_estimated_bytes = flintmax;
            largeCfg.imbalance.component.enable = true;
            largeCfg.imbalance.component.weights = 1;
            largeCfg = normalize_and_validate_config( ...
                largeCfg, 'imbalanced');
            registry = build_jamming_registry(largeCfg);
            testCase.verifyError(@() build_imbalanced_manifest( ...
                largeCfg, registry, 'train'), ...
                'JammingManifest:SplitTooLarge');

            capacityCfg = TestManifestBuilders.makeManifestConfig( ...
                {'FMB', 'MFT', 'ISRJ'}, 1, 1, 2, 'all');
            capacityCfg.dataset.samples_per_pattern = 2;
            capacityCfg.output.max_estimated_bytes = 1000000;
            capacityCfg.imbalance.power.enable = true;
            capacityCfg = normalize_and_validate_config( ...
                capacityCfg, 'imbalanced');
            registry = build_jamming_registry(capacityCfg);
            testCase.verifyError(@() build_imbalanced_manifest( ...
                capacityCfg, registry, 'train'), ...
                'JammingConfig:EstimatedSizeExceeded');
        end

        function openSetBuilderCreatesFourProtocolSplits(testCase)
            % 验证开放集协议按固定顺序返回已知训练/校准及混合验证/测试。
            cfg = TestManifestBuilders.makeOpenSetConfig(20, 8, 16, 20);
            registry = build_jamming_registry(cfg);

            splits = build_openset_manifest(cfg, registry);

            expectedFields = {'train_known'; 'calibration_known'; ...
                'validation_mixed'; 'test_mixed'};
            testCase.verifyEqual(fieldnames(splits), expectedFields);
            testCase.verifyEqual(height(splits.train_known), 20);
            testCase.verifyEqual(height(splits.calibration_known), 8);
            testCase.verifyEqual(height(splits.validation_mixed), 16);
            testCase.verifyEqual(height(splits.test_mixed), 20);
            testCase.verifyTrue(all(~splits.train_known.unknown_present));
            testCase.verifyTrue(all( ...
                splits.train_known.unknown_component_count == 0));
            testCase.verifyTrue(all( ...
                ~splits.calibration_known.unknown_present));
            testCase.verifyTrue(any( ...
                splits.validation_mixed.unknown_present));
            testCase.verifyTrue(any(splits.test_mixed.unknown_present));
            testCase.verifyTrue(any( ...
                splits.test_mixed.unknown_component_count > 1));
        end

        function mixedRatiosUseLargestRemainder(testCase)
            % 验证二十行测试集按最大余数法精确分配四类开放集组成。
            cfg = TestManifestBuilders.makeOpenSetConfig(2, 2, 4, 20);
            registry = build_jamming_registry(cfg);

            splits = build_openset_manifest(cfg, registry);
            category = TestManifestBuilders.classifyOpenSetRows( ...
                splits.test_mixed, cfg.types.known, cfg.types.unknown);

            actualCounts = arrayfun(@(value) sum(category == value), 1:4);
            testCase.verifyEqual(actualCounts, [5, 5, 7, 3]);
            testCase.verifyTrue(all(category >= 1 & category <= 4));
        end

        function splitSeedsDisjointAndUseOffsets(testCase)
            % 验证四表继承普通构建器偏移、确定性及全局随机状态隔离。
            cfg = TestManifestBuilders.makeOpenSetConfig(20, 8, 16, 20);
            registry = build_jamming_registry(cfg);
            originalState = rng;
            cleanup = onCleanup(@() rng(originalState));
            rng(13579, 'twister');
            stateBefore = rng;

            first = build_openset_manifest(cfg, registry);
            stateAfter = rng;
            second = build_openset_manifest(cfg, registry);

            testCase.verifyEqual(stateAfter, stateBefore);
            testCase.verifyEqual(rng, stateBefore);
            testCase.verifyEqual(first, second);
            splitNames = fieldnames(first);
            offsets = [0, 1000000, 2000000, 3000000];
            counts = [20, 8, 16, 20];
            for splitIndex = 1:numel(splitNames)
                manifest = first.(splitNames{splitIndex});
                expectedSeeds = cfg.seed + offsets(splitIndex) + ...
                    (1:counts(splitIndex))';
                testCase.verifyEqual(manifest.sample_seed, expectedSeeds);
            end
            allSeeds = cellfun(@(name) first.(name).sample_seed, ...
                splitNames, 'UniformOutput', false);
            testCase.verifyEqual(numel(unique(vertcat(allSeeds{:}))), ...
                sum(counts));
        end

        function labelsFollowUserPartition(testCase)
            % 验证未知真值完全依据用户分区派生，而不绑定固定干扰代码。
            cfg = TestManifestBuilders.makeOpenSetConfig(4, 4, 40, 40);
            cfg.types.known = {'FMB', 'SinFM', 'VFT'};
            cfg.types.unknown = {'MFT', 'ISRJ'};
            cfg = normalize_and_validate_config(cfg, 'openset');
            registry = build_jamming_registry(cfg);

            splits = build_openset_manifest(cfg, registry);
            mixedNames = {'validation_mixed', 'test_mixed'};
            for splitIndex = 1:numel(mixedNames)
                manifest = splits.(mixedNames{splitIndex});
                expectedCounts = cellfun(@(codes) sum(ismember( ...
                    codes, cfg.types.unknown)), manifest.active_type_codes);
                testCase.verifyEqual(manifest.unknown_component_count, ...
                    expectedCounts);
                testCase.verifyEqual(manifest.unknown_present, ...
                    expectedCounts > 0);
                category = TestManifestBuilders.classifyOpenSetRows( ...
                    manifest, cfg.types.known, cfg.types.unknown);
                testCase.verifyTrue(all(category >= 1 & category <= 4));
            end
        end

        function openSetTablesPreserveBalancedContract(testCase)
            % 验证四表保持普通 manifest 的列名、类型和注册表规范排列。
            cfg = TestManifestBuilders.makeOpenSetConfig(6, 4, 12, 12);
            registry = build_jamming_registry(cfg);
            referenceCfg = cfg;
            referenceCfg.dataset.num_samples = 1;
            referenceCfg.dataset.combination_mode = 'random';
            reference = build_balanced_manifest( ...
                referenceCfg, registry, 'train');

            splits = build_openset_manifest(cfg, registry);
            splitNames = fieldnames(splits);
            registryIds = double([registry.id]);
            registryCodes = {registry.code};
            for splitIndex = 1:numel(splitNames)
                manifest = splits.(splitNames{splitIndex});
                testCase.verifyEqual(manifest.Properties.VariableNames, ...
                    reference.Properties.VariableNames);
                for columnIndex = 1:width(reference)
                    columnName = reference.Properties.VariableNames{ ...
                        columnIndex};
                    testCase.verifyClass(manifest.(columnName), ...
                        class(reference.(columnName)));
                end
                for rowIndex = 1:height(manifest)
                    ids = manifest.active_type_ids{rowIndex};
                    codes = manifest.active_type_codes{rowIndex};
                    testCase.verifyEqual(ids, sort(ids));
                    testCase.verifyEqual(numel(unique(ids)), numel(ids));
                    testCase.verifyTrue(all(ismember( ...
                        codes, cfg.types.enabled)));
                    [found, locations] = ismember(ids, registryIds);
                    testCase.verifyTrue(all(found));
                    testCase.verifyEqual(codes, registryCodes(locations));
                    testCase.verifyEqual( ...
                        manifest.combination_id{rowIndex}, ...
                        strjoin(codes, '+'));
                end
            end
        end

        function impossibleOpenSetCompositionsAreRejected(testCase)
            % 验证非零类别在分区或最大分量数不足时于骨架分配前被拒绝。
            pureUnknownCfg = TestManifestBuilders.makeOpenSetConfig( ...
                2, 2, 1, 1);
            pureUnknownCfg.openset.mixed_ratios = [0, 1, 0, 0];
            pureUnknownCfg = normalize_and_validate_config( ...
                pureUnknownCfg, 'openset');
            pureUnknownCfg.types.enabled = {'FMB'};
            pureUnknownCfg.types.known = {'FMB'};
            pureUnknownCfg.types.unknown = {};
            registry = build_jamming_registry(pureUnknownCfg);
            testCase.verifyError(@() build_openset_manifest( ...
                pureUnknownCfg, registry), ...
                'JammingOpenSet:InvalidComposition');

            oneUnknownCfg = TestManifestBuilders.makeOpenSetConfig( ...
                2, 2, 1, 1);
            oneUnknownCfg.openset.mixed_ratios = [0, 0, 1, 0];
            oneUnknownCfg.dataset.max_components = 1;
            oneUnknownCfg = normalize_and_validate_config( ...
                oneUnknownCfg, 'openset');
            oneUnknownCfg.openset.validation_mixed = 1000000;
            registry = build_jamming_registry(oneUnknownCfg);
            testCase.verifyError(@() build_openset_manifest( ...
                oneUnknownCfg, registry), ...
                'JammingOpenSet:InvalidComposition');

            multipleUnknownCfg = TestManifestBuilders.makeOpenSetConfig( ...
                2, 2, 1, 1);
            multipleUnknownCfg.types.known = {'FMB', 'MFT', 'ISRJ', 'SinFM'};
            multipleUnknownCfg.types.unknown = {'VFT'};
            multipleUnknownCfg.openset.mixed_ratios = [0, 0, 0, 1];
            multipleUnknownCfg = normalize_and_validate_config( ...
                multipleUnknownCfg, 'openset');
            registry = build_jamming_registry(multipleUnknownCfg);
            testCase.verifyError(@() build_openset_manifest( ...
                multipleUnknownCfg, registry), ...
                'JammingOpenSet:InvalidComposition');

            lowMaximumCfg = TestManifestBuilders.makeOpenSetConfig( ...
                2, 2, 1, 1);
            lowMaximumCfg.openset.mixed_ratios = [0, 0, 0, 1];
            lowMaximumCfg.dataset.max_components = 2;
            lowMaximumCfg = normalize_and_validate_config( ...
                lowMaximumCfg, 'openset');
            registry = build_jamming_registry(lowMaximumCfg);
            testCase.verifyError(@() build_openset_manifest( ...
                lowMaximumCfg, registry), ...
                'JammingOpenSet:InvalidComposition');

            noKnownCfg = TestManifestBuilders.makeOpenSetConfig( ...
                2, 2, 1, 1);
            noKnownCfg.openset.mixed_ratios = [0, 0, 1, 0];
            noKnownCfg = normalize_and_validate_config( ...
                noKnownCfg, 'openset');
            noKnownCfg.types.enabled = {'SinFM', 'VFT'};
            noKnownCfg.types.known = {};
            noKnownCfg.types.unknown = {'SinFM', 'VFT'};
            noKnownCfg.openset.train_known = 0;
            noKnownCfg.openset.calibration_known = 0;
            registry = build_jamming_registry(noKnownCfg);
            testCase.verifyError(@() build_openset_manifest( ...
                noKnownCfg, registry), ...
                'JammingOpenSet:InvalidComposition');

            noUnknownCfg = TestManifestBuilders.makeOpenSetConfig( ...
                2, 2, 1, 1);
            noUnknownCfg.openset.mixed_ratios = [0, 0, 1, 0];
            noUnknownCfg = normalize_and_validate_config( ...
                noUnknownCfg, 'openset');
            noUnknownCfg.types.enabled = {'FMB'};
            noUnknownCfg.types.known = {'FMB'};
            noUnknownCfg.types.unknown = {};
            noUnknownCfg.openset.train_known = 0;
            noUnknownCfg.openset.calibration_known = 0;
            registry = build_jamming_registry(noUnknownCfg);
            testCase.verifyError(@() build_openset_manifest( ...
                noUnknownCfg, registry), ...
                'JammingOpenSet:InvalidComposition');

            noKnownMultipleCfg = noKnownCfg;
            noKnownMultipleCfg.openset.mixed_ratios = [0, 0, 0, 1];
            registry = build_jamming_registry(noKnownMultipleCfg);
            testCase.verifyError(@() build_openset_manifest( ...
                noKnownMultipleCfg, registry), ...
                'JammingOpenSet:InvalidComposition');

            highMinimumCfg = TestManifestBuilders.makeOpenSetConfig( ...
                2, 2, 1, 1);
            highMinimumCfg.types.known = {'FMB'};
            highMinimumCfg.types.unknown = {'MFT', 'ISRJ', 'SinFM', 'VFT'};
            highMinimumCfg.dataset.min_components = 2;
            highMinimumCfg = normalize_and_validate_config( ...
                highMinimumCfg, 'openset');
            registry = build_jamming_registry(highMinimumCfg);
            testCase.verifyError(@() build_openset_manifest( ...
                highMinimumCfg, registry), ...
                'JammingOpenSet:InvalidComposition');
        end

        function invalidOpenSetPartitionUsesUnifiedError(testCase)
            % 验证重叠、遗漏或注册表映射失败统一归类为非法开放集分区。
            cfg = TestManifestBuilders.makeOpenSetConfig(2, 2, 4, 4);
            registry = build_jamming_registry(cfg);

            overlapCfg = cfg;
            overlapCfg.types.unknown{end + 1} = 'FMB';
            testCase.verifyError(@() build_openset_manifest( ...
                overlapCfg, registry), 'JammingOpenSet:InvalidPartition');

            missingCfg = cfg;
            missingCfg.types.known = {'FMB', 'MFT'};
            testCase.verifyError(@() build_openset_manifest( ...
                missingCfg, registry), 'JammingOpenSet:InvalidPartition');

            unmappedCfg = cfg;
            unmappedCfg.types.enabled{end + 1} = 'NotRegistered';
            unmappedCfg.types.known{end + 1} = 'NotRegistered';
            testCase.verifyError(@() build_openset_manifest( ...
                unmappedCfg, registry), 'JammingOpenSet:InvalidPartition');
        end

        function zeroOpenSetSplitsKeepCompleteEmptyContract(testCase)
            % 验证四个零计数划分返回类型完整的 0 行普通 manifest 契约。
            cfg = TestManifestBuilders.makeOpenSetConfig(2, 2, 4, 4);
            registry = build_jamming_registry(cfg);
            cfg.openset.train_known = 0;
            cfg.openset.calibration_known = 0;
            cfg.openset.validation_mixed = 0;
            cfg.openset.test_mixed = 0;

            splits = build_openset_manifest(cfg, registry);
            referenceCfg = cfg;
            referenceCfg.dataset.num_samples = 1;
            reference = build_balanced_manifest( ...
                referenceCfg, registry, 'train');
            splitNames = fieldnames(splits);
            for splitIndex = 1:numel(splitNames)
                manifest = splits.(splitNames{splitIndex});
                testCase.verifyEqual(height(manifest), 0);
                testCase.verifyEqual(manifest.Properties.VariableNames, ...
                    reference.Properties.VariableNames);
                for columnIndex = 1:width(reference)
                    columnName = reference.Properties.VariableNames{ ...
                        columnIndex};
                    testCase.verifyClass(manifest.(columnName), ...
                        class(reference.(columnName)));
                end
            end
        end

        function mixedRowsRespectConfiguredMinimumComponents(testCase)
            % 验证四类混合样本都同时服从类别物理下界和配置最小阶数。
            cfg = TestManifestBuilders.makeOpenSetConfig(2, 2, 20, 20);
            cfg.types.enabled = {'FMB', 'MFT', 'ISRJ', ...
                'SinFM', 'VFT', 'SN'};
            cfg.types.known = {'FMB', 'MFT', 'ISRJ'};
            cfg.types.unknown = {'SinFM', 'VFT', 'SN'};
            cfg.dataset.min_components = 3;
            cfg.dataset.max_components = 3;
            cfg = normalize_and_validate_config(cfg, 'openset');
            registry = build_jamming_registry(cfg);

            splits = build_openset_manifest(cfg, registry);
            mixedNames = {'validation_mixed', 'test_mixed'};
            for splitIndex = 1:numel(mixedNames)
                manifest = splits.(mixedNames{splitIndex});
                testCase.verifyEqual(manifest.combination_size, ...
                    3 * ones(height(manifest), 1));
                category = TestManifestBuilders.classifyOpenSetRows( ...
                    manifest, cfg.types.known, cfg.types.unknown);
                testCase.verifyTrue(all(ismember(1:4, category)));
            end
        end

        function mixedCompositionChecksUseActualAllocatedCounts(testCase)
            % 验证只校验最大余数法实际分到行数的类别，并保持并列稳定顺序。
            onlyUnknownCfg = TestManifestBuilders.makeOpenSetConfig( ...
                2, 2, 1, 0);
            onlyUnknownCfg.openset.mixed_ratios = [0.1, 0.9, 0, 0];
            onlyUnknownCfg.types.enabled = {'SinFM'};
            onlyUnknownCfg.types.known = {};
            onlyUnknownCfg.types.unknown = {'SinFM'};
            onlyUnknownCfg.openset.train_known = 0;
            onlyUnknownCfg.openset.calibration_known = 0;
            registry = build_jamming_registry(onlyUnknownCfg);

            onlyUnknown = build_openset_manifest(onlyUnknownCfg, registry);

            testCase.verifyEqual(height(onlyUnknown.validation_mixed), 1);
            testCase.verifyEqual( ...
                onlyUnknown.validation_mixed.active_type_codes, {{'SinFM'}});
            testCase.verifyEqual( ...
                onlyUnknown.validation_mixed.unknown_component_count, 1);

            tieCfg = TestManifestBuilders.makeOpenSetConfig(2, 2, 2, 0);
            tieCfg.openset.mixed_ratios = [0.25, 0.25, 0.25, 0.25];
            tieCfg.openset.train_known = 0;
            tieCfg.openset.calibration_known = 0;
            registry = build_jamming_registry(tieCfg);
            tieSplits = build_openset_manifest(tieCfg, registry);
            tieCategory = TestManifestBuilders.classifyOpenSetRows( ...
                tieSplits.validation_mixed, tieCfg.types.known, ...
                tieCfg.types.unknown);
            tieCounts = arrayfun(@(value) sum(tieCategory == value), 1:4);
            testCase.verifyEqual(tieCounts, [1, 1, 0, 0]);

            zeroMixedCfg = TestManifestBuilders.makeOpenSetConfig( ...
                1, 1, 0, 0);
            zeroMixedCfg.types.enabled = {'FMB'};
            zeroMixedCfg.types.known = {'FMB'};
            zeroMixedCfg.types.unknown = {};
            registry = build_jamming_registry(zeroMixedCfg);
            zeroMixed = build_openset_manifest(zeroMixedCfg, registry);
            testCase.verifyEqual(height(zeroMixed.validation_mixed), 0);
            testCase.verifyEqual(height(zeroMixed.test_mixed), 0);
        end

        function registryIdsMustBeUniquePositiveIntegers(testCase)
            % 验证重复、零、非有限及分数注册表 ID 统一归类为非法分区。
            cfg = TestManifestBuilders.makeOpenSetConfig(2, 2, 4, 4);
            registry = build_jamming_registry(cfg);
            duplicateRegistry = registry;
            duplicateRegistry(2).id = duplicateRegistry(1).id;
            testCase.verifyError(@() build_openset_manifest( ...
                cfg, duplicateRegistry), ...
                'JammingOpenSet:InvalidPartition');

            invalidIds = {0, -1, NaN, Inf, 1.5, '1', ...
                complex(1, 1), [1, 2]};
            for invalidIndex = 1:numel(invalidIds)
                invalidRegistry = registry;
                invalidRegistry(1).id = invalidIds{invalidIndex};
                testCase.verifyError(@() build_openset_manifest( ...
                    cfg, invalidRegistry), ...
                    'JammingOpenSet:InvalidPartition');
            end
        end

        function directMixedSkeletonInheritsBalancedSafetyChecks(testCase)
            % 验证直接骨架保留 uint32、百万行及真实输出容量安全边界。
            cfg = TestManifestBuilders.makeOpenSetConfig(2, 2, 0, 1);
            cfg.openset.train_known = 0;
            cfg.openset.calibration_known = 0;
            cfg.openset.mixed_ratios = [1, 0, 0, 0];
            maximumSeed = double(intmax('uint32'));
            cfg.seed = maximumSeed - 3000000 - 1;
            registry = build_jamming_registry(cfg);

            maximumValid = build_openset_manifest(cfg, registry);
            testCase.verifyEqual( ...
                maximumValid.test_mixed.sample_seed, maximumSeed);

            overflowCfg = cfg;
            overflowCfg.seed = maximumSeed - 3000000;
            testCase.verifyError(@() build_openset_manifest( ...
                overflowCfg, registry), 'JammingManifest:SeedOverflow');

            oversizedCfg = cfg;
            oversizedCfg.seed = 1;
            oversizedCfg.openset.validation_mixed = 1000001;
            testCase.verifyError(@() build_openset_manifest( ...
                oversizedCfg, registry), 'JammingManifest:SplitTooLarge');

            capacityCfg = cfg;
            capacityCfg.seed = 1;
            bytesPerSample = capacityCfg.derived.sample_length * ...
                (16 + 8 * double(capacityCfg.output.save_legacy));
            capacityCfg.output.max_estimated_bytes = bytesPerSample - 1;
            testCase.verifyError(@() build_openset_manifest( ...
                capacityCfg, registry), ...
                'JammingConfig:EstimatedSizeExceeded');

            invalidSeeds = {'7', -1, 1.5, NaN, Inf, ...
                complex(1, 1), 2^32};
            for seedIndex = 1:numel(invalidSeeds)
                invalidSeedCfg = cfg;
                invalidSeedCfg.seed = invalidSeeds{seedIndex};
                testCase.verifyError(@() build_openset_manifest( ...
                    invalidSeedCfg, registry), ...
                    'JammingManifest:InvalidSeed');
            end

            zeroCfg = cfg;
            zeroCfg.openset.test_mixed = 0;
            zeroCfg.seed = maximumSeed - 3000000;
            zeroSplits = build_openset_manifest(zeroCfg, registry);
            testCase.verifyEqual(height(zeroSplits.validation_mixed), 0);
            testCase.verifyEqual(height(zeroSplits.test_mixed), 0);

            zeroOverflowCfg = zeroCfg;
            zeroOverflowCfg.seed = zeroOverflowCfg.seed + 1;
            testCase.verifyError(@() build_openset_manifest( ...
                zeroOverflowCfg, registry), ...
                'JammingManifest:SeedOverflow');
        end

        function mixedSplitDoesNotCallBalancedBuilder(testCase)
            % 验证 mixed 路径静态采用直接安全骨架，不先生成再覆盖普通组合。
            source = fileread(which('build_openset_manifest'));
            functionStart = strfind(source, ...
                'function manifest = build_mixed_split');
            testCase.assertNotEmpty(functionStart);
            remainingSource = source(functionStart(1):end);
            nextFunction = regexp(remainingSource(2:end), ...
                '\nfunction ', 'once');
            testCase.assertNotEmpty(nextFunction);
            functionBody = remainingSource(1:nextFunction);
            testCase.verifyEmpty(strfind( ...
                functionBody, 'build_balanced_manifest'));
        end
    end

    methods (Static, Access = private)
        function cfg = createSmallV2Fixture()
            %CREATESMALLV2FIXTURE 构造仅启用三种新增类型的小型 v2 配置。
            cfg = create_default_config(2);
            cfg.types.enabled = ...
                {'CombSpectrum', 'ClutterPulse', 'SpectralDispersion'};
            cfg.types.known = {'CombSpectrum', 'ClutterPulse'};
            cfg.types.unknown = {'SpectralDispersion'};
            cfg.dataset.num_samples = 12;
            cfg.dataset.min_components = 1;
            cfg.dataset.max_components = 3;
            cfg.dataset.combination_mode = 'random';
            cfg.dataset.samples_per_pattern = 1;
            cfg.params.common.JNR_dB = [20, 20];
            cfg.params.common.arrival_delay_s = [0, 0];
            % 夹具只有一个未知类，不能构造“已知加多个未知”类别。
            cfg.openset.mixed_ratios = [0.25, 0.25, 0.5, 0];
            cfg.output.save_legacy = false;
        end

        function cfg = makeOpenSetConfig(trainCount, calibrationCount, ...
                validationCount, testCount)
            %MAKEOPENSETCONFIG 构造五类启用、三类已知、两类未知的开放集配置。
            cfg = create_default_config();
            cfg.types.enabled = {'FMB', 'MFT', 'ISRJ', 'SinFM', 'VFT'};
            cfg.types.known = {'FMB', 'MFT', 'ISRJ'};
            cfg.types.unknown = {'SinFM', 'VFT'};
            cfg.dataset.min_components = 1;
            cfg.dataset.max_components = 3;
            cfg.dataset.combination_mode = 'random';
            cfg.openset.train_known = trainCount;
            cfg.openset.calibration_known = calibrationCount;
            cfg.openset.validation_mixed = validationCount;
            cfg.openset.test_mixed = testCount;
            cfg.openset.mixed_ratios = [0.25, 0.25, 0.35, 0.15];
            cfg = normalize_and_validate_config(cfg, 'openset');
        end

        function category = classifyOpenSetRows(manifest, known, unknown)
            %CLASSIFYOPENSETROWS 按开放集四类组成规则为每行赋予 1 到 4。
            rowCount = height(manifest);
            category = zeros(rowCount, 1);
            for rowIndex = 1:rowCount
                codes = manifest.active_type_codes{rowIndex};
                unknownCount = sum(ismember(codes, unknown));
                hasKnown = any(ismember(codes, known));
                allUnknown = ~isempty(codes) && all( ...
                    ismember(codes, unknown));
                if unknownCount == 0
                    category(rowIndex) = 1;
                elseif allUnknown
                    category(rowIndex) = 2;
                elseif unknownCount == 1 && hasKnown
                    category(rowIndex) = 3;
                elseif unknownCount >= 2 && hasKnown
                    category(rowIndex) = 4;
                end
            end
        end

        function cfg = makeBalancedConfig(enabledTypes, sampleCount, ...
                minComponents, maxComponents, combinationMode)
            %MAKEBALANCEDCONFIG 构造体量较小且通过标准化的普通组合配置。
            cfg = TestManifestBuilders.makeManifestConfig(enabledTypes, ...
                sampleCount, minComponents, maxComponents, ...
                combinationMode);
            cfg = normalize_and_validate_config(cfg, 'balanced');
        end

        function cfg = makeManifestConfig(enabledTypes, sampleCount, ...
                minComponents, maxComponents, combinationMode)
            %MAKEMANIFESTCONFIG 构造尚未标准化的轻量 manifest 测试配置。
            cfg = create_default_config();
            cfg.types.enabled = enabledTypes;
            cfg.types.known = enabledTypes;
            cfg.types.unknown = {};
            cfg.dataset.num_samples = sampleCount;
            cfg.dataset.min_components = minComponents;
            cfg.dataset.max_components = maxComponents;
            cfg.dataset.combination_mode = combinationMode;
        end

        function verifyPowerDefaults(testCase, manifest)
            %VERIFYPOWERDEFAULTS 验证功率轴关闭时的逐行零值与逻辑假掩码。
            testCase.verifyTrue(all(cellfun(@(powerDb) ...
                all(powerDb == 0), ...
                manifest.component_target_power_dB)));
            testCase.verifyTrue(all(cellfun(@(mask) ...
                ~any(mask), manifest.weak_component_mask)));
        end


        function cfg = makeLargeCountConfig(combinationMode, sampleCount, ...
                samplesPerPattern)
            %MAKELARGECOUNTCONFIG 构造只用于前置规模拒绝测试的大计数配置。
            cfg = create_default_config();
            cfg.types.enabled = {'FMB'};
            cfg.types.known = {'FMB'};
            cfg.types.unknown = {};
            cfg.dataset.num_samples = sampleCount;
            cfg.dataset.min_components = 1;
            cfg.dataset.max_components = 1;
            cfg.dataset.combination_mode = combinationMode;
            cfg.dataset.samples_per_pattern = samplesPerPattern;
            cfg.output.max_estimated_bytes = flintmax;
            cfg = normalize_and_validate_config(cfg, 'balanced');
        end
    end
end
