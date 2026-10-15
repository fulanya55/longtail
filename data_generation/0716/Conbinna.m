clear;
arrays = {1:3, 4:6, 7:9, 10:12, 13:15, 16:18, 19:21, 22:24};

c = [];
% 获取组合的元素数量（从 2 到 8）
% for k = 1:length(arrays)
for k = 1:length(arrays)
    % 获取当前组合的元素数量
    currentCombinationSize = k;
    
    % 获取所有可能的组合索引
    combinationIndices = nchoosek(1:length(arrays), currentCombinationSize);
    
    % 对于每个组合索引，提取对应的数组并合并
    for i = 1:size(combinationIndices, 1)
        selectedArrays = combinationIndices(i, :);
        combinedArray = zeros(1,3);
        d = zeros(1,3);
        for j = 1:length(selectedArrays)
%             combinedArray = [combinedArray ;arrays{selectedArrays(j)}];
            combinedArray = arrays{selectedArrays(j)};
            d = d + combinedArray;
        end
        c = [c;d];
        % 输出当前组合
        fprintf('组合大小为 %d 的第 %d 个组合：\n', currentCombinationSize, i);
        disp(combinedArray);
        fprintf('\n');
    end

end