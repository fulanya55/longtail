function output = zuhe_label(num_classes, zhuhe_num)
%% 产生组合的one-hot标签
%   class_num:单一类别数
%   zhube_num:组合类别数
output = [] ;
for r = 1:zhuhe_num
        % 生成组合
        combinations = nchoosek(1:num_classes, r); % nchoosek 生成组合
        num_combinations = size(combinations, 1); % 组合的数量
        one_hot_matrix = zeros(num_combinations, num_classes); % 初始化 one-hot 矩阵

        % 填充 one-hot 编码
        for i = 1:num_combinations
            one_hot_matrix(i, combinations(i, :)) = 1;
        end
        output  = [output ; one_hot_matrix];
end