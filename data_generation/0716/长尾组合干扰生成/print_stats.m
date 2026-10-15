function print_stats(label, cfg, m)
N = size(label,1);
k = sum(label,2);

fprintf('\n[Stats] Total samples: %d\n', N);

fprintf('[Stats] Samples by active-label count k:\n');
for kk = 1:cfg.class_num
    fprintf('  k=%d : %d\n', kk, sum(k==kk));
end

marg = sum(label,1);
fprintf('[Stats] Marginal positives by component (count / target m):\n');
for i = 1:cfg.class_num
    fprintf('  %-6s : %d   (target approx %.4f)\n', cfg.class_names{i}, marg(i), m(i));
end

fprintf('[Stats] Top combination patterns (multi-hot):\n');
[uniq_rows,~,ic] = unique(label, 'rows');
cnt = accumarray(ic,1);
[sorted_cnt, idx] = sort(cnt, 'descend');

topK = min(20, numel(sorted_cnt));
for t = 1:topK
    pat = uniq_rows(idx(t),:);
    fprintf('  %s : %d\n', mat2str(pat), sorted_cnt(t));
end
end
