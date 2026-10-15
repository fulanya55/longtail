function [patterns, counts] = allocate_pattern_counts(N_total, m, enable_repair, max_iters)
C = numel(m);

% 列出所有非空模式（1..2^C-1）
patterns = zeros(2^C - 1, C);
for v = 1:(2^C - 1)
    patterns(v,:) = bitget(v, 1:C);
end

% 解 s:  s = 1 - Π(1 - m_i*s)
s = min(0.99, max(0.2, sum(m)/numel(m)));  % 初值
for it = 1:200
    s_new = 1 - prod(1 - (m(:)'.*s));
    if abs(s_new - s) < 1e-12
        break;
    end
    s = s_new;
end
s = max(min(s, 0.999999), 1e-8);
p  = m(:) * s;

% 计算各模式的无条件概率 u，再得到条件概率 q=u/s
u = ones(size(patterns,1),1);
for j = 1:size(patterns,1)
    pat = patterns(j,:);
    u(j) = prod((p').^pat) * prod((1-p').^(1-pat));
end
q = u / s;
q = q / sum(q);

% largest remainder 取整
raw = N_total * q;
counts = floor(raw);
remain = N_total - sum(counts);
[~, idx] = sort(raw - counts, 'descend');
counts(idx(1:remain)) = counts(idx(1:remain)) + 1;

if ~enable_repair
    return;
end

% 计数微调：让边际计数更贴近目标 N_total*m
target = round(N_total * m(:));
mc = patterns' * counts;

for it = 1:max_iters
    diff = target - mc;
    if all(diff == 0)
        break;
    end

    need = find(diff > 0, 1, 'first');
    if isempty(need), break; end

    bestImprovement = 0;
    bestA = 0; bestB = 0;

    for a = 1:size(patterns,1)
        if counts(a) <= 0, continue; end
        if patterns(a,need) == 1, continue; end

        for b = 1:size(patterns,1)
            if patterns(b,need) == 0, continue; end

            delta = (patterns(b,:) - patterns(a,:))';
            new_mc = mc + delta;

            % 不允许超过 target（严格贴合）
            if any(new_mc > target)
                continue;
            end

            oldDist = sum(abs(target - mc));
            newDist = sum(abs(target - new_mc));
            improvement = oldDist - newDist;

            if improvement > bestImprovement
                bestImprovement = improvement;
                bestA = a; bestB = b;
            end
        end
    end

    if bestImprovement <= 0
        break;
    end

    counts(bestA) = counts(bestA) - 1;
    counts(bestB) = counts(bestB) + 1;
    mc = patterns' * counts;
end
end
