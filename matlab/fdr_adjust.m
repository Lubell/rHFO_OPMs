function padj = fdr_adjust(p)
% FDR_ADJUST  Benjamini-Hochberg adjusted p-values (same shape as p).
[ps, order] = sort(p(:));
m = numel(ps);
adj = min(1, cummin(ps .* m ./ (1:m)', 'reverse'));
padj = zeros(size(p));
padj(order) = adj;
end
