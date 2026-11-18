function plot_mi_sensitivity_map(r, outFile)
% PLOT_MI_SENSITIVITY_MAP  Sensor map of MI with the ERG (output of mi_window).
%   Colour: mean MI over participants. Marker size: inverse of its SD.
%   ** marks channels with significant grand-average MI.
%   Uses the cmocean 'algae' colormap if available.

c = study_config();
x = [2 3 4 1 2 3 4 5 3 4 5 2 3 4 1] - 1;   % sensor positions, same order as c.barGrid
y = [4 4 4 3 3 3 3 3 2 2 2 1 1 1 2];

figure;
scatter(x, y, 1 ./ (std(r.mi, [], 2)' * 14^-3), mean(r.mi, 2)', 'filled');
caxis([0.5 1]);
colorbar('eastoutside');
if exist('cmocean', 'file'), colormap(cmocean('algae')); end
set(gca, 'xlim', [0 5], 'visible', 'off', 'FontSize', 22);
sig = find(r.p_ga < c.alpha);
for ch = sig
    text(x(ch), y(ch), ['**'; '**'], 'Color', 'r');
end

if nargin > 1
    set(gcf, 'color', 'w');
    print(gcf, '-depsc', '-painters', outFile);
end
end
