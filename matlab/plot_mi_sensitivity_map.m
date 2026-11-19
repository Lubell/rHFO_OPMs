function plot_mi_sensitivity_map(r, outFile)
% PLOT_MI_SENSITIVITY_MAP  Sensor map of MI with the ERG (output of mi_window).
%   Colour: mean MI over participants. Marker size: inverse of its SD.
%   ** marks channels with significant grand-average MI.
%   Uses the cmocean 'algae' colormap if available.

c = study_config();
% Marker positions from the physical array layout (c.arrayGrid, a 4 x 5 grid).
% The former hard-coded x/y followed c.barGrid, which placed channel k in the slot
% of sensor label k, so 13 of the 14 OPMs were drawn in the wrong position.
slot = c.arrayGrid(c.sensorChannels);
x = mod(slot - 1, 5);              % column 0-4
y = 4 - floor((slot - 1) / 5);     % row, top = 4

% Marker area proportional to 1/SD across participants (as published), rescaled so
% the largest marker is MAX_AREA points^2 and neighbouring markers do not overlap.
MAX_AREA = 2500;
area = 1 ./ std(r.mi, [], 2)';
area = MAX_AREA * area / max(area);
figure;
scatter(x, y, area, mean(r.mi, 2)', 'filled');
caxis([0.5 1]);
colorbar('eastoutside');
if exist('cmocean', 'file'), colormap(cmocean('algae')); end
set(gca, 'xlim', [0 5], 'visible', 'off', 'FontSize', 22);
for ch = c.sensorChannels         % print the sensor label on every marker
    text(x(ch), y(ch) - 0.3, c.sensorLabels{ch}, 'HorizontalAlignment', 'center', 'FontSize', 12);
end
sig = find(r.p_ga < c.alpha);
for ch = sig
    text(x(ch), y(ch), ['**'; '**'], 'Color', 'r');
end

if nargin > 1
    set(gcf, 'color', 'w');
    print(gcf, '-depsc', '-painters', outFile);
end
end
