function r = mi_window(signal, seed)
% MI_WINDOW  Mutual information between each sensor and the under-eye ERG, -10 to 120 ms.
%   r = mi_window('highF_env_ev130_145')   (default; also 'highF_75_145')
%   r = mi_window(signal, seed)            seed for the permutations (default 1)
%
% The window has 131 samples (1 kHz). Per participant, and for the group as
%   'ga'  : MI of the grand-average waveforms (mean of the 9 participants;
%           131 samples per channel). This is the published group result (Figure 6).
%   'vec' : MI of all participants' samples concatenated (9 x 131 = 1179 samples)
% Bin count: Freedman-Diaconis, averaged over channels/participants. The group bin
% count is estimated on the concatenated data and used for both 'ga' and 'vec'.
% Significance: z against surrogates in which the ERG bin sequence is split at a
% random point (10-90% of its length) and the two segments swapped and/or reversed;
% successive surrogates are applied to the previous one (mutualinformationx).
% Requires mutualinformationx.m (see README).

if nargin < 1, signal = 'highF_env_ev130_145'; end
if nargin < 2, seed = 1; end
rng(seed, 'twister');                            % surrogates are random: fix them
c = study_config();
data = load_evoked_group(c.dataDir, signal, c.participants);
win = dsearchn(c.times', -0.01):dsearchn(c.times', 0.12);
nP = numel(c.participants);
nC = numel(c.sensorChannels);
erg = squeeze(data(:, c.ergChannel, win));      % participants x samples

%% per participant
bins = zeros(nC, nP);
for ch = 1:nC
    for p = 1:nP
        [~, ~, bins(ch, p)] = mutualinformationx(squeeze(data(p, ch, win)), erg(p, :)');
    end
end
nBins = round(mean(bins, 'all'));

r.mi = zeros(nC, nP);
r.z = zeros(nC, nP);
for ch = 1:nC
    for p = 1:nP
        [r.z(ch, p), H] = mutualinformationx(squeeze(data(p, ch, win)), erg(p, :)', nBins, 2000);
        r.mi(ch, p) = H(1) + H(2) - H(3);
    end
end

%% group
gaData = squeeze(mean(data, 1));
binsGroup = zeros(1, nC);
for ch = 1:nC
    [~, ~, binsGroup(ch)] = mutualinformationx(data(:, ch, win), data(:, c.ergChannel, win));
end
nBinsGroup = round(mean(binsGroup));

[r.mi_ga, r.z_ga, r.mi_vec, r.z_vec] = deal(zeros(1, nC));
for ch = 1:nC
    [r.z_vec(ch), H] = mutualinformationx(data(:, ch, win), data(:, c.ergChannel, win), nBinsGroup, 5000);
    r.mi_vec(ch) = H(1) + H(2) - H(3);
    [r.z_ga(ch), H] = mutualinformationx(gaData(ch, win), gaData(c.ergChannel, win), nBinsGroup, 5000);
    r.mi_ga(ch) = H(1) + H(2) - H(3);
end
r.p = 1 - normcdf(r.z);
r.p_ga = 1 - normcdf(r.z_ga);
r.p_vec = 1 - normcdf(r.z_vec);

% first-order (Miller-Madow) bias of the plug-in MI, (bins - 1)^2 / (2 N ln 2),
% for the number of samples N actually entering each estimate
nWin = numel(win);
mi_bias = @(b, n) (b - 1)^2 / (2 * n * log(2));
fprintf('MI bias ~%.2g bits per participant (%d bins, %d samples)\n', mi_bias(nBins, nWin), nBins, nWin);
fprintf('MI bias ~%.2g bits grand average (%d bins, %d samples)\n', mi_bias(nBinsGroup, nWin), nBinsGroup, nWin);
fprintf('MI bias ~%.2g bits concatenated (%d bins, %d samples)\n', mi_bias(nBinsGroup, nP * nWin), nBinsGroup, nP * nWin);
r.seed = seed;
r.signal = signal;
r.nBins = nBins;
r.nBinsGroup = nBinsGroup;

%% plots
figure('Name', 'MI per participant');
for ch = 1:nC
    subplot(4, 5, c.arrayGrid(ch));   % physical array position (was c.barGrid)
    x = 1:3:3 * nP;
    bar(x, r.mi(ch, :), 0.4); hold on
    sig = r.p(ch, :) < c.alpha;
    text(x(sig), r.mi(ch, sig) + 0.05, '*');
    set(gca, 'xtick', x, 'xticklabel', 1:nP, 'ylim', [0 1.5]);
    title(sensor_title(c.sensorLabels{ch}));
    if ch == nC, xlabel('Participant'); ylabel('Mutual information (bits)'); end
end

figure('Name', 'MI of the grand average');
x = 1:3:3 * nC;
bar(x, r.mi_ga, 0.4); hold on
sig = r.p_ga < c.alpha;
text(x(sig), r.mi_ga(sig) + 0.05, '*');
set(gca, 'xtick', x, 'xticklabel', c.sensorLabels(1:nC), 'FontSize', 22);
xlabel('Sensor label'); ylabel('Mutual information (bits)');
title(sprintf('Grand-average MI, %s, -10 to 120 ms', strrep(signal, '_', ' ')));
end

function t = sensor_title(label)
% 'OPM 5' for an OPM, the label itself for an ERG electrode
if all(isstrprop(label, 'digit')), t = ['OPM ' label]; else, t = label; end
end
