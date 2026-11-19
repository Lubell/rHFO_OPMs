function r = mi_itc_over_time(nPermutations)
% MI_ITC_OVER_TIME  Sensor-ERG mutual information of ITC time courses, per frequency (Figure 7).
%   r = mi_itc_over_time()       published figure only (MI in bits, no permutations)
%   r = mi_itc_over_time(1000)   also the permutation-based statistics
%
% Input: itc_110_145_9cycles.mat (python/export_mat.py), ITC magnitude,
% participants x 16 channels x 10 frequencies x 501 samples (-200 to 300 ms).
% MI between each sensor and the under-eye ERG in a sliding window of 27 samples
% (+-13 around each time point, 1 ms steps), 4 bins (mutualinformationx).
%
%   r.bits  : published Figure 7. MI in bits per participant, -120 to 235 ms;
%             r.bits.mean is the mean over participants, plotted -10 to 120 ms
%             on one shared colour scale.
% With nPermutations > 0, also:
%   r.group : MI as z against nPermutations surrogates per participant; at each
%             time point from -50 to 150 ms the 9 z-values are compared (rank-sum)
%             with all -50-0 ms values; FDR over time per channel and frequency.
%   r.ga    : MI (z) of the grand-average ITC (Fisher z averaged), -150 to 200 ms,
%             two-tailed p from z, FDR over that whole window; plotted -10 to 120 ms.
% Requires mutualinformationx.m (see README).

if nargin < 1, nPermutations = 0; end
c = study_config();
s = load(fullfile(c.dataDir, 'itc_110_145_9cycles.mat'));
fn = fieldnames(s);
itc = s.(fn{1});
[nP, ~, nF, ~] = size(itc);
nC = numel(c.sensorChannels);
half = 13;                                   % 27-sample window
nBins = 4;
r.freqs = logspace(log10(110), log10(145), 10);
plotWin = [-0.01 0.12];

%% per participant: MI in bits (and z if permuting)
t = -0.12:0.001:0.235;
centre = dsearchn(c.times', t');
[bits, z] = deal(zeros(nP, nC, nF, numel(t)));
for p = 1:nP
    for ch = 1:nC
        for f = 1:nF
            x = squeeze(itc(p, ch, f, :));
            y = squeeze(itc(p, c.ergChannel, f, :));
            bt = zeros(1, numel(t));
            zt = zeros(1, numel(t));
            parfor k = 1:numel(t)
                w = centre(k) - half:centre(k) + half;
                if nPermutations > 0
                    [zt(k), H] = mutualinformationx(x(w), y(w), nBins, nPermutations);
                else
                    [~, H] = mutualinformationx(x(w), y(w), nBins);
                end
                bt(k) = H(1) + H(2) - H(3);
            end
            bits(p, ch, f, :) = bt;
            z(p, ch, f, :) = zt;
        end
    end
    fprintf('participant %d done\n', p);
end
r.bits.times = t;
r.bits.mi = bits;
r.bits.mean = squeeze(mean(bits, 1));       % channels x freqs x time

%% published figure: mean MI over participants, -10 to 120 ms, shared colour scale
show = t >= plotWin(1) - 1e-9 & t <= plotWin(2) + 1e-9;
vals = r.bits.mean(:, :, show);
lim = [min(vals, [], 'all') max(vals, [], 'all')];
figure('Name', 'ITC MI, mean over participants (bits)');
for ch = 1:nC
    subplot(4, 5, c.arrayGrid(ch));
    contourf(t(show), r.freqs, squeeze(r.bits.mean(ch, :, show)), 40, 'linecolor', 'none');
    set(gca, 'clim', lim, 'yscale', 'log', 'ytick', round(r.freqs), 'xlim', plotWin);
    title(panel_title(c.sensorLabels{ch}));
end
cb = colorbar('Position', [0.92 0.11 0.015 0.2]);
cb.Label.String = 'Mutual information (bits)';
if nPermutations == 0, return; end

%% group rank-sum on the per-participant z against the baseline
base = dsearchn(t', -0.05):dsearchn(t', 0);
test = dsearchn(t', -0.05):dsearchn(t', 0.15);
[r.group.z, r.group.p_fdr] = deal(zeros(nC, nF, numel(test)));
for ch = 1:nC
    for f = 1:nF
        pooled = z(:, ch, f, base);
        p = zeros(1, numel(test));
        for k = 1:numel(test)
            [p(k), ~, st] = ranksum(z(:, ch, f, test(k)), pooled(:));
            r.group.z(ch, f, k) = st.zval;
        end
        r.group.p_fdr(ch, f, :) = fdr_adjust(p);
    end
end
r.group.times = t(test);
r.group.mi_z = z;

%% grand-average ITC
tga = -0.15:0.001:0.2;
centre = dsearchn(c.times', tga');
ga = squeeze(tanh(mean(atanh(itc), 1)));            % channels x freqs x samples
zga = zeros(nC, nF, numel(tga));
for ch = 1:nC
    for f = 1:nF
        x = squeeze(ga(ch, f, :));
        y = squeeze(ga(c.ergChannel, f, :));
        zt = zeros(1, numel(tga));
        parfor k = 1:numel(tga)
            w = centre(k) - half:centre(k) + half;
            zt(k) = mutualinformationx(x(w), y(w), nBins, nPermutations);
        end
        zga(ch, f, :) = zt;
    end
end
r.ga.times = tga;
r.ga.z = zga;
r.ga.p_fdr = zeros(size(zga));
for ch = 1:nC
    for f = 1:nF
        r.ga.p_fdr(ch, f, :) = fdr_adjust(2 * (1 - normcdf(abs(squeeze(zga(ch, f, :))))));
    end
end

%% statistics plots
figure('Name', 'ITC MI, group rank-sum z');
for ch = 1:nC
    subplot(4, 5, c.arrayGrid(ch));
    contourf(r.group.times, r.freqs, squeeze(r.group.z(ch, :, :)), 40, 'linecolor', 'none');
    set(gca, 'yscale', 'log', 'ytick', round(r.freqs));
    title(panel_title(c.sensorLabels{ch}));
end
figure('Name', 'ITC MI of grand average, FDR p');
% The original plotted samples 1:151 of this time axis, i.e. -150 to 0 ms (baseline
% only): the index range was carried over from the -120 ms time axis above.
show = tga >= plotWin(1) - 1e-9 & tga <= plotWin(2) + 1e-9;
for ch = 1:nC
    subplot(4, 5, c.arrayGrid(ch));
    contourf(tga(show), r.freqs, squeeze(r.ga.p_fdr(ch, :, show)), 40, 'linecolor', 'none');
    set(gca, 'clim', [0 1], 'yscale', 'log', 'ytick', round(r.freqs));
    title(panel_title(c.sensorLabels{ch}));
end
end

function t = panel_title(label)
% 'OPM 5' for an OPM, the label itself for an ERG electrode
if all(isstrprop(label, 'digit')), t = ['OPM ' label]; else, t = label; end
end
