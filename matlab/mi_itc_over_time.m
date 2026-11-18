function r = mi_itc_over_time(nPermutations)
% MI_ITC_OVER_TIME  Sensor-ERG mutual information of ITC time courses, per frequency (Figure 6c).
%   r = mi_itc_over_time(1000)
%
% Input: itc_110_145_9cycles.mat (python/export_mat.py), ITC magnitude,
% participants x 16 channels x 10 frequencies x 501 samples.
% MI in a sliding window of 27 samples (+-13 around each time point), 4 bins,
% z against nPermutations surrogates (mutualinformationx).
%   r.group: per participant, -120 to 235 ms; at each time point from -50 to 150 ms
%            the 9 z-values are compared (rank-sum) with all -50-0 ms values.
%   r.ga   : on the grand-average ITC (Fisher z averaged), -150 to 200 ms;
%            two-tailed p from z.
% p-values are FDR corrected over time, per channel and frequency.
% Requires mutualinformationx.m (see README).

if nargin < 1, nPermutations = 1000; end
c = study_config();
s = load(fullfile(c.dataDir, 'itc_110_145_9cycles.mat'));
fn = fieldnames(s);
itc = s.(fn{1});
[nP, ~, nF, ~] = size(itc);
nC = numel(c.sensorChannels);
half = round(25 / 2);
nBins = 4;
r.freqs = logspace(log10(110), log10(145), 10);

%% per participant, then group rank-sum against the baseline
t = -0.12:0.001:0.235;
centre = dsearchn(c.times', t');
z = zeros(nP, nC, nF, numel(t));
for p = 1:nP
    for ch = 1:nC
        for f = 1:nF
            x = squeeze(itc(p, ch, f, :));
            y = squeeze(itc(p, c.ergChannel, f, :));
            parfor k = 1:numel(t)
                w = centre(k) - half:centre(k) + half;
                z(p, ch, f, k) = mutualinformationx(x(w), y(w), nBins, nPermutations);
            end
        end
    end
    fprintf('participant %d done\n', p);
end
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
        parfor k = 1:numel(tga)
            w = centre(k) - half:centre(k) + half;
            zga(ch, f, k) = mutualinformationx(x(w), y(w), nBins, nPermutations);
        end
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

%% plots
figure('Name', 'ITC MI, group rank-sum z');
for ch = 1:nC
    subplot(4, 5, c.arrayGrid(ch));
    contourf(r.group.times, r.freqs, squeeze(r.group.z(ch, :, :)), 40, 'linecolor', 'none');
    set(gca, 'yscale', 'log', 'ytick', round(r.freqs));
end
figure('Name', 'ITC MI of grand average, FDR p');
show = tga <= 0;  % original plot: -150 to 0 ms (first 151 samples)
for ch = 1:nC
    subplot(4, 5, c.arrayGrid(ch));
    contourf(tga(show), r.freqs, squeeze(r.ga.p_fdr(ch, :, show)), 40, 'linecolor', 'none');
    set(gca, 'clim', [0 1], 'yscale', 'log', 'ytick', round(r.freqs));
end
end
