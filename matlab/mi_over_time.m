function r = mi_over_time(signal, nPermutations)
% MI_OVER_TIME  Sensor-ERG mutual information in a sliding window (27 samples, +-13 ms), -120 to 235 ms.
%   r = mi_over_time('highF_env_ev130_145', 3000)
%   signal: 'highF_75_145', 'highF_130_145' or 'highF_env_ev130_145'
%
% Three ways of combining participants are compared:
%   ga_of_mi : MI per participant, then averaged
%   mi_of_ga : MI of the grand-average waveforms
%   mi_vec   : MI of all participants' samples concatenated
% With nPermutations > 0 the z-scores against surrogates are returned too.
% Requires mutualinformationx.m (see README).

if nargin < 1, signal = 'highF_env_ev130_145'; end
if nargin < 2, nPermutations = 3000; end
c = study_config();
data = load_evoked_group(c.dataDir, signal, c.participants);
nP = numel(c.participants);
nC = numel(c.sensorChannels);

r.times = -0.12:0.001:0.235;
half = round(25 / 2);                                    % half window, samples
centre = dsearchn(c.times', r.times');
nT = numel(r.times);
seg = @(x, t) x(:, :, centre(t) - half:centre(t) + half);  % window around time t
erg = data(:, c.ergChannel, :);
ga = mean(data, 1);

%% bin counts (Freedman-Diaconis, averaged)
[binsP, binsGA, binsVec] = deal(zeros(nP, nC, nT), zeros(nC, nT), zeros(nC, nT));
for ch = 1:nC
    for t = 1:nT
        for p = 1:nP
            [~, ~, binsP(p, ch, t)] = mutualinformationx(seg(data(p, ch, :), t), seg(erg(p, :, :), t));
        end
        [~, ~, binsGA(ch, t)] = mutualinformationx(seg(ga(1, c.ergChannel, :), t), seg(ga(1, ch, :), t));
        [~, ~, binsVec(ch, t)] = mutualinformationx(seg(erg, t), seg(data(:, ch, :), t));
    end
end

%% MI
[mi, z] = deal(zeros(nP, nC, nT));
[r.mi_of_ga, r.z_of_ga, r.mi_vec, r.z_vec] = deal(zeros(nC, nT));
nbGA = round(mean(binsGA, 'all'));
nbVec = round(mean(binsVec, 'all'));
for ch = 1:nC
    for p = 1:nP
        nb = round(mean(binsP(p, ch, :)));
        for t = 1:nT
            [z(p, ch, t), H] = mutualinformationx(seg(data(p, ch, :), t), seg(erg(p, :, :), t), nb, nPermutations);
            mi(p, ch, t) = H(1) + H(2) - H(3);
        end
    end
    for t = 1:nT
        [r.z_of_ga(ch, t), H] = mutualinformationx(seg(ga(1, ch, :), t), seg(ga(1, c.ergChannel, :), t), nbGA, nPermutations);
        r.mi_of_ga(ch, t) = H(1) + H(2) - H(3);
        [r.z_vec(ch, t), H] = mutualinformationx(seg(data(:, ch, :), t), seg(erg, t), nbVec, nPermutations);
        r.mi_vec(ch, t) = H(1) + H(2) - H(3);
    end
    fprintf('channel %d done\n', ch);
end
r.mi = mi;
r.ga_of_mi = squeeze(mean(mi, 1));
r.ga_of_z = squeeze(mean(z, 1));
r.signal = signal;

%% plot the three flavours (MI and, if permuted, z)
show = r.times >= -0.1 & r.times <= 0.23;
vals = {{r.mi_vec, r.mi_of_ga, r.ga_of_mi, 'MI (bits)'}};
if nPermutations > 0
    vals{end + 1} = {r.z_vec, r.z_of_ga, r.ga_of_z, 'z'};
end
for v = 1:numel(vals)
    figure('Name', sprintf('%s over time, %s', vals{v}{4}, signal));
    for ch = 1:nC
        subplot(4, 5, c.arrayGrid(ch));
        plot(r.times(show), [vals{v}{1}(ch, show); vals{v}{2}(ch, show); vals{v}{3}(ch, show)], 'LineWidth', 2);
        if ch == 11, legend('Concatenated', 'MI of GA', 'GA of MI'); end
    end
end
end
