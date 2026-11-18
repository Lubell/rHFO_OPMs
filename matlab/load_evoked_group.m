function data = load_evoked_group(dataDir, name, participants)
% LOAD_EVOKED_GROUP  Stack <pid>_<name>.mat files (written by python/export_mat.py).
%   data = load_evoked_group(dataDir, 'highF_env_ev130_145', 10:19)
%   returns participants x 16 channels x 501 samples (-200 to 300 ms, 1 kHz).

data = zeros(numel(participants), 16, 501);
for i = 1:numel(participants)
    s = load(fullfile(dataDir, sprintf('PO%d_%s.mat', participants(i), name)));
    fn = fieldnames(s);              % variable name differs between export versions
    data(i, :, :) = s.(fn{1});
end
end
