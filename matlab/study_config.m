function c = study_config()
% STUDY_CONFIG  Settings shared by the MATLAB analyses (mirrors python/config.py).

c.dataDir = getenv('OPM_ERG_DATA_DIR');
if isempty(c.dataDir)
    error('Set the environment variable OPM_ERG_DATA_DIR to the data folder.');
end
c.participants = [10 11 13 14 15 16 17 18 19];   % PO12 excluded (OPMs not initialised)
c.times = -0.2:0.001:0.3;                         % epoch time axis (s)
c.ergChannel = 16;                                % under-eye ERG electrode
c.sensorChannels = 1:15;                          % 14 OPMs + upper ERG electrode
c.alpha = 0.0025;                                 % threshold on permutation z (one-sided)
% Sensor label of each channel index (python/config.py SENSOR_LABELS). Figures name
% sensors by these labels, never by channel index.
c.sensorLabels = {'8','6','11','5','4','1','2','7','3','9','10','14','13','12','ERG up','ERG low'};
% Former MI bar-plot layout: it put channel k in the slot of SENSOR LABEL k, so 13 of
% the 14 OPMs were drawn in the wrong position. Kept only to reproduce the old figure.
c.barGrid = [2 3 4 6 7 8 9 10 13 14 15 17 18 19 11];
% subplot slot in the mounted-array layout (python/config.py GRID_POS)
c.arrayGrid = [10 8 15 7 6 2 3 9 4 13 14 19 18 17 11 16];
end
