clc, clear; 

%% Use readNPYheader.m & readNPY.m for reading from .npy
im = readNPY('train_data.npy');
label = im(:, 10001);
im = im(:, 1:10000);

%% Filtering : bandstop, lowpass, highpass

Fs = 500;
d1 = designfilt('bandstopiir','FilterOrder',2, ...
               'HalfPowerFrequency1',49,'HalfPowerFrequency2',51, ...
               'DesignMethod','butter','SampleRate',Fs);

d2 = designfilt('lowpassiir', 'FilterOrder', 12, 'HalfPowerFrequency', ...
                30, 'SampleRate', Fs, 'DesignMethod', 'butter');
     
d3 = designfilt('highpassiir', 'FilterOrder', 12, 'HalfPowerFrequency', ...
                0.5, 'SampleRate', Fs, 'DesignMethod', 'butter');
            
im_filtered1 = filtfilt(d1, im');
im_filtered2 = filtfilt(d2, im_filtered1);
TrainData = filtfilt(d3, im_filtered2);

%% Save preprocessed data

save('Matlab_Data', 'TrainData');

