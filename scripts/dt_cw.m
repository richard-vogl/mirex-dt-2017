%% This function follows the submission guidelines of MIREX wiki:
% http://www.music-ir.org/mirex/wiki/2017:Drum_Transcription
% input:
%   inputfolder: str, the path to the folder that contains the audio files
%   outputfolder: str, the path to the folder that stores the resulting txt files
% Note:
%   BD 0
%   SD 1
%   HH 2
% CW @ GTCMT 2017

function dt_cw(inputfolder, outputfolder, method)

in_folder = ~iscell(inputfolder);

if in_folder
    allList = recursiveFileList(inputfolder, 'wav');
else
    allList = inputfolder;
end

for i = 1:length(allList)
    
    if in_folder
       filePath = allList(i).path;
       fileName = allList(i).name;
    else
       [path name ext] = fileparts(allList{i});
       filePath = [path '/' name ext];
       fileName = [name ext];
    end
    fprintf(['processing file: ' fileName ' '])
    %==== call my method
    [hh, bd, sd] = NmfDrum(filePath, method);
        
    %==== save file to the outputfolder
    if strcmp(outputfolder(end), '/')
    else
        outputfolder = [outputfolder, '/'];
    end
    txt_filename = [outputfolder, fileName(1:end-4), '.txt'];
    genMirexOutput(txt_filename, hh, bd, sd);
    
end
fprintf('Finished\n');

end