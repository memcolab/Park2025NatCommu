%--------------------------------------------------------------------------
% Custom MATLAB script (Park et al., 2025, Nature Communications)
%--------------------------------------------------------------------------
clear;clc;

%% --------------------------------------------------------------------------
% Discrimination Index (within-task)
%--------------------------------------------------------------------------
nConds = 5;% number of conditions (e.g. number of stimuli to decode)
nROIs = 4;% number of target ROIs
ROISizes = [300,400,250,550];% sizes (number of voxels) of each ROI
nSplits = 3;% number of run splits (e.g. 3 ways of spliting 4 runs into two)

% load fMRI data
load('datafilename.mat','patternMatrix_odd','patternMatrix_even'); % matrix size: nConds x max(ROISizes) x nROIs x nSplits

correlMatrixFull = zeros(nConds,nConds,nROIs,nSplits);
for cSplit = 1:nSplits
    for cROI = 1:nROIs
        oddMatrix = squeeze(patternMatrix_odd(:,1:ROISizes(cROI),cROI,cSplit));
        evenMatrix = squeeze(patternMatrix_even(:,1:ROISizes(cROI),cROI,cSplit));

        % cocktail blank removal (pattern normalization)
        for cVoxel = 1:size(oddMatrix,2)
            evenMatrix(:,cVoxel) = evenMatrix(:,cVoxel) - mean(evenMatrix(:,cVoxel),1);
            oddMatrix(:,cVoxel) = oddMatrix(:,cVoxel) - mean(oddMatrix(:,cVoxel),1);
        end
        
        % derive pattern correlation matrix
        tempPatterns = [oddMatrix;evenMatrix]';
        tempCorr = corrcoef(tempPatterns);

        % Fisher's Z transformation, z = 0.5 * log((1+r)/(1-r))
        trans_tempCorr = 0.5 * log((1+tempCorr)./(1-tempCorr));

        correlMatrixFull(:,:,cROI,cSplit) = trans_tempCorr(1:nConds,nConds+1:end);
        for i = 1:size(correlMatrixFull,1)
            for j = i+1:size(correlMatrixFull,2)
                correlMatrixFull(i,j,cROI,cSplit) = (correlMatrixFull(i,j,cROI,cSplit) + correlMatrixFull(j,i,cROI,cSplit))/2;
                correlMatrixFull(j,i,cROI,cSplit) = correlMatrixFull(i,j,cROI,cSplit);
            end
        end
    end
end

omniMatrixCorr = squeeze(mean(correlMatrixFull,4));

% template diagonal and offdiagonal matrices
object_diag_I = eye(nConds);
object_diag_I(object_diag_I==0) = nan;
object_offdiag_I = triu(ones(nConds),1);
object_offdiag_I(object_offdiag_I==0) = nan;

% calculate discrimination index from the pattern correlation matrix
objectDiag = zeros(nConds,nConds,nROIs);
objectOffDiag = zeros(nConds,nConds,nROIs);
for cROI = 1:size(omniMatrixCorr,3)
    objectDiag(:,:,cROI) = object_diag_I .* omniMatrixCorr(:,:,cROI);
    objectOffDiag(:,:,cROI) = object_offdiag_I .* omniMatrixCorr(:,:,cROI);
end
crossDIMatrix = nanmean(reshape(objectDiag,nConds^2,nROIs),1)-nanmean(reshape(objectOffDiag,nConds^2,nROIs),1);

% save data
saveName = 'DiscriminationIndex_data.mat';
save(saveName,'crossDIMatrix');



%% --------------------------------------------------------------------------
% Cross-task Discrimination Index
%--------------------------------------------------------------------------
nConds = 5;% number of conditions (e.g. number of stimuli to decode)
nROIs = 4;% number of target ROIs
ROISizes = [300,400,250,550];% sizes (number of voxels) of each ROI

% load fMRI data of task A and B
load('datafilename.mat','taskA_patternMatrix_all','taskB_patternMatrix_all'); % matrix size: nConds x max(ROISizes) x nROIs

correlMatrixFull = zeros(nConds,nConds,nROIs);
for cROI = 1:nROIs
    taskA_patternMatrix = squeeze(taskA_patternMatrix_all(:,1:ROISizes(cROI),cROI));% Conds X Voxels X cROI
    taskB_patternMatrix = squeeze(taskB_patternMatrix_all(:,1:ROISizes(cROI),cROI));% Conds X Voxels X cROI

    % cocktail blank removal (pattern normalization)
    for cVoxel = 1:size(taskB_patternMatrix,2)
        taskA_patternMatrix(:,cVoxel) = taskA_patternMatrix(:,cVoxel) - mean(taskA_patternMatrix(:,cVoxel),1);
        taskB_patternMatrix(:,cVoxel) = taskB_patternMatrix(:,cVoxel) - mean(taskB_patternMatrix(:,cVoxel),1);
    end

    % derive pattern correlation matrix
    tempPatterns = [taskA_patternMatrix;taskB_patternMatrix]';
    tempCorr = corrcoef(tempPatterns);

    % Fisher transformatio z = 0.5 * log((1+r)/(1-r))
    trans_tempCorr = 0.5 * log((1+tempCorr)./(1-tempCorr));

    correlMatrixFull(:,:,cROI) = trans_tempCorr(1:nConds,nConds+1:end); %correlation between conditions. at each voxel
    for i = 1:size(correlMatrixFull,1)
        for j = i+1:size(correlMatrixFull,2)
            correlMatrixFull(i,j,cROI) = (correlMatrixFull(i,j,cROI) + correlMatrixFull(j,i,cROI))/2;
            correlMatrixFull(j,i,cROI) = correlMatrixFull(i,j,cROI);
        end
    end
end

% template diagonal and offdiagonal matrices
object_diag_I = eye(nConds);
object_diag_I(object_diag_I==0) = nan;
object_offdiag_I = triu(ones(nConds),1);
object_offdiag_I(object_offdiag_I==0) = nan;

% cross-task discrimination index
objectDiag = zeros(nConds,nConds,nROIs);
objectOffDiag = zeros(nConds,nConds,nROIs);
for cROI = 1:size(correlMatrixFull,3)
    objectDiag(:,:,cROI) = object_diag_I .* correlMatrixFull(:,:,cROI);
    objectOffDiag(:,:,cROI) = object_offdiag_I .* correlMatrixFull(:,:,cROI);
end
crossDIMatrix = nanmean(reshape(objectDiag,nConds^2,nROIs),1)-nanmean(reshape(objectOffDiag,nConds^2,nROIs),1);

% save data
saveName = 'Cross-task_DiscriminationIndex_data.mat';
save(saveName,'crossDIMatrix');



%% --------------------------------------------------------------------------
% Cross-region RSA
%--------------------------------------------------------------------------

% Cross-region representational similarity between ROI 1 and ROI 2
ROI1 = 1;
ROI2 = 2;

nConds = 5;% number of conditions (e.g. number of stimuli to decode)
nROIs = 4;% number of target ROIs
ROISizes = [300,400,250,550];% sizes (number of voxels) of each ROI

% load fMRI data of task A and B
load('datafilename.mat','taskA_patternMatrix_all','taskB_patternMatrix_all'); % matrix size: nConds x max(ROISizes) x nROIs

correlMatrixFull = zeros(nConds,nConds,nROIs);
for cROI = 1:nROIs
    taskA_patternMatrix = squeeze(taskA_patternMatrix_all(:,1:ROISizes(cROI),cROI));
    taskB_patternMatrix = squeeze(taskB_patternMatrix_all(:,1:ROISizes(cROI),cROI));

    % derive pattern correlation matrix
    tempPatterns = [taskA_patternMatrix;taskB_patternMatrix]';
    tempCorr = corrcoef(tempPatterns);

    % Fisher transformatio z = 0.5 * log((1+r)/(1-r))
    trans_tempCorr = 0.5 * log((1+tempCorr)./(1-tempCorr));

    correlMatrixFull(:,:,cROI) = trans_tempCorr(1:nConds,nConds+1:end); %correlation between conditions. at each voxel
    for i = 1:size(correlMatrixFull,1)
        for j = i+1:size(correlMatrixFull,2)
            correlMatrixFull(i,j,cROI) = (correlMatrixFull(i,j,cROI) + correlMatrixFull(j,i,cROI))/2;
            correlMatrixFull(j,i,cROI) = correlMatrixFull(i,j,cROI);
        end
    end
end

% derive RDM (representational dissimilarity matrix) for each ROI
offdiag_I = logical(ones(5)-eye(5));
RSM_ROI1 = squeeze(correlMatrixFull(:,:,ROI1));
RSM_ROI2 = squeeze(correlMatrixFull(:,:,ROI2));
RDM_ROI1 = 1 - RSM_ROI1(offdiag_I);
RDM_ROI2 = 1 - RSM_ROI2(offdiag_I);

% calculate representational similarity between two ROIs
tmp_corr = corr(RDM_ROI1,RDM_ROI2);

% Fisher transformation z = 0.5 * log((1+r)/(1-r))
trans_corr = 0.5 * log((1+tmp_corr)./(1-tmp_corr));% diagonal --> Inf

crossROI_RS = trans_corr;

% save data
saveName = 'Cross-region-RSA_data.mat';
save(saveName,'crossROI_RS');


