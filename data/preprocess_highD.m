  %% Process dataset into mat files %%

clear;
clc;




%% Fields: 

%{ 
1: Dataset Id
2: Vehicle Id
3: Frame Number
4: Local X
5: Local Y
6: velocity
7: acceleration
8: Lane Id
9: class
10: Lateral maneuver
11: Longitudinal maneuver
12-50: Neighbor Car Ids at grid location
%}

%% Load data and add dataset id
dataset_to_use = 120;
disp('Loading data...')

for k = 1:dataset_to_use
    if mod(k,2)
        % forward side:
        dataset_name = sprintf('raw_highd_ngsim_format/%02d-fwd.csv', ceil(k/2));
    else
        % backward side:
        dataset_name = sprintf('raw_highd_ngsim_format/%02d-bck.csv', ceil(k/2));
    end
    csv{k}  = readtable(dataset_name);
    traj{k} = csv{k}{:,1:14};
    % Add dataset id at the 1st column
    traj{k} = single([k*ones(size(traj{k},1),1),traj{k}]);

    traj{k} = traj{k}(:,[1,2,3,6,7,13,14,15,12]);

    vehTrajs{k} = containers.Map;
    vehTimes{k} = containers.Map;
end    
%% Parse fields (listed above):
disp('Parsing fields...')

for ii = 1:dataset_to_use
    vehIds = unique(traj{ii}(:,2));

    for v = 1:length(vehIds)
        vehTrajs{ii}(int2str(vehIds(v))) = traj{ii}(traj{ii}(:,2) == vehIds(v),:);
    end
    
    timeFrames = unique(traj{ii}(:,3));

    for v = 1:length(timeFrames)
        vehTimes{ii}(int2str(timeFrames(v))) = traj{ii}(traj{ii}(:,3) == timeFrames(v),:);
    end
    
    for k = 1:length(traj{ii}(:,1))        
        time = traj{ii}(k,3);
        dsId = traj{ii}(k,1);
        vehId = traj{ii}(k,2);
        vehtraj = vehTrajs{ii}(int2str(vehId));
        ind = find(vehtraj(:,3)==time);
        ind = ind(1);
        lane = traj{ii}(k,8);
        
        
       %% Get lateral maneuver:
        ub = min(size(vehtraj,1),ind+40);
        lb = max(1, ind-40);
        if vehtraj(ub,8)>vehtraj(ind,8) || vehtraj(ind,8)>vehtraj(lb,8)
            traj{ii}(k,10) = 3;
        elseif vehtraj(ub,8)<vehtraj(ind,8) || vehtraj(ind,8)<vehtraj(lb,8)
            traj{ii}(k,10) = 2;
        else
            traj{ii}(k,10) = 1;
        end
        
        
       %% Get longitudinal maneuver:
        ub = min(size(vehtraj,1),ind+50);
        lb = max(1, ind-30);
        if ub==ind || lb ==ind
            traj{ii}(k,11) =1;
        else
            vHist = (vehtraj(ind,5)-vehtraj(lb,5))/(ind-lb);
            vFut = (vehtraj(ub,5)-vehtraj(ind,5))/(ub-ind);
            if vFut/vHist <0.8
                traj{ii}(k,11) =2;
            elseif vFut/vHist > 1.25
                traj{ii}(k,11) = 3;
            else
                traj{ii}(k,11) =1;
            end
        end
        % Get 
        % Get grid locations:
        t = vehTimes{ii}(int2str(time));
        frameEgo = t(t(:,8) == lane,:);
        frameL = t(t(:,8) == lane-1,:);
        frameR = t(t(:,8) == lane+1,:);
        if ~isempty(frameL)
            for l = 1:size(frameL,1)
                y = frameL(l,5)-traj{ii}(k,5);
                if abs(y) <90
                    gridInd = 1+round((y+90)/15);
                    traj{ii}(k,11+gridInd) = frameL(l,2);
                end
            end
        end
        for l = 1:size(frameEgo,1)
            y = frameEgo(l,5)-traj{ii}(k,5);
            if abs(y) <90 && y~=0
                gridInd = 14+round((y+90)/15);
                traj{ii}(k,11+gridInd) = frameEgo(l,2);
            end
        end
        if ~isempty(frameR)
            for l = 1:size(frameR,1)
                y = frameR(l,5)-traj{ii}(k,5);
                if abs(y) <90
                    gridInd = 27+round((y+90)/15);
                    traj{ii}(k,11+gridInd) = frameR(l,2);
                end
            end
        end
        
    end
end

save('allData_s.mat','traj');


%% Split train, validation, test
load('allData_s.mat','traj');
disp('Splitting into train, validation and test sets...')

tracks = {};
trajAll = [];
for k = 1:dataset_to_use
    vehIds = unique(traj{k}(:, 2));
    for l = 1:length(vehIds)
        vehTrack = traj{k}(traj{k}(:, 2)==vehIds(l), :);
        tracks{k,vehIds(l)} = vehTrack(:, 3:11)'; % features and maneuver class id
        filtered = vehTrack(30+1:end-50, :);
        trajAll = [trajAll; filtered];
    end
end
clear traj;

trajTr=[];
trajVal=[];
trajTs=[];
for ii = 1:dataset_to_use
    no = trajAll(find(trajAll(:,1)==ii),:);
    len1 = round(length(no)*0.7);
    len2 = round(length(no)*0.8);
    trajTr = [trajTr;no(1:len1,:)];
    trajVal = [trajVal;no(len1:len2,:)];
    trajTs = [trajTs;no(len2:end,:)];
end

disp('Saving mat files...')
%%
traj = trajTr;
save('TrainSet','traj','tracks');

traj = trajVal;
save('ValSet','traj','tracks');

traj = trajTs;
save('TestSet','traj','tracks');










