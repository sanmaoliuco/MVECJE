f = load ('Handwritten_numerals.mat');
X = f.data;
Ground_truth = f.labels;
View_1 = X{2};
View_2 = X{1};
View_3 = X{3};
View_4 = X{5};

tic;
gamma = 0.001;

View_1 =  (View_1) ./ repmat(std(View_1), size(View_1,1), 1);
View_1 = imgaussfilt(View_1); % gaussian image-filter
K_View_1 = pairwise_kernels(View_1, View_1,  'rbf', gamma);
K_View_1 = K_View_1./mean(pdist(K_View_1).^2); 

View_2 =  (View_2) ./ repmat(std(View_2), size(View_2,1), 1);
View_2 = imgaussfilt(View_2); % gaussian image-filter
K_View_2 = pairwise_kernels(View_2, View_2,  'rbf', gamma);
K_View_2 = K_View_2./mean(pdist(K_View_2).^2);


View_3 =  (View_3) ./ repmat(std(View_3), size(View_3,1), 1);
View_3 = imgaussfilt(View_3); % gaussian image-filter
K_View_3 = pairwise_kernels(View_3, View_3,  'rbf', gamma);
K_View_3 = K_View_3./mean(pdist(K_View_3).^2); 

View_4 =  (View_4) ./ repmat(std(View_4), size(View_4,1), 1);
View_4 = imgaussfilt(View_4); % gaussian image-filter
K_View_4 = pairwise_kernels(View_4, View_4,  'rbf', gamma);
K_View_4 = K_View_4./mean(pdist(K_View_4).^2);


data = [K_View_1;K_View_2;K_View_3;K_View_4];

%%%%%%%%%%%%%%%%%
K = data;
Clusters= 10; %number of clusters.
View_num=4; %number of views present in the dataset.
View_data_num=size(K,1)/View_num;
p_list = [1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 50];


Cluster_labels_MVKKM = [];
w_list_MVKKM = [];
Clustering_errors_MVKKM = [];


for i=1:length(p_list(:))
    %MVKKM
    [Cluster_elem_MVKKM,w_MVKKM,Clustering_error_MVKKM] = MVClustering(Clusters, View_num, p_list(i), K, 'MVKKM');
    Cluster_labels_MVKKM = [Cluster_labels_MVKKM Cluster_elem_MVKKM]; % colms represent dif values of p
    w_list_MVKKM = [w_list_MVKKM ;w_MVKKM]; % rows represent dif values of p
    Clustering_errors_MVKKM = [Clustering_errors_MVKKM Clustering_error_MVKKM];  % colms represent dif values of p
end
members = Cluster_labels_MVKKM;
gt = f.labels;
[N, poolSize] = size(members);

%% Parameter
para_theta = 1;

%% Settings
% Ensemble size M
M = 10;
cntTimes = 10; 
% You can set cntTimes to a greater (or smaller) integer if you want to run
% the algorithms more (or less) times.

% For each run, M base clusterings will be randomly drawn from the pool.
% Each row in bcIdx corresponds to an ensemble of M base clusterings.
bcIdx = zeros(cntTimes, M);
for i = 1:cntTimes
    tmp = randperm(poolSize);
    bcIdx(i,:) = tmp(1:M);
end

%%
% The numbers of clusters.
clsNums = [2:30];
% Scores
nmiScoresBestK_LWEA = zeros(cntTimes, 1);
nmiScoresTrueK_LWEA = zeros(cntTimes, 1);

for runIdx = 1:cntTimes
    disp('**************************************************************');
    disp(['Run ', num2str(runIdx),':']);
    disp('**************************************************************');
    
    %% Construct the ensemble of M base clusterings
    % baseCls is an N x M matrix, each row being a base clustering.
    baseCls = members(:,bcIdx(runIdx,:));
    
    %% Get all clusters in the ensemble
    [bcs, baseClsSegs] = getAllSegs(baseCls);
    
    %% Compute CUI
    disp('Compute CUI ... '); 
 
    CUI = computeCUI(bcs, baseClsSegs, para_theta);
  
    
    %% Compute LWCA
    LWCA= computeLWCA(baseClsSegs, CUI, M);
    

    %% Perform LWEA
    disp('Run the LWEA algorithm ... '); 
    resultsLWEA = runLWEA(LWCA, clsNums);
    % The i-th column in resultsLWEA represents the consensus clustering 
    % with clsNums(i) clusters by LWEA.
    disp('--------------------------------------------------------------');
    
    
    %% Display the clustering results.    
    disp('##############################################################'); 
    scoresLWEA = computeNMI(resultsLWEA,gt);
    
    nmiScoresBestK_LWEA(runIdx) = max(scoresLWEA);
    trueK = numel(unique(gt));
    nmiScoresTrueK_LWEA(runIdx) = scoresLWEA(clsNums==trueK);
    
    
    disp(['The Scores at Run ',num2str(runIdx)]);
    disp('    ---------- The NMI scores w.r.t. best-k: ----------    ');
    disp(['LWEA : ',num2str(nmiScoresBestK_LWEA(runIdx))]);
    
    disp('    ---------- The NMI scores w.r.t. true-k: ----------    ');
    disp(['LWEA : ',num2str(nmiScoresTrueK_LWEA(runIdx))]);
    
    disp('##############################################################'); 
    
   
end

disp('**************************************************************');
%disp(['** Average Performance over ',num2str(cntTimes),' runs on the ',dataName,' dataset **']);
disp(['Data size: ', num2str(N)]);
disp(['Ensemble size: ', num2str(M)]);
disp('   ---------- Average NMI scores w.r.t. best-k: ----------   ');
disp(['LWEA   : ',num2str(mean(nmiScoresBestK_LWEA))]);
disp('   ---------- Average NMI scores w.r.t. true-k: ----------   ');
disp(['LWEA   : ',num2str(mean(nmiScoresTrueK_LWEA))]);
disp('**************************************************************');
disp('**************************************************************');

 toc;

idx = 1;
 for i = 1:size(resultsLWEA,2)
     if min(resultsLWEA(:,i)) > 0
          ress = Clustering8Measure(resultsLWEA(:,i),gt);
          res = ress(end,:);
          Result_LWEA(idx,:) = res;
          idx = idx + 1;
      end
 end
 maxresult = max(Result_LWEA,[],1);







