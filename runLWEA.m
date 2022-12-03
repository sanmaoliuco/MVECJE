function resultsLWEA = runLWEA(S, ks)
% Input: the co-association matrix
%        and the numbers of clusters.
% Output: clustering results by LWEA-AL.

N = size(S,1);

d = stod2(S); clear S %convert similarity matrix to distance vector
% average linkage 
Zal = linkage(d,'average'); clear d

resultsLWEA = zeros(N, numel(ks));
disp('.');
 for i = 1:numel(ks)
     K = ks(i);
    disp(['Obtain ',num2str(K),' clusters by LWEA.']); 
    tic;
    resultsLWEA(:,i) = cluster(Zal,'maxclust',K);
    toc;
 end
disp('.');
 
function d = stod2(S)
%==========================================================================
% FUNCTION: d = stod(S)
% DESCRIPTION: This function converts similarity values to distance values
%              and change matrix's format from square to vector (input
%              format for linkage function)
%
% INPUTS:   S = N-by-N similarity matrix
%
% OUTPUT:   d = a distance vector
%==========================================================================
% copyright (c) 2010 Iam-on & Garrett
%==========================================================================

N = size(S,1);
s = zeros(1,N*(N-1)/2);
nextIdx = 1;
for a = 1:N-1 %change matrix's format to be input of linkage fn
    s(nextIdx:nextIdx+(N-a-1)) = S(a,[a+1:end]);
    nextIdx = nextIdx + N - a;
end
d = 1 - s; %compute distance (d = 1-sim)