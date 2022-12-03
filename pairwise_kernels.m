function K = pairwise_kernels(X,X2,type,gamma)
%PAIRWISE_KERNELS 此处显示有关此函数的摘要
%   此处显示详细说明
if strcmp(type,'rbf')
    r2 = repmat(sum(X.^2,2),1,size(X2,1)) + repmat(sum(X2.^2,2),1,size(X,1))'-2*X*X2';
    K = exp(-gamma*r2);
end    

if strcmp(type,'lin')
    K = X*X2';    
end

% n1s = sum(prof_corr_0169,1);