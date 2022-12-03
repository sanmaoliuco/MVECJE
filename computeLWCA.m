function LWCA=computeLWCA(baseClsSegs,ECI,M)
% Get locally weighted co-association matrix

baseClsSegs = baseClsSegs';
N = size(baseClsSegs,1);

% LWCA = (baseClsSegs.*repmat(ECI',N,1)) * baseClsSegs' / M;
LWCA = (bsxfun(@times, baseClsSegs, ECI')) * baseClsSegs' / M;

LWCA = LWCA-diag(diag(LWCA))+eye(N);