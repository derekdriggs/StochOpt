% Returns a storage-optimal representation of the gradient:
%   a scalar c such that W(i,:)*c is the gradient of the i^th function.
% Must run this code with storage-optimal SAGA.
function grad = igrad_nnPCA_SAGA_Lin(x, i, W)
    
    m    = size(W,1);
    
    grad = -W(i,:)*x/m;

end