% Returns a storage-optimal representation of the gradient of l2 loss:
%   a scalar c such that W(i,:)*c is the gradient of the i^th function.
% Must run this code with storage-optimal SAGA.
function grad = igrad_l2_SAGA_Lin(x, i, W, Y)

    grad = (W(i,:)*x - Y(i));

end