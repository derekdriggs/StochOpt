% Full gradient of L2 loss
function grad = grad_l2(x, W, Y)

    grad = W'*(W*x - Y);

end