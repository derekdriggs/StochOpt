% Full gradient of L2 loss in lasso
function grad = grad_lasso(x, W, Y)

    grad = W'*(W*x - Y);

end