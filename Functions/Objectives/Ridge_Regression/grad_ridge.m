function grad = grad_ridge(x, W, Y, lambda)

    grad = W'*(W*x - Y) + lambda * x;

end