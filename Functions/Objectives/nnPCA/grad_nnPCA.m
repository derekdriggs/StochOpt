function grad = grad_nnPCA(x, W)

    grad = -W'*(W*x);

end