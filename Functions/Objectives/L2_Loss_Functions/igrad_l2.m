% Gradient of L2 loss
function grad = igrad_l2(x, i, W, Y)

    grad = W(i,:)'*(W(i,:)*x - Y(i));

end