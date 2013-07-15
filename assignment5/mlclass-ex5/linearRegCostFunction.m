function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

theta0 = theta(1,:);
theta1 = theta(2:end,:);
J = sum((X * theta - y).^2) / (2 * m) + lambda * sum(theta1.^2) / (2 * m);

X0 = X(:,1);
X1 = X(:,2:end);

% For J = 0
grad1 = X0' * (X * theta - y) / m;
% For j >= 1
grad2 = X1' * (X * theta - y) / m + lambda * theta1 / m;

grad = [grad1; grad2];



% =========================================================================

grad = grad(:);

end
