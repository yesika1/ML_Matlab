
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% ==========================================================================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% =========================================================================


function J = computeCostMulti(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples
J = 0;

h = X * theta;              % predictions of hypothesis on examples
sqrErrors = (h - y).^2; % squared errors

J = 1/(2*m) * sum(sqrErrors);

end
