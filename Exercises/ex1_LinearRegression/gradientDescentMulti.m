
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha


    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    % ============================================================



function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    h = X * theta;
    % mxn * nx1 = mx1
    
    stderror = h - y;
    % mx1

    % The change in theta (the "gradient")
    % theta = theta - (alpha/m) * (stderror' * X)';
    % theta: nx1, stderror':1xm * X:mxn = 1xn ' =nx1
    
    % OR another way to do it: by multiplying every element in error by X'
    % theta = theta - (alpha/m) * (X'*X*theta - X'*y);

    % OR another way
    delta = X' * stderror;
    theta = theta - (alpha / m) * delta;

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end


end

