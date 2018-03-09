
%GRADIENTDESCENT 
% Performs gradient descent to learn theta

%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) 
%   updates theta by taking num_iters gradient steps with learning rate alpha


    % ============================================================
    % Instructions: Perform a single gradient step on the parameter vector theta. 
    %               
    % 'm' is the number of training examples (the rows of X), 
    % 'n' is the number of features (the columns of X). 
    % 'n' is also the size of the theta vector (n x 1).
    % The hypothesis is a vector, formed by multiplying the X matrix and the theta vector.
    % X has size (m x n), and theta is (n x 1), so the product is (m x 1), same size as 'y'
    % The "errors vector" is the difference between the 'h' vector and the 'y' vector.
    % The change in theta (the "gradient") is the sum of the product of X and the "errors vector", 
    % scaled by alpha and 1/m.
    % Since X is (m x n), and the error vector is (m x 1), 
    % and the result you want is the same size as theta (which is (n x 1), 
    % you need to transpose X before you can multiply it by the error vector.
    %
    % The vector multiplication automatically includes calculating the sum of the products.
    % theta = theta - theta_change;
    %
    % ============================================================


function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    h = X * theta;
    % mxn * nx1 = mx1
    
    stderror = h - y;
    % mx1

    % The change in theta (the "gradient")
    %theta = theta - (alpha/m) * (stderror' * X)';
    % theta: nx1, stderror':1xm * X:mxn = 1xn ' =nx1

    % OR another way to do it: by multiplying every element in error by X'
    % theta = theta - (alpha/m) * (X'*X*theta - X'*y);

    % OR another way
    delta = X' * stderror;
    theta = theta - (alpha / m) * delta;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end


end
