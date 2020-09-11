function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
reg1 = 0;
reg11 = 0;
reg2 = 0;
reg22 = 0;
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X1 = [ones(m,1) X];
z2 = X1 * Theta1';
a2 = sigmoid(z2);
A2 = [ones(size(a2,1),1) a2];
z3 = A2 * Theta2';
A3 = sigmoid(z3);

Y = zeros(size(y,1), max(y));

for p = 1:m
Y(p,y(p)) = 1 ;
endfor

for i = 1:size(A3,1)
  
  for k = 1:size(A3,2)
    
    J1 = (-1/m)*((Y(i,k) * log(A3(i,k))) + ((1-Y(i,k)) * (log(1-A3(i,k)))));
    
    J = J + J1 ;
  endfor
   
endfor

for i = 1 : (hidden_layer_size)
  for j = 2 : (input_layer_size+1)
    reg11 = (Theta1(i,j))^2;
  
  reg1 = reg1 + reg11;
  endfor
  
endfor

##reg1

for i = 1 : (num_labels)

  for j = 2 : (hidden_layer_size+1)
    
    reg22 = (Theta2(i,j))^2;
  
  reg2 = reg2 + reg22;

  endfor
  
endfor

##reg2

reg =  (lambda/(2*m)) * (reg1 +reg2);

J = J + reg;

%====================================================
%       BackPropagation Algorithm
%====================================================
## size_theta1 = size(Theta1)
## size_theta2 = size(Theta2)
## size_a3 = size(a3)
## size_a2 = size(a2)

 del3 = A3 - Y;
 del2 = (del3 * (Theta2(:,2:end))) .* sigmoidGradient(z2);
 Delta2 = del3' * A2;
 Delta1 = del2' * X1;
 Theta2_grad = Delta2 / m ;
 Theta1_grad = Delta1 / m ;
 Theta2_grad(:, 2: end) = Theta2_grad(:,2:end) + ((lambda/m) * Theta2(:,2:end));
 Theta1_grad(:, 2: end) = Theta1_grad(:,2:end) + ((lambda/m) * Theta1(:,2:end));
  
  
## size_Delta_2 = size(Delta_2)
## size_Delta_1 = size(Delta_1)
##   Theta2_grad = delta_2(2:end,:) / m ;
##   Theta1_grad = delta_1(2:end,:) / m ;
## size_Theta2_grad = size(Theta2_grad)
## size_Theta1_grad = size(Theta1_grad)

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
