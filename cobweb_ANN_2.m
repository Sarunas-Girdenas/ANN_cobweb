% In this script we simulate cobweb model with ANN with sigmoid activation function
% ANN learns parameters alpha2 and beta2 and converges to REE values
% References used for this code are Evans & Honkapohja (2001) and notes from briandolhansky.com
% code is written by Sarunas Girdenas, August, 2014, sg325@exeter.ac.uk
% note that this file requires gridfit.m script for 3D plotting. It could be obtained from Matlab Central

% Firstly, we load the same values of exogenous variable g 

g_var  = load('w_lag.txt');       % g (exogenous) variable
Shocks = load('Shocks_var.txt');  % shocks

% Specify the parameters

alpha_1  = 5;                      % price equation intercept
c        = -0.5;                   % beta_0 + beta_1 in Economic Model
sigma    = 0.5;                    % variance of shock
time     = 300; 				   % simulation horizon
a        = zeros(time,1); 		   % expected price level
p        = zeros(time,1);          % actual price level
alpha_2  = zeros(time,1);		   % parameter alpha_2 in Economic Model
beta_2   = zeros(time,1);          % parameter beta_2 in Economic Model
delta    = 1; 					   % define delta next to w in price equation
g_lag    = zeros(length(g_var),1); % exogenous variable g, lagged by 1, g(t-1)

% create the g_lag now

for z = 2:length(g_var)
	g_lag(z,1) = g_var(z-1);
end

% REE values

a2 = alpha_1/(1-c);
b2 = delta/(1-c);

% initialize OLS parameters

beta_2_initial 	= 2; 
alpha_2_initial = 1; 

% initialize p and a variables of the model

a(1) = alpha_2_initial+beta_2_initial*g_lag(1);
p(1) = alpha_1+c*a(1)+delta*g_lag(1)+sigma*Shocks(1,1); %Initial values of p and a

% Neural Network initialization

max_iter = 100;               % no of network iterations
alpha_n  = 0.01;              % gradient descent learning rate, calibrate it to change convergence properties of ANN
w        = zeros(1,2);        % weights for Neural Network 
grad_t_h = zeros(time,1);     % store network activation function
h_hist   = zeros(max_iter,1); % save loss function

% simulate the model

for i = 2:time

	sim_no = i
	
	% macroeconomic model

	a(i) = alpha_2(i-1)+beta_2(i-1)*g_var(i-1);
	p(i) = alpha_1+c*a(i)+delta*g_var(i-1)+sigma*Shocks(i,1);

	% update parameters beta_2 and alpha_2 using Neural Network

		for k = 1:max_iter

			X = [ones(i,1) g_lag(1:i,1)]; % create X variable for Neural Network
			Y = p(1:i,1);                 % create Y variable for Neural Network
			grad_t = zeros(1,2);          % initialize gradient descent

				for t = 1:i               % loop over each observation in sample

					x_t = X(t,:);
					y_t = Y(t);

					% compute hypothesis

					h = w*x_t' - y_t;                                               % compute hypothesis for each observation
					grad_t = grad_t + 2*h*x_t*exp(-w*x_t')/((1+exp(-w*x_t'))^(2));  % sum hypothesis (we use sigmoid function here)
					B = 1/(1+exp(-w*x_t'));                                         % store activation function
					
				end

			w = w - (1/t)*grad_t;    % update gradient descent

			h_hist(k,1) = sum(h.^2); % save loss function
		end

		% update economic model estimates

		alpha_2(i)  = w(1,1);
		beta_2(i)   = w(1,2);

	% store variables

	grad_t_h(i,:) = B;      % storing activation function and two weights from NN
	
end


% plot results

% model

figure;
subplot(2,2,1)
plot(p,'k')
xlabel('Simulation Horizon')
title('Actual Price Level')
subplot(2,2,2)
plot(a,'k')
xlabel('Simulation Horizon')
title('Expected Price Level')
subplot(2,2,3)
plot(alpha_2,'k')
hold;
refline([0 a2])
xlabel('Simulation Horizon')
title('Parameter \alpha_2')
subplot(2,2,4)
plot(beta_2,'k')
hold;
refline([0 b2])
xlabel('Simulation Horizon')
title('Parameter \beta_2')
legend('Simulated Series from ANN', 'REE Value')

% activation function

figure;
plot(grad_t_h,'k');
xlabel('Simulation Horizon')

% plot 3D loss function, requires gridfit.m script

x1 = alpha_2(1:max_iter);
y1 = beta_2(1:max_iter);
z  = -h_hist;
gx = min(x1):0.02:max(x1);
gy = min(y1):0.02:max(y1);
g  = gridfit(x1,y1,z,gx,gy);
figure;
surf(gx,gy,g);

% setting axis

xlabel('Parameter \alpha_2');
ylabel('Parameter \beta_2');
zlabel('Loss Function');

% rotate title of axis

set(get(gca,'xlabel'),'rotation',15); 
set(get(gca,'ylabel'),'rotation',-25); 
set(get(gca,'zlabel'),'rotation',90); 
colormap(hot);