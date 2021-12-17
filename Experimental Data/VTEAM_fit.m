clc; clear all;

data = importdata('pentacenesingle200_4_slow_k2400.txt');
%Generate reference data
t = linspace(data.data(1,2),data.data(end,2),length(data.data(:,2)));%Mohan's data %seconds
V_ref = data.data(:,3)';%Insert Mohan's data;
I_ref = data.data(:,4)'; %Insert Mohan's data here;

figure(1);
plot(V_ref, I_ref);
hold on;

% Calculate euclidean norms
V_ref_norm = sqrt(sum(V_ref.^2));
I_ref_norm = sqrt(sum(I_ref.^2));

% Define parameters
alpha_off = 20;
alpha_on = 3;
V_on = -1.1437;
V_off = 1.1699;
R_off = 5.0553e7;
R_on = 1.5238e3;
w_on = 0;
w_off = 10;
k_off = 1;
k_on = -30;

% Define the initial guess parameters
parameters(1) = alpha_off;
parameters(2) = alpha_on;
parameters(3) = V_on;
parameters(4) = V_off;
parameters(5) = R_off;
parameters(6) = R_on;
parameters(7) = w_on;
parameters(8) = w_off;
cparameters(1) = k_off;
cparameters(2) = k_on;

% Paramaterise RMS_error functions for optimisation
f = @(x) RMS_err(parameters,x,V_ref,I_ref,t);

% Calculate optimisation
X = fminsearch(f, cparameters);
error = RMS_err(parameters,X,V_ref,I_ref,t); %This is a seperate function file

% Calculate dw/dt and w
w = zeros(1,length(V_ref));
dw_dt = zeros(1,length(V_ref));
% Calculate the voltage and current characteristics
for i = 1:length(V_ref)
    if (V_ref(i)< V_on)
        dw_dt(i) = k_on*(V_ref(i)/V_on -1)^alpha_on;   
    elseif(V_on < V_ref(i) && V_ref(i) < V_off)
        dw_dt(i) = 0;      
    elseif(V_off < V_ref(i))
        dw_dt(i) = k_off*(V_ref(i)/V_off -1)^alpha_off;     
    end
    % Calculate the w parameter
    if i ==1
        w(i) = 0;
    else
        w(i) = w(i-1)+ (t(2)-t(1))*trapz(dw_dt(i-1:i));
    end   
    % Bound the state parameter
    if w(i)<w_on
        w(i) = w_on;
    elseif w(i)>w_off
        w(i) = w_off;
    end  
end
R = R_on + (R_off - R_on)/(w_off-w_on).*(w-w_on);
V = V_ref;
I = V./R;

%Plot VTEAM model onto figure with data as well.
plot(V,I,'r--');
hold off;

figure(2);
plot(V, I, 'r--');

