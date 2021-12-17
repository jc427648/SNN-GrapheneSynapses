clc; clear all;

data = importdata('pentacenesingle200_4_slow_k2400.txt');

%Ideal STDP window
IdealCurr = load('IdealSTDP.mat').Interpdata(2,:);
V_ref = data.data(:,3)';
I_ref = data.data(:,4)';
dx = 0.01;
x = [-80:dx:80];
delta_t = [-80:0.5:80];

%Initial parameters for optimisation
InitParams = [-25.0, 25, 0.75, 1.1 5 30 1.0 1.0];
options = optimset('MaxFunEvals',20,'MaxIter',20);
[a, Initial, delta_t] = GenSTDP(InitParams);

%Ideal STDP window
IdealCurr = load('IdealSTDP.mat').Interpdata(2,:);

%Plot all relevant data.
figure(2)
% plot(delta_t, Imax,'b:')
xlabel('\DeltaT (ms)')
ylabel('\DeltaI (A)')
hold off
plot(delta_t, IdealCurr,'r')
hold on
plot(delta_t, Initial,'g')
hold off

figure(3)
plot(V_ref,I_ref)
xlabel('Voltage (V)')
ylabel('Current (A)')

figure(4)
hold off
plot(x,f_actP(x,InitParams(1:6)))
xlabel('Time (ms)')
ylabel('Voltage (V)')

%Save the STDP window
out(1,:) = delta_t;
out(2,:) = Initial;%Imax;
save('STDP_Window2.mat','out')

function [err, Imax, delta_t] = GenSTDP(params)
data = importdata('pentacenesingle200_3_slow_k2400.txt');

%Ideal STDP window
IdealCurr = load('IdealSTDP.mat').Interpdata(2,:);
V_ref = data.data(:,3)';
I_ref = data.data(:,4)';
dx=0.01;
x=[-80:dx:80];
delta_t=[-80:0.5:80];
for j = 1:length(delta_t)
    V(j,:) = params(7)*f_actP(x,params(1:6)) - params(8)*f_actP(x + delta_t(j),params(1:6));
end

%Find the maximum absolute voltage and the index at which it is located.
[V_max,I] = max(abs(V),[],2);

%Apply the interpolation to determine the current
for n = 1:length(V_max)
    %Use the index to find actual maximum voltage (to account for negatives.)
    V_max(n) = V(n,I(n));
    J = find((V_ref>V_max(n)-0.05) & (V_ref<V_max(n) +0.05),1,'last');
    I_max(n) = I_ref(J);
end
err = sqrt(sum((I_max - IdealCurr).^2));
Imax = I_max;
end






