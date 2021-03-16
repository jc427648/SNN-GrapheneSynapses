%% Current Summing analysis
%As opposed to raw data analysis, Current summing analysis might provide a
%better STDP window.
clc; clear all;

%Import the data
Data1 = importdata('pentacenesingle200_slow_k2400.txt');
Data2 = importdata('pentacenesingle200_2_slow_k2400.txt');
Data3 = importdata('pentacenesingle200_3_slow_k2400.txt');
Data4 = importdata('pentacenesingle200_4_slow_k2400.txt');

%Ideal STDP window
% IdealCurr = load('IdealSTDP.mat').Interpdata(2,:);


%Define the reference voltages

V_ref = Data3.data(:,3)';
I_ref = Data3.data(:,4)';



dx=0.01;
x=[-80:dx:80];
delta_t=[-80:0.5:80];


%We'll use the data 4 like previous.

%Initial parameters for optimisation
InitParams = [-28.0, 45, 2.2, 1.1 30 30 1.0 0.4];
%Fminsearch suggests parameters should be [-35.43 14.24 2.81 1.50]
%[-30.0, 45, 2.2, 1.1 30 30 1.0 0.4] This vector almost works

options = optimset('MaxFunEvals',20,'MaxIter',20);

% OutParams = fminsearch(@GenSTDP,InitParams,options);

% [err, Imax, delta_t] = GenSTDP(OutParams);
[a, Imax, delta_t] = GenSTDP(InitParams);

%Ideal STDP window
% IdealCurr = load('IdealSTDP.mat').Interpdata(2,:);

%Plot all relevant data.

figure(2)
hold off
% plot(delta_t, Imax,'b:')
xlabel('\DeltaT (ms)')
ylabel('\DeltaI (A)')
% plot(delta_t, IdealCurr,'r')
hold on
plot(-delta_t, Imax,'g')
hold off

figure(3)
plot(V_ref,I_ref)
xlabel('Voltage (V)')
ylabel('Current (A)')

figure(4)
hold off
plot(x,f_actP(x,InitParams))
xlabel('Time (ms)')
ylabel('Voltage (V)')
% hold on
% plot(x,f_actP_lin(x))

%Save the STDP window
out(1,:) = delta_t;
out(2,:) = Imax(end:-1:1);

save('STDP_Window.mat','out')

%Below code was used to determine how the waveform changes for varying
%delta_t values.
%
figure(5)
for k=1:length(delta_t)
    plot(x,InitParams(7)*f_actP_lin(x,InitParams) + InitParams(8)*f_actP_lin(x - delta_t(k),InitParams))
    pause(0.1)
end




%Define the waveforms that will be used. Each row represents a different
%delta_t value.
function [err, I_sum, delta_t] = GenSTDP(params)
%Import the data
Data1 = importdata('pentacenesingle200_slow_k2400.txt');
Data2 = importdata('pentacenesingle200_2_slow_k2400.txt');
Data3 = importdata('pentacenesingle200_3_slow_k2400.txt');
Data4 = importdata('pentacenesingle200_4_slow_k2400.txt');

%Ideal STDP window
% IdealCurr = load('IdealSTDP.mat').Interpdata(2,:);


%Define the reference voltages

V_ref = Data3.data(:,3)';
I_ref = Data3.data(:,4)';



dx=0.01;
x=[-80:dx:80];
delta_t=[-80:0.5:80];
for j = 1:length(delta_t)
    V(j,:) = params(7)*f_actP(x,params(1:6)) + params(8)*f_actP(x + delta_t(j),params(1:6));
    
end
WaveCurr = zeros(length(delta_t),length(x));

%For each voltage in the waveform, need to find corresponding current in
%I-V data.
for j =1:numel(V)
    if V(j)> 5
        V(j) = 5;
    elseif V(j) <-5
        V(j) = -5;
    end
    J = find((V_ref>V(j)-0.05) & (V_ref<V(j) + 0.05),1,'first');
    WaveCurr(j) = I_ref(J);
end

I_sum = dx*trapz(WaveCurr,2)';

% err = sqrt(sum((I_sum - IdealCurr).^2));
err=1;


end






