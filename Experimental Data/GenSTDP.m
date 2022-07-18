
%Generate STDP window
clc;clear all;
%Define shapes of pre and post-synaptic potentials
PreShape = [-0.0055, 0.0718, 1.1987, 0.5590, 0.0360, 0.0349];
PostShape = [-0.0078, 0.0650, 1.1478, 0.4682, 0.0407, 0.0392];

dt = 1e-3;
x = -100e-3:dt:100e-3;
delta_t = -80e-3:dt:80e-3;
i = 1;
k = 1;
z=1;

devices = struct();
devices.w = zeros(1, 1);
devices.SetPState = zeros(1, 1);
devices.SetNState = zeros(1, 1);
devices.ResetPState = zeros(1, 1);
devices.ResetNState = zeros(1, 1);



for del_t_value = delta_t
    devices.w(1) = 1;
    V_ref = fActP(x - del_t_value,PostShape)-fActP(x,PreShape);
    [integrated_current, w, SetPState, SetNState, ResetPState, ResetNState] = apply_voltage(V_ref, dt,devices.w(k, z), devices.SetPState(k, z), devices.SetNState(k, z), devices.ResetPState(k, z), devices.ResetNState(k, z),21805,1580,-0.1138,0.799998,-1.1478,1,1.8482,0.4498,-1.033,-1.4849,220.69,64.919,-0.000287,-0.000960);
    DelW(i) = integrated_current;
    i = i+1;
end
%Plot STDP window
figure(1)
plot(delta_t,-DelW)

%Plot Action potential shapes
figure(2)
hold off
plot(x,-fActP(x,PreShape),'b')
hold on
plot(x,fActP(x,PostShape),'r')