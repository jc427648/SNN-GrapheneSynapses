%Generate STDP window
clc;clear all;
% Define shapes of pre and post-synaptic potentials
% PreShape = [-0.0063672136168335,0.0573223105515885,1.0791473330305086,0.8252277014693181,0.0351388170474768,0.0382991785352078];
% PostShape = [-0.0063672136168335,0.0573223105515885,1.0791473330305086,0.8252277014693181,0.0351388170474768,0.0382991785352078];
% PostShape = [-0.3,0.2 1.079147330305086,0.8252277014693181,50e-3,110e-3];
% 
% PreShape = [-0.0573223105515885,0.0063672136168335,0.8252277014693181,1.0791473330305086,0.0382991785352078 ,0.0351388170474768];
% PostShape = [-0.0573223105515885,0.0063672136168335,0.8252277014693181,1.0791473330305086, 0.0382991785352078,0.0351388170474768];
 
% PreShape = [-0.020225919,0.07829250,1.20527,0.414112,0.0801100439,0.0054362];
% PostShape = [-0.009534344,0.0787184,1.2237019,0.26342,0.09389235,0.03861868];



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

figure(1)
plot(delta_t,DelW)

%Debugging
figure(2)
hold off
plot(x,-fActP(x,PreShape),'b')
hold on
plot(x,fActP(x,PostShape),'r')




