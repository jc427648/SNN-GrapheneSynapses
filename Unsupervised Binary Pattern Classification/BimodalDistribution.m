%Generate Bimodal Distribution
clc; clear all;
% Define paramaters
num_neurons = 1;
num_patterns = 1;
num_inputs = 25;
epochs = 20;
VthMin = 2.5e-4;
VthMax = 20e-4;
Ve = 0.2e-4;
tau = 0.030e0; % Timing constant
R = 1e3; % Membrane resistance
dt = 0.001; % Timestep (s)
T = 0:dt:5; % Simulation timesteps
% Homeostatic plasticity parameters
Target = 25; % Ideal neuron firing rate
gamma = 35; % Multiplicative constant


refrac = zeros(1,num_neurons);%Refractory period (s).
refPer = 0.1;%Refractory period (s).

%Define shapes of pre and post-synaptic potentials
PreShape = [-0.020225919,0.0782925,1.20527,0.41411,0.093892356,0.0054362];
PostShape = [-0.00953449981,0.0787184394,1.2237,0.26342,0.08011004,0.0386186865];

% PreShape = [-0.015,0.015,-0.85,-1.0,0.005,0.010];
% PostShape = [-0.060,0.060,-1.0,-0.85,0.020,0.010];

%For this code, we want to demonstrate learning based off of STDP window.
% fmax = 20;%Hz
% fmin = 5;%Hz

Ap = 1.5e-7;
Tp = 50e-3;

An = -1.0e-7;
Tn = 110e-3;

% dt = 1e-3;
STDPDelT = -500e-3:dt:500e-3;

%Define an STDP window, which will be used as a lookup.
STDP = zeros(1,length(STDPDelT));
STDP(STDPDelT >= 0) = Ap*exp(-STDPDelT(STDPDelT >= 0)/Tp);
STDP(STDPDelT < 0) = An*exp(STDPDelT(STDPDelT < 0)/Tn);

%Set low and high frequencies
hr = 20;
lr = 5;

figure(3)
hold off
plot(STDPDelT,STDP,'b')
hold on
plot(1/hr*ones(1,length(STDPDelT)),linspace(An,Ap,length(STDPDelT)),'r--')
plot(-1/hr*ones(1,length(STDPDelT)),linspace(An,Ap,length(STDPDelT)),'r--')




% Define maximum and minimum levels of current
Imax = 5e-6;
Imin = 0e-6;
c_p_w = 10; % Current pulse width (index value)
i_p_w = 15; % Inhibition pulse width (index value)

% Define an inhibition current
inhib = 185e-6;



%Setting Init Weight
w= 1;

% Define the input as a matrix (for character recognition).
% Use 10Hz for whitened pixels and 200Hz for darkened pixels. Each row
% represents a column of pixels.
zero = [lr, hr, hr, hr, lr; hr, lr, lr, lr, hr; hr, lr, lr, lr, hr; hr, ...
    lr, lr, lr, hr; lr, hr, hr, hr, lr];
input = zero;

% Pre allocate output vector
memVOut = zeros(num_neurons, length(T), num_neurons);

devices = struct();
devices.w = zeros(num_inputs, num_neurons);
devices.SetPState = ones(num_inputs, num_neurons);
devices.SetNState = zeros(num_inputs, num_neurons);
devices.ResetPState = zeros(num_inputs, num_neurons);
devices.ResetNState = zeros(num_inputs, num_neurons);

% Initialise with random currents
init_w = Imin / 1 + (Imax) / 1 .* rand(5, 5, num_neurons); %Make random currents between zero and Imax.
Vth = VthMin .* ones(num_neurons, 1); % Threshold of neuron (V)

% Have a reshaped initial weight for ease of computation
init_w_rs = reshape(permute(init_w, [2 1 3]), [num_inputs 1 num_neurons]);

% Plot the initial random weights
H2 = figure(1);
bax = tiledlayout(2, 5);
map = redblue(100);

for n = 1:num_neurons
    ax = nexttile;
    imagesc(1:5, 1:5, init_w(:, :, n)');
    pbaspect(ax, [2, 2 2])
    xlim([0.5, 5.5]);
    ylim([0.5, 5.5]);
    view(0, 90);
    set(gca, 'xtick', [])
    set(gca, 'xticklabel', [])
    set(gca, 'ytick', [])
    set(gca, 'yticklabel', [])
    colormap(ax, map);
    title(sprintf('Neuron %d', n), 'fontsize', 18, 'fontname', 'arial')
end

cb = colorbar('FontSize', 16, 'Limits', [Imin, Imax]);
cb.Ticks = -0.025:0.005:0.025;
cb.Layout.Tile = 'east';
set(H2, 'Position', [500 500 1400 550]);

Activity = zeros(num_neurons, length(T), num_neurons);

for run = 1:epochs
    % Define current and membrane potential vectors.
    current = ones(num_inputs,length(T),num_neurons).*init_w_rs;
    Vm = zeros(num_neurons, length(T));
    current_sum = zeros(1, length(T), num_neurons);

    % Generate patterns for simulation
    input_tr = rand(num_inputs, length(T), num_patterns);
    % Develop the current pulses from the input spikes
    %         for i = 1:num_inputs
    %
    %             for j = 1:length(T)
    %
    %                 if input_tr(i, j, z) >= 1
    %
    %                     if j + c_p_w > length(T)
    %                         curr_tr(i, j:end, z) = 1;
    %                     else
    %                         curr_tr(i, j:j + c_p_w, z) = 1;
    %                     end
    %
    %                 end
    %
    %             end
    %
    %         end

    for z = 1:num_patterns
        for i = 1:num_inputs
            input_tr(i, :, z) = input_tr(i, :, z) <= input(i) * dt;
        end
    end

    input_tr = 1 * input_tr; % Convert from logical.
    for z = 1:num_patterns
        %Begin injecting currents into the network.
        for i = 1:length(T)
            % Determine where spiking events occur
            spk = find(Vm(:, i) > Vth);
            refrac(spk) = refrac(spk) + refPer;
            refrac_idx = find(refrac>0);
            No_spk = find(Vm(:, i) < Vth);
            % Update Activity values for each neuron.
            Vm(spk, i + 1) = Ve;
            Activity(spk, i, z) = 1;
            Activity(No_spk, i, z) = 0;
            if not(isempty(spk))



                for n = spk
                    % Calculate the changes in current based on spike times.
                    for k = 1:num_inputs
                        sp_time = find(input_tr(k, :, z) > 0);
                        delta_t = -T(sp_time) + T(i);
                        a = delta_t < 0;
                        b = delta_t > 0;
                        lll = 0;

                        if not(isempty(delta_t))

                            if isempty(delta_t(a))
                                del_t(1) = -85e-3;
                            else
                                del_t(1) = max(delta_t(a));
                            end

                            if isempty(delta_t(b))
                                del_t(2) = 85e-3;
                            else
                                del_t(2) = min(delta_t(b));
                            end

                            curr_change = [0, 0];

                            for p = 1:2
                                del_t_value = del_t(p);

                                if (abs(del_t_value) < 500e-3)
                                    %BELOW IS USING ACTUAL DEVICE, NOT WINDOW.
                                    %shape = [-0.020,0.020,1.0,1.0,0.020,0.020];
                                    %x = -100e-3:1e-3:100e-3;%dt = 0.01e-3
                                    %V_ref = fActP(x - del_t_value,PostShape)-fActP(x,PreShape); % This should be determined using an additional external function- Vref should be the applied voltage to the given device.
                                    %%[integrated_current, w, SetPState, SetNState, ResetPState, ResetNState] = apply_voltage(V_ref, dt, devices.w(k, z), devices.SetPState(k, z), devices.SetNState(k, z), devices.ResetPState(k, z), devices.ResetNState(k, z), params.data(1), params.data(2), params.data(3), params.data(4), params.data(5), params.data(6), params.data(7), params.data(8), params.data(9), params.data(10), params.data(11), params.data(12), params.data(13), params.data(14));
                                    %%[integrated_current, w, SetPState, SetNState, ResetPState, ResetNState] = apply_voltage(V_ref, dt,devices.w(k, z), devices.SetPState(k, z), devices.SetNState(k, z), devices.ResetPState(k, z), devices.ResetNState(k, z),20846,1602,0.122,0.8395,-1.048,1,1.573,0.332,-1.023,-1.484,-216.47,-88.3778,0.000282,0.000963);
                                    %[integrated_current, w, SetPState, SetNState, ResetPState, ResetNState] = apply_voltage(V_ref, 0.01e-3,devices.w(k, z), devices.SetPState(k, z), devices.SetNState(k, z), devices.ResetPState(k, z), devices.ResetNState(k, z),21805,1580,-0.1138,0.799998,-1.1478,1,1.8482,0.4498,-1.033,-1.4849,220.69,64.919,-0.000287,-0.000960);
                                    %devices.w(k, z) = w;
                                    %devices.SetPState(k, z) = SetPState;
                                    %devices.SetNState(k, z) = SetNState;
                                    %devices.ResetPState(k, z) = ResetPState;
                                    %devices.ResetNState(k, z) = ResetNState;
                                    %curr_change(p) = integrated_current;
                                    %                                 end

                                    %BELOW IS USING WINDOW AND NOT DEVICE.
                                    del_t_idx = find((del_t_value-dt<STDPDelT)&(STDPDelT<del_t_value+dt),1,'first');
                                    curr_change(p) = STDP(del_t_idx);
                                end

                            end

                            % Potentiation
                            current(k, i:end, n) = current(k, i:end, n) + curr_change(2);
                            % Depression
                            current(k, (i - round(del_t(1) / dt)):end, n) = current(k, (i - round(del_t(1) / dt)):end, n) +curr_change(1);
                        end
                    end
                end
            end

            % Determine if current has exceeded either limit
            current(current > Imax) = Imax;
            current(current < Imin) = Imin;

            current_sum(:, i:end, :) = sum(current(:, i:end, :), 1);
            csum = squeeze(permute(current_sum, [3 2 1]));

            Vm(No_spk, i + 1) = Vm(No_spk, i) + dt / tau * (Ve - Vm(No_spk, i) + (csum(No_spk, i)) * R);
            Vm(Vm < Ve) = Ve;
            %Ensure neurons that are still within refractory period are not
            %modified by above line of code.
            Vm(refrac_idx,i+1) = Ve;


            %Decrement refractory period
            refrac = refrac - dt;
        end

        %END OF PATTERN PRESENTATION

        % Need to re-initialise the weights.
        for n = 1:num_neurons
            % For each input
            for k = 1:num_inputs

                if not(isempty(find(current(k, :, n) ~= 0, 1, 'last')))
                    init_w_rs(k, :, n) = current(k, find(current(k, :, n) ~= 0, 1, 'last'), n);
                end

            end

        end

    end
    % Update the thresholds based on homeostatic plasticity
    Vth = Vth + dt * gamma .* (sum(Activity, [2, 3]) - Target);
    Vth(Vth < VthMin) = VthMin;
    Vth(Vth > VthMax) = VthMax;

    % Define the length of time to present all inputs
    tvec = linspace(0, T(end) * num_neurons, length(memVOut(n, :)));
    % Calculate the final weights and store them for next run.
    finalW = reshape(init_w_rs, [5 5 num_neurons]);
    % Generate a rastor plot
    H1 = figure(2);
    rastor = reshape(Activity, [num_neurons num_neurons * length(T)]);

    for n = 1:num_neurons
        rastor(n, :) = n * rastor(n, :);
    end

    plot(tvec, rastor, 'bo')
    axis([0 tvec(end) 0.5 num_neurons + 0.5])
    xticks(0:T(end):T(end) * 10);
    xlabel('Time (s)', 'FontSize', 13, 'FontWeight', 'bold')
    ylabel('Neuron', 'FontSize', 13, 'FontWeight', 'bold')
    data = [tvec; rastor]';
    writematrix(data, sprintf('epoch_%d.csv', run));
    % Update the receptive field
    figure(1);
    bax = tiledlayout(2, 5);

    for n = 1:num_neurons
        ax = nexttile;
        imagesc(1:5, 1:5, finalW(:, :, n)',[Imin Imax]);
        pbaspect(ax, [2, 2 2])
        xlim([0.5, 5.5]);
        ylim([0.5, 5.5]);
        view(0, 90);
        set(gca, 'xtick', [])
        set(gca, 'xticklabel', [])
        set(gca, 'ytick', [])
        set(gca, 'yticklabel', [])
        colormap(ax, map);
        title(sprintf('Neuron %d', n), 'fontsize', 18, 'fontname', 'arial');
    end

    cb = colorbar('FontSize', 16, 'Limits', [Imin, Imax]);
    cb.Ticks = -0.025:0.005:0.025;
    cb.Layout.Tile = 'east';
    set(H2, 'Position', [500 500 1400 550]);
end







