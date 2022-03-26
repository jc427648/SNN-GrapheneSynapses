clc; clear all;
rng('default');
rng(101);

% Define paramaters
num_neurons = 10;
epochs = 100;
VthMin = 10e-3;
VthMax = 0.029254;
Ve = 0.0095464;
tau = 0.0015744;
R = 499.12;
dt = 2e-4;
T = 0:dt:0.1;
% Homeostatic plasticity parameters
Target = 16.121;
gamma = 0.029254;

% Define maximum and minimum levels of current
Imax = 10e-6;
Imin = -10e-6;
c_p_w = 10; % Current pulse width (index value)
i_p_w = 27; % Inhibition pulse width (index value)

% Define an inhibition current
inhib = 6.0241e-05;

%Set low and high frequencies
hr = 200;
lr = 5;

% Define the input as a matrix (for character recognition).
% Use 10Hz for whitened pixels and 200Hz for darkened pixels. Each row
% represents a column of pixels.
zero = [lr, hr, hr, hr, lr; hr, lr, lr, lr, hr; hr, lr, lr, lr, hr; hr, ...
    lr, lr, lr, hr; lr, hr, hr, hr, lr];
one = [lr, lr, lr, lr, lr; lr, hr, lr, lr, lr; hr, hr, hr, hr, hr; lr, ...
    lr, lr, lr, lr; lr, lr, lr, lr, lr];
two = [lr, hr, lr, lr, hr; hr, lr, lr, lr, hr; hr, lr, lr, hr, hr; hr, ...
    lr, hr, lr, hr; lr, hr, lr, lr, hr];
three = [lr, lr, lr, lr, lr; hr, lr, lr, lr, hr; hr, lr, hr, lr, hr; ...
    hr, lr, hr, lr, hr; lr, hr, lr, hr, lr];
four = [lr, lr, hr, hr, lr; lr, hr, lr, hr, lr; hr, lr, lr, hr, lr; hr, ...
    hr, hr, hr, hr; lr, lr, lr, hr, lr];
five = [hr, hr, hr, lr, hr; hr, lr, hr, lr, hr; hr, lr, hr, lr, hr; hr, ...
    lr, hr, lr, hr; hr, lr, lr, hr, lr];
six = [lr, hr, hr, hr, lr; hr, lr, hr, lr, hr; hr, lr, hr, lr, hr; hr, ...
    lr, hr, lr, hr; lr, lr, lr, hr, lr];
seven = [hr, lr, lr, lr, lr; hr, lr, lr, lr, lr; hr, lr, hr, hr, hr; ...
    hr, hr, lr, lr, lr; hr, lr, lr, lr, lr];
eight = [lr, hr, lr, hr, lr; hr, lr, hr, lr, hr; hr, lr, hr, lr, hr; ...
    hr, lr, hr, lr, hr; lr, hr, lr, hr, lr];
nine = [lr, lr, lr, lr, lr; hr, hr, hr, lr, lr; hr, lr, hr, lr, lr; hr, ...
    lr, hr, lr, lr; hr, hr, hr, hr, hr];

% Define the number of pixels
num_inputs = numel(nine);

% Pre allocate output vector
memVOut = zeros(num_neurons, length(T), num_neurons);

% Initialise with random currents
init_w = Imin / 1 +(Imax - Imin) / 1. *rand(5, 5, num_neurons);
Vth = VthMin .* ones(num_neurons, 1); % Threshold of neuron (V)

% Have a reshaped initial weight for ease of computation
init_w_rs = reshape(permute(init_w, [2 1 3]), [num_inputs 1 num_neurons]);

% Plot the initial random weights
H2 = figure(1);
bax = tiledlayout(2,5);
map = redblue(100);
for n = 1:num_neurons
    ax = nexttile;
    imagesc(1:5, 1:5, init_w(:, :, n)');
    pbaspect(ax,[2, 2 2])
    xlim([0.5, 5.5]);
    ylim([0.5, 5.5]);
    view(0, 90);
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    colormap(ax, map);
    title(sprintf('Neuron %d',n), 'fontsize', 18, 'fontname', 'arial')
end
cb = colorbar('FontSize', 16, 'Limits', [Imin, Imax]);
cb.Ticks = -0.025:0.005:0.025;
cb.Layout.Tile = 'east';
set(H2,'Position',[500 500 1400 550]);

% Load the STDP data
STDP_data = importdata("current.txt");
Activity = zeros(num_neurons,length(T),num_neurons);

for run = 1:epochs
    % Create a random sequence for inputs
    inpu = 1:num_neurons;
    [x,y] = sort(rand(1,num_neurons));
    inpu = inpu(y);
    % Define current and membrane potential vectors.
    current = zeros(num_inputs,length(T),num_neurons);
    Vm = zeros(num_neurons,length(T));
    current_sum = zeros(1,length(T),num_neurons);
    z=0;
    % Generate patterns for simulation
    input_tr = rand(num_inputs,length(T),num_neurons);
    curr_tr = zeros(num_inputs,length(T),num_neurons);
    for inp = inpu
        z = z+1; 
        switch inp
            case 1
                input = one;
            case 2
                input = two;
            case 3
                input = three;
            case 4
                input = four;
            case 5
                input = five;
            case 6
                input = six;
            case 7
                input = seven;
            case 8
                input = eight;
            case 9
                input = nine;
            case 10
                input = zero;
        end
        for i = 1:num_inputs
            input_tr(i,:,z) = input_tr(i,:,z)<=input(i)*dt;
        end
        input_tr = 1*input_tr; % Convert from logical.      
        % Develop the current pulses from the input spikes
        for i = 1:num_inputs
            for j =1:length(T)
                if input_tr(i,j,z)>= 1
                    if j+c_p_w>length(T)
                        curr_tr(i,j:end,z) = 1;
                    else
                        curr_tr(i,j:j+c_p_w,z) = 1;
                    end
                end
            end
        end
    end
    % Execute neuron simulation for single pattern
    inhibition = zeros(num_neurons, length(T), num_neurons);
    for z = 1:num_neurons
        % Generate the current waveforms for each neuron
        current = curr_tr(:,:,z).*init_w_rs;
        ind = find(current==0);
        for i = 1:length(T)
            % Determine where spiking events occur
            spk = find(Vm(:,i)>Vth);
            No_spk = find(Vm(:,i)<Vth);
            % Update Activity values for each neuron.
            Vm(spk,i+1) = Ve;
            Activity(spk,i,z) = 1;
            Activity(No_spk,i,z) = 0;
            if not(isempty(spk))
                for n = spk 
                    % Calculate the changes in current based on spike times.
                    for k = 1:num_inputs
                        sp_time = find(input_tr(k,:,z)>0);
                        delta_t = -T(sp_time) + T(i);
                        a = delta_t<0;
                        b = delta_t>0; 
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
                                if (abs(del_t(p)) <= 85e-3)
                                    curr_change(p) = STDP_data(2, find(STDP_data(1, :) == interp1(STDP_data(1, :),STDP_data(1, :),del_t(p), 'nearest'), 1));
                                end
                            end
                            % Potentiation
                            current(k,i:end,n) = current(k,i:end,n) + curr_change(2);
                            % Depression
                            current(k,(i-round(del_t(1)/dt)):end,n) = current(k,(i-round(del_t(1)/dt)):end,n) +curr_change(1); 
                            % Inhibition
                            for ne = 1:num_neurons
                                if (n ~= ne) & i+i_p_w>length(T)
                                    inhibition(ne,i:end,z) = -inhib;
                                elseif (n~=ne)
                                    inhibition(ne,i:i+i_p_w,z) = -inhib;
                                end
                            end
                        end
                    end 
                end
            end 
            % Determine if current has exceeded either limit
            current(current>Imax) = Imax;
            current(current<Imin) = Imin;
            % With knowledge of where the indexes of zero are, can reset
            current(ind) = 0;
            current_sum(:,i:end,:) = sum(current(:,i:end,:),1);
            csum = squeeze(permute(current_sum, [3 2 1]));
            Vm(No_spk,i+1) = Vm(No_spk,i) + dt/tau*(Ve-Vm(No_spk,i) + (csum(No_spk,i)+inhibition(No_spk,i,z))*R);
            Vm(Vm<Ve) = Ve;
            i = i + 1;   
        end
        % Need to re-initialise the weights.
        for n = 1:num_neurons
            % For each input
            for k = 1:num_inputs
                if not(isempty(find(current(k,:,n)~=0,1,'last')))
                    init_w_rs(k,:,n) = current(k,find(current(k,:,n)~=0,1,'last'),n);
                end
            end
        end
    end
    % Update the thresholds based on homeostatic plasticity
    Vth = Vth + dt*gamma.*(sum(Activity,[2,3])-Target);
    Vth(Vth<VthMin)=VthMin;
    Vth(Vth>VthMax) = VthMax;
    % Define the length of time to present all inputs
    tvec = linspace(0,T(end)*num_neurons,length(memVOut(n,:)));
    % Calculate the final weights and store them for next run.
    finalW = reshape(init_w_rs, [5 5 num_neurons]);
    % Generate a rastor plot
    H1 = figure(2);
    rastor  = reshape(Activity,[num_neurons num_neurons*length(T)]);
    for n = 1:num_neurons
        rastor(n,:) = n*rastor(n,:);
    end
    plot(tvec,rastor,'bo')
    axis([0 tvec(end) 0.5 num_neurons+0.5])
    xticks(0:T(end):T(end)*10);
    xlabel('Time (s)','FontSize',13,'FontWeight','bold')
    ylabel('Neuron','FontSize',13,'FontWeight','bold')
    data = [tvec; rastor]';
    writematrix(data, sprintf('epoch_%d.csv', run));
    % Update the receptive field
    fig = figure(1);
    bax = tiledlayout(2,5);
    for n = 1:num_neurons
        ax = nexttile;
        imagesc(1:5, 1:5, finalW(:, :, n)');
        pbaspect(ax,[2, 2 2])
        xlim([0.5, 5.5]);
        ylim([0.5, 5.5]);
        view(0, 90);
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        colormap(ax, map);
        title(sprintf('Neuron %d',n), 'fontsize', 18, 'fontname', 'arial');
    end
    cb = colorbar('FontSize', 16, 'Limits', [Imin, Imax]);
    cb.Layout.Tile = 'east';
    set(H2,'Position',[500 500 1400 550]);
    saveas(fig, sprintf('epoch_%d', run), 'svg');
end
