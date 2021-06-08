%% Homeostatic plasticity

%This code is the same as char_recog_network but with homeostatic
%plasticity added.

clc;clear all;

numPatterns = 10;
numNeurons = 10;
%Define paramaters
VthMin = 0.25;
VthMax = 20;
Ve = 0.2;
tau = 0.020;%Timing constant
R = 5e2;%Membrane resistance
dt = 0.0002; %Difference in time 0.2ms
T = 0:dt:0.1;%Simulation time
Tlength = length(T);%Useful variable.

%Define homeostatic plasticity parameters
%For queriloz article, it says activity is simply the number of times it
%should fire in an extended period of time. I'm going to define a rate that
%is suitable and then using the time define the appropriate target
%activity.
Target = 25/0.1*T(end);%30*dt/T(end)/num_neurons;%0.010%Ideal neuron firing rate
gamma = 1e-2;%Multiplicative constant
Activity = zeros(numPatterns,length(T),numPatterns);

%Define maximum and minimum levels of current
Imax = 25e-3;
Imin = -25e-3;

c_p_w = 10;%Current pulse width (index value)
i_p_w = 15;%Inhibition pulse width (index value)

%Define an inhibition current
inhib = 185e-3;%185e-3;

%Set low and high frequencies
hr = 200;
lr = 20;

%define the input as a matrix (for character recognition).
%use 10Hz for whitened pixels and 200Hz for darkened pixels. Each row
%represents a column of pixels.
zero = [lr,hr,hr,hr,lr; hr,lr,lr,lr,hr; hr,lr,lr,lr,hr; hr,lr,lr,lr,hr;...
    lr,hr,hr,hr,lr];
one = [lr,lr,lr,lr,lr; lr,hr,lr,lr,lr; hr,hr,hr,hr,hr; lr,lr,lr,lr,lr;...
    lr,lr,lr,lr,lr];
two = [lr,hr,lr,lr,hr; hr,lr,lr,lr,hr; hr,lr,lr,hr,hr; hr,lr,hr,lr,hr;...
    lr,hr,lr,lr,hr];
three = [lr,lr,lr,lr,lr; hr,lr,lr,lr,hr; hr,lr,hr,lr,hr; hr,lr,hr,lr,hr;...
    lr,hr,lr,hr,lr];
four = [lr,lr,hr,hr,lr; lr,hr,lr,hr,lr; hr,lr,lr,hr,lr; hr,hr,hr,hr,hr;...
    lr,lr,lr,hr,lr];
five = [hr,hr,hr,lr,hr; hr,lr,hr,lr,hr; hr,lr,hr,lr,hr; hr,lr,hr,lr,hr;...
    hr,lr,lr,hr,lr];
six = [lr,hr,hr,hr,hr; hr,lr,hr,lr,hr; hr,lr,hr,lr,hr; lr,lr,hr,lr,hr;...
    lr,lr,lr,hr,lr];
seven = [hr,lr,lr,lr,lr; hr,lr,lr,lr,lr; hr,lr,hr,hr,hr; hr,hr,lr,lr,lr;...
    hr,lr,lr,lr,lr];
eight = [lr,hr,lr,hr,lr; hr,lr,hr,lr,hr; hr,lr,hr,lr,hr; hr,lr,hr,lr,hr;...
    lr,hr,lr,hr,lr];
nine = [lr,lr,lr,lr,lr; hr,hr,hr,lr,lr; hr,lr,hr,lr,lr; hr,lr,hr,lr,lr;...
    hr,hr,hr,hr,hr];

%Create two, non-overlapping patterns to test the network.
test1 = [hr,hr,hr,hr,hr; hr,hr,hr,hr,hr; lr,lr,lr,lr,lr; lr,lr,lr,lr,lr;...
    lr,lr,lr,lr,lr]; %all left hand side columns
test2 = [lr,lr,lr,lr,lr; lr,lr,lr,lr,lr; lr,lr,lr,lr,lr; hr,hr,hr,hr,hr;...
    hr,hr,hr,hr,hr]; %all right hand side columns

%Define the number of pixels
numPixels = numel(nine);


%Pre allocate output vector
memVOut = zeros(numPatterns,Tlength,numPatterns);

%Either initialise with random currents or use the previously stored values
%of current to re-update network
x = input('Start over?','s');
if x == 'y'
    init_w =   Imin/1 +(Imax-Imin)/1.*rand(5,5,numPatterns);
    Vth = 1.5.*ones(numPatterns,1);%Volts, threshold of neuron.
elseif x == 'n'
    init_w = load('PreviousWeights.mat').finalW;
    Vth = load('Vth.mat').Vth;
end

%Have a reshaped initial weight for ease of computation
init_w_rs = reshape(permute(init_w, [2 1 3]) ,[numPixels 1 numPatterns]);


%plot the initial random weights
figure(14)
sgtitle('Initial weights')

map = redblue(100);
for n = 1:numPatterns
    subplot(2,5,n)
    heatmap(init_w(:,:,n), 'Colormap',map,'Colorlimit',[Imin Imax]);
    title(sprintf('Neuron %d',n))
end

%Debug
% delta_t_out = zeros(num_inputs*num_neurons,length(T)*num_neurons);
% curr_change_out = zeros(num_inputs,length(T)*num_neurons,num_neurons);
%Introduce curr change out as ell.

%Load the STDP data
STDP_data = load('STDP_Window.mat');
Interpdata = STDP_data.out;

for epochs = 1:20
    %     Activity = zeros(num_neurons,length(T),num_neurons);
    
    %Create random sequence for inputs
    inpu = 1:numPatterns;
    [x,y] = sort(rand(1,numPatterns));
    inpu = inpu(y);
    
    %Define current and membrane potential vectors.
    current = zeros(numPixels,Tlength,numPatterns);
    Vm = zeros(numNeurons,Tlength);
    current_sum = zeros(1,Tlength,numPatterns);
    z=0;
    
    %% Generate patterns for simulation
    
    %Each row of input/curr train represents a different pixel.
    input_tr = rand(numPixels,Tlength,numPatterns);
    curr_tr = zeros(numPixels,Tlength,numPatterns);
    
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
        %Generate the input and current trains, where the z dimension
        %represents the current/input train for the zth pattern introduced.
        [input_tr(:,:,z), curr_tr(:,:,z)] = GenInput(input,dt,Tlength,c_p_w);
    end
    
    
    %% Execute neuron simulation for single pattern
    %Define inhibition current
    inhibition = zeros(numNeurons,length(T),numPatterns);
    sumAct = 0;
    tic
    for z = 1:numPatterns
        %Generate the current waveforms for each neuron based on their initial
        %weight
        current = curr_tr(:,:,z).*init_w_rs;%The third dimension is the neuron.
        ind = (current==0);%Should alter so that index is only for particular z.
        current_sum = sum(current,1);
        csum = squeeze(permute(current_sum, [3 2 1])); %No inhibition yet presented.
%         
%         currFunc = cumsum(csum,2); %Obviously also no inhibition yet either.
%         Vm = Ve.*(1-exp(-T/tau))+(1-exp(-T/tau)).*R.*currFunc +Ve;
        
        for i = 1:Tlength
            spk = find(Vm(:,i)>Vth);
            No_spk = find(Vm(:,i)<Vth);
            
            Vm(spk,i+1) = Ve;
            
            Activity(spk,i,z) = 1;
            Activity(No_spk,i,z) = 0;
            
            if not(isempty(spk))
                for n = spk
                    %Calculate the changes in current based on spike times.
                    [curr_change, del_t] = CalcSTDP(input_tr(:,:,z),numPixels,T,Interpdata,i);
                    
                    %Potentiation
                    current(:,i:end,n) = current(:,i:end,n) + curr_change(:,2)+curr_change(:,1);
                    %depression
%                     for k  = 1:numPixels
%                         current(k,(i-round(del_t(k,1)/dt)):end,n) = current(k,(i-round(del_t(k,1)/dt)):end,n) +curr_change(k,1);
%                     end
                    
                    %Inhibition
                    for ne = 1:numPatterns
                        if (n ~= ne) & i+i_p_w>length(T)
                            inhibition(ne,i:end,z) = -inhib;
                        elseif (n~=ne)
                            inhibition(ne,i:i+i_p_w,z) = -inhib;
                        end
                    end
                end
                
                %Determine if current has exceeded either limit
                current(current>Imax)=Imax;
                current(current<Imin)=Imin;
                
                %With knowledge of where the indexes of zero are, can reset
                current(ind) = 0;
                
                
                %For next section of code, could consider working out values in
%                 %advance, using vectorisation, and then implementing.
%                 current_sum(:,i:end,:) = sum(current(:,i:end,:),1);
%                 csum = squeeze(permute(current_sum, [3 2 1])) +inhibition(:,:,z);
%                 currFunc = cumsum(csum,2);
%                 currFunc(n,i:Tlength) = currFunc(n,i:Tlength)-currFunc(n,i); 
                %Now only update the membrane potential for the spiking neuron,
                %as it will have changed due to current changes.
                
%                 Tvec = (T(i):dt:T(end));
%                 Tvec = Tvec - T(i);
%                 Vm(spk,i:Tlength) = (1-exp(-Tvec/tau)).*(Ve+ R.*currFunc(:,i:Tlength));
%                 Vm(Vm<Ve) = Ve;
                
                %As this won't change without spiking events.
                sumAct = sum(Activity,[2,3]);
                
            end
%             
                        Vm(No_spk,i+1) = Vm(No_spk,i) + dt/tau*(Ve-Vm(No_spk,i) + (csum(No_spk,i)+inhibition(No_spk,i,z))*R);
%             Ensure membrane potential is not below minimum.
                        Vm(Vm<Ve) = Ve;
            
            
            %This is the Qeuriloz model for homeostasis.
            Vth = Vth + dt*gamma.*(sumAct-Target);
            Vth(Vth<VthMin)=VthMin;
            Vth(Vth>VthMax) = VthMax;
            %Note, the above homeostasis needs to be implemented per time
            %step, and is difficult to vectorise based off what I've done.
            %It should however also be possible.
            
        end
        %Need to re-initialise the weights.
        for n = 1:numPatterns
            %For each input
            for k = 1:numPixels
                if not(isempty(find(current(k,:,n)~=0,1,'last')))
                    init_w_rs(k,:,n) = current(k,find(current(k,:,n)~=0,1,'last'),n);
                end
            end
        end
        
        
    end
    
    %Define the length of time to present all inputs
    tvec = linspace(0,T(end)*numPatterns,length(memVOut(n,:)));
    
    %Calculate the final weights and store them for next run.
    finalW = reshape(init_w_rs, [5 5 numPatterns]);
    
    
    %Generate a rastor plot
    figure(12)
    rastor  = reshape(Activity,[numPatterns numPatterns*length(T)]);
    for n = 1:numPatterns
        rastor(n,:) = n*rastor(n,:);
    end
    
    plot(tvec,rastor,'bo')
    axis([0 tvec(end) 0.5 numPatterns+0.5])
    xticks(0:T(end):T(end)*10);
    
    %Generate a heatmap for all 10 neurons
    figure(13)
    sgtitle('Final synaptic weights')
    for n = 1:numPatterns
        finalW(:,:,n) = finalW(:,:,n)';
        subplot(2,5,n)
        heatmap(finalW(:,:,n), 'Colormap',map,'Colorlimit',[Imin Imax]);
        title(sprintf('Neuron %d',n))
    end
    t1 = toc
    
end
%Save the final weights.
save('PreviousWeights.mat','finalW')
%Save threshold values
save('Vth.mat','Vth')

%     string = sprintf('rastor%d.mat',run)
%     save(string,'rastor')