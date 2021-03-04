%% Homeostatic plasticity

%This code is the same as char_recog_network but with homeostatic
%plasticity added.

clc;clear all;


tic
num_neurons = 10;
%Define paramaters
VthMin = 0.25;
VthMax = 20;
Ve = 0.2;
tau = 0.020;%Timing constant
R = 5e2;%Membrane resistance
dt = 0.0002; %Difference in time 0.2ms
T = 0:dt:0.1;%Simulation time

%Define homeostatic plasticity parameters
Target = 65*dt/T(end)/num_neurons;%0.010%Ideal neuron firing rate
gamma = 30;%Multiplicative constant
Activity = zeros(num_neurons,length(T),num_neurons);


%Define maximum and minimum levels of current
Imax = 25e-3;
Imin = -25e-3;%-10e-5;

c_p_w = 10;%Current pulse width (index value)
i_p_w = 15;%Inhibition pulse width (index value)

%Define an inhibition current
inhib = 185e-3;%185e-3;

%Set low and high frequencies
hr = 200; %These frequencies worked for graphene device, 100,10 worked for ideal.
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
num_inputs = numel(nine);


%Pre allocate output vector
memVOut = zeros(num_neurons,length(T),num_neurons);

%Either initialise with random currents or use the previously stored values
%of current to re-update network
x = input('Start over?','s');
if x == 'y'
    init_w =   Imin/1 +(Imax-Imin)/1.*rand(5,5,num_neurons);
    Vth = 3.5.*ones(num_neurons,1);%Volts, threshold of neuron.
elseif x == 'n'
    init_w = load('PreviousWeights.mat').finalW;
    Vth = load('Vth.mat').Vth
end

%Have a reshaped initial weight for ease of computation
init_w_rs = reshape(permute(init_w, [2 1 3]) ,[num_inputs 1 num_neurons]);


%plot the initial random weights
figure(14)
sgtitle('Initial weights')

map = redblue(100);
for n = 1:num_neurons
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

% %     %Debugging, to see the effects of modifying STDP window.
% Interpdata(2,161) = 0.0000;
% %  figure(15)
% %  hold off
% %  plot(Interpdata(1,:),Interpdata(2,:))
% %  hold on
% Ap = 5.8e-3;%5.8e-3;
% An = -3.0e-3;
% Tp = 0.80e-2;
% Tn = 1.8e-2;
% %Testing modifications to the STDP window.
% Interpdata(2,1:160) = An.*exp(Interpdata(1,1:160).*1e-3./Tn);
% Interpdata(2,162:end) = Ap.*exp(-Interpdata(1,162:end).*1e-3./Tp);
%The below code flips the STDP window, and should not work. However, it
%produces the best convergence results. I tried flipping delta_t, and it
%doesn't produce these results either, it is strange.
%  Interpdata(2,:) = 3*Interpdata(2,:);
% Interpdata(2,1:87) = 0;
% Interpdata(2,174:end) = 0;

for run = 1:20
    Activity = zeros(num_neurons,length(T),num_neurons);
    
    %Create random sequence for inputs
    inpu = 1:num_neurons;
    [x,y] = sort(rand(1,num_neurons));
    inpu = inpu(y);
    
    %Define current and membrane potential vectors.
    current = zeros(num_inputs,length(T),num_neurons);
    Vm = zeros(num_neurons,length(T));
    current_sum = zeros(1,length(T),num_neurons);
    
    %Define a refractory period
    %  th = zeros(1,num_neurons);
    %  ref_per = 5;
    
    z=0;
    %Choose which input is presented to the network
    
    %% Generate patterns for simulation
    
    %Each row of input/curr train represents a different pixel.
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
        input_tr = 1*input_tr; %Convert from logical.
        
        %Develop the current pulses from the input spikes
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
        
        
        
        % Below code was used to check the inputs.
        % inp_tr_out(:,length(input_tr)*(inp-1)+1:length(input_tr)*inp) = input_tr;
        %     figure(4)
        %         for k = 1:numel(input)
        %         subplot(5,5,k)
        %         plot(T,input_tr(k,:,z))
        %          axis([0 2 0 2])
        %      xlabel('Time')
        %      title('Spike train')
        %         end
        
    end
    
    
    %% Execute neuron simulation for single pattern
    
    %Define inhibition current
    inhibition = zeros(num_neurons,length(T),num_neurons);
    
    % current_sum = sum(current,1);

    %   th = zeros(num_neurons,1);
    for z = 1:num_neurons
        %Generate the current waveforms for each neuron based on their initial
        %weight
        current = curr_tr(:,:,z).*init_w_rs;
        ind = find(current==0);%Should alter so that index is only for particular z.
        
        for i = 1:length(T)
            
            %Note this loop starts at a single point in time and then loops through
            %all neuron inputs.
            spk = find(Vm(:,i)>Vth);
            No_spk = find(Vm(:,i)<Vth);
            
            Vm(spk,i+1) = Ve;
            Activity(spk,i,z) = 1;
            if not(isempty(spk))
                for n = spk
                    
                    %            th(n) = i;
                    %Calculate the changes in current based on spike times.
                    %%%%%%%%%%%%%%%%%
                    
                    for k = 1:num_inputs
                        sp_time = find(input_tr(k,:,z)>0);
                        delta_t = -T(sp_time) + T(i);
                        
                        %I need times for both pre and post.
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
                            
                            %Now we interpolate on the STDP and find the value. Then change
                            %the current.
                            for p = 1:2
                                time_ind = find(del_t(p)-0.25e-3<1e-3*Interpdata(1,:) &  1e-3*Interpdata(1,:)<del_t(p)+0.25e-3);
                                if isempty(time_ind)
                                    curr_change(p) = 0;
                                else
                                    curr_change(p) = Interpdata(2,time_ind);
                                end
                            end
                            
                            
                            %                     for p = 1:2
                            %                         if del_t(p)>-80e-3 && del_t(p)<80e-3
                            %                             time_ind = find(del_t(p)-0.5e-3<1e-3*Interpdata(1,:) &  1e-3*Interpdata(1,:)<del_t(p)+0.5e-3);
                            %
                            %                             %In case value is exact
                            %                             if length(time_ind)==1
                            %                                 curr_change(p) = Interpdata(2,time_ind);
                            %                             else
                            %                                 %We now have to interpolate linearly, rather than pick a point
                            %                                 m = (Interpdata(2,time_ind(2)) - Interpdata(2,time_ind(1)))/...
                            %                                     (Interpdata(1,time_ind(2)) - Interpdata(1,time_ind(1)));
                            %                                 c = Interpdata(2,time_ind(1)) - m*Interpdata(1,time_ind(1));
                            %
                            %                                 curr_change(p) = m*del_t(p)*1e3 + c;
                            %                             end
                            %                         else
                            %                             %Check to see if curr_change is empty
                            %                             curr_change(p) = 0;
                            %                         end
                            %
                            %                     end
                            
                            
                            
                            
                            
                            % pos_change_out = curr_change(1)
                            %Potentiation
                            current(k,i:end,n) = current(k,i:end,n) + curr_change(2);
                            %depression
                            current(k,(i-round(del_t(1)/dt)):end,n) = current(k,(i-round(del_t(1)/dt)):end,n) +curr_change(1);
                            
                            %Inhibition
                            for ne = 1:num_neurons
                                if (n ~= ne) & i+i_p_w>length(T)
                                    inhibition(ne,i:end,z) = -inhib;
                                elseif (n~=ne)
                                    inhibition(ne,i:i+i_p_w,z) = -inhib;
                                end
                            end
                        end
                        
                    end
                    %%%%%%%%%%%%
                    
                    
                end
            end
            
            %Determine if current has exceeded either limit
            current(current>Imax)=Imax;
            current(current<Imin)=Imin;
            
            %With knowledge of where the indexes of zero are, can reset
            
            current(ind) = 0;
            %            if th(n)+ref_per<i
            
            current_sum(:,i:end,:) = sum(current(:,i:end,:),1);
            csum = squeeze(permute(current_sum, [3 2 1]));
            Vm(No_spk,i+1) = Vm(No_spk,i) + dt/tau*(Ve-Vm(No_spk,i) + (csum(No_spk,i)+inhibition(No_spk,i,z))*R);
            %            end; %Introduce a refractory period of no current change.
            
            Vm(Vm<Ve) = Ve;
            i = i+1;
            
        end
        
        
        
        
        t3 = toc
        
        
        %Determine the output current waveform
        % current_out(:,length(T)*(z-1)+1:length(T)*z,:) = current;
        
        %     memVOut(:,:,z) = Vm;
        
        
        %Need to re-initialise the weights.
        for n = 1:num_neurons
            %For each input
            for k = 1:num_inputs
                if not(isempty(find(current(k,:,n)~=0,1,'last')))
                    init_w_rs(k,:,n) = current(k,find(current(k,:,n)~=0,1,'last'),n);
                end
            end
        end
        
        % % %Debugging
        %     init_w = reshape(init_w_rs, [5 5 5]);
        %     for n = 1:num_neurons
        %         init_w(:,:,n) = init_w(:,:,n)';
        %     end
        %
        %
        %     figure(13)
        %     for n = 1:num_neurons
        %         subplot(2,5,n)
        %         heatmap(init_w(:,:,n), 'Colormap',map,'Colorlimit',[Imin Imax]);
        %         title(sprintf('Neuron %d',n))
        %     end
        
        
        
        
        
        
    end
    %Update the thresholds based on homeostatic plasticity
    Vth = Vth + gamma.*(sum(Activity,[2,3])./z/i-Target);
    Vth(Vth<VthMin)=VthMin;
    Vth(Vth>VthMax) = VthMax;
    Vth %Print to command window
    
    %Define the length of time to present all inputs
    tvec = linspace(0,T(end)*num_neurons,length(memVOut(n,:)));
    % figure(11)
    % hold off
    %
    %Plot the membrane potential for all neurons.
    % figure(1)
    % for n=1:num_neurons
    %     subplot(5,2,n)
    %     plot(tvec,memVOut(n,:))
    %     hold on
    %     plot(tvec,Vth.*ones(1,length(tvec)))
    %     hold off
    %     %plot(T,Vm);
    %     xlabel('Time (s)')
    %     ylabel('Membrane Voltage')
    % end
    
    %Plot the current waveforms for each neuron's inputs. This is effectively a
    %plot of the synaptic weight change as a function of time.
    % for n = 1:num_neurons
    % figure(n)
    % hold off
    % sgtitle(sprintf('Neuron %d',n))
    % for k =1:num_inputs
    %     subplot(num_inputs/5,5,k)
    %     plot(T,current(num_inputs*(n-1)+k,:))
    %     plot(linspace(0,T(end)*num_neurons,length(current_out(25*(n-1)+k,:))),current_out(25*(n-1)+k,:))
    %     hold on
    %     ind = find(memVOut(n,:)>Vth);
    %     plot(tvec(ind),current_out(25*(n-1)+k,ind),'r.')
    %     hold off
    %     axis([0 T(end)*num_neurons Imin Imax])
    %     xlabel('Time (s)')
    %     ylabel('Current (A)')
    %     title(sprintf('Pixel %d',k))
    % end
    % end
    
    
    
    
    
    %Calculate the final weights and store them for next run.
    finalW = reshape(init_w_rs, [5 5 num_neurons]);
    
    
    %Generate a rastor plot
    figure(12)
    rastor  = reshape(Activity,[num_neurons num_neurons*length(T)]);
    for n = 1:num_neurons
        rastor(n,:) = n*rastor(n,:);
    end
    
    plot(tvec,rastor,'bo')
    axis([0 tvec(end) 0.5 num_neurons+0.5])
    xticks(0:T(end):T(end)*10);
    
    %Generate a heatmap for all 10 neurons
    
    
    figure(13)
    sgtitle('Final synaptic weights')
    for n = 1:num_neurons
        finalW(:,:,n) = finalW(:,:,n)';
        subplot(2,5,n)
        heatmap(finalW(:,:,n), 'Colormap',map,'Colorlimit',[Imin Imax]);
        title(sprintf('Neuron %d',n))
    end
    
    
    % figure(1)
    % plot(Interpdata(1,:),Interpdata(2,:))
    
    
    
    
    %Save the final weights.
    save('PreviousWeights.mat','finalW')
    %Save threshold values
    save('Vth.mat','Vth')
    string = sprintf('rastor%d.mat',run)
    save(string,'rastor')
    
    
    
    % outn = An*Tn
    % outp = Ap*Tp
    
    tend = toc
    
    
end
% figure(1)
% plot(t5)