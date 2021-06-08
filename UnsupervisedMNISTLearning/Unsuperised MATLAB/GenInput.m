function [input_tr, curr_tr] = GenInput(input,dt,Tlength,c_p_w)
%Generate the input spike and current trains based off of the pattern in
%input. Input is a matrix consisting of the Poissonian spiking rates for
%each pixel. dt is the difference in time. The current train is just
%normalised to 1 and 0, it does not incorporate the weight. YOU SHOULD
%CONSIDER OUTPUTTING AN INDEX FOR SPIKES ON INPUT_TR. The width of the
%current pulses in curr_tr are defined by the CurrentPulseWidth (c_p_w).

%Define number of pixels and sizes of input/current trains.
numPixels = numel(input);
input_tr = rand(numPixels,Tlength);
curr_tr = zeros(numPixels,Tlength);

%reshape input so that it is easier to multiply by.
input = reshape(input,[numPixels, 1]);


%Below is to generate Poissonian spike trains. The idea is that the
%probability of a spike ocurring in a given discrete time block is equal to
%Spiking rate * dt.
input_tr = input_tr<=input*dt;
input_tr = 1*input_tr;%Convert from logical

%Develop the current pulses from the input spikes
for i = 1:numPixels
    for j =1:Tlength
        if input_tr(i,j)>= 1
            if j+c_p_w>Tlength
                curr_tr(i,j:end) = 1;
            else
                curr_tr(i,j:j+c_p_w) = 1;
            end
        end
    end
end





