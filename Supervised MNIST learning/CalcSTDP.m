function [curr_change del_t] = CalcSTDP(input_tr,numPixels,T,Interpdata,i)
%Calculates the change in current for a particular neuron due to the STDP
%window presented to it. The value in index 1 of curr_change should be the
%negative change in current whilst the second index should be the positive
%change in current. input_tr is the spiking train of inputs, numPixels is
%the number of pixels in the pattern (or the number of different spike
%trains acting as inputs) T is the particular point in time of neuronal
%firing and Interpdata is the STDP data used to interpolate.

curr_change = zeros(numPixels,2);
del_t = zeros(numPixels,2);

for k = 1:numPixels
    sp_time = find(input_tr(k,:)>0);
    delta_t = -T(sp_time) + T(i);
    
    %I need times for both pre and post.
    a = delta_t<0;
    b = delta_t>0;
    
    if not(isempty(delta_t))
        
        if isempty(delta_t(a))
            del_t(k,1) = -85e-3;
        else
            del_t(k,1) = max(delta_t(a));
        end
        
        if isempty(delta_t(b))
            del_t(k,2) = 85e-3;
        else
            del_t(k,2) = min(delta_t(b));
        end
        
        %Now we interpolate on the STDP and find the value. Then change
        %the current.
        for p = 1:2
            time_ind = find(del_t(k,p)-0.25e-3<1e-3*Interpdata(1,:) &  1e-3*Interpdata(1,:)<del_t(k,p)+0.25e-3);
            if isempty(time_ind)
                curr_change(k,p) = 0;
            else
                curr_change(k,p) = Interpdata(2,time_ind);
            end
        end
    end
end
%%%%Below codes is to produce linear interpolation between points, as
%%%%opposed to nearest point.Technically more accurate, but longer to run.
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