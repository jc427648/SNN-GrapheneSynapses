function [integrated_current, w, SetPState, SetNState, ResetPState, ResetNState] = apply_voltage(V_ref, dt, w, SetPState, SetNState, ResetPState, ResetNState, ROff, ROn, VResetn, VResetp, VSetn, VSetp, alphaResetn, alphaResetp, alphaSetn, alphaSetp, kResetn, kResetp, kSetn, kSetp)
    I_out = zeros(length(V_ref), 1);
    for i = 1:length(I_out)
        ReturnZeroCurrent = true;
        % If above positive setting voltage
        if (V_ref(i) > VSetp)
            dw_dt = kSetp * (V_ref(i) / VSetp - 1) ^ alphaSetp;
            w = w + dt * dw_dt;
            if w > 1
                w = 1;
            elseif w < 0
                w = 0;
            end
            SetPState = 1;
            SetNState = 0;
            ResetPState = 0;
            ResetNState = 0;
            ReturnZeroCurrent = false;
        % If below negative setting voltage
        elseif (V_ref(i) < VSetn)
            dw_dt = kSetn * (V_ref(i) / VSetn - 1) ^ alphaSetn;
            w = w + dt * dw_dt;
            if w > 1
                w = 1;
            elseif w < 0
                w = 0;
            end
            SetPState = 0;
            SetNState = 1;
            ResetPState = 0;
            ResetNState = 0;
            ReturnZeroCurrent = false;
        % If  being reset once in positive voltage region
        elseif (((0 < V_ref(i))&& (V_ref(i) < VResetp)) && ((ResetPState == 1) || (SetPState == 1)))
            dw_dt = kResetp * (V_ref(i) / VResetp) ^ alphaResetp;
            w = w + dt * dw_dt;
            if w > 1
                w = 1;
            elseif w < 0
                w = 0;
            end
            SetPState = 0;
            SetNState = 0;
            ResetPState = 1;
            ResetNState = 0;
            ReturnZeroCurrent = false;
        % If being reset once in negative voltage region
        elseif (((0 > V_ref(i))&& (V_ref(i)) > VResetn) && ((ResetNState == 1 || SetNState == 1)))
            dw_dt = kResetn * (V_ref(i) / VResetn) ^ alphaResetn;
            w = w + dt * dw_dt;
            if w > 1
                w = 1;
            elseif w < 0
                w = 0;
            end
            SetPState = 0;
            SetNState = 0;
            ResetPState = 0;
            ResetNState = 1;
            ReturnZeroCurrent = false;
        end
        resistance = update_resistance(w, ROn, ROff);
        if (ReturnZeroCurrent)
            I_out(i) = 0;
        else
            % Calculate current vector
            I_out(i) = V_ref(i) / resistance;
        end

    end
    integrated_current = trapz(0:dt:((length(I_out) - 1)*dt),I_out);
end

