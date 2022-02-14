function resistance = update_resistance(w, ROn, ROff)
    resistance = ROn + (ROff - ROn) / (1) * (w);
end