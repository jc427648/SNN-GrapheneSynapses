%% Plotting the entire raster plot
figure(2)
hold off
odd = 1;
for run = 1:20
    string = sprintf('rastor%d.mat',run);
    rastor = load(string).rastor;
    time = (run-1)*15010+1:run*15010;
    if odd == 1
        str = 'bo';
        odd = 0;
    elseif odd == 0
        str = 'ro'
        odd = 1
    end
    
    plot(time,rastor,str)
    hold on
end
ylim([0.5 10.5])
xticks(0:15010:15010*20)
ax = gca;
ax.XAxis.MinorTickValues = 0:1501:15010*20;
grid on
grid minor
