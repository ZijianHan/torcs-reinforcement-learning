clear all;
load 'cumreward.csv';
ep = 1:length(cumreward);
figure(1)
scatter(ep,cumreward,10,'filled')



figure('Name','track_map')
trackWidth = 8; %[m]
x_straight = 0:999;
y_straight1_left = (100+trackWidth/2)*ones(1,length(x_straight));
y_straight1_right = (100-trackWidth/2)*ones(1,length(x_straight));
y_straight1 = 0 * ones(1,length(x_straight));
hold on;
plot(x_straight,y_straight1_left,'Color',[0.0,0.0,0.0]);
plot(x_straight,y_straight1_right,'Color',[0.0,0.0,0.0]);
xlim([-10,120])
ylim([-10,120])
