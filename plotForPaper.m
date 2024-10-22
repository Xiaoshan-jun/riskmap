clear all
figure(1)
% Define the coordinates of the centers of each cell
x = [0.5 1.5; 0.5 1.5];
y = [1.5 1.5; 0.5 0.5];

% Plot the grid lines
line([0 0], [0 2], 'Color', 'k');
line([0 2], [0 0], 'Color', 'k');
line([0 2], [1 1], 'Color', 'k');
line([0 2], [2 2], 'Color', 'k');
line([1 1], [0 2], 'Color', 'k');
line([2 2], [0 2], 'Color', 'k');

% Add the values to each cell
text(x(1,:), y(1,:), {'0.1','0.05'}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 16);
text(x(2,:), y(2,:), {'0','0.05'}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 16);
title('example risk map','FontSize',16)
axis equal; % set the aspect ratio to 1:1
axis off; % turn off the axis ticks and labels
print('exampleRisk.png', '-dpng', '-r900');

%%
% Define the x and y values of the plane
figure(2)
[x,y] = meshgrid(-0.5:1:1.5);

% Define the z values of the plane
z = ones(size(x))*0.9;

% Plot the surface
surf(x,y,z);
hold on; % Hold the plot for adding points and figure
alpha(0.3);
% Define the data for the line and points
x = [0 1 1];
y = [0 0  1 ];
z = [1 0.95 0.9025];

% Plot the line
plot3(x, y, z, 'b-',LineWidth=3);

plot3([0 0], [0 1], [1 0.9], 'r-',LineWidth=3);

plot3([0 1], [1 1], [0.9 0.855], 'r-',LineWidth=3);
% Plot the points
scatter3(x, y, z, 100,'filled', 'MarkerFaceColor', 'g');
positions = [0,0,1;1,0,0.95;1,1,0.9025];
for i = 1:length(x)
    text(x(i), y(i), z(i), sprintf('(%0.f, %0.f, %0.4f)', positions(i,:)), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Color', 'k');
end
x = [0 1]; 
y = [1 1];
z = [0.9 0.855];
scatter3([0 1], [1 1], [0.9 0.855], 100,'filled', 'MarkerFaceColor', 'r');
positions = [0,1,0.9;1,1,0.855];
for i = 1:length(x)
    text(x(i), y(i), z(i), sprintf('(%0.f, %0.f, %0.4f)', positions(i,:)), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Color', 'k');
end

% Define the data for the surface figure
%x2 = [0 1 1];
%y2 = [0 0  1 ];
%z2 = [1 0.95 0.9025];

% Plot the surface figure
%fill3(x2, y2, z2, 'b');
axis off
hold off; % Release the plot

title('example path','FontSize',16)
view(-30, 10);
print('exampleRiskpath.png', '-dpng', '-r900');
