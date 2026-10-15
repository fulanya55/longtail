%% 拼接数据
clear; clc;
dirPath = 'D:\魏晶晶工作记录-20200320\魏晶晶项目\35项目\仿真数据生成\20200107-换6000\data';
filePath = [dirPath '\data'];
numClass = 9
number = 1000
dataArr = single(zeros(number * numClass, 6000));
label = zeros(number * numClass, 1);
for i = 1 : numClass 
   cd(dirPath);
   dataPath = [filePath num2str(i)];
   load(dataPath);
   dataArr(number * (i - 1) + 1 : number * i, :) = data(1 : number, :);
   label(number * (i - 1) + 1 : number* i) = i;
end
data = dataArr;
save data.mat data;
label = single(label);
save label.mat label;