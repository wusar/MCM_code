

T1=1e-3; % 初始温度，10的10次方！需要设定一个很大的温度。！！！！！！！！！！！！！！
T0=1e11;% 终止温度！！！！！！！！！！！！！！！！！！
l=2; % 各温度下的迭代次数
q=0.993;%降温速率
syms x;
eq = 1000*(0.9)^x == num2str(T1);
Time=ceil(double(solve(eq,x)));  % 计算迭代的次数
%接下来初始化取的点的坐标
point1=zeros(2,1);
obj = zeros(Time,1);%代价函数值储存矩阵初始化
obj0=fun_value(point1);%计算初始值
count=0;%初始化计数值
track = zeros(2,Time);
 
while T0>T1
    count =count+1;
    point2=new_point();%更新取点的函数
     % 2. Metropolis法则判断是否接受新解
    [point1,R] = Metropolis(point1,point2,T0);%Metropolis 抽样算法
     if count == 1 || R > obj(count-1)
        obj(count) = R;           %如果当前温度下函数值小于上一路程则记录当前函数值及对应的点!!!!!!!大于
     else
         obj(count) = obj(count-1);%如果当前温度下函数值大于上一路程则记录上一函数值！！！！小于
    end
    track(:,count) = point1;
    T0 = q*T0; 
end
figure
plot(1:count,obj)
xlabel('迭代次数')
ylabel('函数值')
title('优化过程')
 
disp('最优解:')
S = track(:,end)
obj(end)
