function [S,R] = Metropolis(S1,S2,T)
 
% S1：  当前解
% S2:   新解
% D:    距离矩阵（点的函数值）
% T:    当前温度
 
% S：   下一个当前解
% R：   下一个当前解的函数值
 
R1 = fun_value(S1);  %计算点的函数值
 
 
R2 = fun_value(S2);  %计算点的函数值
dC = R2 - R1;   %计算函数值之差
if dC > 0       %如果函数值降低 接受新点,！！！！！！！！！！！！！！！！！！！！这里进行了修改
    S = S2;
    R = R2;
elseif exp(-dC/T)>= rand   %以exp(-dC/T)概率接受新点
    S = S2;
    R = R2;
else        %不接受新点
    S = S1;
    R = R1;
end

end
