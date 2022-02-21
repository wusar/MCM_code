function value = fun_value(point) %求函数值的函数
M = csvread('pred.csv');
Bp=M(:,2);
Br=M(:,1);
N = csvread('gold_pred_new.csv');
Gp=N(:,2);
Gr=N(:,1);

hyper1=2;
%alpha=point(1);
%beta=point(2);
alpha=27;
beta=0;
alphaG=0.02;
alphaB=0.01;
pro=1000;
cash=pro;
Wb=0;
Wg=0;
count_=0;
for i=1:200
   gpn=Gp(i+1);
   bpn=Bp(i+1);
   gr=Gr(i);
   br=Br(i);
   grn=Gr(i+1);
   brn=Br(i+1);
   
   A=alpha*(gpn-gr)/gr;
   B=beta*(bpn-br)/br;
   
   j=mod(i,7);
   if j~=5 && j~=6 
      if A>=0 && B>=0 && A*B~=0%这里考虑那个赚的多
          cash2gold=cash*A/(A+B);
          cash2bit=cash*B/(A+B);
          if A<=B%黄金小于比特币时，减持一部分黄金给比特币,如果cash可以直接支付这一部分，就不减持了
              gold2bit=Wg*(B-A)/(A+B);%应该转化的黄金数量,这里可能还是要继续考虑
              if gold2bit*gr*(1-alphaG)<cash2gold
                  cash2gold=cash2gold-gold2bit*gr*(1-alphaG);
                  cash2bit=cash-cash2gold;
                  gold2bit=0;
              else
                  gold2bit=gold2bit-cash2gold/(gr*(1-alphaG));%最后需要转化的黄金数量就是减去要转化成相应数量的现金的黄金的数量
                  cash2bit=cash+gold2bit*gr*(1-alphaG);
                  cash2gold=0;
              end
              Wgn=Wg-gold2bit+cash2gold*(1-alphaG)/gr;
              Wbn=Wb+cash2bit*(1-alphaB)/br;
              Wg=Wgn;
              Wb=Wbn;
              cash=0;
              pro=Wbn*brn*(1-alphaB)+Wgn*grn*(1-alphaG)+cash;%计算到了第二天真实收益
          else%黄金大于比特币时，减持一部分比特币给黄金,如果cash可以直接支付这一部分，就不减持了
             bit2gold=Wb*(A-B)/(A+B);%应该转化的黄金数量
             if bit2gold*br*(1-alphaB)<cash2bit
                  cash2bit=cash2bit-bit2gold*br*(1-alphaB);
                  cash2gold=cash-cash2bit;
                  bit2gold=0;
             else
                  bit2gold=bit2gold-cash2bit/(gr*(1-alphaG));%最后需要转化的比特币数量就是减去要转化成相应数量的现金的比特币的数量
                  cash2gold=cash+bit2gold*br*(1-alphaB);
                  cash2bit=0;
             end
             Wbn=Wb-bit2gold+cash2bit*(1-alphaB)/br;
             Wgn=Wg+cash2gold*(1-alphaG)/gr;
             Wg=Wgn;
             Wb=Wbn;
             cash=0;
             pro=Wbn*brn*(1-alphaB)+Wgn*grn*(1-alphaG)+cash;%计算到了第二天真实收益
          end
        
      elseif A>=0 && B<=0 
       %flag=0;
       %gamma=(A-B)/(A-hyper1*B);
       
       Wbn=Wb*(1+max(-1,B));%选择减持比特币
       delta_cash=(Wb-Wbn)*br*(1-alphaB);
       delta_Wg=(cash+delta_cash)/gr*(1-alphaG);%计算可以增持的最大黄金数量
       Wgn=delta_Wg*(-A)/(B-A)+Wg;;%选择增持黄金
       cash=delta_cash-(Wgn-Wg)/(1-alphaG)*gr+cash;
       pro=Wbn*brn*(1-alphaB)+Wgn*grn*(1-alphaG)+cash;%计算到了第二天真实收益
       Wg=Wgn;
       Wb=Wbn;
       
      elseif A<=0 && B>=0
       %flag=0;
       %gamma=(-A)/(B-hyper1*A);
       Wgn=Wg*(1+max(-1,A));%选择减持黄金
       delta_cash=(Wg-Wgn)*gr*(1-alphaG);%计算减持带来的cash增值量
       delta_Wb=(cash+delta_cash)/br*(1-alphaB);%计算可以增持的最大比特币数量
       Wbn=delta_Wb*(-A)/(B-A)+Wb;%选择增持比特币
       cash=delta_cash-(Wbn-Wb)/(1-alphaB)*br+cash;
       pro=Wbn*brn*(1-alphaB)+Wgn*grn*(1-alphaG)+cash;
       Wg=Wgn;
       Wb=Wbn;
      else%黄金和比特币都会下跌的情况
       %flag=1
       Wgn=Wg*(1+max(-1,A));%选择减持黄金
       Wbn=Wb*(1+max(-1,B));%选择减持比特币
       cash=cash+(Wg-Wgn)*gr*(1-alphaG)+(Wb-Wbn)*br*(1-alphaB);
       pro=Wbn*brn*(1-alphaB)+Wgn*grn*(1-alphaG)+cash;
       Wg=Wgn;
       Wb=Wbn;
      end
      %pro=cash+Wg*(1-alphaG)*gr+Wb*(1-alphaB)*br;%计算出当前总财产
    %if(flag==1)
       % Wb=0;
        %Wg=0;
        %cash=pro;%两个都亏，选择全部换成现金
    %else%否则选择将财产全部转换成比特币和黄金
       %Wgn=gamma*pro/gr*(1-alphaG);%这里有问题
       %Wbn=(1-gamma)*pro/br*(1-alphaB);
       %cash=0;
       
       %pro=Wgn*grn*(1-alphaG)+Wbn*brn*(1-alphaB)-abs(Wgn-Wg)*gr*alphaG-abs(Wbn-Wb)*br*alphaB;
       
       %Wg=Wgn;
       %Wb=Wbn;
  else%黄金不能交易的日子里
      if B>=0%比特币预测上涨
          Wgn=Wg;
          Wbn=Wb+cash/br*(1-alphaB)*min(1,hyper1*B/beta);
          pro=cash+Wgn*grn*(1-alphaG)+Wbn*brn*(1-alphaB);
          Wg=Wgn;
          Wb=Wbn;
      else%比特币预测下跌
          Wgn=Wg;
          Wbn=Wb*(1+min(-1,hyper1*B/beta));%选择减持比特币
          cash=(Wb-Wbn)*br*(1-alphaB)+cash;
          pro=cash+Wgn*grn*(1-alphaG)+Wbn*brn*(1-alphaB);
          Wg=Wgn;
          Wb=Wbn;
      end
   end
   %if i<10
    %pro,A,B,cash,Wg,Wb,gr,br,grn,brn
   %end
end
if pro>5000
    value=pro
    alpha,beta
end
    value=pro;
end