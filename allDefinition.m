clc                     %清屏
close all;              %关闭当前所有figure图像
SamNum=20;              %输入样本数量为20
TestSamNum=20;          %测试样本数量为20
ForcastSamNum=2;        %预测样本数量为2
HiddenUnitNum=8;        %中间层隐节点为8，比直接调用多了一个节点
InDim=3;                %网络输入维度为3
OutDim=2;               %网络输出维度为2

%原始数据
load highway.mat;
p=[sqrs;sqjdcs;sqglmj];  %输入数据矩阵
t=[glkyl;glhyl];         %目标数据矩阵
[SamIn,minp,maxp,tn,mint,maxt]=premnmx(p,t); %原始样本初始化

rand('state',sum(100*clock));    %根据系统时钟种子产生随机数
NoiseVar=0.01;                  %噪声强度0.01(目的防止过度拟合)
Noise=NoiseVar*randn(2,SamNum); %生成噪音
SamOut=tn+Noise;                %将噪声添加到输出样本上

TestSamIn=SamIn;                %这里取输入样本与测试样本相同，因为样本容量少
TestSamOut=SamOut;              %输出样本和测试样本一样

MaxEpochs=100;                %最多训练50000次
lr=0.035;                       %学习速率0.035
EO=0.65*10^(-3);                %设置目标误差
W1=0.5*rand(HiddenUnitNum,InDim)-0.1; %初始化输入层与隐含层的权值
B1=0.5*rand(HiddenUnitNum,1)-0.1;     %初始化输入层与隐含层的阈值
W2=0.5*rand(OutDim,HiddenUnitNum)-0.1;%初始化隐含层与输出层的权值
B2=0.5*rand(OutDim,1)-0.1;            %初始化隐含层与输出层的阈值
ErrHistory=[];                        %中间变量，预先占据内存

for i=1:MaxEpochs
    HiddenOut=logsig(W1*SamIn+repmat(B1,1,SamNum));%隐含层输出
    NetworkOut=W2*HiddenOut+repmat(B2,1,SamNum);   %输出层输出
    Error=SamOut-NetworkOut;                       %实际输出与网络之差
    %SSE=sumsqr(Error)                              %代价函数
    SSSE=Error.*Error;
    SSE=sum(SSSE(:))
    
    
    ErrHistory=[ErrHistory SSE];
    if SSE<EO,break,end                %达到误差要求，跳出学习网络
        
    %以下6行是BP网络最核心的程序
    %他们是权值（阈值）根据代价函数负梯度下降算法，动态调整
    Delta2=Error;
    Delta1=W2'*Delta2.*HiddenOut.*(1-HiddenOut);%sigmoid的导数是f'(x)=f(x)(1-f(x))
    %隐含层
    dW2=Delta2*HiddenOut';
    dB2=Delta2*ones(SamNum,1);
    %输入层
    dW1=Delta1*SamIn';
    dB1=Delta1*ones(SamNum,1);
    %对隐含层与输出层之间的权值、阈值进行修正
    W2=W2+lr*dW2;
    B2=B2+lr*dB2;
    %对输入层与隐含层之间的权值、阈值进行修正
    W1=W1+lr*dW1;
    B1=B1+lr*dB1;
end

HiddenOut=logsig(W1*SamIn+repmat(B1,1,TestSamNum));%隐含层输出最终结果
NetWorkOut=W2*HiddenOut+repmat(B2,1,TestSamNum);  %输出层输出最终结果
a=postmnmx(NetworkOut,mint,maxt);                  %还原网络输出层的结果
x=1990:2009;                                       %时间轴刻度
newk=a(1,:);                                       %网络输出客运量
newh=a(2,:);                                       %网络输出货运量
figure;
subplot(2,1,1);plot(x,newk,'r-o',x,glkyl,'b--+');   %绘制公路客运量对比图
legend('网络输出客运量','实际客运量','Location','northwest');
xlabel('年份');ylabel('客运量/万人');
title('源程序神经网络 客运量学习和测试对比图');
         
subplot(2,1,2);plot(x,newh,'r-o',x,glhyl,'b--+');   %绘制公路货运量对比图
legend('网络输出货运量','实际货运量','Location','northwest');
xlabel('年份');ylabel('货运量/万吨');
title('源程序神经网络 货运量学习和测试对比图'); 

figure;
plot(1:MaxEpochs,ErrHistory,'r-o')
xlabel('迭代次数');ylabel('误差');
title('训练误差')
%%
%利用训练好的模型进行预测
%利用训练好的BP，对新数据进行仿真

%利用训练好的网络进行预测
%当用训练好的网络对新的数据pnew进行预测时，也应做相应的处理
pnew=[73.39 75.55;3.9635 4.0975 ;0.9880 1.0268];%2010年2011相关数据
pnewn=tramnmx(pnew,minp,maxp);   %利用原始输入的归一化参数对新数据归一化
HiddenOut = logsig(W1*pnewn+repmat(B1,1,ForcastSamNum));
anewn=W2*HiddenOut+repmat(B2,1,ForcastSamNum);            
anew=postmnmx(anewn,mint,maxt) %把网络预测得到的数据返归一到原数量级