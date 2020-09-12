clc                     %����
close all;              %�رյ�ǰ����figureͼ��
SamNum=20;              %������������Ϊ20
TestSamNum=20;          %������������Ϊ20
ForcastSamNum=2;        %Ԥ����������Ϊ2
HiddenUnitNum=8;        %�м�����ڵ�Ϊ8����ֱ�ӵ��ö���һ���ڵ�
InDim=3;                %��������ά��Ϊ3
OutDim=2;               %�������ά��Ϊ2

%ԭʼ����
load highway.mat;
p=[sqrs;sqjdcs;sqglmj];  %�������ݾ���
t=[glkyl;glhyl];         %Ŀ�����ݾ���
[SamIn,minp,maxp,tn,mint,maxt]=premnmx(p,t); %ԭʼ������ʼ��

rand('state',sum(100*clock));    %����ϵͳʱ�����Ӳ��������
NoiseVar=0.01;                  %����ǿ��0.01(Ŀ�ķ�ֹ�������)
Noise=NoiseVar*randn(2,SamNum); %��������
SamOut=tn+Noise;                %��������ӵ����������

TestSamIn=SamIn;                %����ȡ�������������������ͬ����Ϊ����������
TestSamOut=SamOut;              %��������Ͳ�������һ��

MaxEpochs=100;                %���ѵ��50000��
lr=0.035;                       %ѧϰ����0.035
EO=0.65*10^(-3);                %����Ŀ�����
W1=0.5*rand(HiddenUnitNum,InDim)-0.1; %��ʼ����������������Ȩֵ
B1=0.5*rand(HiddenUnitNum,1)-0.1;     %��ʼ������������������ֵ
W2=0.5*rand(OutDim,HiddenUnitNum)-0.1;%��ʼ����������������Ȩֵ
B2=0.5*rand(OutDim,1)-0.1;            %��ʼ������������������ֵ
ErrHistory=[];                        %�м������Ԥ��ռ���ڴ�

for i=1:MaxEpochs
    HiddenOut=logsig(W1*SamIn+repmat(B1,1,SamNum));%���������
    NetworkOut=W2*HiddenOut+repmat(B2,1,SamNum);   %��������
    Error=SamOut-NetworkOut;                       %ʵ�����������֮��
    %SSE=sumsqr(Error)                              %���ۺ���
    SSSE=Error.*Error;
    SSE=sum(SSSE(:))
    
    
    ErrHistory=[ErrHistory SSE];
    if SSE<EO,break,end                %�ﵽ���Ҫ������ѧϰ����
        
    %����6����BP��������ĵĳ���
    %������Ȩֵ����ֵ�����ݴ��ۺ������ݶ��½��㷨����̬����
    Delta2=Error;
    Delta1=W2'*Delta2.*HiddenOut.*(1-HiddenOut);%sigmoid�ĵ�����f'(x)=f(x)(1-f(x))
    %������
    dW2=Delta2*HiddenOut';
    dB2=Delta2*ones(SamNum,1);
    %�����
    dW1=Delta1*SamIn';
    dB1=Delta1*ones(SamNum,1);
    %���������������֮���Ȩֵ����ֵ��������
    W2=W2+lr*dW2;
    B2=B2+lr*dB2;
    %���������������֮���Ȩֵ����ֵ��������
    W1=W1+lr*dW1;
    B1=B1+lr*dB1;
end

HiddenOut=logsig(W1*SamIn+repmat(B1,1,TestSamNum));%������������ս��
NetWorkOut=W2*HiddenOut+repmat(B2,1,TestSamNum);  %�����������ս��
a=postmnmx(NetworkOut,mint,maxt);                  %��ԭ���������Ľ��
x=1990:2009;                                       %ʱ����̶�
newk=a(1,:);                                       %�������������
newh=a(2,:);                                       %�������������
figure;
subplot(2,1,1);plot(x,newk,'r-o',x,glkyl,'b--+');   %���ƹ�·�������Ա�ͼ
legend('�������������','ʵ�ʿ�����','Location','northwest');
xlabel('���');ylabel('������/����');
title('Դ���������� ������ѧϰ�Ͳ��ԶԱ�ͼ');
         
subplot(2,1,2);plot(x,newh,'r-o',x,glhyl,'b--+');   %���ƹ�·�������Ա�ͼ
legend('�������������','ʵ�ʻ�����','Location','northwest');
xlabel('���');ylabel('������/���');
title('Դ���������� ������ѧϰ�Ͳ��ԶԱ�ͼ'); 

figure;
plot(1:MaxEpochs,ErrHistory,'r-o')
xlabel('��������');ylabel('���');
title('ѵ�����')
%%
%����ѵ���õ�ģ�ͽ���Ԥ��
%����ѵ���õ�BP���������ݽ��з���

%����ѵ���õ��������Ԥ��
%����ѵ���õ�������µ�����pnew����Ԥ��ʱ��ҲӦ����Ӧ�Ĵ���
pnew=[73.39 75.55;3.9635 4.0975 ;0.9880 1.0268];%2010��2011�������
pnewn=tramnmx(pnew,minp,maxp);   %����ԭʼ����Ĺ�һ�������������ݹ�һ��
HiddenOut = logsig(W1*pnewn+repmat(B1,1,ForcastSamNum));
anewn=W2*HiddenOut+repmat(B2,1,ForcastSamNum);            
anew=postmnmx(anewn,mint,maxt) %������Ԥ��õ������ݷ���һ��ԭ������