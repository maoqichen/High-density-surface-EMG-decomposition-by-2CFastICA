function [S,Mu]=CFICA2(EMG,param,Mu)
% $$$ Documentation of 2CFastICA  $$$
% Author:Maoqi Chen
% Email:hiei@mail.ustc.edu.cn  or  maoqi.chen@uor.edu.cn
% Update:2024.02.13
% Please cite:
% [1]M. Chen and P. Zhou, "2CFastICA: A Novel Method for High Density Surface EMG Decomposition Based on 
%    Kernel Constrained FastICA and Correlation Constrained FastICA," in IEEE Transactions on Neural 
%    Systems and Rehabilitation Engineering, vol. 32, pp. 2177-2186, 2024, doi: 10.1109/TNSRE.2024.3398822.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Full format: [S,Mu]=CFICA2(EMG,param,Mu);
%Suggested format : [S,Mu]=CFICA2(EMG);
%                   [S,Mu]=CFICA2(EMG,param);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT:
% EMG, multi-channel EMG signal data matrix (channels X samples).
%      Before feeding the EMG signal into the program, invalid channels 
%      should be removed and each channel (row) should be properly filtered 
%      (especially baseline wander) and (preferably) normalized (mean=0, std=1).
% param, a structure contains the important parameters of 2CfastICA. 
%       (PS: The names of parameters in "param" should be the same as this documentation.)
%  param.delay, the delay factor of FastICA, defaults to 10 samples.
%  param.fs, the sampling frequency of EMG signal, defaults to 2048 Hz . 
%  param.wavelength, the wavelength of the MUAPs, defaults to fix(fs/20).
%  param.peakinterval, the minimum sampling interval of successive spikes of one motor unit, defaults to fix(fs/40).
%  param.convergethreshold, the converge threshold of FastICA, defaults to 1e-6.
%  param.valleymode, feature selection of valley-seeking ((0 [PCA] or 1 [RMS DASDV] or 2 [TSNE])), defaults to 0.
% Mu, a structure array contains the information of identified motor units.
%     This input works when user wish to continue the decomposition based on the existing outcomes.
%  Mu.MUpulse, the FastICA output (MU index X samples), defaults to [].
%  Mu.S, the firing trains extracted from Mu.MUpulse (MU index X samples), defaults to [] . 
%  Mu.Wave, the estimated waveform matrix. Column i contains the waveform information of all identified MUs in channel i
%           (each segment of length param.wavelength corresponds to a MUAP waveform.), defaults to [].
% OUTPUT:
% S, the output matrix of identified spike trains, the same as "Mu.S".
% Mu, the same as "Mu" in the input arguments.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CFICA2 is an interactive multi-channel EMG decomposition program that sequentially
% estimates the motor unit firing trains by kernel constrained FastICA. Upon identifying 
% each motor unit spike train, the program generates a figure displaying this spike train
% alongside the firing train extracted by an automatically determined threshold. Then users 
% can provide specific instructions based on the results and prompts shown in the figure 
% to determine how the program should process the current result and proceed to next step.
% With different operations, the user can perform five actions:
% (1)	Just press "enter": accept current result.
% (2)	Input "0": discard the current spike train and look for the next one.
% (3)	Input "a positive real number (except 66 and 88)": manually set the threshold as 
%       the input number to extract the firing train from the current spike train and update
%       it by correlation constrained FastICA. After that, operation (1) or (2) or (3) or (5)
%       can be repeated until satisfactory treatment of the result is obtained.
% (4)	Input "66": use valley-seeking clustering and estimate new spike train after clustering. 
%       After that, operation (1) or (2) or (3) can be repeated.
% (5)	Input "88": discard the current result and end the program (when no reliable spike train can be found). 
%       After that, the program will display the final decomposition result (of a random selected channel)
%       and ask the users if there are any unreliable motor units that need to be discarded. 
%       If so, input the array of motor unit indexes to be discarded; if not, just press "enter".
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin==1
    param=struct();
    Mu.Wave=[];Mu.MUpulse=[];Mu.S=[];
end
if nargin==2
    Mu.Wave=[];Mu.MUpulse=[];Mu.S=[];
end

if (~isfield(param,'delay'))
    param.delay=10;
end

if (~isfield(param,'fs'))
    param.fs=2048;
end
if (~isfield(param,'wavelength'))
    param.wavelength=fix(param.fs/20);
end
if (~isfield(param,'peakinterval'))
    param.peakinterval=fix(param.fs/40);
end
if (~isfield(param,'valleymode'))
    param.valleymode=0;
end

if (~isfield(param,'convergethreshold'))
    param.convergethreshold=1e-6;
end

[Xt] = Whitening_PCA(EMG,param.delay);
if nargin<3
    C=[];YT=[];MUnum=0;
else
    YT=yanchi(Mu.MUpulse,param.delay) ;
    C=YT*Xt';
    MUnum=size(Mu.S,1);
end

Total_num=floor(size(Xt,1)/(2*param.delay+1));
while MUnum<Total_num
    [Y,~]=ker_fastica(Xt,C,param.convergethreshold);
    [ths,St]=thresh_cov(Y,param.peakinterval);
    figure;set(gcf,'Position',get(0,'ScreenSize'));subplot(211);plot(Y);
    title({'Press "enter": accept current firing train.' ;'Input "0": discard the current spike train and look for the next one.' ...
        ;'Input "66": use valley-finding clustering.';'Input "88": discard the current result and end the program.';'Input "other positive real number": manually set the threshold.'});
    subplot(212);plot(St);title(strcat('previous threshold:',num2str(ths)));
    do=input('');close
    if isempty(do)
        [Y,~]=CICA(Xt,St,param.convergethreshold);
    end
    if ~isempty(do) && do(1)==66
        F=getspike(Y,thresh_cov(Y,param.peakinterval),1,param.fs,param.valleymode);
        if ~isempty(F)
            yc=zeros(size(F,2),size(Xt,2));
            for ll=1:size(F,2)
                yc(ll,F{ll})=1;
            end
        end
        for ll=1:size(F,2)
            [Y,~]=CICA(Xt,yc(ll,:),param.convergethreshold);
            [ths,St]=thresh_cov(Y,param.peakinterval);
            figure;set(gcf,'Position',get(0,'ScreenSize'));subplot(211);plot(Y);
            title({'Press "enter": accept current firing train.' ;'Input "0": discard the current spike train and look for the next one.' ...
                ;'Input "other positive real number": manually set the threshold.'});
            subplot(212);plot(St);title(strcat('previous threshold:',num2str(ths)));
            do2=input('valley_seeking');close

            if isempty(do2)
                [Y,~]=CICA(Xt,St,param.convergethreshold);
            end
            while (~isempty(do2) && do2(1)~=0 )
                St=searchspike_in(Y,do2,param.peakinterval);
                [Y,~]=CICA(Xt,St);
                figure;set(gcf,'Position',get(0,'ScreenSize'));subplot(211);plot(Y);
                title({'Press "enter": accept current firing train.' ;'Input "0": discard the current spike train and look for the next one.' ...
                    ;'Input "other positive real number": manually set the threshold.'});
                subplot(212);plot(St);title(strcat('previous threshold:',num2str(do2)));
                do2=input('valley_seeking');close
            end
            if do2==0
                continue;
            end
            YT=[YT;yanchi(Y,param.delay)];
            C=YT*Xt';
            Mu.MUpulse=[Mu.MUpulse;Y];
            MUnum=MUnum+1;
            Mu.S=[Mu.S;St];
        end
        continue;
    end
    while (~isempty(do) && do(1)~=0 && do(1)~=88)
        St=searchspike_in(Y,do,param.peakinterval);
        [Y,~]=CICA(Xt,St);
        figure;set(gcf,'Position',get(0,'ScreenSize'));subplot(211);plot(Y);
        title({'Press "enter": accept current firing train.' ;'Input "0": discard the current spike train and look for the next one.' ...
            ;'Input "88": discard the current result and end the program.';'Input "other positive real number": manually set the threshold.'});
        subplot(212);plot(St);title(strcat('previous threshold:',num2str(do)));
        do=input('');close
    end
    if do==0
        continue;
    elseif do==88
        break;
    end
    YT=[YT;yanchi(Y,param.delay)];
    C=YT*Xt';
    Mu.MUpulse=[Mu.MUpulse;Y];
    MUnum=MUnum+1;
    Mu.S=[Mu.S;St];
end
[Mu.S,Mu.MUpulse]=dis_spike(Mu.S,Mu.MUpulse);

[Mu.Wave,~,~,~]=PEELOFF(EMG,Mu.S,param.wavelength,1);
do=input('Any unreliable MUs?(enter their index):');
if ~isempty(do)
    Mu.S(do,:)=[];Mu.MUpulse(do,:)=[];Mu.Wave=[];
    if ~isempty(Mu.S)
        [Mu.Wave,~,~,~]=PEELOFF(EMG,Mu.S,param.wavelength,1);
    end
end
S=Mu.S;

    function [y,Q,xt,Cond] = Whitening_PCA(x,delay,pro_trun)
        if nargin==1
            delay=0;
            pro_trun=0;
        end
        if nargin==2
            pro_trun=0;
        end
        [Nsignal, Nsample] = size(x);
        xt = zeros((delay+1)*Nsignal,Nsample);
        for k= 1:Nsignal
            xt((delay+1)*(k-1)+1:(delay+1)*k,:)=toeplitz([x(k,1),zeros(1,delay)],x(k,:));
        end
        xt=xt-mean(xt,2)*ones(1,Nsample);
        x_cov=cov(xt');
        Cond=cond(x_cov);
        [E,D]=eig(x_cov);
        lamda=diag(D);
        num=length(lamda);trun=fix(num*(1-pro_trun));
        [ww,j]=sort(lamda,'descend');E=E(:,j(1:trun)');D=diag(ww(1:trun));

        Q=sqrt(D)\(E)';
        y=Q*xt;
    end

    function S=searchspike_in(x,high,peak_interval)
        S=zeros(size(x));
        if high>0
            [~,L] = findpeaks(x,'minpeakheight',high,'MINPEAKDISTANCE',peak_interval);
            S(1,L)=1;
        elseif high<0
            [~,L] = findpeaks(-x,'minpeakheight',-high,'MINPEAKDISTANCE',peak_interval);
            S(1,L)=1;
        else S=[];
        end
    end

    function [xt] = yanchi(x,R)
        [Nsignal, Nsample] = size(x);
        xt = zeros((2*R+1)*Nsignal,Nsample);
        for i = 1:Nsignal
            xt((2*R+1)*(i-1)+1:(2*R+1)*i,:)=toeplitz([x(i,(R+1):-1:1),zeros(1,R)],[x(i,(R+1):end),zeros(1,R)]);
        end
    end

    function [W,residual,huifu,MUAPT]=PEELOFF(y,S,wavelength,p)
        a=randi(size(y,1),1,1);
        if isempty(S)
            residual=y;
            W=[];
            huifu=y;
            return
        end
        y=y-mean(y,2)*ones(1,size(y,2));
        [~,xt,~,CC]=boxing(y(a,:),S,wavelength,p,a);
        W=CC*y';
        if p>=2
            for k=1:size(W,2)
                W2=reshape(W(:,k),[wavelength,size(S,1)]);
                W2=wavesurgery(W2')';
                W(:,k)=W2(:);
            end
        end
        MUAPT=zeros(size(y,1),size(S,2),size(S,1));
        for iiii=1:size(S,1)
            MUAPT(:,:,iiii)=W((iiii-1)*wavelength+1:iiii*wavelength,:)'*xt((iiii-1)*wavelength+1:iiii*wavelength,:);
        end
        huifu=sum(MUAPT,3);
        residual=y-huifu;
        function [W,xt,huifu,CC]=boxing(x,S,wavelength,p,a)
            [n,T]=size(S);
            yanchi=fix(0.4*wavelength);
            xt=zeros(n*wavelength,T);
            for j=1:n
                xt((j-1)*wavelength+1:j*wavelength,:)=toeplitz([S(j,(yanchi+1):-1:1),zeros(1,(wavelength-yanchi-1))],[S(j,(yanchi+1):end),zeros(1,yanchi)]);
            end
            %xt=sparse(xt);
            %C=(xt*xt')\xt;
            CC=pinv(xt');
            W=CC*x';
            if p>=2
                Wt=reshape(W,[wavelength,n]);
                Wt=wavesurgery(Wt')';W=Wt(:);
            end
            huifu=W'*xt;
            if (p>=1)
                figure;
                maxi=1.5*max(max(abs(x)),max(abs(huifu)));
                subplot(2,4,2:4);plot(x);hold on;
                plot(huifu-maxi);hold on;
                plot(x-huifu-2*maxi);
                hold on;
                title(strcat('Ch',32, num2str(a)));
                subplot(2,4,6:8);
                for iii=1:size(S,1)
                    plot(S(iii,:)-3*iii);hold on;
                end
                hold on;
                subplot(n,4,1:4:4*n-3);jg=max(abs(W));
                for j=1:n
                    plot(W(1+wavelength*(j-1):wavelength*j)-1.5*jg*j,'linewidth',3);hold on;
                end
                pause(2);
            end
        end
    end
    function [S,MUpulse]=dis_spike(y,Mupulse,p)
        if nargin==2
            p=0.4;
        end
        S=[];MUpulse=[];
        while ~ isempty(y)
            S=[S;y(1,:)];MUpulse=[MUpulse;Mupulse(1,:)];
            Co=zeros(size(y,1),1);
            for j=1:size(y,1)
                Co(j)=max(xcorr(y(1,:),y(j,:),'coeff'));
            end
            y(Co>=p,:)=[];
            Mupulse(Co>=p,:)=[];
        end
    end

    function [Y,w]=ker_fastica(X,A,convergethreshold1,ini)
        if nargin<3
            convergethreshold1=1e-6;
        end
        c1=0;

        if isempty(A)
            if nargin<4
                w=randn(size(X,1),1);w=normalize(w,'norm');
            else
                w=mean(X(:,ini),2);w=normalize(w,'norm');
            end
            Y=w'*X;w0=zeros(size(w));
            K=1;
        else
            K=null(A);
            X=K'*X;
            if nargin<4
                w=randn(size(X,1),1);w=normalize(w,'norm');
            else
                w=mean(X(:,ini),2);w=normalize(w,'norm');
            end
            Y=w'*X;w0=zeros(size(w));
        end

        while (abs(1-abs(w0'*w))>convergethreshold1)&&(c1<1000)
            w0=w;
            w=X*tanh(Y)'-sum((1-(tanh(Y)).^2))*w0;

            w=normalize(w,'norm');
            Y=w'*X;
            c1=c1+1;
        end
        w=K*w;
        if skewness(Y')<0
            Y=-Y;
            w=-w;
        end
        if c1>=1000
            disp('warn:FASTICA does not converge whin 1000 iterations.');
        end
    end


    function [y,w]=CICA(xw,S,convergethreshold1)
        if nargin<3
            convergethreshold1=1e-6;
        end
        c1=0;
        T=size(xw,2);miu=0.3*T;
        r=xw*S';r=normalize(r,'norm');
        w=r;w0=zeros(size(w));
        y=w'*xw;
        while (abs(1-abs(w0'*w))>convergethreshold1)&&(c1<1000)
            w0=w;
            w=miu*r-xw*tanh(y)'+sum((1-(tanh(y)).^2))*w0;
            %w=miu*r-xw*tanh(y)';
            w=normalize(w,'norm');
            y=w'*xw;
            c1=c1+1;
        end

        if skewness(y')<0
            y=-y;
            w=-w;
        end
        if c1>=1000
            disp('warn:CICA does not converge whin 1000 iterations.');
        end
    end
    function [F,Spike,Sc,L,C]=getspike(x,th,p,fs,mode)
        interval=fix(fs/100);
        win=fix(fs/200);
        [~,L] = findpeaks(x(2*win+1:end-2*win),'minpeakheight',th,'MINPEAKDISTANCE',interval);
        L=L+2*win;
        if numel(L)<5
            F=[];Spike=[];Sc=[];L=[];C=[];
            return;
        end

        Spike=zeros(length(L),2*win+1);
        for ii=1:length(L)
            [~,pos]=max(x(L(ii)-win:L(ii)+win));
            pos=pos+L(ii)-win-1;
            Spike(ii,:)=x(pos-win:pos+win);
        end

        if mode==0
            [SC_pca] = Whitening_PCA(Spike');
            Sc=SC_pca(1:2,:)';
        elseif mode==1
            Sc(:,1)=sqrt(mean(Spike.^2,2));
            Sc(:,2)=sqrt(mean(diff(Spike,1,2).^2,2));
        end

        [C]=valley_seeking(Sc,15,0.3,2);
        if p==1
            figure;set(gcf,'Position',get(0,'ScreenSize'));
            for ii=1:max(C)
                scatter(Sc(C==ii,1),Sc(C==ii,2)); hold on
            end
            scatter(Sc(C==-1,1),Sc(C==-1,2));hold on
            pause(0.5);close
        end
        F=cell(1,max(C));
        for ii=1:max(C)
            F{ii}=L(C==ii);
        end

    end
    function [C,Cd,J]=valley_seeking(x,t1,t2,t3)
        D=squareform(pdist(x));d=size(D,1);
        [~,l]=sort(D,2);L=zeros(d,d);
        for ii=1:d
            L(ii,l(ii,:))=0:d-1;
        end
        Ss=(L+L')/2;
        J=abs(L-L')./Ss.^(1+1/d);[~,L_s]=sort(L,2);J(isnan(J))=0;
        D2=zeros(d,d);
        for ii=1:d-1
            for j=ii+1:d
                D2(ii,j)=(L(ii,j)*sum(L(L_s(ii,1:L(ii,j)),ii))+L(j,ii)*sum(L(L_s(j,1:L(j,ii)),j)))/(L(ii,j)*L(ii,j)*(L(ii,j)-1)+L(j,ii)*L(j,ii)*(L(j,ii)-1));
            end
        end
        D2=D2+D2';
        D2(isnan(D2))=1;
        Cd=(Ss<=t1).*(J<=t2).*(D2<=(t3/2));
        Total=1:d;
        C=zeros(d,1);k=1;
        while sum(C==0)
            X=Total(C==0);
            A=X(1);
            B=find(sum(Cd(A,:),1));
            while (~isempty(setdiff(B,A)))
                A=union(A,B);
                B=find(sum(Cd(A,:),1));
            end
            if numel(A)<d/5
                C(A)=-1;
            else
                C(A)=k;
                k=k+1;
            end
        end

    end
    function [th,Spike]=thresh_cov(x,peak_interval)
        pp=200;b=sort(x,'descend');
        t=linspace (2,b(20),pp);
        cov=ones(size(t));
        for i=1:length(t)
            [~,d]=findpeaks(x,'minpeakheight',t(i),'MINPEAKDISTANCE',peak_interval);
            dd=diff(d);
            cov(i)=std(dd)/mean(dd);
        end
        [~,index]=min(cov(cov>0.05));
        th=t(index);
        Spike=zeros(size(x));
        [~,d]=findpeaks(x,'minpeakheight',th,'MINPEAKDISTANCE',peak_interval);
        Spike(d)=1;
    end

end
