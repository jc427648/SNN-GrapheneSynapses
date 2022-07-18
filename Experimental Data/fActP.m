
function out = fActP(x,params)
    n=length(x);
    x1 = params(1);%x1=-45;
    x2 = params(2);%x2=45;
    
    amp_p = params(4);%amp_p=-1.4;
    amp_n = params(3);%amp_n=-1.4;
    xop = params(5);%xop=20;
    xon = params(6);%xon=10;
    out=zeros(n,1);
    if x1>0
        exit(0);
    elseif x2<0
        exit(0);
    end
    %quick note: A+ is highest positive amplitude, Tau+ is timing constant for
    %+. This relates to amp_n and xon respectively.A-T->A+T+.
    i1=find(x>x1&x<0);
    out(i1)=-amp_n*(exp(x(i1)/xon)-exp(x1/xon))/(1-exp(x1/xon));
    i2=find(x>0&x<x2);
    out(i2)=amp_p*(exp(-x(i2)/xop)-exp(-x2/xop))/(1-exp(-x2/xop));