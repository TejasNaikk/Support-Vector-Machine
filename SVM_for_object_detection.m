[trD, trLb, valD, valLb, trRegs, valRegs]   = HW2_Utils.getPosAndRandomNeg();


x=trD;
y=trLb;

[d,n]=size(x);

kernel = x'*x;
f = -1.*ones(n,1);
beq = 0;
aeq = y';
y_diag = diag(y);
H = y_diag*kernel*y_diag;
lb = zeros(n,1);
C = 0.1;
ub = C.*ones(n,1);

alpha = quadprog(H,f,[],[],aeq,beq,lb,ub);

alpha_new=diag(alpha);

temp=alpha_new*y;
w=x*temp;

temp1=y(1);
b=min(y-(w'*x)')/2;

HW2_Utils.genRsltFile(w,b,'./val','outfile');
[ap,precision,recall] = HW2_Utils.cmpAP('outfile','./val');

ap