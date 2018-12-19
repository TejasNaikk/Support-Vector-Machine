dataset=load('C:/Users/tjnai/Downloads/hw2data/hw2data/q3_1_data.mat');

x=dataset.trD;
y=dataset.trLb;

[d,n]=size(x);
kernel = x'*x;
f = -1.*ones(n,1);
beq = 0;
aeq = y';
y_diag = diag(y);
H = y_diag*kernel*y_diag;
lb = zeros(n,1);
C = 10;
ub = C.*ones(n,1);

[alpha,objective] = quadprog(H,f,[],[],aeq,beq,lb,ub);

alpha_new=diag(alpha);

temp=alpha_new*y;
w=x*temp;

temp1=y(1);
b=min(y-(w'*x)')/2;

x_val=dataset.valD;
y_val=dataset.valLb;

bias_1=b*ones(1,size(y_val,1));
y_pred = sign(w'*x_val+bias_1)';
accuracy = sum(y_pred==y_val)/size(y_val,1);

support_vectors = w'*x_val + bias_1;
num_sv = sum(and(support_vectors<1,support_vectors>-1));

[C,order]=confusionmat(y_val,y_pred);

