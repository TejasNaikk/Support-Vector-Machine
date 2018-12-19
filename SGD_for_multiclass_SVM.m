dataset=load('C:/Users/tjnai/Downloads/hw2data/hw2data/q3_1_data.mat');

x=dataset.trD;
y=dataset.trLb;
loss = zeros(2000);

eta0 = 1;
eta1 = 100;
k = size(unique(y),1); %2
[d,n] = size(x);
w = zeros(d,k);

c=0.01;
m=containers.Map([1,-1],[1,2]);


for epoch=1:2000
    eta = eta0/(eta1+epoch);
    randindex = randperm(n);%random indices
    totalloss=0;
    for i=1:n
        %Finding Y hat first
        index=randindex(i);
        x_i=x(:,index);
        y_i=m(y(index));
        temp_w=w;
        temp_w(:,y_i)=(-1*inf);
        [temp_val,y_hat]=max(temp_w'*x_i);
        %getting l
        l = max((w(:,y_hat)'*x_i-w(:,y_i)'*x_i+1),0);

        for j=1:k
            if j==y_i
                if l>0
                    der_y_i=(w(:,y_i))./n - c.*(x_i);
                else
                    der_y_i = (w(:,y_i))./n;
                end            
                w(:,j) = w(:,j) - (eta*der_y_i);
            elseif j==y_hat
                if l>0
                    der_y_hat=(w(:,y_hat))./n + c.*(x_i);
                else
                    der_y_hat =(w(:,y_hat))./n;
                end            
                w(:,j) = w(:,j) - (eta*der_y_hat);
            else
                w(:,j) = w(:,j) - (eta*(w(:,j))./n); 
            
            end
        end
                
        l = max((w(:,y_hat)'*x_i-w(:,y_i)'*x_i+1),0); %Calculating again loss
        totalloss = totalloss + (sum(vecnorm(w).^2))/(2*n) + c*l;
        
    end
    loss(epoch) = totalloss;
end
plot(loss);

[y_hat_val,y_hat_ind] = max(w'*x);
y_hat_ind=y_hat_ind';
y_actual =zeros(n,1);
for i=1:n
    y_actual(i)=m(y(i));
end

accuracy=sum(y_actual==y_hat_ind)/n;
error=sum(y_actual ~= y_hat_ind)/n;
    


