[trD, trLb, valD, valLb, trRegs, valRegs]   = HW2_Utils.getPosAndRandomNeg();

%HW2_Utils.genRsltFile(w,b,'./val','outfile');

pos_x = trD(:,1:size(trLb(trLb==1)));
pos_y = trLb(1:size(trLb(trLb==1)))';

neg_x = trD(:,size(trLb(trLb==1))+1:size(trD,2));
neg_y = trLb(size(trLb(trLb==1))+1:size(trD,2))';

x_train = [pos_x neg_x];
y_train = [pos_y neg_y]';

%[weights,bias] = train_quadratic(trainingX,trainingY);

x=x_train;
y=y_train;

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

obj_list = zeros(1,10);
ap_list = zeros(1,10);

for epoch = 1:8
    
    original_val = size(neg_x,2);
    A_temp = (w'*neg_x)+b;
    A_temp = A_temp';
    A = and(A_temp>-1,A_temp<1);
    neg_x=neg_x(:,A);
    
    imFiles = ml_getFilesInDir(sprintf('%s/%sIms/', HW2_Utils.dataDir, 'train'), 'jpg');
    nIm = length(imFiles);            
    rects = cell(1, nIm);
    startT = tic;
    for i=1:50
       ml_progressBar(i, nIm, 'Ub detection', startT);
       im = imread(imFiles{i});
       rects{i} = HW2_Utils.detect(im, w, b);
       rects_neg = rects{i}(1:4,rects{i}(5,:)<0);
       rects_pos = rects{i}(1:4,rects{i}(5,:)>0);
       
       [imH, imW,~] = size(im);
       badIdxs = or(rects_neg(3,:) > imW, rects_neg(4,:) > imH);
       rects_neg = rects_neg(:,~badIdxs);
                
                
       % Remove random rects that overlap more than 30% with an annotated upper body
       
       nsize = size(rects_pos,2);
       
       for j=1:nsize
           overlap = HW2_Utils.rectOverlap(rects_neg, rects_pos(:,j));                    
           rects_neg = rects_neg(:, overlap < 0.3);
           if isempty(rects_neg)
               break;
           end;
       end;
       
       [D_i, R_i] = deal(cell(1, size(rects_neg,2)));
       for j=1:size(rects_neg,2)
           
           imReg = im(rects_neg(2,j):rects_neg(4,j), rects_neg(1,j):rects_neg(3,j),:);
           imReg = imresize(imReg, HW2_Utils.normImSz);
           R_i{j} = imReg;
           D_i{j} = HW2_Utils.cmpFeat(rgb2gray(imReg));                    
       end
       
       dim = size(pos_x,1);
       num = size(D_i,2);
       B = zeros(dim,num);
       for k=1:num
           B(:,k) = D_i{k};
       end
       
       %if(size(B,2))>1000
       %    B = B(:,1:1000);
       %end
       neg_x = [neg_x B];
    end
    
    new_val = size(neg_x,2);
    temp = new_val-original_val;
    if (temp)>=0
        %if (temp) >1000
        %    neg_y = [neg_y [-1.*ones(1,1000)]];
        %else
        neg_y = [neg_y [-1.*ones(1,(temp))]];
%        end
    else
        neg_y = neg_y(1,1:(size(neg_y,2))+temp);
    end
    
    x_train = [pos_x neg_x];
    y_train = [pos_y neg_y]';

    x=x_train;
    y=y_train;

    [d,n]=size(x);

    kernel = x'*x;
    f = -1.*ones(n,1);
    beq = 0;
    aeq = y';
    y_diag = diag(y);
    H = double(y_diag*kernel*y_diag);
    lb = zeros(n,1);
    C = 0.1;
    ub = C.*ones(n,1);

    [alpha,obj] = quadprog(H,f,[],[],aeq,beq,lb,ub);

    alpha_new=diag(alpha);

    temp=alpha_new*y;
    w=x*temp;

    temp1=y(1);
    b=min(y-(w'*x)')/2;

    -1*obj
    obj_list(epoch) = -1*obj;
    [ap,precision,recall] = HW2_Utils.cmpAP('outfile','./val');
    ap_list(epoch) = ap;
end

plot(obj_list);
plot(ap_list);
HW2_Utils.genRsltFile(w,b,'./test','4_2_file');


%[ap,precision,recall] = HW2_Utils.cmpAP('outfile','./val');

