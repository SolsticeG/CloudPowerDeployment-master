%��ͼ �Զ��庯��drawplot ���� x y 
function drawplot(x, y1,t1,y2,t2,y3,t3,y4,t4,xtext,ytext,titletext)
plot(x, y1,'-g',x,y2,':r',x,y3,'--b',x,y4,'-.k'); %%ʹ��matlab����plot()��ͼ
legend(t1,t2,t3,t4)
grid on %������ʾ
xlabel(xtext)
ylabel(ytext)
title(titletext)