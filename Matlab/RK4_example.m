%%----------------------------------------------
%              dy/dt = y - t^2 + 1  0 <= t <= 2
%               y(0) = 0.5          h = 0.5
%%----------------------------------------------

%%-define constant symbol
t0 = 0;
w0 = 0.5;
h0 = 0.2;
%%-----------------------

rk45(t0,w0,h0)

function rk45(t,w,h)
epsilon = 0.00001;
i = 0;

while t<2
    h = min(h, 2-t);
    k1 = h*f(t,w);
    k2 = h*f(t+h/4,w+k1/4);
    k3 = h*f(t+3/8*h,w+k1*3/32+k2*9/32);
    k4 = h*f(t+12/13*h,w+1932/2197*k1-7200/2197*k2+7296/2197*k3);
    k5 = h*f(t+h, w+439*k1/216-8*k2+3680*k3/513-845*k4/4104);
    k6 = h*f(t+h/2, w-8*k1/27+2*k2-3544*k3/2565+1859*k4/4104-11*k5/40);
    w1 = w + 25*k1/216+1408*k3/2565+2197*k4/4104-k5/5;
    w2 = w + 16*k1/135+6656*k3/12825+28561*k4/56430-9*k5/50+2*k6/55;
    R = 1/h*abs(w1-w2);
    delta = 0.84*(epsilon/R)^1/4;
    

if R <= epsilon
    t = t + h; 
    w = w1;
    i = i+1;
    Tt(i) = t;
    It(i) = i;
    fprintf('Step %d: t = %6.4f, w = %18.15f\n', i, t, w);
    h = delta*h;
    fprintf('step delta = %18.15f\n',delta);
    deltap(i) = delta;
else
    h = delta*h;
end 
end
figure
plot(Tt,deltap)
end



function [] = rungekutta(t,y,h)
fprintf('Step 0: t = %6.4f, y = %18.15f\n', t, y);
for i=1:40
k11 = h*f(t,y);
k22 = h*f(t+h/2,y+k11/2);
k33 = h*f(t+h/2,y+k22/2);
k44 = h*f(t+h,y+k33);
y = y + 1/6*(k11+2*k22+2*k33+k44);
t = t + h;
fprintf('Step %d: t = %6.4f, y = %18.15f\n', i, t, y);
end


end

function v = f(t,y)
v = y - t^2 + 1;
end


