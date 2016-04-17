function KernelCCA
%kernel CCA
%% Generate Data
N = 200;
t = 2 * pi .* rand(N, 1);
n1 = 0.01 .* randn(N, 1); n2 = 0.01 .* randn(N, 1);
x = sin(t) + n1; y = cos(t) + n2;
plot(x, y, '.')
%% implement Kernel CCA 
gamma = 1; reg = 0.1; eta = 0.000000001;
K = GK(x, gamma); L = GK(y, gamma);
H = eye(N) - ones(N) ./ N;
Kt = H * K * H; Lt = H * L * H;
z = zeros(N);
A = [z, Kt * Lt / N;
     Lt * Kt / N, z];
B = [Kt * Kt + reg * Kt, z;
     z, Lt * Lt + reg * Lt];
%% Incomplete Cholesky Solution
R = inChole(B, eta);
At = pinv(R') * A * pinv(R);
[V, D] = eig(At);
coco_cca = abs(real(D(1, 1))) * N
e1 = pinv(R) * V(:, 1);
alpha = e1(1 : N); beta = e1(N + 1 : 2 * N);
f = map(alpha, K);
g = map(beta, L);
figure();
plot(x, f, '.', 'markersize', 15); xlabel('x'); ylabel('f(x)');
figure();
plot(y, g, '.', 'markersize', 15); xlabel('y'); ylabel('g(x)');
figure();
plot(f, g, '.', 'markersize', 15); xlabel('f(x)'); ylabel('g(x)');
%% Original Solution
[V, D] = eig(A, B);
coco_cca = abs(real(D(1, 1))) * N
e1 = V(:, 1);
alpha = e1(1 : N); beta = e1(N + 1 : 2 * N);
f = map(alpha, K);
g = map(beta, L);
figure();
plot(x, f, '.', 'markersize', 15); xlabel('x'); ylabel('f(x)');
figure();
plot(y, g, '.', 'markersize', 15); xlabel('y'); ylabel('g(x)');
figure();
plot(f, g, '.', 'markersize', 15); xlabel('f(x)'); ylabel('g(x)');

%Gaussian Kernel
function K = GK(x, gamma)
n = length(x);
K = zeros(n);
for i = 1 : n
    for j = 1 : n
        K(i, j) = exp(-(x(i) - x(j))^2 / gamma^2);
    end
end

%incomplete cholesky decomposition
function R = inChole(K, eta)
[ell, ell1] = size(K);
j = 0;
R = zeros(ell,ell);
d = diag(K);
[a,I(j+1)] = max(d);
while a > eta
  j = j + 1;
  nu(j) = sqrt(a);
  for i = 1:ell
    R(j,i) = (K(I(j),i) - R(:,i)'*R(:,I(j)))/nu(j);
    d(i) = d(i) - R(j,i)^2;
  end
  [a,I(j+1)] = max(d);
end
T = j;
R = R(1:T,:);
