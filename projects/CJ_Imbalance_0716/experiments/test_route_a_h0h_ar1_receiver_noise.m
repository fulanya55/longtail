function test_route_a_h0h_ar1_receiver_noise()
%TEST_ROUTE_A_H0H_AR1_RECEIVER_NOISE Deterministic noise-process contract.
n = 200000; rho = 0.80; seed = 968677743;
a = route_a_h0h_ar1_receiver_noise(seed,n,1,rho);
b = route_a_h0h_ar1_receiver_noise(seed,n,1,rho);
assert(isequal(a,b),'AR(1) noise is not deterministic by receiver seed.');
power = mean(abs(a).^2);
pseudocov = abs(mean(a.^2));
lagOne = real(mean(a(2:end).*conj(a(1:end-1))) / mean(abs(a).^2));
assert(abs(power-1) < 0.02,'AR(1) power normalization contract failed.');
assert(pseudocov < 0.02,'AR(1) proper-complex pseudo-covariance contract failed.');
assert(abs(lagOne-rho) < 0.02,'AR(1) lag-one correlation contract failed.');
white = route_a_h0h_white_receiver_noise(seed,3000,1);
assert(abs(mean(abs(white).^2)-1) < 0.15,'Reference-white formula contract failed.');
fprintf('ROUTE_A_H0H_AR1_NOISE_TEST=PASS\n');
end
