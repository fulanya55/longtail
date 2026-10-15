function noise = route_a_h0h_ar1_receiver_noise(receiverNoiseSeed, sampleLength, noisePower, rho)
%ROUTE_A_H0H_AR1_RECEIVER_NOISE Deterministic proper-complex AR(1) receiver noise.
% This project-local wrapper deliberately mirrors the reference generator's
% receiver seed contract but never modifies reference-generator source code.
if ~(isscalar(receiverNoiseSeed) && isfinite(receiverNoiseSeed) && receiverNoiseSeed >= 0)
    error('RouteA:H0hSeed','receiverNoiseSeed must be a non-negative scalar.');
end
if ~(isscalar(sampleLength) && sampleLength == floor(sampleLength) && sampleLength > 1)
    error('RouteA:H0hLength','sampleLength must be an integer above one.');
end
if ~(isscalar(noisePower) && isfinite(noisePower) && noisePower > 0)
    error('RouteA:H0hPower','noisePower must be finite and positive.');
end
if ~(isscalar(rho) && isfinite(rho) && abs(rho) < 1)
    error('RouteA:H0hRho','rho must satisfy abs(rho)<1.');
end
stream = RandStream('mt19937ar','Seed',double(receiverNoiseSeed));
innovation = (randn(stream,1,sampleLength) + 1i*randn(stream,1,sampleLength)) / sqrt(2);
unit = complex(zeros(1,sampleLength));
unit(1) = innovation(1);
scale = sqrt(1-rho^2);
for index=2:sampleLength
    unit(index) = rho*unit(index-1) + scale*innovation(index);
end
noise = sqrt(noisePower) * unit;
noise = complex(double(real(noise)),double(imag(noise)));
end
