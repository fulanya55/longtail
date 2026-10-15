function noise = route_a_h0h_white_receiver_noise(receiverNoiseSeed, sampleLength, noisePower)
%ROUTE_A_H0H_WHITE_RECEIVER_NOISE Exact reference compose_jamming_sample noise formula.
stream = RandStream('mt19937ar','Seed',double(receiverNoiseSeed));
noise = sqrt(noisePower/2) * (randn(stream,1,sampleLength) + 1i*randn(stream,1,sampleLength));
noise = complex(double(real(noise)),double(imag(noise)));
end
