function rnn_write_generated_to_audio(generated, feeder, filename)
%WRITE_GENERATED_TO_AUDIO synthesizes audio and writes it to an audiofile.
% GENERATED is the generated output from your model. Generally speaking,
%   it os produced by calling RNN_GENERATE witjh appropriate arguments.
% FEEDER is the instance of the MIDIFeeder class that was used to buld your
%   model.
% FILENAME is the path to store the audio file. It should end in .wav

nm = feeder.onehot_decode(generated);
w = nmat2snd(nm);
w = w - mean(w);
w = w / max(abs(w));
audiowrite(filename, w, 22050);
end