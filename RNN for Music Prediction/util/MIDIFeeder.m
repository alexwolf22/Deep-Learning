classdef MIDIFeeder < handle
  properties(SetAccess = private, Hidden = true)
    % quantize to one step = sixteenth note
    quantize_onsets = 1/16;
    % quantize durations of MIDI notes to sixteenth note
    quantize_durs = 1/16;
    % lowest pssible pitch; 48 corresonds to C3
    low_note = 48;
    % highest possible pitch; 84 corresponds to C6
    high_note = 84;
    % number of pitches covered
    range;
  end % private hidden properties
  
  properties
    epochs_seen = 0;
    startix = 1;
    % root directories for MIDI files
    train_root;
    validation_root;
    test_root;
    % default bpm when inverting back to MIDI
    default_bpm = 120;
    % default velocity when inverting back to MIDI
    default_velocity = 127;
    % cell array of note matrices
    nm;
    % cell array of one-hot encoded note matrices
    encoded;
    % reshaped encoded matrices concatenated along first dimension
    data;
    % how much weight to associate with step events, which occur much more
    % frequently than note events
    step_weight;
    % class label for time step
    STEP_END;
    % total number of classes (2*range + 1)
    nclasses;
    % imdb format for MatConvNet
    imdb;
    % encode/decode only note on events, so that note continuation events
    % are also encoded as note on. This is the default right now, as note
    % continuaton is not implemented.
    note_on_only;
  end % public properties
  
  methods
    function obj = MIDIFeeder(train_root, validation_root, test_root, ...
        step_weight, note_on_only)
      if nargin < 5
        obj.note_on_only = false;
      else
        obj.note_on_only = note_on_only;
      end
      if nargin < 4
        obj.step_weight = 1;
      else
        obj.step_weight = step_weight;
      end
      if nargin < 1
        error('Constructor expects one input argument.');
      end
      rng(1234);
      obj.nm = {};
      obj.imdb = struct();
      obj.imdb.images = struct('set', []);
      obj.imdb.meta.sets = {}; 
      if exist('train_root', 'var') && ~isempty(train_root)
        obj.train_root = train_root;
        oldFolder = cd(train_root);
        try
          nm = dir2coll();
          n = length(nm);
          obj.nm = cat(2, obj.nm, nm);
          obj.imdb.images.set = cat(2, obj.imdb.images.set, ones(1, n));
          obj.imdb.meta.sets{1} = 'train'; 
        catch ME
          cd(oldFolder);
          rethrow(ME);
        end
        cd(oldFolder);
      end      
      if exist('validation_root', 'var') && ~isempty(validation_root)
        obj.validation_root = validation_root;
        oldFolder = cd(validation_root);
        try
          nm = dir2coll();
          n = length(nm);
          obj.nm = cat(2, obj.nm, nm);
          obj.imdb.images.set = cat(2, obj.imdb.images.set, 2*ones(1, n));
          obj.imdb.meta.sets{2} = 'val'; 
        catch ME
          cd(oldFolder);
          rethrow(ME);
        end
        cd(oldFolder);
      end
      if exist('test_root', 'var') && ~isempty(test_root)
        obj.train_root = test_root;
        oldFolder = cd(test_root);
        try
          nm = dir2coll();
          n = length(nm);
          obj.nm = cat(2, obj.nm, nm);
          obj.imdb.images.set = cat(2, obj.imdb.images.set, 3*ones(1, n));
          obj.imdb.meta.sets{3} = 'test'; 
        catch ME
          cd(oldFolder);
          rethrow(ME);
        end
        cd(oldFolder);
      end
      
      obj.range = obj.high_note - obj.low_note + 1;
      if obj.note_on_only
        obj.STEP_END = obj.range + 1;
      else
        obj.STEP_END = 2*obj.range + 1;
      end
      obj.nclasses = obj.STEP_END;
      preprocess(obj);
      encode_all(obj);
    end % constructor
    
    function preprocess(obj)
      empties = [];
      for i = 1:length(obj.nm)
        % remove note events occurring on channel 9, which is typically reserved
        % for drums
        obj.nm{i}(obj.nm{i}(:,3) == 9,:) = [];

        if isempty(obj.nm{i})
          empties = [empties, i]; %#ok<AGROW>
          continue;
        end
        
%         % transpose to c
%         obj.nm{i} = transpose2c(obj.nm{i});
% 
%         if isempty(obj.nm{i})
%           empties = [empties, i]; %#ok<AGROW>
%           continue;
%         end
        
        % remove notes below low_note
        obj.nm{i}(obj.nm{i}(:,4) < obj.low_note,:) = [];
        
        % remove notes above high_note
        obj.nm{i}(obj.nm{i}(:,4) > obj.high_note,:) = [];

        if isempty(obj.nm{i})
          empties = [empties, i]; %#ok<AGROW>
          continue;
        end
        
        % quantize notes and their durations
        obj.nm{i} = quantize(obj.nm{i}, obj.quantize_onsets, obj.quantize_durs);

        % remove notes with 0 duration
        obj.nm{i}(obj.nm{i}(:,2) == 0,:) = [];

        if isempty(obj.nm{i})
          empties = [empties, i]; %#ok<AGROW>
          continue;
        end
        % sort the array by rows
        [~, ix] = sort(obj.nm{i}(:,1));
        obj.nm{i} = obj.nm{i}(ix,:);        
      end
      obj.nm(empties) = [];
      obj.imdb.images.set(empties) = [];
    end % preprocess
    
    function encode_all(obj)
      % encode all MIDI files with the followingshape: 1xSxDxT, where S is
      % the sequence length, D is the number of classes, and T is the
      % number of obserations
      obj.encoded = cell(size(obj.nm));
      for i = 1:length(obj.nm)
        obj.encoded{i} = reshape(onehot_encode(obj, i), 1, [], obj.nclasses);
      end
    end % encode_all
    

    function [batchx, batchy] = get_batch(obj, ~, batch, seq_len)
      min_seq_len = inf;
      for b = batch
        min_seq_len = min(min_seq_len, size(obj.encoded{b},2));
      end

      % if no seq_len argument is given, all sequences in this batch will be
      % truncated to the length of the shortest sequence. Otherwise. all
      % sequences will be truncated or padded to seq_len.
      if nargin < 4
        seq_len = min_seq_len;
      end
      
      obj.data = zeros(1, seq_len, obj.nclasses, numel(batch));

      i = 1;
      for b = batch
        diff = size(obj.encoded{b}, 2) - seq_len;
        % if this example is longer than seq_len truncate, otherwise pad
        if diff > 0
          obj.data(:,:,:,i) = obj.encoded{b}(:,1:seq_len,:,:);
        elseif diff < 0
          obj.data(:,:,:,i) = cat(2, obj.encoded{b}, ...
            zeros(1, -diff, obj.nclasses));
        else
          obj.data(:,:,:,i) = obj.encoded{b};
        end
      i = i +1;   
      end
      
      batchx = obj.data(1,1:end-1,:,:);
      batchy = zeros(1,seq_len-1, 2, numel(batch));
      for i = 2:seq_len
        for j = 1:numel(batch)
          batchy(1,i-1,:,j) = ...
            obj.one_hot_to_class(obj.data(1,i,:,j));
        end
      end
    end % get_batch

    function cl = one_hot_to_class(obj, one_hot)
      cl = find(one_hot);
      if isempty(cl)
        cl = 0;
      end
      if cl == obj.STEP_END
        w = obj.step_weight;
      else
        w = 1;
      end
      cl = [cl w];
    end % one_hot_to_class
    
    function cl = one_hot_to_class_simple(~, one_hot)
      cl = find(one_hot);
      if isempty(cl)
        cl = 0;
      end
    end % one_hot_to_class
        
    function encoded = onehot_encode(obj, ix)
      % One-hot encoding of midi notes
      % nm_ is in the following format:
      % beat |  dur (beats) | channel | pitch | vel | secs | dur (secs)
      %
      % We want to encode each note as a one-hot vector, where:
      %  encoded(note+1) = 1 if this is a new instance of note
      %  encoded(obj.STEP_END) = 1 if a step just ended
      
      nm_ = obj.nm{ix};
      % vector storing the last seen notes
      notes_ctr = zeros(obj.range, 1);
      % total number of steps, where there are 1/obj.quantize_onsets/4 steps
      % per beat
      nsteps = (nm_(end, 1) - nm_(1, 1)) / obj.quantize_onsets / 4 + 1;
      % number of note events
      nnotes = size(nm_, 1);
      % guestimate of number of rows
      nrows = nnotes + nsteps + 2;
      encoded = zeros(nrows, obj.nclasses);

      % assumes nm_ has at least two rows
      six = 1;
      eix = 2;
      n = 1;
      while eix <= nnotes
        % find ending index of notes in this step
        while nm_(six,1) == nm_(eix,1)
          eix = eix + 1;
          if eix > nnotes
            break;
          end
        end
        % extract notes for this step
        notes = nm_(six:eix-1,:);
        
        % figure out which ones are new
        new_notes = notes(notes_ctr(notes(:,4)-obj.low_note+1) == 0, 4);
        
        % increment the durations for notes in the counter
        notes_ctr(notes(:, 4)-obj.low_note+1) = ...
          notes_ctr(notes(:, 4)-obj.low_note+1) + notes(:, 2);
        
        if eix <= nnotes
          nsteps = (nm_(eix, 1) - nm_(eix-1, 1))  / obj.quantize_onsets / 4;
        else
          nsteps = max(nm_(six:end, 2)) / obj.quantize_onsets / 4;
        end
        for i = 1:nsteps
          ix = find(notes_ctr > 0);
          if ~isempty(ix)
            for j = flip(ix)'
              % if this is a new note
              if ismember(j + obj.low_note - 1, new_notes)
                encoded(n, j) = 1;
              % otherwise this is a continued note
              else
                if obj.note_on_only
                  encoded(n, j) = 1;
                else
                  encoded(n, obj.range+j) = 1;
                end
              end
              n = n + 1;
            end
          end
          new_notes = [];
          notes_ctr(notes_ctr > 0) = ...
            notes_ctr(notes_ctr > 0) - obj.quantize_onsets * 4;
          % step_end marker
          encoded(n, obj.STEP_END) = 1;
          n = n + 1;
        end
        six = eix;
        eix = eix + 1;
      end
      if n-1 < nrows
        encoded(n:end, :) = [];
      end
    end % onehot_encode

    function nm = onehot_decode(obj, encoded, bpm)
      if nargin < 3
        bpm = obj.default_bpm;
      end
      beat = 0;
      secs = 0;
      secs_onsets_per_step = 60 / bpm * obj.quantize_onsets * 4;
      secs_durs_per_step = 60 / bpm * obj.quantize_durs * 4;
      encoded = reshape_for_decode(obj, encoded);
      six = 1;
      eix = 2;
      nm = zeros(size(encoded, 1), 7);
      notes_ix = zeros(obj.range, 1);
      while six < size(encoded, 1)
        % find the end of the time step
        while encoded(eix, obj.STEP_END) ~= 1
          if eix == size(encoded, 1)
            break;
          end
          eix = eix + 1;
        end
        notes_to_kill = ones(obj.range, 1);
        for i = six:eix-1
          new_note = find(encoded(i,:));
          % if this is a note-on event
          % must protect against repeated note-on events 
          if new_note + obj.low_note - 1 <= obj.high_note && ...
              notes_to_kill(new_note)
            notes_ix(new_note) = i;
            notes_to_kill(new_note) = 0;
            nm(i, :) = [beat, obj.quantize_durs, randi(9)-1, ...
              new_note + obj.low_note - 1, obj.default_velocity, ...
              secs, secs_durs_per_step];
          % if this is a note continuation event
          elseif ~obj.note_on_only && ...
               new_note + obj.low_note - 1 > obj.high_note && ...
              new_note <= 2*obj.range
            new_note = new_note - obj.range;
            ix = notes_ix(new_note);
            notes_to_kill(new_note) = 0;
            % even though this is a note contnuation event, there may not
            % have been a previous corresponding note onset
            if ix
              nm(ix, 2) = nm(ix, 2) + obj.quantize_durs;
              nm(ix, 7) = nm(ix, 7) + secs_durs_per_step;
            % if not, then treat it as a new note
            else
              notes_ix(new_note) = i;
              nm(i, :) = [beat, obj.quantize_durs, randi(9)-1, ...
                new_note + obj.low_note - 1, obj.default_velocity, ...
                secs, secs_durs_per_step];
            end
          end
        end
        notes_ix(logical(notes_to_kill)) = 0;
        beat = beat + obj.quantize_onsets;
        secs = secs + secs_onsets_per_step;
        six = eix;
        while encoded(six, obj.STEP_END) == 1
          if six == size(encoded, 1)
            break;
          end
          six = six + 1;
        end
        eix = six;
      end
    end % onehot_decode
    
    function reshaped = reshape_for_decode(obj, encoded)
      reshaped = reshape(permute(encoded, [2,4,3,1]), [], obj.nclasses);
    end % reshape_for_decode
    
  end % methods

end % classdef