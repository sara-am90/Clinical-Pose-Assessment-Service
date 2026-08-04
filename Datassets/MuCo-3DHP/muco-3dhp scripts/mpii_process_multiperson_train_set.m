%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to sample frames from MPI-INF-3DHP dataset and create MuCo-3DHP composites
% The code and data is made available only for academic research purposes 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1928);
addpath('./util/') %The utilities come from MPI-INF-3DHP code
mpii_config_paths

%Path where MPI-INF-3DHP is downloaded
mpi_inf_3dhp_path = mpii_data_path; %See mpii_config_paths.m

%Path where the composited images should be written out
out_data_path = %Provide a path ;

%Paths where textures for augmentation are kept
bg_data_path = %Provide the path to a directory with background textures;
fg_data_path = %Provide the path to directory with foreground textures;

%Sample the next frame after some joint has moved by atleast this amount since
%the previously sampled frame
joint_thresh = 175; %mm

%Frame number to start sampling from 
starting_frame = 180; %Give some time for the tracking to stabilize perhaps 

%Cameras to use from MPI-INF-3DHP 
camera_set = mpii_get_camera_set('relevant');

%The subset of joints to use, joint parent indices, and the joint names 
[joint_set, o1_parents, o2_parents, joint_names] = mpii_get_joint_set('extended');
all_joints = mpii_get_joint_set('all');

subject_id = {
              'S1', 1; 
              'S2', 2;
              'S3', 3;
              'S4', 4;
              'S5', 5; 
              'S6', 6;
              'S7', 7;
              'S8', 8;
               };
num_sequences = 2;  %2 sequences per person

%Go over the sequences and collect the sampled frames into individual sets
%Then go and randomly select pairs and pass this information to the data
%creation script
data_info = cell(length(camera_set), length(subject_id)*num_sequences);

%% There is some stuff happening here for legacy reasons, ignore that
dat_idx = 0;
for i = 1:size(subject_id,1)
    fprintf('%s\n', subject_id{i,1});
    for j = 1:num_sequences
      dat_idx = dat_idx + 1;  
      %Read in the annotations
      dat = load([mpi_inf_3dhp_path filesep 'S' int2str(subject_id{i,2}) filesep 'Seq' int2str(j) filesep 'annot.mat']);
      %Filter the useful indices 
      master_filt_idx = int16(mpii_filter_3D_pose(dat.univ_annot3{1}, joint_thresh, starting_frame));
      
      for ci = 1:length(camera_set)
          seq_info = cell(1,6); %Cols for img_name, crop_size, 2D annotation, 3D annotation, 3D O1, 3D O2, Chair Position
          idx = 1;
          filt_idx = int16((rand()-0.5)*50)+master_filt_idx;
          filt_idx(filt_idx > length(dat.frames)) = length(dat.frames);
          frames = dat.frames(filt_idx);
          cam = camera_set(ci);
          c = find(dat.cameras == cam);
          fprintf('Doing camera %d, cidx %d\n', cam, c);
          %Get the correct camera index and filter annotation frames
          annot2 = dat.annot2{c}(filt_idx,:);
          annot2 = reshape(annot2', 2, length(all_joints), 1, []);
          univ_annot3 = dat.univ_annot3{c}(filt_idx,:);
          univ_annot3 = reshape(univ_annot3', 3, length(all_joints), 1, []);
          
          for f = 1:length(filt_idx)
             frame = frames(f);
             %%Check if something is in the frame, if it is, go cwaaaazy
             %if( ~isempty(find(annot2(:,:,:,f)<2048,1)) && ~isempty(find(annot2(:,:,:,f) < 0,1)) )
             if( sum(  prod((annot2(:,:,:,f)<2048) & (annot2(:,:,:,f)> 0)) ) > 0  )
                 %compute crop size. You can use this information if you want to create crops around subjects
                  annot = annot2(:,:,:,f)';
                  min2D = min(annot)-320; 
                  max2D = max(annot)+320;
                  min2D(1) = max(1,min2D(1));
                  min2D(2) = max(1,min2D(2));
                  max2D(1) = min(2048,max2D(1));
                  max2D(2) = min(2048,max2D(2));
                  rect = [min2D, (max2D-min2D)];
                 %Pass subject ID, sequence, camera and frame information out 
                 seq_info{idx,1} = [subject_id{i,2} j cam frame];
                 seq_info{idx,2} = rect;
                 seq_info{idx,3} = annot2(:,joint_set,:,f);
                 seq_info{idx,4} = univ_annot3(:,joint_set,:,f);
                 seq_info{idx,5} = seq_info{idx,4} - seq_info{idx,4}(:,o1_parents,:);
                 seq_info{idx,6} = seq_info{idx,4} - seq_info{idx,4}(:,o2_parents,:);
                 idx = idx + 1;
             end
          end % Useful frames
          data_info{ci,dat_idx} = seq_info;
      end %Selected cameras
    end %Sequences per subject
end %Subjects

%% Saving it so that most of the code above this need not be rerun multiple times
save([out_data_path filesep 'dataset_info.mat'], 'data_info');
%load([out_data_path filesep 'dataset_info.mat']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to create composites of 4 subjects 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
composite_info = cell(1,4);
num_quadruples = 60000;
for ci = 1: length(camera_set)
   quadruples = randi(size(data_info,2), num_quadruples, 4);
   for j = 1:num_quadruples
       composite_info((ci-1) * num_quadruples + j, :) = {data_info{ci,quadruples(j,1)}(randi(size(data_info{ci,quadruples(j,1)},1), 1, 1), :), data_info{ci,quadruples(j,2)}(randi(size(data_info{ci,quadruples(j,2)},1), 1, 1), :), data_info{ci,quadruples(j,3)}(randi(size(data_info{ci,quadruples(j,3)},1), 1, 1), :), data_info{ci,quadruples(j,4)}(randi(size(data_info{ci,quadruples(j,4)},1), 1, 1), :)};
   end      
end
rng(1999);
composite_info = composite_info(randperm(size(composite_info,1), size(composite_info,1)), :);

%% Saving it so that most of the code above this need not be rerun multiple times
save([out_data_path filesep 'sampled_dataset_info.mat'], 'composite_info', '-v7.3');
%load([out_data_path filesep 'sampled_dataset_info.mat']);

%%%%%%%%%%%
%% Dump full composited frames of the dataset
%%%%%%%%%%%
ts = 1;
%Augmented data
mpii_create_muco_3dhp_composites(composite_info(1:10,:), out_data_path , sprintf('augmented_set_%03d',ts),  bg_data_path, fg_data_path, fg_data_path);
%Unaugmented data
mpii_create_muco_3dhp_composites(composite_info(1:10,:), out_data_path , sprintf('unaugmented_set_%03d',ts),  [], [], []);
