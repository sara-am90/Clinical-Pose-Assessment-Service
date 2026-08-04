% Read in images and annotations provided in data_info and dump
% out the 2D and 3D annotations chunkwise 
%!!!!!!!! APPLIES IMAGE CORRECTION TOO !!!!!!!!!!!!!!!
function [] = mpii_create_muco_3dhp_composites(composite_info, out_path, folder_prefix, bg_data_path, fg_data_path, chair_data_path)
    mpii_config_paths

num_people = size(composite_info(1,:),2);
rng(311);
do_bg_augmentation = true;
if(isempty(bg_data_path))
    do_bg_augmentation = false;
else
    bg_paths_ = dir(bg_data_path);
    bg_paths_ = bg_paths_(6:end);
    bg_paths = cell(length(bg_paths_),1);
    for i = 1:length(bg_paths_)
        bg_paths{i} = [bg_data_path filesep bg_paths_(i).name];
    end
end
do_fg_augmentation = true;
if(isempty(fg_data_path))
    do_fg_augmentation = false;
else
    fg_paths_ = dir(fg_data_path);
    fg_paths_ = fg_paths_(6:end);
    fg_paths = cell(length(fg_paths_),1);
    for i = 1:length(fg_paths_)
        fg_paths{i} = [fg_data_path filesep fg_paths_(i).name];
    end    
end
do_chair_augmentation = true;
if(isempty(chair_data_path))
    do_chair_augmentation = false;
else
    chair_paths_ = dir(chair_data_path);
    chair_paths_ = chair_paths_(6:end);
    chair_paths = cell(length(chair_paths_),1);
    for i = 1:length(chair_paths_)
        chair_paths{i} = [chair_data_path filesep chair_paths_(i).name];
    end        
end


%Chunk Size (just the number of images per 
nimg_in_chunk = 500;  
chunk_size = nimg_in_chunk; %500
N = size(composite_info,1);
num_chunk = ceil(N / nimg_in_chunk);

num_joints = size(composite_info{1,1}{1,3},2);

files = {};
gauss_filt_size = 2;
gauss_filt = fspecial('gaussian', 5, 0.8);

current_input_idx = 1;

out_folder = fullfile(out_path, folder_prefix); 
system(sprintf('mkdir %s', out_folder)); 
for chunk = 1:num_chunk
    
    n = min(chunk_size, (N-current_input_idx+1) );
    
    %Order these front to back
    joint_loc3 = zeros(3,num_joints,num_people,n);
    joint_loc2 = zeros(2,num_joints,num_people,n);
    img_names = cell(1,n);
    
    fprintf('Doing Chunk %d of %d\n', chunk, num_chunk);

	current_out_idx = 1;
    for nc = 1:n
        
        people_depth = [];
        for np = 1:num_people
            people_depth = [people_depth, composite_info{current_input_idx,np}{1,4}(3,15)];
        end
        [~, people_index] = sort(people_depth);

       
        for np = 1 : num_people
            pidx = people_index(np);
            %crop_rect{np} = composite_info{current_input_idx,pidx}{1,2};
            sub_id = composite_info{current_input_idx,pidx}{1,1}(1);
            seq = composite_info{current_input_idx,pidx}{1,1}(2);
            [bg_augmentable, ub_augmentable, lb_augmentable, chair_augmentable, ~, ~] = mpii_get_sequence_info(sub_id, seq);
            img_base = fullfile(mpii_data_path, sprintf('S%d/Seq%d', sub_id, seq), 'imageSequence');
            fgmask_base = fullfile(mpii_data_path, sprintf('S%d/Seq%d', sub_id, seq), 'FGmasks');
            chairmask_base = fullfile(mpii_data_path, sprintf('S%d/Seq%d', sub_id, seq), 'ChairMasks');
            img_path = [img_base filesep sprintf('img_%d_%06d.jpg', composite_info{current_input_idx,pidx}{1,1}(3), composite_info{current_input_idx,pidx}{1,1}(4))];
            input_img = imadjust(imread(img_path),[],[],0.7 + rand()/10);   
            fg_mask_path = [fgmask_base filesep sprintf('img_%d_%06d.jpg', composite_info{current_input_idx,pidx}{1,1}(3), composite_info{current_input_idx,pidx}{1,1}(4))];
            fg_mask{np} = imgaussfilt(imread(fg_mask_path),gauss_filt_size);
            chairmask_path = [chairmask_base filesep sprintf('img_%d_%06d.jpg', composite_info{current_input_idx,pidx}{1,1}(3), composite_info{current_input_idx,pidx}{1,1}(4))];
            chair_mask{np} = imread(chairmask_path);
            chair_mask{np} = double(repmat(chair_mask{np}(:,:,1), 1, 1, 3))/255;


            if(do_fg_augmentation)
                if(ub_augmentable || lb_augmentable)
                    %Remove all color from the image
                    input_gray = repmat(rgb2gray(input_img),1,1,3);
                    %input_img = a*input_gray + (1-a)*input_img;
                    if(ub_augmentable)
                        upper_mask = repmat(double(fg_mask{np}(:,:,2)), 1, 1, 3)/255;
                        fg_img = imresize(read_rgb_image(fg_paths{randi(length(fg_paths))}), [size(input_img,1), size(input_img,2)]);
                        input_gray = double(input_gray) .* (double(upper_mask)/2 + 0.5) + double(fg_img) .*(0.5 - double(upper_mask)/2);    
                        input_img = double(input_img) .* (upper_mask ) + double(input_gray) .* (1-(upper_mask));

                    end
                    if(lb_augmentable)
                        lower_mask = repmat(double(fg_mask{np}(:,:,3)), 1, 1, 3)/255;
                        fg_img = imresize(read_rgb_image(fg_paths{randi(length(fg_paths))}), [size(input_img,1), size(input_img,2)]);
                        input_gray = double(input_gray) .* (lower_mask/2 +0.5) + double(fg_img) .*(0.5 - lower_mask/2); 
                        input_img = double(input_img) .* (lower_mask) + double(input_gray) .* (1-(lower_mask));
                    end

                end

            end
            if(do_chair_augmentation && chair_augmentable)
                chair_img = imresize(read_rgb_image(chair_paths{randi(length(chair_paths))}), [size(input_img,1), size(input_img,2)]);
                input_img = double(input_img) .* (chair_mask{np}/2 +0.5) + double(chair_img) .*(0.5 - chair_mask{np}/2);       
            end
                
            image{np} = input_img;
            joint_loc2(:,:,np,current_out_idx) = composite_info{current_input_idx,pidx}{1,3};
            joint_loc3(:,:,np,current_out_idx) = bsxfun(@minus, composite_info{current_input_idx,pidx}{1,4}, composite_info{current_input_idx,pidx}{1,4}(:,15));

        end
        %Now we have the individual images FG and chair augmented and we
        %have the masks available, now composite the two together and BG
        %augment the image behind
        if(do_bg_augmentation && bg_augmentable)
            bg_mask = repmat(double(fg_mask{end}(:,:,1)), 1, 1, 3)/255;
            bg_img = imresize(read_rgb_image(bg_paths{randi(length(bg_paths))}), [size(image{end},1), size(image{end},2)]);
            image{end} = double(image{end}) .* bg_mask + double(bg_img) .*(1 - bg_mask);
        end        

        input_img = image{end};
        for np = (num_people-1):-1:1
            bg_mask = repmat(double(fg_mask{np}(:,:,1)), 1, 1, 3)/255;
            input_img = double(input_img) .* chair_mask{np} + double(image{np}) .*(1 - chair_mask{np});
            input_img = double(image{np}) .* bg_mask + double(input_img) .*(1 - bg_mask);
        end
        

        img_names{current_out_idx} = sprintf('%06d.jpg',current_input_idx);
        imwrite(input_img/255, fullfile(out_folder, img_names{current_out_idx}));
                
        fprintf('Doing image %d out of %d\n',current_input_idx, N);

        %Flag to track input index and output index separately if creating multiple ouputs with
        %the same input, such as with different augmentations for the same input 
        current_out_idx = current_out_idx +1;

	    current_input_idx = current_input_idx+1;
      end %Images in current chunk

     save(fullfile(out_folder, sprintf('chunk_%05d_annot.mat',chunk)), 'img_names', 'joint_loc3', 'joint_loc2');  

end %All Chunks

end
    
function [img] = read_rgb_image(filename)
     try
        img = imread(filename);
     catch
         img = cat(3, rand*ones(2000),rand*ones(2000),rand*ones(2000));
     end
     if(size(img,3) < 3)
         img = repmat(img(:,:,1),1,1,3);
     end
end
