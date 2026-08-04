function [ frame_idx ] = mpii_filter_3D_pose( annot3d, threshold, starting_frame)

% Since this filters pose in 3 dimensions, one need only provide the
% annotatins from one camera
% Starting from the first frame, it only keeps frames where the 
% movement was more than the threshold on atleast one joint


%frame_idx = [];
frame_idx = starting_frame;
prev_frame_idx = starting_frame;

for i = (starting_frame+1):size(annot3d,1)
    %Compute the distance per joint wrt the previous frame
    joint_distances = sqrt(sum((reshape(annot3d(prev_frame_idx,:),3,[]) - reshape(annot3d(i,:),3,[])).^2,1));
    if(~isempty(find(joint_distances > threshold, 1)))
        frame_idx = [frame_idx; i];
        prev_frame_idx = i;
    end
end

end

