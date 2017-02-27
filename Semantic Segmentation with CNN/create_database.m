function database = create_database(data_path)

% Load the dataset
data= load(strcat(data_path, '/semantic_segmentation_dataset.mat'));

% Initialize the struct database with the correct fields
pics= data.images_tr;

%switch from BGR to RGB
[~,~,~,t]= size(pics);

blues= pics(:,:,1,:);
reds= pics(:,:,3,:);
for i=1:t
    pics(:,:,1,t)= reds(:,:,:,t);
    pics(:,:,3,t)= blues(:,:,:,t);
end

set= data.sets_tr;
label= data.anno_tr -1;

images= struct('data', pics, 'label', label, 'set', set);
sets= struct('name', {'train', 'val', 'test'}, 'id', [1,2,3]);
classes= struct('name',data.clNames, 'id', (1:36));
meta= struct();

database= struct('images', images, 'sets', sets, 'meta', meta, 'classes', classes);
