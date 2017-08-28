import os

saved_models_path = '/playpen/ammirato/Data/Detections/saved_models/'
base_name = 'FRA_TD_1-28_archA2_0'
file_extension = '.h5'

model_names = os.listdir(saved_models_path)

for name in model_names:

    if not name[:len(base_name)] == base_name:
        continue

    new_name = name[:len(base_name)]
    suffix = name[len(base_name):]
    under_ind1 = suffix.index('_') 
    try:
        under_ind2 = suffix.index('_',1) 
    except:
        continue #file has already been renamed
    epoch = suffix[under_ind1+1:under_ind2]
    new_name = new_name + '_' + epoch + file_extension
    os.rename(saved_models_path+name, saved_models_path+new_name)
