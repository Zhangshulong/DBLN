import os
import numpy as np
# Load UEA dataset from arff files
def load_from_arff_file(full_file_path_and_name, replace_missing_vals_with="NaN"):
    instance_list = []
    class_val_list = []
    data_started = False
    is_multi_variate = False
    is_first_case = True
    n_cases = 0
    n_channels = 1
    with open(full_file_path_and_name, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                if (
                    is_multi_variate is False
                    and "@attribute" in line.lower()
                    and "relational" in line.lower()
                ):
                    is_multi_variate = True

                if "@data" in line.lower():
                    data_started = True
                    continue
                # if the 'data tag has been found, the header information
                # has been cleared and now data can be loaded
                if data_started:
                    line = line.replace("?", replace_missing_vals_with)

                    if is_multi_variate:
                        line, class_val = line.split("',")
                        class_val_list.append(class_val.strip())
                        channels = line.split("\\n")
                        channels[0] = channels[0].replace("'", "")
                        if is_first_case:
                            n_channels = len(channels)
                            n_timepoints = len(channels[0].split(","))
                            is_first_case = False
                        elif len(channels) != n_channels:
                            raise ValueError(
                                f" Number of channels not equal in "
                                f"dataset, first case had {n_channels} channel "
                                f"but case number {n_cases+1} has "
                                f"{len(channels)}"
                            )
                        inst = np.zeros(shape=(n_channels, n_timepoints))
                        for c in range(len(channels)):
                            split = channels[c].split(",")
                            inst[c] = np.array([float(i) for i in split])
                    else:
                        line_parts = line.split(",")
                        if is_first_case:
                            is_first_case = False
                            n_timepoints = len(line_parts) - 1
                        class_val_list.append(line_parts[-1].strip())
                        split = line_parts[: len(line_parts) - 1]
                        inst = np.zeros(shape=(n_channels, n_timepoints))
                        inst[0] = np.array([float(i) for i in split])
                    instance_list.append(inst)
    return np.asarray(instance_list), np.asarray(class_val_list)


def uea_load_data(dataroot, dataset):
    train_data_path = os.path.join(dataroot, dataset, dataset + "_TRAIN.arff")
    test_data_path = os.path.join(dataroot, dataset, dataset + "_TEST.arff")

    X_train, y_train = load_from_arff_file(train_data_path) # <class 'numpy.ndarray'> [n_cases, n_channels, n_timepoints]
    X_test, y_test = load_from_arff_file(test_data_path)
    num_classes = len(np.unique(y_train))
   
    sum_dataset = np.concatenate((X_train, X_test), axis=0) # [n_cases, n_channels, n_timepoints] 
    sum_target = np.concatenate((y_train, y_test), axis=0)
    
    _, sum_target = np.unique(sum_target, return_inverse=True)
    sum_target = sum_target.astype(np.int64)

    return sum_dataset, sum_target, num_classes