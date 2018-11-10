def preprocess_monk(filename_train,filename_test,new_filename):

    data_in_train = {}
    with open(filename_train) as f:

        for line in f:

            example = line.split(' ')[-1]
            data_in_train[example] = 1

    with open(filename_test) as f:
        with open(new_filename,"w+") as new:
            for line in f:

                example = line.split(' ')[-1]
                if not example in  data_in_train:
                    new.write(line)
