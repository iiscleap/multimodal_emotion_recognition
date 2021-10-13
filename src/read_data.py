import pickle5 as pickle
import numpy as np
import pandas as pd

def create_test_mask(videos, labels, max_utt_length, file_name):
    mask = np.empty([len(videos), max_utt_length])
    for ind, vid in enumerate(videos):
        act_length = len(labels[vid])
        rem_length = max_utt_length - act_length
        ones = np.ones((act_length, 1))
        zeros = np.zeros((rem_length, 1))
        total = np.concatenate((ones, zeros), axis = 0)
        mask[ind, :] = total[:, 0]
    with open(file_name, 'wb') as f:
        np.save(f, mask)

def create_mask(videos, labels, max_utt_length, file_name, ses_no, fold_num):
    """Mask saved for each utterance in the video. 1 means true utterance, 0 means 
    padding in the utterance
    """
    train_num, val_num = 0, 0
    for vid in videos:
        if ses_no in vid:
            val_num += 1
        else:
            train_num += 1
    mask_val = np.empty([val_num, max_utt_length])
    mask_train = np.empty([train_num, max_utt_length])
    train_ind, val_ind = 0, 0
    for ind, vid in enumerate(videos):
        act_length = len(labels[vid])
        rem_length = max_utt_length - act_length
        ones = np.ones((act_length, 1))
        zeros = np.zeros((rem_length, 1))
        total = np.concatenate((ones, zeros), axis = 0)
        if ses_no in vid:
            mask_val[val_ind, :] = total[:, 0]
            val_ind += 1
        else:
            mask_train[train_ind, :] = total[:, 0]
            train_ind += 1
    val_file_name = file_name.replace("train", "val" + str(fold_num))
    train_file_name = val_file_name.replace("val", "train")
    with open(val_file_name, 'wb') as f:
        np.save(f, mask_val)
    with open(train_file_name, 'wb') as f:
        np.save(f, mask_train)

def create_test_array(videos, max_utt_length, mat, feature_length, mode, file_name):
    """Creates the npy file for test for the appropriare modality passed in the argument.
    All the utterances are padded to the maximum length available in the dataset."""
    if mode == 'label':
        Y = np.empty([len(videos), max_utt_length])
    else:
        Y = np.empty([len(videos), feature_length, max_utt_length])
    for ind, vid in enumerate(videos):
        if mode == 'label':
            target = np.array(mat[vid]).reshape(-1, 1)
            rem_length = max_utt_length - target.shape[0]
            mask = -1*np.ones((rem_length, 1))
            target = np.concatenate((target, mask), axis = 0)
            target[target == 4] = 0
            Y[ind, :] = target[:,0]
        else:
            target = np.array(mat[vid]).T
            rem_length = max_utt_length - target.shape[1]
            mask = np.zeros((feature_length, rem_length))
            target = np.concatenate((target, mask), axis = 1)
            Y[ind, :, :] = target
    with open(file_name, 'wb') as f:
        np.save(f, Y)

def create_array(videos, max_utt_length, mat, feature_length, mode, file_name, ses_no, fold_num):
    """Creates the npy file for train/valid for the appropriare modality passed in the argument.
    All the utterances are padded to the maximum length available in the dataset.
    """
    train_num, val_num = 0, 0
    for vid in videos:
        if ses_no in vid:
            val_num += 1
        else:
            train_num += 1
    if mode == 'label':
        Y_train = np.empty([train_num, max_utt_length])
        Y_val = np.empty([val_num, max_utt_length])
    else:
        Y_train = np.empty([train_num, feature_length, max_utt_length])
        Y_val = np.empty([val_num, feature_length, max_utt_length])
    train_ind, val_ind = 0, 0
    for ind, vid in enumerate(videos):
        if mode == 'label':
            target = np.array(mat[vid]).reshape(-1, 1)
            rem_length = max_utt_length - target.shape[0]
            mask = -1*np.ones((rem_length, 1))
            target = np.concatenate((target, mask), axis = 0)
            target[target == 4] = 0
            if ses_no in vid:
                Y_val[val_ind, :] = target[:,0]
                val_ind += 1
            else:
                Y_train[train_ind, :] = target[:,0]
                train_ind += 1
        else:
            target = np.array(mat[vid]).T
            rem_length = max_utt_length - target.shape[1]
            mask = np.zeros((feature_length, rem_length))
            target = np.concatenate((target, mask), axis = 1)
            if ses_no in vid:
                Y_val[val_ind, :, :] = target
                val_ind += 1
            else:
                Y_train[train_ind, :, :] = target
                train_ind += 1
    val_file_name = file_name.replace("train", "val" + str(fold_num))
    train_file_name = val_file_name.replace("val", "train")
    with open(val_file_name, 'wb') as f:
        np.save(f, Y_val)
    with open(train_file_name, 'wb') as f:
        np.save(f, Y_train)    

def create_text_csv(videos, labels, sentences, path):
    #Creates the csv file for training the text unimodal model
    df = pd.DataFrame(columns = ['Vid_name','Utterance', 'Emotion'])
    for vid_id in videos:
        utterance_list = sentences[vid_id]
        emotion_list = labels[vid_id]
        df2 = pd.DataFrame()
        df2['Utterance'] = utterance_list
        df2['Emotion'] = emotion_list
        df2['Vid_name'] = [vid_id]*len(emotion_list)
        df = df.append(df2, ignore_index = True) 
    df["Emotion"] = df["Emotion"].replace(4, 0)
    df = df[df["Emotion"] < 4]
    df.to_csv(path, index=False)

def create_data(pickle_path, train_aud, train_img, train_text, train_lab, test_aud,
                test_img, test_text, test_lab, path_train, path_test, train_csv, test_csv):
    """Reads the pickle file and filters out utterances with emotions happy(0), sad(1),
    neutral(2), angry(3), excited(4). Creates npy files for each of the text, audio and image
    modalities, though image is not used. For Session 5 as test, each of the other 4 sessions 
    are considered as validation even though finally only session 1 is used for validation.
    """

    f = pickle.load(open(pickle_path, 'rb'), encoding= 'latin1')
    labels, text, audio, image = f[2], f[3], f[4], f[5]
    sentences, train_videos, test_videos = f[6], f[7], f[8]
    for k, v in labels.items():
        new_labels, text_new, audio_new, img_new = [], [], [], []
        sentences_new = []
        for i in range(len(v)):
            if v[i] <= 4:
                new_labels.append(v[i])
                text_new.append(text[k][i])
                audio_new.append(audio[k][i])
                img_new.append(image[k][i])
                sentences_new.append(sentences[k][i])
        labels[k] = new_labels
        text[k] = text_new
        audio[k] = audio_new
        image[k] = img_new
        sentences[k] = sentences_new

    max_utt_length_train = max(len(labels[i]) for i in train_videos)
    max_utt_length_test = max(len(labels[i]) for i in test_videos)
    max_utt_length = max(max_utt_length_train, max_utt_length_test)

    audio_feature_length = audio[list(train_videos)[0]][0].shape[0]
    text_feature_length = text[list(train_videos)[0]][0].shape[0]
    image_feature_length = image[list(train_videos)[0]][0].shape[0]

    train_videos = list(train_videos)
    train_videos.sort()
    sessions = ['Ses01', 'Ses02', 'Ses03', 'Ses04']
    for ind, ses_no in enumerate(sessions):
        create_mask(train_videos, labels, max_utt_length, path_train, ses_no, ind)
        create_array(train_videos, max_utt_length, labels, 1, 'label', train_lab, ses_no, ind)
        create_array(train_videos, max_utt_length, audio, audio_feature_length, 'audio', train_aud, ses_no, ind)
        create_array(train_videos, max_utt_length, text, text_feature_length, 'text', train_text, ses_no, ind)
        create_array(train_videos, max_utt_length, image, image_feature_length, 'image', train_img, ses_no, ind)
    create_text_csv(train_videos, labels, sentences, train_csv)

    test_videos = list(test_videos)
    test_videos.sort()
    create_test_mask(test_videos, labels, max_utt_length, path_test)
    create_test_array(test_videos, max_utt_length, labels, 1, 'label', test_lab)
    create_test_array(test_videos, max_utt_length, audio, audio_feature_length, 'audio', test_aud)
    create_test_array(test_videos, max_utt_length, text, text_feature_length, 'text', test_text)
    create_test_array(test_videos, max_utt_length, image, image_feature_length, 'image', test_img)
    create_text_csv(test_videos, labels, sentences, test_csv)
