import pandas as pd

class DataLoader():
    
    def __init__(self, path_labels, path_train=None, path_val=None, path_test=None):
        self.path_labels = path_labels
        self.path_train = path_train
        self.path_val = path_val
        self.path_test = path_test
        self.get_labels(path_labels)
        if self.path_train:
            self.train_df = self.load_video_labels(self.path_train)
        if self.path_val:
            self.val_df = self.load_video_labels(self.path_val)
        if self.path_test:
            self.test_df = self.load_video_labels(self.path_test, labels_exist=False)
        
    def get_labels(self, path_labels):
        self.labels_df = pd.read_csv(path_labels, names=['label'])
        self.labels = [str(label[0]) for label in self.labels_df.values]
        self.n_labels = len(self.labels)
        self.label_to_int = dict(zip(self.labels, range(self.n_labels)))
        self.int_to_label = dict(enumerate(self.labels))
        
    def load_video_labels(self, path_subset, labels_exist=True):
        df = pd.read_csv(path_subset)
        df = df[['video_id', 'label']]
        if labels_exist:
            return df[df.label.isin(self.labels)]
        else:
            return df