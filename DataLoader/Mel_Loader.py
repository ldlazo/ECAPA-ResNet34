from torch.utils.data import Dataset
import librosa

class AudioDataset(Dataset):
    def __init__(self, df, data_path):
        super(Dataset, self).__init__()
        self.df = df
        self.data_path = str(data_path)
    
    #Number of items in dataset
    
    def __len__(self):
        return len(self.df)
    
    
    #Getting i-th item from dataset
    
    def __getitem__(self, idx):
        
        audio_file = self.data_path + self.df.iloc[idx]["ID"]
        
        speaker = self.df.iloc[idx]["class_id"]
        
        audio, sample_rate = librosa.load(audio_file, offset=self.df.iloc[idx]['Start'], duration=2)
        
        mel_spec = librosa.feature.melspectrogram(audio, sample_rate, n_mels=80)
        log_mels = librosa.power_to_db(mel_spec)
        
        return log_mels, speaker