import torch
import pandas as pd
from Bio import SeqIO
import esm
from torch.utils.data import DataLoader, TensorDataset


class ESM2FeatureExtractor:
    def __init__(self, model_name='esm2_t12_35M_UR50D', device=None):
        """
        严格保持与原始代码一致的ESM2特征提取器
        参数:
            model_name: 必须使用'esm2_t12_35M_UR50D'保持与原有代码一致
            device: 若不指定则自动检测
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 严格使用原有代码的加载方式
        self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.model = self.model.to(self.device)
        self.model.eval()

        # 保持原有batch_converter初始化方式
        self.batch_converter = self.alphabet.get_batch_converter()

    def clean_sequence(self, sequence):
        """完全复制原有清洗逻辑（只替换O为X）"""
        return sequence.replace('O', 'X')

    def parse_fasta(self, file_path):
        """完全复制原有FASTA解析逻辑"""
        data = []
        with open(file_path, 'r') as file:
            lines = file.readlines()

        protein_name = ""
        sequence = ""
        for line in lines:
            line = line.strip()
            if line.startswith(">"):
                if protein_name:
                    data.append((protein_name, sequence))
                protein_name = line[1:]
                sequence = ""
            else:
                sequence += line

        if protein_name:
            data.append((protein_name, sequence))

        return data

    def extract_features(self, fasta_file, batch_size=8):
        """
        严格保持原有特征提取流程
        返回:
            torch.Tensor: 与原有代码完全相同的embeddings
            list: 序列ID列表
        """
        # 保持原有处理流程
        sequences = self.parse_fasta(fasta_file)
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
        batch_tokens = batch_tokens.to(self.device)

        # 完全复制原有的DataLoader处理方式
        dataset = TensorDataset(batch_tokens)
        data_loader = DataLoader(dataset, batch_size=batch_size)

        all_embeddings = []
        for batch in data_loader:
            with torch.no_grad():
                results = self.model(batch[0], repr_layers=[12])  # 固定使用第12层

            # 完全相同的特征提取方式
            embeddings = results["representations"][12]
            embeddings = embeddings.mean(dim=1)  # 保持原有平均方式
            all_embeddings.append(embeddings)
            torch.cuda.empty_cache()

        # 保持相同的合并方式
        return torch.cat(all_embeddings, dim=0), [seq[0] for seq in sequences]

    def save_features(self, embeddings, output_file):
        """完全复制原有的保存方式"""
        embeddings_np = embeddings.cpu().numpy()
        pd.DataFrame(embeddings_np).to_csv(output_file, index=False)
        return output_file


# 提供与原有代码完全兼容的独立函数
def load_model(device):
    """保持原有函数接口"""
    extractor = ESM2FeatureExtractor(device=device)
    return extractor.model, extractor.alphabet


def extract_features(fasta_file, model, alphabet, device, batch_size=8):
    """保持原有函数接口"""
    extractor = ESM2FeatureExtractor(device=device)
    return extractor.extract_features(fasta_file, batch_size)


def save_features_as_csv(embeddings, output_file):
    """保持原有函数接口"""
    extractor = ESM2FeatureExtractor()
    return extractor.save_features(embeddings, output_file)