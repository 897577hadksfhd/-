import numpy as np
import pandas as pd
from Bio import SeqIO


class OneHotEncoderProtein:
    def __init__(self):
        """初始化One-Hot编码器"""
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_index = {aa: i for i, aa in enumerate(self.amino_acids)}

    def clean_sequence(self, sequence):
        """清理序列中的非常见氨基酸(替换为X)"""
        return ''.join([aa if aa in self.aa_index else 'X'
                        for aa in sequence.upper()])

    def one_hot_encode(self, sequence):
        """
        对单个序列进行One-Hot编码
        返回: 展平后的特征向量 (seq_len * 20)
        """
        seq_len = len(sequence)
        # 初始化全零矩阵
        encoded = np.zeros((seq_len, len(self.amino_acids)))
        for i, aa in enumerate(sequence):
            if aa in self.aa_index:
                encoded[i, self.aa_index[aa]] = 1
        return encoded.flatten()

    def parse_txt_fasta(self, file_path):
        """解析TXT/FASTA文件，返回(序列ID列表, 序列列表)"""
        seq_ids = []
        sequences = []
        current_id = ""
        current_seq = ""

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):  # 标题行
                    if current_id:  # 保存上一个序列
                        seq_ids.append(current_id)
                        sequences.append(self.clean_sequence(current_seq))
                    current_id = line[1:]  # 去掉">"
                    current_seq = ""
                else:  # 序列行
                    current_seq += line

        # 添加最后一个序列
        if current_id:
            seq_ids.append(current_id)
            sequences.append(self.clean_sequence(current_seq))

        return seq_ids, sequences

    def process_file(self, txt_path):
        """
        处理整个文件
        返回: (特征矩阵numpy数组, 序列ID列表)
        """
        seq_ids, sequences = self.parse_txt_fasta(txt_path)
        features = np.array([self.one_hot_encode(seq) for seq in sequences])
        return features, seq_ids