import os

class Tokenizer:
    def __init__(self, vocab_path):
        self.token_to_id = {}
        self.id_to_token = []
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip().split('\t')[0]  # sentencepiece .vocab format
                self.token_to_id[token] = idx
                self.id_to_token.append(token)

    def encode(self, text):
        # 공백은 SentencePiece가 '▁'로 처리하므로 수동으로 대응
        return [
            self.token_to_id.get(ch if ch != ' ' else '▁', self.unk_id)
            for ch in text
        ]

    def decode(self, ids):
        tokens = [self.id_to_token[i] for i in ids if i < len(self.id_to_token)]
        return ''.join(tokens).replace('▁', ' ').strip()
    
    @property
    def vocab_size(self):
        return len(self.id_to_token)

    @property
    def pad_id(self):
        return self.token_to_id.get('<pad>', 0)

    @property
    def blank_id(self):
        return self.token_to_id.get('<blank>', 0)

    @property
    def unk_id(self):
        return self.token_to_id.get('<unk>', 0)
