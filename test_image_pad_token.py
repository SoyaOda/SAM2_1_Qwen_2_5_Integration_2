#!/usr/bin/env python3
"""
image_padトークンのIDを確認するスクリプト
"""

from transformers import AutoTokenizer, AutoProcessor

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

print("=== 特殊トークンの確認 ===")
print(f"Vocabulary size: {len(tokenizer)}")
print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
print(f"BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")

# image関連のトークンを探す
print("\n=== image関連トークンの検索 ===")
vocab = tokenizer.get_vocab()
image_tokens = {k: v for k, v in vocab.items() if 'image' in k.lower()}
for token, token_id in sorted(image_tokens.items(), key=lambda x: x[1]):
    print(f"'{token}': {token_id}")

# |image_pad|>の確認
print("\n=== <|image_pad|>トークンの確認 ===")
test_tokens = ["<|image_pad|>", "<|image|>", "<image_pad>", "image_pad"]
for token in test_tokens:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"'{token}' -> ID: {token_id}")
    except:
        print(f"'{token}' -> Not found")

# トークナイズテスト
print("\n=== トークナイズテスト ===")
text = "This is a test <|image_pad|> text"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")

# プロセッサーの設定を確認
print("\n=== プロセッサー設定 ===")
if hasattr(processor, 'image_processor'):
    print(f"Image processor type: {type(processor.image_processor)}")
    if hasattr(processor.image_processor, 'image_pad_id'):
        print(f"Image pad ID: {processor.image_processor.image_pad_id}")