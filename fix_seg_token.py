#!/usr/bin/env python3
"""
SEGトークン問題の修正スクリプト
1. ProcessorのトークナイザーにもSEGトークンを追加
2. SEGトークン埋め込みのrequires_gradを確実に有効化
"""

import torch
from pathlib import Path
import sys

# プロジェクトのパスを追加
sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.models.lisa_model import LISA_Model
from src.utils.tokenizer_utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor

def main():
    print("=" * 80)
    print("SEGトークン問題の修正")
    print("=" * 80)
    
    # 設定
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        torch_dtype="auto",
        freeze_qwen=True,
        freeze_sam=True,
        train_seg_token=True,
        use_flash_attention=False
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # トークナイザーの準備
    print("\n1. トークナイザーの準備")
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    seg_token_id = tokenizer.convert_tokens_to_ids(config.seg_token)
    print(f"SEGトークンID: {seg_token_id}")
    
    # プロセッサの準備
    print("\n2. プロセッサの準備と修正")
    processor = AutoProcessor.from_pretrained(config.qwen_model_name)
    
    # ProcessorのトークナイザーにもSEGトークンを追加（重要！）
    if config.seg_token not in processor.tokenizer.get_vocab():
        print(f"ProcessorのトークナイザーにSEGトークンを追加します...")
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [config.seg_token]})
        print(f"追加後のProcessor語彙サイズ: {len(processor.tokenizer)}")
    else:
        print(f"ProcessorのトークナイザーにはすでにSEGトークンが存在します")
    
    # モデルの初期化
    print("\n3. モデルの初期化")
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    model = model.to(device)
    
    # SEGトークン埋め込みの確認と修正
    print("\n4. SEGトークン埋め込みの確認と修正")
    if model.seg_token_id is not None:
        word_embeddings = model.qwen.get_input_embeddings()
        seg_embedding = word_embeddings.weight[model.seg_token_id]
        
        print(f"修正前のrequires_grad: {seg_embedding.requires_grad}")
        
        if config.train_seg_token and not seg_embedding.requires_grad:
            print("SEGトークン埋め込みを学習可能に設定...")
            # 直接requires_gradを設定
            model.qwen.get_input_embeddings().weight[model.seg_token_id].requires_grad_(True)
            
            # 再確認
            seg_embedding = model.qwen.get_input_embeddings().weight[model.seg_token_id]
            print(f"修正後のrequires_grad: {seg_embedding.requires_grad}")
    
    # テスト：トークン化の確認
    print("\n5. トークン化のテスト")
    
    # 修正されたprocessorでテスト
    from PIL import Image
    dummy_image = Image.new('RGB', (448, 448), color=(128, 128, 128))
    text_with_seg = "Segment the red object in this image. <SEG>"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": text_with_seg}
            ]
        }
    ]
    
    processed = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    input_ids = processed['input_ids']
    print(f"input_ids shape: {input_ids.shape}")
    
    # SEGトークンが含まれているか確認
    seg_positions = (input_ids == seg_token_id).nonzero(as_tuple=True)
    if len(seg_positions[1]) > 0:
        print(f"✅ SEGトークンが正しくトークン化されました！位置: {seg_positions[1].tolist()}")
    else:
        print(f"❌ SEGトークンがまだトークン化されていません")
        
        # デバッグ：入力テキストを直接トークン化
        print("\n直接トークン化テスト:")
        direct_tokens = processor.tokenizer(text_with_seg, return_tensors="pt")
        direct_seg_pos = (direct_tokens['input_ids'] == seg_token_id).nonzero(as_tuple=True)
        if len(direct_seg_pos[1]) > 0:
            print(f"直接トークン化では成功！位置: {direct_seg_pos[1].tolist()}")
        else:
            print("直接トークン化でも失敗")
    
    print("\n" + "=" * 80)
    print("修正提案")
    print("=" * 80)
    
    print("""
問題と解決策：

1. **Processorのトークナイザー更新が必要**
   - minimal_train.pyでprocessorを作成した後、processorのトークナイザーにもSEGトークンを追加する必要があります
   
2. **SEGトークン埋め込みのrequires_grad設定**
   - set_tokenizerメソッドで設定されるはずが効いていない可能性があります
   - requires_grad_(True)を使用して確実に設定する必要があります

3. **データセットでのトークン化**
   - HybridDatasetでprocessorを使用する際、更新されたトークナイザーを使用する必要があります
""")

if __name__ == "__main__":
    main()