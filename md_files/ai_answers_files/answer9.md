調査結果
minimal_train.py の orig_hw 取得方法について

現在の minimal_train.py 内で save_visualization() を呼び出す際に使用している orig_h, orig_w（元画像サイズ） は、データセットおよび collate 関数からバッチに渡されています
GitHub
。具体的には、HybridDataset で各サンプルに対して original_image（PIL形式の元画像）が含まれており、MultiModalDataCollator（もしくは collate_fn）によってそれらが batch['original_images'] リストにまとめられます
GitHub
。save_visualization() 関数ではまずバッチ内に original_images が存在するかチェックし、存在すればその最初の画像から orig_h, orig_w を算出します
GitHub
。つまり 元画像の高さ・幅情報は HybridDataset→collate_fn を通じて伝搬 されており、可視化時に別途ファイルから読み込むことはしていません（データローダから受け取った情報をそのまま使用しています）。なお、万一 original_images がバッチに無い場合はフォールバックとして、正規化解除した pixel_values のテンソルから画像を再構成し、そのサイズを近似的に元画像サイズとして使用しています
GitHub
（ただしこれはあくまで近似であり、可能な限り original_image を提供して正確なサイズを保持しています）。

Qwenのimage_grid_thw/Token-FPN出力サイズとSAM出力(1024x1024)の対応

Qwen2.5-VL 側の視覚特徴は動的解像度に対応しており、入力画像の大きさに応じてトークン数（パッチ数）が変化します。そのため、Qwenのビジョンエンコーダ出力をToken-FPNでマルチスケール特徴に変換する際には、まず image_grid_thw（各サンプルの元パッチグリッドサイズ情報）を使って1次元トークン列を対応する高さH_raw×幅W_rawの2D特徴マップに再構成します
GitHub
。再構成後、Token-FPNはFPN構造で4つのスケールの特徴を生成し、SAMの画像エンコーダ出力と互換性のある解像度に調整します
GitHub
。具体的には、SAM2.1 の画像エンコーダが 1024×1024入力に対して 64×64 の特徴マップを出力するため、Token-FPNの最も高解像度の出力（P0）を取得し、そのサイズが 64×64 でない場合は 双線形補間（torch.nn.functional.interpolate）で 64×64 にリサイズしています
GitHub
。このようにして、Qwen側の特徴マップをSAM側の期待する空間解像度に合わせる処理を行っています。なお、Token-FPN内ではその後ストライド8（2倍）・ストライド4（4倍）のアップサンプル特徴も計算し、SAMのMask Decoderが必要とする複数解像度の特徴（High-Res Feature）として提供しています
GitHub
。以上により、動的解像度のQwen特徴と固定解像度のSAM特徴とのサイズ対応付けは、Token-FPN内の補間処理によって適切に行われています。

可視化出力時のGTマスクと予測マスクの解像度差について

Visualizer（save_visualization）の出力を見ると、一見GTマスクと予測マスクのサイズが異なるように見える場合があります。しかし、コード上ではどちらのマスクも最終的に元画像の解像度に揃えられる設計になっています。sam_postprocess_mask 呼び出し時に渡している orig_size=(orig_h, orig_w) は、前述の通り SAM2.1に入力する前の元画像のサイズ（高さ・幅）です
GitHub
。この値はHybridDatasetで保持していた元画像の解像度であり、後処理で補完的に推測しているものではありません。実際、予測マスク（pred_mask）はまず sam_postprocess_mask 関数で右下パディング部分を除去し、元画像サイズ (orig_h, orig_w) にリサイズされます
GitHub
。その結果得られたポストプロセス済み予測マスクは orig_h×orig_w のサイズとなり、GTマスクについても同じサイズにリサイズして比較しています
GitHub
。コードでは、GTマスク (batch['mask_labels']) の numpy 配列を取得した後、そのシェイプが (orig_h, orig_w) と異なる場合に cv2.resize（最近傍補間）で orig_h×orig_w に拡大・縮小しています
GitHub
。このように、可視化時にはGTマスク・予測マスクともに元の画像解像度に揃えてから描画しており、本来であればサイズ不一致は起こらないはずです。もし現状の実装で不一致が見られる場合、それはおそらく初期の実装上の問題か、可視化用データ取得のタイミングに起因する可能性がありますが、基本的な設計としては orig_hw（元画像サイズ）を保持しそれを使って両マスクを同一解像度にしています。

SAM2のマスク出力（低解像度）を可視化する際のアップサンプリング方法

SAM2.1（Hiera-Large）のMask Decoderは内部で256×256の低解像度マスクを推論し、それをモデル内部で4倍アップサンプルして1024×1024の出力マスク（入力画像と同じパディング込み解像度）を得ています
GitHub
。本実装では、モデルから得られる outputs.mask_logits は 基本的に [1, 1024, 1024]（またはマルチマスク時 [M, 1024, 1024]）のテンソルとして扱っています
GitHub
。可視化の際には、このマスクテンソルに対し独自に用意した後処理関数 sam_postprocess_mask を適用しています。sam_postprocess_mask 内ではまずSAM標準の右下パディングを考慮して不要なパディング部分を切り取り、その後**torch.nn.functional.interpolate（双線形補間）によって元の画像サイズ (orig_hw) に拡大しています
GitHub
。したがって、SAM2の低解像度マスクを可視化用にアップサンプリングする処理は、F.interpolate を用いて実現しています（公式の SamPredictor などにある postprocess_masks 相当の処理を自前で実装しています
GitHub
）。この処理により、256×256 → 1024×1024（モデル内部）→ 元画像サイズという段階を経て、最終的にオリジナル解像度に復元されたマスク**が得られます
GitHub
。なお、当コードでは SAM2Transforms 等のクラスは直接使用しておらず、すべて自前の sam_postprocess_mask 関数で完結しています
GitHub
。以上のように、可視化時にはF.interpolateによるアップサンプリングとパディング除去でマスク解像度を元画像に揃える形になっています。
