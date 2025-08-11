# CLAUDE.md



This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
serena MCPが日本語で対応すること！
日本語で応答すること！
Web searchの際は、必ずo3-query MCPを利用すること！

## o3-query MCP について
エラーや未知の質問、技術的な調査が必要な場合は**必ず**o3-queryツールを呼び出し、
返ってきたクエリを私に表示して停止してください。
必ずクエリ生成→人間が結果を入力のフローに従ってください
※Queryの最後に「これらについて、Qwen2.5-VL、SAM2.1、LISA等の公式実装を参考にした解決策を教えてください。」という文言を入れること

### 使用例
- 技術的なエラーの解決方法を調べたい時
- 最新の実装方法を調査したい時  
- 公式ドキュメントやベストプラクティスを確認したい時
- 具体的なライブラリやモデルの使用方法を調べたい時

[Introduction]
将来的な目標：LISA→FoodLMMの成功を受けて、最新のVLMと最新のSAMを統合して、LISAのような深い次元で画像と言語を理解した基盤モデル（LISA改）を作り、それをFoodLMMの学習方法を参考にファインチューニングし、FoodLMM改を作り、写真内の料理や食材の量の推定を精度高く行わせる予定。VLMとSAMのようなモデルの統合モデルについて、最新の知見も導入しながらポテンシャルの高いモデル実装を目指している。

今までの過去の実装方針・実際の実装ログ（実装指針に関しては一部未実装もあるかも、実装更新もありうる、いずれにせよ現在の実際のコードScript内容が一番信頼性が高い）はmd_filesにまとまってあるので、おおまかな実装の把握に役立てること。詳細部分は実際にコード（minimal_train.pyやその関連Script）で把握すること。

[命令]
〇事前タスク：minimal_train.py, test_a1_oracle_sam_fixed.pyとその関連Scriptをすべてよく読んで（長い場合は分割しながら読んで）現状の実装を理解して。そのあと、学習データセットの生データからどのようにデータが処理されていくか、入力データとモデル・ラベルデータの整合性を意識して把握して。

〇本タスク：minimal_train.pyを以下のコマンドを実行しているが、seg lossが全く下がらずマスク生成も全く向上しない（参照：outputs/minimal_train_20250810_191150）。原因究明のため、テスト用にtest_a1_oracle_sam_fixed.pyを作って実行するときちんとseg lossの現象と精度の高いMask生成が確認できている（参照：test_outputs/test_a1_20250811_102514）。

minimal_train.pyの実行コマンド：python minimal_train.py --dataset_types "sem_seg" --samples_per_epoch 10000 --batch_size 2 --gradient_accumulation_steps 16 --num_epochs 4 --seg_loss_weight 1.0 --adapter_lr 5e-3 --seg_token_lr 1e-3 --lora_lr 1.5e-4 --save_steps 2500 --use_wandb --wandb_project minimal_train_full-semseg

test_a1_oracle_sam_fixed.pyの実行コマンド：python test_a1_oracle_sam_fixed.py

そこでGTP-5proにより現状のレビューとネクストステップとしてのTest Scriptの実装方針をmd_files/current/Focus_Seg_Loss20250811_spec.mdに作成したので、参考にして実装を行って！

※作業の途中でminimal_train.pyでseg lossが下がらない理由の可能性のあるバグを発見したら報告すること。

※実装の際に注意すること
・自信のない部分は適宜正規（Qwen, SAM, Huggingface, Pytorchなど公式の実装）の実装をWebでしらべながら予想や自前の実装を少なくして実装すること
・Webリサーチを積極的に行い、Qwen, SAMの正規の実装をできるだけ用いること
・フォールバック的もしくはダミーコードはエラーを隠蔽するので適切にエラーを出して止め、次のデバッグに繋がる情報を提供するように修正すること
・解決した問題に関するデバッグログや必要のないデバッグログは削除、デバッグ作業により解決した部分はシンプル（分岐のない確定的な実装）にして、コードをシンプルに保つようにすること