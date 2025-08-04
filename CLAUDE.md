# CLAUDE.md



This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
serena MCPが日本語で対応すること！
日本語で応答すること！
Web searchの際は、必ずo3-query MCPを利用すること！

## o3-query MCP について
エラーや未知の質問、技術的な調査が必要な場合は**必ず**o3-queryツールを呼び出し、
返ってきたクエリを私に表示して停止してください。
必ずクエリ生成→人間が結果を入力のフローに従ってください。

### 使用例
- 技術的なエラーの解決方法を調べたい時
- 最新の実装方法を調査したい時  
- 公式ドキュメントやベストプラクティスを確認したい時
- 具体的なライブラリやモデルの使用方法を調べたい時

[Introduction]
将来的な目標：LISA→FoodLMMの成功を受けて、最新のVLMと最新のSAMを統合して、LISAのような深い次元で画像と言語を理解した基盤モデル（LISA改）を作り、それをFoodLMMの学習方法を参考にファインチューニングし、FoodLMM改を作り、写真内の料理や食材の量の推定を精度高く行わせる予定。VLMとSAMのようなモデルの統合モデルについて、最新の知見も導入しながらポテンシャルの高いモデル実装を目指している。



[命令]
①事前タスク：以下のファイルをよく読んで現状を理解して
現状実装が完了した指針
・md_files/past/o3_spec1.md
・md_files/past/high_res_features_spec.md

実装レコード
・md_files/implementation_record_20250804.md

走ることが確認できているTest Script（必要があれば実際に実行してログを確認すること）
・tests/training_integration_test.py

今回参照したい指針
・md_files/current/o3_spec2.md

②本番タスク
大まかな指針としてはo3_spec2.mdの指針を参考に実装を進めてほしい。ただし、reference_filesにはほかの統合モデル（Gemma＋SAM）のプロジェクトのDataset関連のScriptsを格納しているが、本プロジェクトと全く同様のDatasetを用いているので細かいデータへのアクセスや前処理の仕方はreference_filesの方が参考になるはず。


※実装の際に注意すること
・自信のない部分は適宜正規（Qwen2.5-VL(3B), SAM2.1, Huggingface, Pytorchなど公式の実装）の実装をWebでしらべながら予想や自前の実装を少なくして実装すること
・o3_spec1.mdはあくまでおおまかな指針であるので、細かな実装はWebでベストプラクティスをリサーチして[Introduction]に述べている目的に沿うように、本質的に実装を進めること（簡易な実装でとりあえず走るコードは必要ない、本質的に目標を達成するコードが欲しい）
・Webリサーチを積極的に行い、Qwen, SAM, Huggingface, Peftの正規の実装をできるだけ用いること
・フォールバック的もしくはダミーコードはエラーを隠蔽するので適切にエラーを出して止め、次のデバッグに繋がる情報を提供するように修正すること
・解決した問題に関するデバッグログや必要のないデバッグログは削除、デバッグ作業により解決した部分はシンプル（分岐のない確定的な実装）にして、コードをシンプルに保つようにすること