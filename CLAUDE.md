
小田惣也 <odssuu@gmail.com>
16:22 (0 分前)
To 自分

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



[命令]
〇minimal_train.pyとその関連Scriptをすべてよく読んで（長い場合は分割しながら読んで）現状の実装を理解して

〇minimal_train.pyを進めた場合のSEG LOSSがほぼ下がらない問題がある。現状、SEGトークンがo3_spec1-3のいずれかのMDファイルにあるように適切に定義され、モデルに組み込み、学習が進むようになっているかコードをよく読んで必要があればデバッグScriptやデバッグログも使用しつつ確かめてほしい。


