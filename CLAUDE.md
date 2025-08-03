# CLAUDE.md



This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
serena MCPが対応すること！
日本語で応答すること！
Webリサーチはo3をタイムアウトの設定なしで用いること！

[Introduction]
将来的な目標：LISA→FoodLMMの成功を受けて、最新のVLMと最新のSAMを統合して、LISAのような深い次元で画像と言語を理解した基盤モデル（LISA改）を作り、それをFoodLMMの学習方法を参考にファインチューニングし、FoodLMM改を作り、写真内の料理や食材の量の推定を精度高く行わせる予定。VLMとSAMのようなモデルの統合モデルについて、最新の知見も導入しながらポテンシャルの高いモデル実装を目指している。

現状：md_files/current/o3_spec1.mdを元にLISA改を段階的に実装していきたい。

[命令]
md_files/current/o3_spec1.mdと現状のコードを理解して、段階的に実装していきたい。

※実装の際に注意すること
・自信のない部分は適宜正規（Qwen, SAM, Huggingface, Pytorchなど公式の実装）の実装をWebでしらべながら予想や自前の実装を少なくして実装すること
・o3_spec1.mdはあくまでおおまかな指針であるので、細かな実装はWebでベストプラクティスをリサーチして[Introduction]に述べている目的に沿うように、本質的に実装を進めること（簡易な実装でとりあえず走るコードは必要ない、本質的に目標を達成するコードが欲しい）
・Webリサーチを積極的に行い、Qwen, SAM, Huggingface, Peftの正規の実装をできるだけ用いること
・フォールバック的もしくはダミーコードはエラーを隠蔽するので適切にエラーを出して止め、次のデバッグに繋がる情報を提供するように修正すること
・解決した問題に関するデバッグログや必要のないデバッグログは削除、デバッグ作業により解決した部分はシンプル（分岐のない確定的な実装）にして、コードをシンプルに保つようにすること