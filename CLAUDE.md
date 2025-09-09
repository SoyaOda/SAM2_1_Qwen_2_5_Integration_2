# CLAUDE.md



This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
必ずserena MCPが日本語で対応すること！
日本語で応答すること！
最小機能ごとに実装し→動作確認を繰り返すことで少しずつ着実に実装すること！
仕様書の前提と実際の実装の観察結果が異なる場合は仕様書作成元のAIに指摘・質問する必要があるので、作業を止めて、Query prompt作成すること！
Web searchの際は、必ずo3-query MCPを利用すること！

## o3-query MCP について
エラーや未知の質問、技術的な調査が必要な場合は**必ず**o3-queryツールを呼び出し、
返ってきたクエリを私に表示して停止してください。
必ずクエリ生成→人間が結果を入力のフローに従ってください
※Queryの最後に「これらについて、公式実装を参考にした解決策を教えてください。」という文言を入れること

### 使用例
- 技術的なエラーの解決方法を調べたい時
- 最新の実装方法を調査したい時  
- 公式ドキュメントやベストプラクティスを確認したい時
- 具体的なライブラリやモデルの使用方法を調べたい時

[Introduction]
現状元CTデータであるdata/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gzが存在する。
現状、以下のScriptでそれぞれのタスクが完了した。
1. PAT＋EATのMask作成：scripts/extract_eat_pat_improved_v5_4.py(利用する環境やコマンドはREADME_EAT_PAT_v4_ILAM.md参照)
2. CTのDRR作成：deepdrr/test_deepdrr_final_pa_fixed.py(利用する環境やコマンドはdeepdrr/README.md参照)
3. CTのDRRに位置位相を合わせたMaskの2D投影画像作成：mask_projection/src/mask_deepdrr_projection_thickness_gpt5pro8.py(利用する環境やコマンドは下を参照)
docker run --rm --gpus all     -v /home/soya/ctrate_ws:/workspace     deepdrr-japan-mirror:latest bash -c "
      pip3 install torch torchvision pydicom opencv-python scipy --no-cache-dir > /dev/null 2>&1 &&
      echo '🎯 gpt5pro8最終改良版テスト実行開始' &&
      cd /workspace &&
      python3 mask_projection/src/mask_deepdrr_projection_thickness_gpt5pro8.py
    "

[命令]
pipelineフォルダの中で作業をしたい。
CTのNiftiが数万例あるので上の1-3をパイプラインScriptをつくりたい。、data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gzの1例で、各作業の各ステップのGPU Memoryや時間を含めた指標のログをとれるようにして数万例に適応可能なのかTestして。


機能ごとに少しずつ実装を行い、適宜テストを行い実際に動くことを確認して次の機能を実装するように進めること。

もしわからない部分があれば、積極的にAIにクエリするのでプロンプトを作成して。