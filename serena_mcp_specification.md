# Serena MCP 仕様書
## 他プロジェクトでの利用ガイド

## 概要
Serena MCPは、セマンティックなコード検索と編集機能を提供する強力なコーディングエージェントツールキットです。MCP（Model Context Protocol）サーバーとして動作し、Claude CodeやClaude Desktopなどのクライアントから利用できます。

## 主な機能
- **セマンティックコード分析**: LSP（Language Server Protocol）ベースの高度なコード理解
- **シンボル検索・編集**: 関数、クラス、メソッドなどのシンボル単位での操作
- **メモリ管理**: プロジェクト固有の情報を保存・参照
- **多言語対応**: Python, TypeScript/JavaScript, PHP, Go, Rust, C/C++, Java等をサポート

## システム要件
- **OS**: Linux, macOS, Windows (WSL含む)
- **Python**: 3.8以上推奨
- **パッケージマネージャー**: uvまたはpip

## インストール方法

### 方法1: uvxを使用した直接実行（推奨）
```bash
# 最新版を直接実行
uvx --from git+https://github.com/oraios/serena serena start-mcp-server
```

### 方法2: GitHubからローカルインストール
```bash
# リポジトリをクローン
git clone https://github.com/oraios/serena.git
cd serena

# uvを使用してインストール
uv pip install -e .

# または pipを使用
pip install -e .
```

### 方法3: Dockerを使用（実験的）
```bash
docker run --rm -i --network host \
  -v /path/to/your/projects:/workspaces/projects \
  ghcr.io/oraios/serena:latest \
  serena start-mcp-server --transport stdio
```

### 方法4: Nixを使用
```bash
nix run github:oraios/serena -- start-mcp-server --transport stdio
```

## プロジェクト設定

### 1. プロジェクトディレクトリの準備
新しいプロジェクトでSerenaを使用する場合、プロジェクトのルートディレクトリに`.serena`フォルダを作成します。

```bash
cd /path/to/your/project
mkdir .serena
```

### 2. project.yml の作成
`.serena/project.yml`ファイルを作成し、以下の内容を設定します：

```yaml
# プロジェクトの言語を指定
# 対応言語: python, typescript, java, go, rust, cpp, ruby, csharp
language: python

# gitignoreファイルを使用してファイルを無視するか
ignore_all_files_in_gitignore: true

# 追加で無視するパス（gitignore構文）
ignored_paths: []

# 読み取り専用モード（編集ツールを無効化）
read_only: false

# 除外するツール（通常は空のままで良い）
excluded_tools: []

# プロジェクト初期プロンプト
initial_prompt: ""

# プロジェクト名
project_name: "YourProjectName"
```

### 3. メモリファイルの管理（オプション）
プロジェクト固有の情報を保存する場合、`.serena/memories/`ディレクトリを作成：

```bash
mkdir .serena/memories
```

メモリファイルは`.md`形式で保存され、将来の作業で参照できます。

## Claude Codeでの設定

### 1. ユーザーレベル設定（全プロジェクト共通）
`~/.claude/settings.json`または`~/.claude.json`に以下を追加：

```json
{
  "mcpServers": {
    "serena": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/oraios/serena",
        "serena",
        "start-mcp-server"
      ],
      "env": {
        "SERENA_PROJECT_PATH": "/path/to/default/project"
      }
    }
  }
}
```

### 2. プロジェクトレベル設定
プロジェクトディレクトリに`.claude/settings.local.json`を作成：

```json
{
  "mcpServers": {
    "serena": {
      "command": "serena",
      "args": ["start-mcp-server"],
      "cwd": "${projectRoot}",
      "env": {
        "SERENA_PROJECT_PATH": "${projectRoot}"
      }
    }
  }
}
```

## 複数プロジェクトでの利用

### 方法1: プロジェクト切り替え方式
1つのSerenaインスタンスで複数プロジェクトを管理：

```yaml
# グローバル設定で複数プロジェクトを登録
projects:
  - name: "project1"
    path: "/home/user/projects/project1"
  - name: "project2"
    path: "/home/user/projects/project2"
```

`activate_project`ツールを使用してプロジェクトを切り替えます。

### 方法2: 独立インスタンス方式
各プロジェクトで独立したSerenaインスタンスを起動：

```json
{
  "mcpServers": {
    "serena_project1": {
      "command": "serena",
      "args": ["start-mcp-server", "--port", "9121"],
      "cwd": "/path/to/project1"
    },
    "serena_project2": {
      "command": "serena",
      "args": ["start-mcp-server", "--port", "9122"],
      "cwd": "/path/to/project2"
    }
  }
}
```

## 環境変数の利用

設定ファイルで環境変数を使用可能：

```json
{
  "mcpServers": {
    "serena": {
      "command": "${HOME}/.local/bin/serena",
      "args": ["start-mcp-server"],
      "env": {
        "SERENA_API_KEY": "${SERENA_API_KEY}",
        "PROJECT_PATH": "${PWD}"
      }
    }
  }
}
```

## トラブルシューティング

### Serenaが起動しない場合
1. Pythonバージョンを確認（3.8以上）
2. 依存関係をインストール：`pip install uv`
3. ログを確認：`serena start-mcp-server --log-level debug`

### プロジェクトが認識されない場合
1. `.serena/project.yml`が存在することを確認
2. `language`フィールドが正しく設定されていることを確認
3. プロジェクトパスが正しいことを確認

### メモリが読み込まれない場合
1. `.serena/memories/`ディレクトリが存在することを確認
2. メモリファイルが`.md`拡張子であることを確認
3. ファイル権限を確認

## ベストプラクティス

1. **プロジェクト構造の維持**
   - `.serena`フォルダはgitで管理（秘密情報を含まない限り）
   - メモリファイルは必要に応じて共有

2. **パフォーマンス最適化**
   - 不要なファイルは`ignored_paths`で除外
   - 大規模プロジェクトでは`read_only: true`を検討

3. **セキュリティ**
   - APIキーは環境変数で管理
   - `.claude/settings.local.json`はgitignoreに追加

4. **チーム開発**
   - `.serena/project.yml`は共有
   - 個人設定は`settings.local.json`に記載

## 高度な設定

### SSEモードでの起動
```bash
serena start-mcp-server --transport sse --port 9121
```

クライアント側で`http://localhost:9121/sse`に接続します。

### カスタムツール設定
`excluded_tools`でツールを無効化：

```yaml
excluded_tools:
  - execute_shell_command  # シェルコマンド実行を無効化
  - delete_lines          # 行削除を無効化
```

### Web管理画面
デフォルトでlocalhostに管理画面が起動します。ログ表示やサーバー停止が可能です。

## リファレンス

- **公式リポジトリ**: https://github.com/oraios/serena
- **MCP仕様**: https://modelcontextprotocol.io/
- **Claude Code ドキュメント**: https://docs.anthropic.com/en/docs/claude-code/mcp

## サポート言語一覧

| 言語 | 設定値 | LSPサポート |
|------|--------|------------|
| Python | python | ✅ フル対応 |
| TypeScript/JavaScript | typescript | ✅ フル対応 |
| Java | java | ✅ フル対応 |
| Go | go | ✅ フル対応 |
| Rust | rust | ✅ フル対応 |
| C/C++ | cpp | ✅ フル対応 |
| Ruby | ruby | ✅ フル対応 |
| C# | csharp | ✅ フル対応（要.slnファイル）|
| PHP | php | ⚠️ 部分対応 |

## 更新履歴
- 2025年9月: 最新版対応、Docker/Nixサポート追加
- 2025年8月: SSEモード、Web管理画面追加
- 2025年7月: 初期リリース