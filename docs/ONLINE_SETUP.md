# 完全オンライン実行ガイド

Python含め、すべてをオンラインで実行するための設定ガイドです。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                        OneDrive                                 │
│  ┌─────────────┐                      ┌─────────────┐          │
│  │ Camera Roll │ ←─── 写真アップロード │   XMPファイル  │          │
│  │  (写真)     │                      │   (自動同期)   │          │
│  └──────┬──────┘                      └───────▲──────┘          │
└─────────┼─────────────────────────────────────┼─────────────────┘
          │                                     │
          │ rclone同期                          │ rclone同期
          ▼                                     │
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Actions                               │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  深夜2時（JST）自動実行 or 手動実行                      │    │
│  │                                                        │    │
│  │  1. OneDriveから写真ダウンロード (rclone)              │    │
│  │  2. 画像を縮小 (Pillow)                                │    │
│  │  3. AI解析 (Google Cloud Vision API)                   │    │
│  │  4. 翻訳 (DeepL API)                                   │    │
│  │  5. XMP生成                                            │    │
│  │  6. OneDriveへアップロード (rclone)                    │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Mylio Photos                               │
│  XMPサイドカーを自動認識 → 日本語検索可能                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## セットアップ手順

### Step 1: リポジトリをFork

1. このリポジトリをGitHubでFork
2. Fork先の Settings → Actions → General で Actions を有効化

### Step 2: Google Cloud Vision API 設定

1. [Google Cloud Console](https://console.cloud.google.com/) にアクセス
2. 新規プロジェクトを作成
3. 「APIとサービス」→「ライブラリ」→「Cloud Vision API」を有効化
4. 「APIとサービス」→「認証情報」→「サービスアカウント作成」
5. JSONキーをダウンロード

### Step 3: DeepL API 設定

1. [DeepL API](https://www.deepl.com/pro-api) でアカウント作成
2. 無料プラン（月50万文字）または有料プランを選択
3. 認証キーを取得

### Step 4: OneDrive (rclone) 設定

ローカルでrcloneを設定してトークンを取得:

```bash
# rcloneインストール
curl https://rclone.org/install.sh | sudo bash

# OneDrive設定（対話式）
rclone config

# 設定内容:
# - name: onedrive
# - type: onedrive
# - 認証フローに従う（ブラウザが開く）
```

設定後、以下のファイルからトークンを取得:
```bash
cat ~/.config/rclone/rclone.conf
```

`token = {...}` の部分をコピー

### Step 5: GitHub Secrets 設定

リポジトリの Settings → Secrets and variables → Actions で以下を追加:

| Secret名 | 内容 |
|----------|------|
| `GOOGLE_CREDENTIALS_JSON` | サービスアカウントJSONの全内容 |
| `DEEPL_API_KEY` | DeepL APIキー |
| `ONEDRIVE_TOKEN` | rcloneで取得したトークン（`{...}`部分） |
| `ONEDRIVE_DRIVE_ID` | OneDriveのドライブID（rclone configに表示） |

### Step 6: ワークフロー設定確認

`.github/workflows/batch-process.yml` を確認:

```yaml
on:
  schedule:
    # 毎日17:00 UTC = 02:00 JST
    - cron: '0 17 * * *'
```

必要に応じて時間を変更

---

## 実行方法

### 自動実行

設定完了後、毎日深夜2時（JST）に自動実行されます。

### 手動実行

1. GitHub → Actions タブ
2. 「Photo Analysis Batch」を選択
3. 「Run workflow」ボタンをクリック
4. オプション:
   - `Force reprocess`: すべての画像を再処理（既存XMPを上書き）

---

## Google Colab でのテスト実行

Colabで即座にテストしたい場合:

```python
# セル1: 依存パッケージインストール
!pip install google-cloud-vision deepl pillow

# セル2: 認証設定
import os

# Google Cloud認証（JSONをアップロード）
from google.colab import files
uploaded = files.upload()  # JSONファイルをアップロード
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = list(uploaded.keys())[0]

# DeepL APIキー
os.environ['DEEPL_API_KEY'] = 'your-api-key-here'

# セル3: テスト画像アップロード
!mkdir -p photos output
uploaded_images = files.upload()  # 画像をアップロード
for name, data in uploaded_images.items():
    with open(f'photos/{name}', 'wb') as f:
        f.write(data)

# セル4: リポジトリクローンと実行
!git clone https://github.com/YOUR_USERNAME/codings.git
%cd codings/ai-photo-analyzer

!python -m src.cloud_runner \
    --input-dir /content/photos \
    --output-dir /content/output

# セル5: 結果ダウンロード
!zip -r /content/xmp_files.zip /content/output
files.download('/content/xmp_files.zip')
```

---

## Power Automate との連携（オプション）

リアルタイム処理が必要な場合、Power Automate + Azure Functions を使用:

### フロー構成

```
[OneDriveトリガー: ファイル作成時]
        ↓
[HTTP: Azure Functions呼び出し]
        ↓
[OneDrive: XMPファイル保存]
```

### Azure Functions 設定

1. Azure Portal で Function App 作成
2. Python ランタイムを選択
3. `src/cloud_runner.py` をベースに HTTP トリガー関数を作成

---

## トラブルシューティング

### GitHub Actions が失敗する

1. **Secrets確認**: すべてのSecretsが正しく設定されているか
2. **ログ確認**: Actions → 失敗したrun → 詳細ログを確認
3. **API制限**: Vision/DeepL APIの使用量制限を確認

### rclone認証エラー

トークンの有効期限切れの可能性:
```bash
# ローカルで再認証
rclone config reconnect onedrive:

# 新しいトークンをGitHub Secretsに更新
```

### XMPがMylio Photosに認識されない

- ファイル名が正確に一致しているか確認
- RAWファイル: `IMG_001.CR2.xmp` 形式
- 通常画像: `IMG_001.xmp` 形式

---

## 料金目安

| サービス | 無料枠 | 超過時 |
|----------|--------|--------|
| GitHub Actions | 2,000分/月 | $0.008/分 |
| Cloud Vision | 1,000回/月 | $1.50/1000回 |
| DeepL | 50万文字/月 | $20/100万文字〜 |
| rclone/OneDrive | 無制限 | - |

月1,000枚程度なら無料枠内で運用可能です。

---

## 関連リンク

- [Google Cloud Vision API ドキュメント](https://cloud.google.com/vision/docs)
- [DeepL API ドキュメント](https://www.deepl.com/docs-api)
- [rclone OneDrive 設定](https://rclone.org/onedrive/)
- [GitHub Actions ドキュメント](https://docs.github.com/actions)
