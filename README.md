# AI Photo Analysis System

Google Photosを超える写真分析・検索システム。OneDriveにアップロードされた写真を自動的にAI解析し、日本語タグ付きのXMPメタデータを生成してMylio Photosでの検索を可能にします。

## 特徴

- **軽量な日中処理**: 写真アップロード時に縮小コピーのみ作成（通信量・バッテリー消費を抑制）
- **深夜バッチ処理**: 深夜の安定した時間帯にAI解析を一括実行
- **高精度ラベリング**: Google Cloud Vision APIによる物体認識・顔検出・ランドマーク検出
- **日本語対応**: DeepL APIによる高品質な翻訳で日本語検索を実現
- **Mylio Photos連携**: XMPサイドカーファイルでシームレスなメタデータ統合
- **RAW画像対応**: CR2, NEF, ARW等のRAW画像を自動検出し、ペアのJPEGを解析してXMPを生成
- **完全オンライン実行**: GitHub Actions / Google Colab でPython環境不要で実行可能

## 実行方法

### 🌐 オンライン実行（推奨）

Pythonのローカルインストール不要で実行できます。

| 方法 | 用途 | リンク |
|------|------|--------|
| **Google Colab** | 手動で数枚〜数十枚処理 | [Colabで開く](notebooks/photo_analyzer_colab.ipynb) |
| **GitHub Actions** | 毎日自動で定期処理 | [設定ガイド](docs/ONLINE_SETUP.md) |

#### Google Colab クイックスタート

1. [Colabノートブック](notebooks/photo_analyzer_colab.ipynb)を開く
2. Google Cloud認証JSON をアップロード
3. DeepL APIキーを入力
4. 写真をアップロードして実行
5. XMPファイルをダウンロード

#### GitHub Actions 自動実行

1. リポジトリをFork
2. Secretsを設定（詳細は[オンライン設定ガイド](docs/ONLINE_SETUP.md)）
3. 毎日深夜2時（JST）に自動実行

### 💻 ローカル実行

従来通りローカルでも実行可能です。詳細は[インストール](#インストール)セクションを参照。

## システムフロー

```
【ステップA：日中（即時・軽量処理）】
1. OneDriveに写真がアップロードされる
2. 自動検知して縮小コピー（短辺640px）を作成
3. 一時フォルダに _shrink ファイルとして保存

【ステップB：深夜（バッチ処理）】
1. スケジュール起動（デフォルト: 午前2時）
2. _shrink ファイルをGoogle Cloud Vision APIで解析
3. 英語ラベルをDeepL APIで日本語に翻訳
4. XMPメタデータファイルを生成（オリジナルと同じフォルダに保存）
5. _shrink ファイルを削除

【Mylio Photos連携】
- IMG_001.jpg と IMG_001.xmp を自動ペアリング
- 日本語タグで検索可能

【RAW画像処理フロー】
1. RAW画像（例：IMG_001.CR2）を検出
2. 同名のJPEG（IMG_001.JPG）を探索
3. JPEGを縮小してAI解析
4. RAW用XMP（IMG_001.CR2.xmp）を生成
   - 「RAW」「RAW画像」「CR2」等のラベルを自動付加
5. Mylio PhotosでRAW画像として認識・検索可能
```

## RAW画像サポート

### 対応RAWフォーマット

| メーカー | 拡張子 |
|---------|--------|
| Canon | .CR2, .CR3, .CRW |
| Nikon | .NEF, .NRW |
| Sony | .ARW, .SRF, .SR2 |
| Fujifilm | .RAF |
| Olympus/OM System | .ORF |
| Panasonic | .RW2 |
| Pentax | .PEF, .DNG |
| Leica | .DNG, .RWL |
| Samsung | .SRW |
| Sigma | .X3F |
| Adobe DNG | .DNG |

### RAW+JPEG撮影時の動作

RAW+JPEGで撮影した場合（例：`IMG_001.CR2` + `IMG_001.JPG`）：

1. システムがRAWファイルを検出
2. 同名のJPEGファイルを自動で探索
3. JPEGを解析用に使用（RAWは大きすぎるため）
4. 解析結果をRAW用XMPとして保存

```
撮影ファイル:
├── IMG_001.CR2    (RAW)
└── IMG_001.JPG    (JPEG)

生成されるXMP:
└── IMG_001.CR2.xmp  ← RAW画像用メタデータ
```

### RAW専用ラベル

RAW画像には以下のラベルが自動的に追加されます：

- 日本語: `RAW`, `RAW画像`, `生データ`, `CR2` (拡張子)
- 英語: `RAW`, `RAW Image`, `Unprocessed`, `CR2`

これにより、Mylio Photosで「RAW」と検索するとRAW画像のみを表示できます。

### 設定

`config/settings.yaml` でRAW処理を設定：

```yaml
raw_processing:
  enabled: true
  raw_extensions:
    - ".cr2"
    - ".nef"
    - ".arw"
    # ... 他の拡張子
  analysis_source_priority:
    - ".jpg"
    - ".jpeg"
    - ".png"
  raw_labels:
    japanese:
      - "RAW"
      - "RAW画像"
    english:
      - "RAW"
      - "RAW Image"
```

## インストール

### 1. 依存パッケージのインストール

```bash
cd ai-photo-analyzer
pip install -r requirements.txt
```

### 2. API設定

`.env.example` を `.env` にコピーして設定：

```bash
cp .env.example .env
```

#### Google Cloud Vision API

1. [Google Cloud Console](https://console.cloud.google.com/) でプロジェクトを作成
2. Cloud Vision APIを有効化
3. サービスアカウントを作成してJSONキーをダウンロード
4. `.env` に設定：
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   ```

#### DeepL API

1. [DeepL API](https://www.deepl.com/pro-api) でアカウントを作成
2. 認証キーを取得
3. `.env` に設定：
   ```
   DEEPL_API_KEY=your-deepl-api-key
   ```

#### OneDriveパス

```
ONEDRIVE_LOCAL_PATH=/path/to/OneDrive
```

### 3. 設定ファイルのカスタマイズ（任意）

`config/settings.yaml` で各種設定を変更できます：

```yaml
# バッチ処理時間
scheduler:
  batch_start_hour: 2    # 午前2時
  batch_start_minute: 0
  timezone: "Asia/Tokyo"

# Vision API設定
vision_api:
  max_labels: 20
  min_confidence: 0.7
```

## 使い方

### サービスとして実行（推奨）

```bash
python -m src.main
```

バックグラウンドで実行する場合：

```bash
nohup python -m src.main > /dev/null 2>&1 &
```

### コマンドラインオプション

```bash
# 既存写真のスキャンとshrink画像作成
python -m src.main --scan

# バッチ処理を即時実行
python -m src.main --process

# 現在のステータスを表示
python -m src.main --status

# 即時処理モードで起動（新規写真を即座にAI解析）
python -m src.main --immediate

# デバッグログを有効化
python -m src.main --debug
```

## プロジェクト構成

```
ai-photo-analyzer/
├── config/
│   └── settings.yaml      # 設定ファイル
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py            # メインアプリケーション
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_processor.py    # 画像縮小処理
│   │   ├── file_watcher.py       # ファイル監視
│   │   ├── vision_service.py     # Google Vision API
│   │   ├── translation_service.py # DeepL翻訳
│   │   ├── xmp_generator.py      # XMPファイル生成
│   │   ├── batch_processor.py    # バッチ処理スケジューラ
│   │   └── raw_processor.py      # RAW画像処理
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # 設定管理
│       └── logger.py             # ログ設定
├── temp/                  # 一時ファイル（shrink画像）
├── logs/                  # ログファイル
├── .env.example          # 環境変数テンプレート
├── requirements.txt      # 依存パッケージ
└── README.md
```

## 生成されるXMPファイルの例

```xml
<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="AI Photo Analyzer">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:lr="http://ns.adobe.com/lightroom/1.0/">
      <dc:subject>
        <rdf:Bag>
          <rdf:li>空</rdf:li>
          <rdf:li>雲</rdf:li>
          <rdf:li>山</rdf:li>
          <rdf:li>風景</rdf:li>
          <rdf:li>Sky</rdf:li>
          <rdf:li>Cloud</rdf:li>
          <rdf:li>Mountain</rdf:li>
          <rdf:li>Landscape</rdf:li>
        </rdf:Bag>
      </dc:subject>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
```

## Mylio Photosでの設定

1. Mylio Photosを起動
2. ソースとしてOneDriveフォルダを追加
3. XMPサイドカーファイルの読み込みを有効化（デフォルトで有効）
4. 同期後、日本語キーワードで検索可能

## トラブルシューティング

### Vision APIエラー

- `GOOGLE_APPLICATION_CREDENTIALS` が正しく設定されているか確認
- サービスアカウントに適切な権限があるか確認
- APIが有効化されているか確認

### DeepL翻訳エラー

- APIキーが正しいか確認
- 使用量制限に達していないか確認

### XMPが認識されない

- ファイル名が完全に一致しているか確認（拡張子以外）
- Mylio Photosでライブラリを再同期

### ログの確認

```bash
tail -f logs/photo_analyzer.log
```

## 注意事項

- Vision APIとDeepL APIには利用料金が発生する場合があります
- 大量の写真を処理する場合は、各APIの料金プランを確認してください
- 深夜のバッチ処理時間はシステム負荷を考慮して設定してください

## ライセンス

MIT License
